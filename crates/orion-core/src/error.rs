//! Error architecture for OrionServe-RS.
//!
//! The engine distinguishes three broad failure classes, because each one has a
//! different correct response at the API boundary:
//!
//! * [`ConfigError`] — the operator misconfigured the server. Fatal at startup.
//! * [`ModelError`]  — a model artifact is missing, malformed, or unsupported.
//!   Fatal at load time, never at request time.
//! * [`EngineError`] — something went wrong while serving a request. Some
//!   variants are caused by the caller (4xx), some by us (5xx), and some are
//!   transient capacity signals that the caller should retry.
//!
//! Keeping them separate means the HTTP layer can map failures to status codes
//! without string matching, and per-request code paths cannot accidentally
//! return a "model file corrupt" error.

use std::path::PathBuf;

/// Errors raised while validating operator-supplied configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("invalid value for `{field}`: {reason}")]
    InvalidValue { field: &'static str, reason: String },

    #[error("`{field}` must be a power of two, got {value}")]
    NotPowerOfTwo { field: &'static str, value: usize },

    #[error("`{field}` must be in {min}..={max}, got {value}")]
    OutOfRange {
        field: &'static str,
        min: u64,
        max: u64,
        value: u64,
    },

    #[error("configuration is internally inconsistent: {0}")]
    Inconsistent(String),
}

/// Errors raised while loading or validating a model artifact.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model path does not exist: {0}")]
    PathNotFound(PathBuf),

    #[error("required model file missing: {0}")]
    MissingFile(PathBuf),

    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("malformed {file}: {reason}")]
    Malformed { file: String, reason: String },

    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("unsupported dtype `{0}` for this backend")]
    UnsupportedDtype(String),

    #[error("tensor `{name}` has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("tensor `{0}` not found in checkpoint")]
    TensorNotFound(String),
}

/// Errors raised while serving an inference request.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The request itself is invalid. Maps to HTTP 400.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The prompt does not fit in the model context window. HTTP 400.
    #[error(
        "prompt of {prompt_tokens} tokens plus {max_tokens} requested output \
         exceeds context length {context_len}"
    )]
    ContextLengthExceeded {
        prompt_tokens: usize,
        max_tokens: usize,
        context_len: usize,
    },

    /// The admission queue is full. HTTP 429 — the caller should retry.
    #[error("server at capacity: {queued} requests already waiting (limit {limit})")]
    QueueFull { queued: usize, limit: usize },

    /// KV cache is exhausted and no request could be preempted to make room.
    /// HTTP 503.
    #[error("KV cache exhausted: needed {needed} blocks, {available} free")]
    CacheExhausted { needed: usize, available: usize },

    /// The request exceeded its deadline while queued or running. HTTP 504.
    #[error("request timed out after {elapsed_ms}ms in state `{state}`")]
    Timeout {
        elapsed_ms: u64,
        state: &'static str,
    },

    /// The client went away, or the request was explicitly aborted.
    #[error("request cancelled")]
    Cancelled,

    /// Tokenizer failure: encoding or decoding produced an error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A compute backend failed. Will carry CUDA errors once that backend lands.
    #[error("backend `{backend}` failed during {stage}: {reason}")]
    Backend {
        backend: &'static str,
        stage: &'static str,
        reason: String,
    },

    /// The engine worker loop is gone. Every in-flight request fails this way.
    #[error("inference engine has shut down")]
    EngineShutdown,

    /// An invariant the engine relies on was violated. Always a bug in
    /// OrionServe, never the caller's fault: logged at ERROR and surfaced as an
    /// opaque 500 so internal state never leaks to clients.
    #[error("internal invariant violated: {0}")]
    Internal(String),
}

impl EngineError {
    /// Whether the caller may usefully retry this request unchanged.
    ///
    /// Used by the API layer to decide between `429`/`503` (with `Retry-After`)
    /// and a terminal `4xx`/`5xx`.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            EngineError::QueueFull { .. } | EngineError::CacheExhausted { .. }
        )
    }

    /// Whether this failure was caused by the caller's input rather than by
    /// server state.
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            EngineError::InvalidRequest(_)
                | EngineError::ContextLengthExceeded { .. }
                | EngineError::Cancelled
        )
    }

    /// A short, stable, machine-readable code. Exposed in API error bodies so
    /// clients can branch on failures without parsing prose.
    pub fn code(&self) -> &'static str {
        match self {
            EngineError::InvalidRequest(_) => "invalid_request",
            EngineError::ContextLengthExceeded { .. } => "context_length_exceeded",
            EngineError::QueueFull { .. } => "queue_full",
            EngineError::CacheExhausted { .. } => "cache_exhausted",
            EngineError::Timeout { .. } => "timeout",
            EngineError::Cancelled => "cancelled",
            EngineError::Tokenizer(_) => "tokenizer_error",
            EngineError::Backend { .. } => "backend_error",
            EngineError::EngineShutdown => "engine_shutdown",
            EngineError::Internal(_) => "internal_error",
        }
    }
}

/// Top-level error type for the engine binary, spanning startup and serving.
#[derive(Debug, thiserror::Error)]
pub enum OrionError {
    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Engine(#[from] EngineError),
}

/// Convenience alias used throughout the workspace.
pub type Result<T, E = EngineError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_errors_are_retryable() {
        assert!(EngineError::QueueFull {
            queued: 10,
            limit: 10
        }
        .is_retryable());
        assert!(EngineError::CacheExhausted {
            needed: 4,
            available: 0
        }
        .is_retryable());
        assert!(!EngineError::Internal("boom".into()).is_retryable());
    }

    #[test]
    fn client_errors_are_classified_separately_from_retryable_ones() {
        let e = EngineError::ContextLengthExceeded {
            prompt_tokens: 5000,
            max_tokens: 200,
            context_len: 4096,
        };
        assert!(e.is_client_error());
        assert!(!e.is_retryable());
    }

    #[test]
    fn error_codes_are_distinct() {
        let codes = [
            EngineError::InvalidRequest(String::new()).code(),
            EngineError::QueueFull {
                queued: 0,
                limit: 0,
            }
            .code(),
            EngineError::Cancelled.code(),
            EngineError::EngineShutdown.code(),
        ];
        let unique: std::collections::HashSet<_> = codes.iter().collect();
        assert_eq!(unique.len(), codes.len());
    }

    #[test]
    fn internal_errors_are_neither_client_nor_retryable() {
        let e = EngineError::Internal("block table desynced".into());
        assert!(!e.is_client_error());
        assert!(!e.is_retryable());
    }
}
