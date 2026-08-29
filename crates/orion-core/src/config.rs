//! Engine configuration and its validation.
//!
//! Every knob an operator can turn lives here, with a default that is safe on
//! a small machine. Validation is total: [`EngineConfig::validate`] is the one
//! place that decides whether a configuration is legal, and it runs before any
//! memory is reserved. Failing at startup with a precise message is much
//! cheaper than discovering an inconsistency mid-flight.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// KV cache sizing and layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    /// Tokens per KV block.
    ///
    /// The central tradeoff of a paged cache: smaller blocks waste less on the
    /// partially-filled last block of each sequence, but multiply block-table
    /// metadata and per-block indirection. 16 is the value `docs/kv-cache.md`
    /// argues for.
    pub block_size: usize,

    /// Fraction of device memory the cache pool may claim, after weights.
    /// Only consulted when `num_blocks` is `None`.
    pub gpu_memory_utilization: f32,

    /// Explicit block count, overriding the memory-fraction heuristic.
    /// Required on backends that cannot report free memory.
    pub num_blocks: Option<usize>,

    /// Enable automatic prefix caching: reuse blocks whose token contents
    /// match a previously computed prefix.
    pub enable_prefix_caching: bool,

    /// Blocks to keep in the CPU swap pool for preempted sequences.
    /// `0` disables swapping, making recompute the only preemption strategy.
    pub num_cpu_swap_blocks: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            gpu_memory_utilization: 0.90,
            num_blocks: None,
            enable_prefix_caching: true,
            num_cpu_swap_blocks: 0,
        }
    }
}

impl CacheConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.block_size == 0 || !self.block_size.is_power_of_two() {
            return Err(ConfigError::NotPowerOfTwo {
                field: "cache.block_size",
                value: self.block_size,
            });
        }
        // Above 256 the internal fragmentation of the tail block dominates for
        // short sequences; below 4 the block table becomes larger than the data
        // it indexes.
        if !(4..=256).contains(&self.block_size) {
            return Err(ConfigError::OutOfRange {
                field: "cache.block_size",
                min: 4,
                max: 256,
                value: self.block_size as u64,
            });
        }
        if !self.gpu_memory_utilization.is_finite()
            || self.gpu_memory_utilization <= 0.0
            || self.gpu_memory_utilization > 1.0
        {
            return Err(ConfigError::InvalidValue {
                field: "cache.gpu_memory_utilization",
                reason: format!("must be in (0.0, 1.0], got {}", self.gpu_memory_utilization),
            });
        }
        if let Some(0) = self.num_blocks {
            return Err(ConfigError::InvalidValue {
                field: "cache.num_blocks",
                reason: "must be at least 1 when set explicitly".into(),
            });
        }
        Ok(())
    }
}

/// How preempted sequences give up their cache blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreemptionMode {
    /// Drop the blocks and recompute the prefill on resumption. No copy cost,
    /// but pays the prefill again.
    Recompute,
    /// Copy blocks to host memory and back. Trades PCIe bandwidth for compute;
    /// only wins when the prompt is long relative to transfer cost.
    Swap,
}

/// Scheduler policy and batch budgets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Maximum sequences resident in the running set.
    pub max_num_seqs: usize,

    /// Token budget for one engine step, across prefill and decode.
    ///
    /// This is the primary throughput/latency dial: it bounds how much work a
    /// single step can do, and therefore how long a decode-only sequence can be
    /// stalled behind a large prefill.
    pub max_num_batched_tokens: usize,

    /// Longest prompt+output the engine will accept. Defaults to the model's
    /// trained context length when `None`.
    pub max_model_len: Option<usize>,

    /// Split long prefills into chunks that fit the token budget, interleaving
    /// them with decodes rather than monopolizing a step.
    pub enable_chunked_prefill: bool,

    /// Requests allowed to wait for admission before the engine sheds load.
    pub max_waiting_requests: usize,

    /// How preemption reclaims blocks.
    pub preemption_mode: PreemptionMode,

    /// Wall-clock budget for a request, queueing included. `None` means the
    /// only limit is the client's own timeout.
    pub request_timeout_secs: Option<u64>,

    /// Bias toward decode over prefill when both are eligible.
    ///
    /// Scheduling decode first keeps inter-token latency smooth for
    /// already-streaming clients at some cost to TTFT for new arrivals.
    pub prioritize_decode: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_model_len: None,
            enable_chunked_prefill: true,
            max_waiting_requests: 1024,
            preemption_mode: PreemptionMode::Recompute,
            request_timeout_secs: Some(300),
            prioritize_decode: true,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_num_seqs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.max_num_seqs",
                reason: "must be at least 1".into(),
            });
        }
        if self.max_num_batched_tokens == 0 {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.max_num_batched_tokens",
                reason: "must be at least 1".into(),
            });
        }
        // Without chunked prefill an entire prompt must fit in one step, so a
        // token budget below the sequence count guarantees starvation.
        if self.max_num_batched_tokens < self.max_num_seqs {
            return Err(ConfigError::Inconsistent(format!(
                "scheduler.max_num_batched_tokens ({}) is below scheduler.max_num_seqs ({}): \
                 a decode-only step could not give every running sequence one token",
                self.max_num_batched_tokens, self.max_num_seqs
            )));
        }
        if let Some(len) = self.max_model_len {
            if len == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "scheduler.max_model_len",
                    reason: "must be at least 1 when set".into(),
                });
            }
            if !self.enable_chunked_prefill && len > self.max_num_batched_tokens {
                return Err(ConfigError::Inconsistent(format!(
                    "scheduler.max_model_len ({len}) exceeds \
                     scheduler.max_num_batched_tokens ({}) with chunked prefill disabled: \
                     a full-length prompt could never be scheduled",
                    self.max_num_batched_tokens
                )));
            }
        }
        if self.max_waiting_requests == 0 {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.max_waiting_requests",
                reason: "must be at least 1".into(),
            });
        }
        if let Some(0) = self.request_timeout_secs {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.request_timeout_secs",
                reason: "must be at least 1 second when set".into(),
            });
        }
        Ok(())
    }
}

/// HTTP server limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Largest request body accepted, in bytes. Bounds the work a single
    /// client can force the tokenizer to do.
    pub max_request_bytes: usize,
    /// Concurrent in-flight HTTP requests, independent of the engine's own
    /// queue limit.
    pub max_concurrent_requests: usize,
    /// Grace period for in-flight requests during shutdown.
    pub shutdown_grace_secs: u64,
    /// Serve `/metrics` in Prometheus text format.
    pub enable_metrics: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8000,
            max_request_bytes: 4 * 1024 * 1024,
            max_concurrent_requests: 1024,
            shutdown_grace_secs: 30,
            enable_metrics: true,
        }
    }
}

impl ServerConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.host.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "server.host",
                reason: "must not be empty".into(),
            });
        }
        if self.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.port",
                reason: "must not be 0".into(),
            });
        }
        if self.max_request_bytes < 1024 {
            return Err(ConfigError::OutOfRange {
                field: "server.max_request_bytes",
                min: 1024,
                max: u64::MAX,
                value: self.max_request_bytes as u64,
            });
        }
        if self.max_concurrent_requests == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.max_concurrent_requests",
                reason: "must be at least 1".into(),
            });
        }
        Ok(())
    }
}

/// Top-level engine configuration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub cache: CacheConfig,
    pub scheduler: SchedulerConfig,
    pub server: ServerConfig,
}

impl EngineConfig {
    /// Validates every section, plus the cross-section invariants that no
    /// single section can check on its own.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.cache.validate()?;
        self.scheduler.validate()?;
        self.server.validate()?;

        if self.scheduler.preemption_mode == PreemptionMode::Swap
            && self.cache.num_cpu_swap_blocks == 0
        {
            return Err(ConfigError::Inconsistent(
                "scheduler.preemption_mode is `swap` but cache.num_cpu_swap_blocks is 0: \
                 there is nowhere to swap blocks to"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(EngineConfig::default().validate().is_ok());
    }

    #[test]
    fn block_size_must_be_a_power_of_two_in_range() {
        for bad in [0usize, 3, 12, 100] {
            let c = CacheConfig {
                block_size: bad,
                ..Default::default()
            };
            assert!(c.validate().is_err(), "accepted block_size {bad}");
        }
        for good in [4usize, 16, 32, 256] {
            let c = CacheConfig {
                block_size: good,
                ..Default::default()
            };
            assert!(c.validate().is_ok(), "rejected block_size {good}");
        }
        // A power of two can still be out of range.
        let c = CacheConfig {
            block_size: 512,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn memory_utilization_must_be_a_sane_fraction() {
        for bad in [0.0, -0.1, 1.5, f32::NAN] {
            let c = CacheConfig {
                gpu_memory_utilization: bad,
                ..Default::default()
            };
            assert!(c.validate().is_err(), "accepted utilization {bad}");
        }
    }

    #[test]
    fn token_budget_below_sequence_count_is_rejected() {
        let s = SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 128,
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn unchunked_prefill_shorter_than_context_is_rejected() {
        let s = SchedulerConfig {
            enable_chunked_prefill: false,
            max_model_len: Some(32768),
            max_num_batched_tokens: 8192,
            ..Default::default()
        };
        assert!(
            s.validate().is_err(),
            "a prompt at max_model_len could never be scheduled"
        );

        // The same setting is fine once chunked prefill is on.
        let s = SchedulerConfig {
            enable_chunked_prefill: true,
            ..s
        };
        assert!(s.validate().is_ok());
    }

    #[test]
    fn swap_preemption_requires_swap_blocks() {
        let cfg = EngineConfig {
            scheduler: SchedulerConfig {
                preemption_mode: PreemptionMode::Swap,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = EngineConfig {
            cache: CacheConfig {
                num_cpu_swap_blocks: 512,
                ..Default::default()
            },
            ..cfg
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn server_limits_are_bounded() {
        assert!(ServerConfig {
            host: "  ".into(),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(ServerConfig {
            port: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(ServerConfig {
            max_request_bytes: 16,
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn round_trips_through_json() {
        let cfg = EngineConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: EngineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn unknown_config_keys_are_rejected_rather_than_silently_ignored() {
        // A typo in a config file must fail loudly, not leave the operator
        // wondering why their setting had no effect.
        let err = serde_json::from_str::<EngineConfig>(r#"{"cache":{"blocksize":16}}"#);
        assert!(err.is_err());
    }
}
