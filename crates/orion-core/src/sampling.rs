//! Sampling parameters and their validation.
//!
//! These values arrive from untrusted API input, so every constructor path runs
//! through [`SamplingParams::validate`]. The engine treats a `SamplingParams`
//! value that exists as already-valid; validation happens once, at the edge.

use serde::{Deserialize, Serialize};

use crate::error::EngineError;
use crate::id::TokenId;

/// How the next token is chosen from the logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    /// Always take the arg-max. Deterministic regardless of seed.
    Greedy,
    /// Sample from the (temperature/top-k/top-p filtered) distribution.
    Stochastic,
}

/// Per-request sampling configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    /// Softmax temperature. `0.0` means greedy decoding.
    pub temperature: f32,
    /// Nucleus sampling mass in `(0.0, 1.0]`. `1.0` disables the filter.
    pub top_p: f32,
    /// Keep only the `k` highest-logit tokens. `0` disables the filter.
    pub top_k: usize,
    /// Penalty applied to tokens already present in the context.
    /// `1.0` disables it; `> 1.0` discourages repetition.
    pub repetition_penalty: f32,
    /// Hard cap on generated tokens for this request.
    pub max_tokens: usize,
    /// Minimum tokens to generate before EOS is allowed to terminate.
    pub min_tokens: usize,
    /// RNG seed. `None` draws from entropy; `Some` makes the request
    /// reproducible.
    pub seed: Option<u64>,
    /// Stop generating when any of these token ids is produced. The model's own
    /// EOS ids are merged in by the engine.
    pub stop_token_ids: Vec<TokenId>,
    /// Stop generating when any of these strings appears in the decoded output.
    pub stop_sequences: Vec<String>,
    /// Include the chosen token's logprob (and top alternatives) in the
    /// response.
    pub logprobs: Option<usize>,
    /// Echo the prompt tokens back in the response.
    pub echo: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_tokens: 16,
            min_tokens: 0,
            seed: None,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
            logprobs: None,
            echo: false,
        }
    }
}

/// Upper bound on `logprobs`, matching the OpenAI API's own limit. Prevents a
/// client from forcing a full-vocabulary sort per token.
pub const MAX_LOGPROBS: usize = 20;

/// Upper bound on the number of stop sequences, so stop-string matching stays
/// cheap in the decode hot loop.
pub const MAX_STOP_SEQUENCES: usize = 16;

impl SamplingParams {
    /// Checks every field against its legal range.
    ///
    /// Returns [`EngineError::InvalidRequest`] naming the offending field, so
    /// the API can hand the client something actionable.
    pub fn validate(&self) -> Result<(), EngineError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(EngineError::InvalidRequest(format!(
                "temperature must be a finite value >= 0.0, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(EngineError::InvalidRequest(format!(
                "top_p must be in (0.0, 1.0], got {}",
                self.top_p
            )));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(EngineError::InvalidRequest(format!(
                "repetition_penalty must be a finite value > 0.0, got {}",
                self.repetition_penalty
            )));
        }
        if self.max_tokens == 0 {
            return Err(EngineError::InvalidRequest(
                "max_tokens must be at least 1".into(),
            ));
        }
        if self.min_tokens > self.max_tokens {
            return Err(EngineError::InvalidRequest(format!(
                "min_tokens ({}) must not exceed max_tokens ({})",
                self.min_tokens, self.max_tokens
            )));
        }
        if let Some(n) = self.logprobs {
            if n > MAX_LOGPROBS {
                return Err(EngineError::InvalidRequest(format!(
                    "logprobs must be <= {MAX_LOGPROBS}, got {n}"
                )));
            }
        }
        if self.stop_sequences.len() > MAX_STOP_SEQUENCES {
            return Err(EngineError::InvalidRequest(format!(
                "at most {MAX_STOP_SEQUENCES} stop sequences are allowed, got {}",
                self.stop_sequences.len()
            )));
        }
        if self.stop_sequences.iter().any(|s| s.is_empty()) {
            return Err(EngineError::InvalidRequest(
                "stop sequences must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// The effective sampling mode.
    ///
    /// Temperature `0.0` is the conventional API spelling of "greedy", so it is
    /// normalized here rather than in the sampler, which then never has to
    /// guard against dividing by zero.
    pub fn mode(&self) -> SamplingMode {
        if self.temperature == 0.0 {
            SamplingMode::Greedy
        } else {
            SamplingMode::Stochastic
        }
    }

    /// Whether the request is reproducible: either greedy, or seeded.
    pub fn is_deterministic(&self) -> bool {
        self.mode() == SamplingMode::Greedy || self.seed.is_some()
    }

    /// Builder-style setter used mainly in tests.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Builder-style setter used mainly in tests.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        assert!(SamplingParams::default().validate().is_ok());
    }

    #[test]
    fn zero_temperature_is_greedy_and_deterministic() {
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(p.mode(), SamplingMode::Greedy);
        assert!(p.is_deterministic());
    }

    #[test]
    fn unseeded_stochastic_sampling_is_not_deterministic() {
        let p = SamplingParams::default();
        assert_eq!(p.mode(), SamplingMode::Stochastic);
        assert!(!p.is_deterministic());
        assert!(p.with_seed(42).is_deterministic());
    }

    #[test]
    fn rejects_non_finite_temperature() {
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let p = SamplingParams {
                temperature: bad,
                ..Default::default()
            };
            assert!(p.validate().is_err(), "accepted temperature {bad}");
        }
    }

    #[test]
    fn rejects_top_p_outside_unit_interval() {
        for bad in [0.0, -0.5, 1.5, f32::NAN] {
            let p = SamplingParams {
                top_p: bad,
                ..Default::default()
            };
            assert!(p.validate().is_err(), "accepted top_p {bad}");
        }
        assert!(SamplingParams {
            top_p: 1.0,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn rejects_zero_max_tokens_and_inverted_bounds() {
        assert!(SamplingParams {
            max_tokens: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            min_tokens: 10,
            max_tokens: 5,
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn rejects_excessive_logprobs_and_stop_sequences() {
        assert!(SamplingParams {
            logprobs: Some(MAX_LOGPROBS + 1),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            stop_sequences: vec!["x".into(); MAX_STOP_SEQUENCES + 1],
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            stop_sequences: vec![String::new()],
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn round_trips_through_json() {
        let p = SamplingParams::default().with_seed(7).with_max_tokens(128);
        let json = serde_json::to_string(&p).unwrap();
        let back: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn deserializes_from_a_partial_object_using_defaults() {
        let p: SamplingParams = serde_json::from_str(r#"{"max_tokens": 64}"#).unwrap();
        assert_eq!(p.max_tokens, 64);
        assert_eq!(p.temperature, 1.0);
        assert!(p.validate().is_ok());
    }
}
