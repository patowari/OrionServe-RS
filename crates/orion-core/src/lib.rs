//! Core domain types, traits and error architecture for OrionServe-RS.
//!
//! This crate is the shared vocabulary of the engine. Everything else in the
//! workspace depends on it, and it depends on nothing but `serde`, `thiserror`
//! and `uuid`. That constraint is deliberate: it keeps the domain model free of
//! any runtime, backend or transport, so the scheduler and cache can be tested
//! without a GPU and the API without a model.
//!
//! # Layout
//!
//! * [`error`] — the three failure classes and their mapping to API responses.
//! * [`id`] — newtype identifiers that stop `usize` indices being mixed up.
//! * [`config`] — operator-facing configuration and total validation.
//! * [`sampling`] — sampling parameters, validated at the API edge.
//! * [`sequence`] — the sequence lifecycle state machine.
//! * [`traits`] — [`Backend`], [`LanguageModel`], [`Sampler`] and the batch
//!   types that cross those boundaries.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod config;
pub mod error;
pub mod id;
pub mod sampling;
pub mod sequence;
pub mod traits;

pub use config::{CacheConfig, EngineConfig, PreemptionMode, SchedulerConfig, ServerConfig};
pub use error::{ConfigError, EngineError, ModelError, OrionError, Result};
pub use id::{BlockId, RequestId, SequenceId, TokenId};
pub use sampling::{SamplingMode, SamplingParams};
pub use sequence::{FinishReason, Sequence, SequenceState, SequenceTimings};
pub use traits::{
    Backend, DType, Device, ForwardBatch, ForwardOutput, LanguageModel, ModelMetadata, Sampler,
};
