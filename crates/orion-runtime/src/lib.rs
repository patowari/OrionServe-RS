//! Execution engine: sampling, batch execution and the engine driver.
//!
//! # Module map
//!
//! * [`sampling`] — logits to token: repetition penalty, temperature, top-k,
//!   top-p, and seeded reproducibility.
//!
//! Batch execution and the engine loop are `planned`; see the roadmap in the
//! README.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod engine;
pub mod sampling;

pub use engine::{spawn, Engine, EngineHandle, EngineStats, GenerationRequest, StreamEvent};
pub use sampling::DefaultSampler;
