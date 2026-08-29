//! OpenAI-compatible HTTP API.
//!
//! Implements `/v1/chat/completions` and `/v1/completions` with Server-Sent
//! Events streaming, plus health, readiness and model-listing endpoints.
//!
//! See [`routes`] for the handlers and [`types`] for the wire format.

#![forbid(unsafe_code)]

pub mod routes;
pub mod types;

pub use routes::{router, ApiError, AppState};
pub use types::*;
