//! Transformer model implementations and checkpoint loading.

// `unsafe` is confined to one call in `loader`: memory-mapping the weight
// file. Every other module is `forbid(unsafe_code)` at module scope via the
// deny below, and the single exception carries its safety argument inline.
#![deny(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod attention;
pub mod backend;
pub mod config;
pub mod loader;
pub mod tensor;
pub mod transformer;

pub use attention::{paged_attention, AttentionArgs, KvStore};
pub use backend::CpuBackend;
pub use config::{Architecture, HfConfig};
pub use loader::CheckpointLoader;
pub use tensor::{Matrix, RopeTable};
pub use transformer::{ModelWeights, TransformerModel};
