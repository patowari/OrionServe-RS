//! The CPU backend.
//!
//! Serves two purposes: it makes the engine runnable with no GPU present, and
//! it is the numerical reference that CUDA kernels will be validated against.
//! Being obviously correct matters more here than being fast.

use orion_core::{Backend, DType, Device, EngineError};

/// Executes on the host CPU in `f32`.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend {
    _private: (),
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    /// Host memory is not reported.
    ///
    /// The engine uses this to size the KV cache pool automatically. On CPU
    /// there is no fixed device budget to divide up, and claiming a fraction of
    /// system RAM would be a worse default than requiring the operator to say
    /// what they want. `None` forces `cache.num_blocks` to be set explicitly.
    fn total_memory(&self) -> Option<u64> {
        None
    }

    fn available_memory(&self) -> Option<u64> {
        None
    }

    /// A no-op: host execution is synchronous, so there is nothing queued.
    fn synchronize(&self) -> Result<(), EngineError> {
        Ok(())
    }

    /// Only `f32` is executed natively. Half-precision checkpoints are
    /// converted at load time rather than computed in reduced precision, since
    /// `f32` arithmetic is what makes this a trustworthy reference.
    fn supports_dtype(&self, dtype: DType) -> bool {
        matches!(dtype, DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cpu_backend_identifies_itself() {
        let b = CpuBackend::new();
        assert_eq!(b.name(), "cpu");
        assert_eq!(b.device(), Device::Cpu);
        assert!(b.device().is_cpu());
    }

    #[test]
    fn synchronize_is_a_no_op() {
        assert!(CpuBackend::new().synchronize().is_ok());
    }

    #[test]
    fn memory_reporting_is_unavailable_so_the_pool_must_be_configured() {
        let b = CpuBackend::new();
        assert!(b.total_memory().is_none());
        assert!(b.available_memory().is_none());
    }

    #[test]
    fn only_f32_is_natively_supported() {
        let b = CpuBackend::new();
        assert!(b.supports_dtype(DType::F32));
        for d in [DType::F16, DType::BF16, DType::I8, DType::I4] {
            assert!(!b.supports_dtype(d), "{d:?} should not be native");
        }
    }
}
