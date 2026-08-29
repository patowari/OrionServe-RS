//! CUDA backend and custom Transformer kernels.
//!
//! # Status: not built, not verified
//!
//! **No code in this crate has been compiled against a CUDA toolkit or executed
//! on a GPU.** The development machine has neither — `nvcc` and `nvidia-smi`
//! are both absent. The kernels in `kernels/cuda/` are written but unbuilt, and
//! this crate's FFI surface is a design, not a working integration.
//!
//! Everything here is behind the `cuda` feature, which is **off by default**.
//! With the feature off the crate compiles to a stub that reports CUDA as
//! unavailable, so the rest of the workspace builds and tests normally on a
//! machine without a GPU — which is the only machine this has ever run on.
//!
//! # What has to happen before any performance claim
//!
//! In order, each gating the next:
//!
//! 1. **It compiles.** `nvcc` accepts the kernels for a real architecture.
//! 2. **It is correct.** Every kernel matches the CPU reference in
//!    `orion-models::tensor` within a stated tolerance, on shapes covering the
//!    edge cases (single row, non-multiple-of-warp widths, extreme values).
//! 3. **Output is unchanged.** A full forward pass on GPU produces the same
//!    logits as the CPU path, within tolerance, for the same input.
//! 4. **It is measured.** Benchmarked against the CPU reference *and* against
//!    an unfused sequence, with `synchronize()` called so an async launch queue
//!    cannot make a kernel look instantaneous.
//! 5. **It is recorded.** Numbers go in `docs/performance-journal.md` with the
//!    GPU, driver, CUDA version and shapes they came from.
//!
//! Until step 5, this crate claims nothing. See `docs/cuda.md`.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use orion_core::{Backend, DType, Device, EngineError};

/// Why CUDA is unavailable, when it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaUnavailable {
    /// The crate was built without the `cuda` feature.
    FeatureDisabled,
    /// The feature is on but no CUDA driver was found.
    NoDriver,
    /// A driver exists but reports no usable devices.
    NoDevices,
    /// The requested device ordinal does not exist.
    NoSuchDevice { requested: usize, available: usize },
}

impl std::fmt::Display for CudaUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaUnavailable::FeatureDisabled => write!(
                f,
                "CUDA support was not compiled in; rebuild with --features cuda"
            ),
            CudaUnavailable::NoDriver => write!(f, "no CUDA driver found"),
            CudaUnavailable::NoDevices => write!(f, "CUDA driver found but no devices available"),
            CudaUnavailable::NoSuchDevice {
                requested,
                available,
            } => write!(
                f,
                "CUDA device {requested} requested but only {available} device(s) present"
            ),
        }
    }
}

impl std::error::Error for CudaUnavailable {}

/// Whether this build can use CUDA at all.
///
/// Always `false` in the current build. Kept as a function rather than a
/// constant so callers read it at runtime and behave correctly under either
/// feature setting.
pub fn is_available() -> bool {
    cfg!(feature = "cuda")
}

/// Number of CUDA devices visible.
pub fn device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        // Would query cudaGetDeviceCount here. Unimplemented and unverified;
        // returning 0 keeps callers on the CPU path rather than letting them
        // proceed into code that has never run.
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// A CUDA device, if one could be opened.
///
/// Construction is the *only* place availability is decided. Once a
/// `CudaBackend` exists, callers may assume the device is usable — the
/// alternative, checking availability at every call site, is how a codebase
/// ends up with paths that were never exercised.
#[derive(Debug)]
pub struct CudaBackend {
    ordinal: usize,
    name: String,
    total_memory: u64,
}

impl CudaBackend {
    /// Opens a CUDA device.
    ///
    /// Currently always fails: the backend is not implemented. Returning an
    /// error rather than a non-functional handle means no caller can
    /// accidentally believe it is running on a GPU.
    pub fn new(ordinal: usize) -> Result<Self, CudaUnavailable> {
        if !is_available() {
            return Err(CudaUnavailable::FeatureDisabled);
        }
        let count = device_count();
        if count == 0 {
            return Err(CudaUnavailable::NoDevices);
        }
        if ordinal >= count {
            return Err(CudaUnavailable::NoSuchDevice {
                requested: ordinal,
                available: count,
            });
        }
        // Unreachable in this build; device_count() is always 0.
        Err(CudaUnavailable::NoDevices)
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    fn total_memory(&self) -> Option<u64> {
        Some(self.total_memory)
    }

    fn available_memory(&self) -> Option<u64> {
        // Would call cudaMemGetInfo. Not implemented.
        None
    }

    /// Blocks until queued work completes.
    ///
    /// Essential for honest benchmarking: CUDA launches are asynchronous, so
    /// without this a kernel appears to take microseconds regardless of what it
    /// does. Every timing in this project's benchmarks brackets its measurement
    /// with a synchronize for exactly that reason.
    fn synchronize(&self) -> Result<(), EngineError> {
        Err(EngineError::Backend {
            backend: "cuda",
            stage: "synchronize",
            reason: "CUDA backend is not implemented".into(),
        })
    }

    fn supports_dtype(&self, dtype: DType) -> bool {
        // The dtypes the kernels are written for. Nothing here is verified.
        matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
    }
}

/// Numerical tolerance for validating a kernel against the CPU reference.
///
/// A GPU kernel will not match bit-for-bit, and demanding that would be wrong:
/// reduction order differs, fused multiply-add changes rounding, and fast-math
/// intrinsics trade accuracy for speed deliberately. The question is whether
/// the difference is small enough not to change model behaviour.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Absolute tolerance, for values near zero where relative error explodes.
    pub abs: f32,
    /// Relative tolerance, for values with meaningful magnitude.
    pub rel: f32,
}

impl Tolerance {
    /// Tolerance for `f32` kernels.
    ///
    /// Tight: an `f32` kernel differing by more than this is doing something
    /// structurally different, not merely reassociating.
    pub const F32: Tolerance = Tolerance {
        abs: 1e-5,
        rel: 1e-5,
    };

    /// Tolerance for `f16` kernels.
    ///
    /// Looser by three orders of magnitude, because `f16` has ~11 bits of
    /// mantissa — roughly 1e-3 relative precision. Demanding `f32` tolerance
    /// from a half-precision kernel would fail on correct code.
    pub const F16: Tolerance = Tolerance {
        abs: 1e-2,
        rel: 1e-2,
    };

    /// Whether two values agree within this tolerance.
    pub fn matches(&self, expected: f32, actual: f32) -> bool {
        if !expected.is_finite() || !actual.is_finite() {
            // Both non-finite in the same way is a match; anything else is not.
            return expected.to_bits() == actual.to_bits();
        }
        let diff = (expected - actual).abs();
        diff <= self.abs || diff <= self.rel * expected.abs()
    }
}

/// Result of comparing a kernel's output against the reference.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub kernel: String,
    pub elements: usize,
    pub mismatches: usize,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    /// Index of the worst mismatch, for debugging.
    pub worst_index: Option<usize>,
}

impl ValidationReport {
    pub fn passed(&self) -> bool {
        self.mismatches == 0
    }

    pub fn render(&self) -> String {
        if self.passed() {
            format!(
                "{}: PASS ({} elements, max abs {:.3e}, max rel {:.3e})",
                self.kernel, self.elements, self.max_abs_error, self.max_rel_error
            )
        } else {
            format!(
                "{}: FAIL — {}/{} elements differ (max abs {:.3e}, max rel {:.3e}, worst at {:?})",
                self.kernel,
                self.mismatches,
                self.elements,
                self.max_abs_error,
                self.max_rel_error,
                self.worst_index
            )
        }
    }
}

/// Compares a kernel's output against the CPU reference.
///
/// This is the gate every kernel must pass before it is benchmarked, let alone
/// used. It is implemented and tested now, with no GPU, so that when hardware
/// arrives the validation path is already trustworthy — writing the checker
/// after the kernel is how a subtly wrong kernel gets declared correct.
pub fn validate(
    kernel: &str,
    expected: &[f32],
    actual: &[f32],
    tol: Tolerance,
) -> Result<ValidationReport, EngineError> {
    if expected.len() != actual.len() {
        return Err(EngineError::Internal(format!(
            "validation shape mismatch for {kernel}: reference has {} elements, kernel produced {}",
            expected.len(),
            actual.len()
        )));
    }

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst_index = None;
    let mut worst_error = 0.0f32;

    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if !tol.matches(e, a) {
            mismatches += 1;
        }
        if e.is_finite() && a.is_finite() {
            let abs = (e - a).abs();
            let rel = if e.abs() > f32::EPSILON {
                abs / e.abs()
            } else {
                0.0
            };
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            if abs > worst_error {
                worst_error = abs;
                worst_index = Some(i);
            }
        }
    }

    Ok(ValidationReport {
        kernel: kernel.to_string(),
        elements: expected.len(),
        mismatches,
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        worst_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_build_has_no_cuda() {
        // The honest statement of what this crate currently is. If this test
        // ever fails, CUDA has genuinely been enabled and every claim in the
        // module docs needs revisiting.
        assert!(!is_available());
        assert_eq!(device_count(), 0);
    }

    #[test]
    fn opening_a_device_fails_rather_than_returning_a_dead_handle() {
        // A non-functional backend that constructs successfully is how callers
        // end up believing they are on a GPU.
        let err = CudaBackend::new(0).unwrap_err();
        assert_eq!(err, CudaUnavailable::FeatureDisabled);
        assert!(err.to_string().contains("--features cuda"));
    }

    #[test]
    fn unavailability_reasons_are_distinguishable() {
        let cases = [
            CudaUnavailable::FeatureDisabled,
            CudaUnavailable::NoDriver,
            CudaUnavailable::NoDevices,
            CudaUnavailable::NoSuchDevice {
                requested: 3,
                available: 1,
            },
        ];
        let messages: std::collections::HashSet<String> =
            cases.iter().map(|c| c.to_string()).collect();
        assert_eq!(messages.len(), cases.len(), "messages must be distinct");
    }

    #[test]
    fn identical_output_validates() {
        let reference = vec![1.0, 2.0, 3.0, -4.0];
        let report = validate("test", &reference, &reference, Tolerance::F32).unwrap();
        assert!(report.passed());
        assert_eq!(report.mismatches, 0);
        assert_eq!(report.max_abs_error, 0.0);
        assert!(report.render().contains("PASS"));
    }

    #[test]
    fn small_differences_pass_within_tolerance() {
        // Reassociated reductions and FMA produce differences of this order.
        let reference = vec![1.0, 100.0, 1000.0];
        let actual = vec![1.000001, 100.0005, 1000.005];
        let report = validate("test", &reference, &actual, Tolerance::F32).unwrap();
        assert!(report.passed(), "{}", report.render());
    }

    #[test]
    fn a_structurally_wrong_kernel_fails() {
        let reference = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0, 30.0];
        let report = validate("test", &reference, &actual, Tolerance::F32).unwrap();

        assert!(!report.passed());
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.worst_index, Some(2));
        assert!(report.render().contains("FAIL"));
    }

    #[test]
    fn half_precision_tolerance_is_looser_than_single() {
        // f16 has ~11 bits of mantissa. Demanding f32 tolerance from a correct
        // f16 kernel would fail it.
        let reference = vec![1.0, 10.0];
        let actual = vec![1.004, 10.04];

        assert!(!validate("t", &reference, &actual, Tolerance::F32)
            .unwrap()
            .passed());
        assert!(validate("t", &reference, &actual, Tolerance::F16)
            .unwrap()
            .passed());
    }

    #[test]
    fn absolute_tolerance_covers_values_near_zero() {
        // Relative error is meaningless at zero; without the absolute term a
        // correct kernel would fail on any near-zero output.
        let reference = vec![0.0, 1e-9];
        let actual = vec![1e-7, 2e-7];
        assert!(validate("t", &reference, &actual, Tolerance::F32)
            .unwrap()
            .passed());
    }

    #[test]
    fn a_nan_only_matches_a_nan() {
        let report = validate("t", &[f32::NAN], &[f32::NAN], Tolerance::F32).unwrap();
        assert!(report.passed(), "NaN should match itself");

        let report = validate("t", &[f32::NAN], &[1.0], Tolerance::F32).unwrap();
        assert!(!report.passed(), "NaN must not match a finite value");

        let report = validate("t", &[1.0], &[f32::INFINITY], Tolerance::F32).unwrap();
        assert!(!report.passed(), "infinity must not match a finite value");
    }

    #[test]
    fn a_shape_mismatch_is_an_error_not_a_failed_comparison() {
        // Comparing different-length outputs is a bug in the harness, not a
        // kernel result.
        let err = validate("t", &[1.0, 2.0], &[1.0], Tolerance::F32).unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn the_report_names_the_worst_offender() {
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let actual = vec![1.0, 2.5, 3.0, 4.0];
        let report = validate("k", &reference, &actual, Tolerance::F32).unwrap();
        assert_eq!(report.worst_index, Some(1));
        assert!((report.max_abs_error - 0.5).abs() < 1e-6);
    }

    #[test]
    fn synchronize_is_required_for_honest_timing() {
        // Documented here as an executable reminder: a CUDA backend that does
        // not synchronize makes every kernel look instantaneous.
        //
        // The backend cannot be constructed in this build, so this asserts the
        // contract on the error path instead.
        assert!(CudaBackend::new(0).is_err());
    }
}
