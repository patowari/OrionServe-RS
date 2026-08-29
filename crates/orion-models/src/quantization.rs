//! Weight quantization.
//!
//! # Status
//!
//! **Implemented and tested:** the INT8 and INT4 quantize/dequantize round trip
//! and its error characteristics, on CPU. This is real, working code with real
//! tests.
//!
//! **Not implemented:** quantized *matrix multiplication*. Loading a quantized
//! checkpoint, dequantizing on the fly, and running the forward pass in reduced
//! precision are all `planned`. Nothing in the engine currently serves a
//! quantized model.
//!
//! The split is deliberate: the numerics can be verified now, without a GPU,
//! and getting them right is the part where mistakes are silent. A quantized
//! matmul that is merely slow announces itself; one that is subtly wrong
//! produces plausible text and is very hard to notice.
//!
//! # Why quantization matters here
//!
//! Weights dominate memory at rest, and memory bandwidth dominates decode time.
//! Llama-3-8B in FP16 is ~16 GB of weights; every decode step reads all of
//! them. INT8 halves that traffic and INT4 quarters it, which is why
//! quantization is a *latency* optimization as much as a capacity one.
//!
//! # Group-wise scaling
//!
//! A single scale for an entire weight matrix is too coarse: one outlier
//! channel forces a scale that crushes everything else to a handful of levels.
//! Scales are therefore per-group along the input dimension, with a group size
//! of 128 — small enough to track local dynamic range, large enough that the
//! scale metadata stays negligible.

use orion_core::EngineError;

/// Elements sharing one scale factor.
///
/// 128 is the value most quantized checkpoint formats use. The tradeoff:
/// smaller groups track outliers better but add metadata (one `f32` per group)
/// and more per-group work in the dequantize inner loop.
pub const GROUP_SIZE: usize = 128;

/// A group-wise quantized tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    /// Packed quantized values. For INT4, two values per byte.
    pub data: Vec<u8>,
    /// One scale per group.
    pub scales: Vec<f32>,
    /// One zero-point per group, for asymmetric quantization.
    pub zero_points: Vec<i32>,
    pub num_elements: usize,
    pub bits: QuantBits,
}

/// Quantization width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    Int8,
    Int4,
}

impl QuantBits {
    pub fn bits(self) -> usize {
        match self {
            QuantBits::Int8 => 8,
            QuantBits::Int4 => 4,
        }
    }

    /// Number of representable levels.
    pub fn levels(self) -> i32 {
        1 << self.bits()
    }

    /// Highest representable unsigned value.
    pub fn max_value(self) -> i32 {
        self.levels() - 1
    }

    pub fn as_str(self) -> &'static str {
        match self {
            QuantBits::Int8 => "int8",
            QuantBits::Int4 => "int4",
        }
    }
}

impl QuantizedTensor {
    /// Bytes occupied, including scale metadata.
    pub fn size_in_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }

    /// Compression ratio against the same tensor in `f32`.
    ///
    /// Includes metadata, so it is always below the naive `32 / bits` figure.
    /// Quoting the naive number would overstate the saving.
    pub fn compression_ratio(&self) -> f64 {
        let original = self.num_elements * 4;
        if self.size_in_bytes() == 0 {
            return 0.0;
        }
        original as f64 / self.size_in_bytes() as f64
    }

    pub fn num_groups(&self) -> usize {
        self.scales.len()
    }
}

/// Quantizes `f32` values group-wise.
///
/// Uses **asymmetric** quantization: each group gets both a scale and a
/// zero-point, mapping `[min, max]` onto the full integer range. Symmetric
/// quantization (zero-point fixed at the midpoint) wastes half the range when a
/// group's values are one-sided, which weight distributions frequently are
/// after activation functions like SiLU.
pub fn quantize(values: &[f32], bits: QuantBits) -> Result<QuantizedTensor, EngineError> {
    if values.is_empty() {
        return Ok(QuantizedTensor {
            data: Vec::new(),
            scales: Vec::new(),
            zero_points: Vec::new(),
            num_elements: 0,
            bits,
        });
    }

    let num_groups = values.len().div_ceil(GROUP_SIZE);
    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);
    let mut quantized: Vec<i32> = Vec::with_capacity(values.len());

    for group in values.chunks(GROUP_SIZE) {
        // A non-finite weight means the checkpoint is corrupt. Failing loudly
        // beats quantizing NaN into a plausible-looking integer.
        //
        // Checked per element rather than on the min/max: `f32::min` and
        // `f32::max` return the *other* operand when one side is NaN, so a NaN
        // passes straight through a fold and leaves finite bounds behind.
        if let Some(bad) = group.iter().find(|v| !v.is_finite()) {
            return Err(EngineError::Internal(format!(
                "cannot quantize a group containing a non-finite value ({bad})"
            )));
        }

        let (min, max) = group
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));

        // A constant group has zero range. Any scale works; 1.0 avoids a
        // division by zero and dequantizes exactly.
        let range = max - min;
        let scale = if range > 0.0 {
            range / bits.max_value() as f32
        } else {
            1.0
        };
        let zero_point = if range > 0.0 {
            (-min / scale).round() as i32
        } else {
            // Represent the constant exactly by folding it into the zero point.
            0
        };

        for &v in group {
            let q = if range > 0.0 {
                ((v / scale).round() as i32 + zero_point).clamp(0, bits.max_value())
            } else {
                0
            };
            quantized.push(q);
        }

        // For a constant group the scale carries the value itself, so
        // dequantization reproduces it exactly.
        scales.push(if range > 0.0 { scale } else { 0.0 });
        zero_points.push(if range > 0.0 {
            zero_point
        } else {
            min.to_bits() as i32
        });
    }

    let data = pack(&quantized, bits);

    Ok(QuantizedTensor {
        data,
        scales,
        zero_points,
        num_elements: values.len(),
        bits,
    })
}

/// Packs quantized integers into bytes.
///
/// INT4 packs two values per byte, low nibble first.
fn pack(values: &[i32], bits: QuantBits) -> Vec<u8> {
    match bits {
        QuantBits::Int8 => values.iter().map(|&v| v as u8).collect(),
        QuantBits::Int4 => {
            let mut out = Vec::with_capacity(values.len().div_ceil(2));
            for pair in values.chunks(2) {
                let lo = (pair[0] as u8) & 0x0F;
                let hi = pair.get(1).map_or(0, |&v| (v as u8) & 0x0F);
                out.push(lo | (hi << 4));
            }
            out
        }
    }
}

/// Unpacks bytes back into quantized integers.
fn unpack(data: &[u8], bits: QuantBits, count: usize) -> Vec<i32> {
    match bits {
        QuantBits::Int8 => data.iter().take(count).map(|&b| b as i32).collect(),
        QuantBits::Int4 => {
            let mut out = Vec::with_capacity(count);
            for &byte in data {
                out.push((byte & 0x0F) as i32);
                if out.len() < count {
                    out.push((byte >> 4) as i32);
                }
                if out.len() >= count {
                    break;
                }
            }
            out
        }
    }
}

/// Reconstructs `f32` values from a quantized tensor.
pub fn dequantize(tensor: &QuantizedTensor) -> Vec<f32> {
    let quantized = unpack(&tensor.data, tensor.bits, tensor.num_elements);
    let mut out = Vec::with_capacity(tensor.num_elements);

    for (group_idx, group) in quantized.chunks(GROUP_SIZE).enumerate() {
        let scale = tensor.scales.get(group_idx).copied().unwrap_or(1.0);
        let zero_point = tensor.zero_points.get(group_idx).copied().unwrap_or(0);

        if scale == 0.0 {
            // Constant group: the zero-point field carries the bit pattern.
            let value = f32::from_bits(zero_point as u32);
            out.extend(std::iter::repeat_n(value, group.len()));
            continue;
        }

        for &q in group {
            out.push((q - zero_point) as f32 * scale);
        }
    }
    out
}

/// Quantization error statistics, for judging whether a scheme is acceptable.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationError {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    /// Root-mean-square error relative to the RMS of the original values.
    ///
    /// More useful than raw RMSE: it is scale-invariant, so the same threshold
    /// applies to weight matrices of very different magnitudes.
    pub relative_rmse: f32,
}

/// Measures how much a quantization round trip changed the values.
pub fn measure_error(original: &[f32], reconstructed: &[f32]) -> QuantizationError {
    if original.is_empty() || original.len() != reconstructed.len() {
        return QuantizationError {
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            relative_rmse: 0.0,
        };
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_orig = 0.0f64;

    for (&o, &r) in original.iter().zip(reconstructed.iter()) {
        let err = (o - r).abs();
        max_abs = max_abs.max(err);
        sum_abs += err as f64;
        sum_sq_err += (err as f64) * (err as f64);
        sum_sq_orig += (o as f64) * (o as f64);
    }

    let n = original.len() as f64;
    let rmse = (sum_sq_err / n).sqrt();
    let rms_orig = (sum_sq_orig / n).sqrt();

    QuantizationError {
        max_abs_error: max_abs,
        mean_abs_error: (sum_abs / n) as f32,
        relative_rmse: if rms_orig > 0.0 {
            (rmse / rms_orig) as f32
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weight-like distribution: roughly normal, centred near zero.
    fn weights(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32 * 0.017 + seed;
                (x.sin() + x.cos() * 0.5) * 0.1
            })
            .collect()
    }

    #[test]
    fn int8_round_trip_is_close() {
        let original = weights(1024, 0.0);
        let q = quantize(&original, QuantBits::Int8).unwrap();
        let back = dequantize(&q);

        assert_eq!(back.len(), original.len());
        let err = measure_error(&original, &back);
        assert!(
            err.relative_rmse < 0.01,
            "int8 relative RMSE {} is too high",
            err.relative_rmse
        );
    }

    #[test]
    fn int4_round_trip_is_coarser_but_bounded() {
        let original = weights(1024, 1.0);
        let q = quantize(&original, QuantBits::Int4).unwrap();
        let back = dequantize(&q);

        let err = measure_error(&original, &back);
        assert!(
            err.relative_rmse < 0.15,
            "int4 relative RMSE {} is too high",
            err.relative_rmse
        );
    }

    #[test]
    fn int4_is_measurably_worse_than_int8() {
        // The tradeoff has to be visible, or the test is not checking anything.
        let original = weights(2048, 2.0);

        let e8 = measure_error(
            &original,
            &dequantize(&quantize(&original, QuantBits::Int8).unwrap()),
        );
        let e4 = measure_error(
            &original,
            &dequantize(&quantize(&original, QuantBits::Int4).unwrap()),
        );

        assert!(
            e4.relative_rmse > e8.relative_rmse,
            "int4 ({}) should be less accurate than int8 ({})",
            e4.relative_rmse,
            e8.relative_rmse
        );
    }

    #[test]
    fn compression_ratios_are_reported_honestly() {
        let original = weights(4096, 0.0);

        let q8 = quantize(&original, QuantBits::Int8).unwrap();
        let q4 = quantize(&original, QuantBits::Int4).unwrap();

        // Naive ratios would be 4x and 8x; metadata makes the real figures
        // lower, and this asserts we report the real ones.
        assert!(q8.compression_ratio() < 4.0);
        assert!(q8.compression_ratio() > 3.7, "{}", q8.compression_ratio());
        assert!(q4.compression_ratio() < 8.0);
        assert!(q4.compression_ratio() > 7.0, "{}", q4.compression_ratio());
    }

    #[test]
    fn packing_halves_the_bytes_for_int4() {
        let original = weights(1000, 0.0);
        let q8 = quantize(&original, QuantBits::Int8).unwrap();
        let q4 = quantize(&original, QuantBits::Int4).unwrap();

        assert_eq!(q8.data.len(), 1000);
        assert_eq!(q4.data.len(), 500);
    }

    #[test]
    fn group_wise_scaling_survives_an_outlier() {
        // The reason for group-wise rather than per-tensor scales: one huge
        // value must not destroy precision everywhere else.
        let mut values = vec![0.01f32; GROUP_SIZE * 4];
        values[GROUP_SIZE * 2] = 1000.0; // outlier in group 2

        let q = quantize(&values, QuantBits::Int8).unwrap();
        let back = dequantize(&q);

        // Groups away from the outlier keep their precision.
        let clean_error = (values[0] - back[0]).abs();
        assert!(
            clean_error < 0.001,
            "an outlier in another group degraded this one: {clean_error}"
        );
        assert_eq!(q.num_groups(), 4);
    }

    #[test]
    fn a_constant_group_is_reproduced_exactly() {
        // Zero range would divide by zero in a naive implementation.
        let values = vec![0.5f32; GROUP_SIZE];
        let q = quantize(&values, QuantBits::Int8).unwrap();
        let back = dequantize(&q);

        for (i, &v) in back.iter().enumerate() {
            assert_eq!(v, 0.5, "element {i} was not reproduced exactly");
        }
    }

    #[test]
    fn all_zeros_round_trip_exactly() {
        let values = vec![0.0f32; 256];
        let back = dequantize(&quantize(&values, QuantBits::Int4).unwrap());
        assert!(back.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn an_empty_tensor_is_handled() {
        let q = quantize(&[], QuantBits::Int8).unwrap();
        assert_eq!(q.num_elements, 0);
        assert!(dequantize(&q).is_empty());
        assert_eq!(q.compression_ratio(), 0.0);
    }

    #[test]
    fn a_partial_final_group_is_handled() {
        // Length not a multiple of GROUP_SIZE, and odd for the int4 packing.
        let original = weights(GROUP_SIZE + 37, 0.0);
        let q = quantize(&original, QuantBits::Int4).unwrap();
        let back = dequantize(&q);

        assert_eq!(back.len(), original.len());
        assert_eq!(q.num_groups(), 2);
        let err = measure_error(&original, &back);
        assert!(err.relative_rmse < 0.2);
    }

    #[test]
    fn non_finite_weights_are_rejected_rather_than_silently_encoded() {
        // A NaN quantized into an integer becomes a plausible-looking value,
        // which is far worse than a load failure.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = weights(GROUP_SIZE, 0.0);
            values[5] = bad;
            assert!(
                quantize(&values, QuantBits::Int8).is_err(),
                "accepted {bad} in a weight tensor"
            );
        }
    }

    #[test]
    fn asymmetric_quantization_uses_the_full_range_on_one_sided_data() {
        // Post-activation weights are frequently one-sided. Symmetric
        // quantization would waste half the levels on values that never occur.
        let values: Vec<f32> = (0..GROUP_SIZE).map(|i| i as f32 * 0.01).collect();
        let q = quantize(&values, QuantBits::Int8).unwrap();
        let back = dequantize(&q);

        let err = measure_error(&values, &back);
        assert!(
            err.relative_rmse < 0.01,
            "one-sided data quantized poorly: {}",
            err.relative_rmse
        );
    }

    #[test]
    fn error_measurement_is_scale_invariant() {
        // The same relative error should be reported whether weights are tiny
        // or huge, which is why relative_rmse exists.
        let small: Vec<f32> = weights(512, 0.0);
        let large: Vec<f32> = small.iter().map(|v| v * 1000.0).collect();

        let e_small = measure_error(
            &small,
            &dequantize(&quantize(&small, QuantBits::Int8).unwrap()),
        );
        let e_large = measure_error(
            &large,
            &dequantize(&quantize(&large, QuantBits::Int8).unwrap()),
        );

        assert!(
            (e_small.relative_rmse - e_large.relative_rmse).abs() < 0.002,
            "{} vs {}",
            e_small.relative_rmse,
            e_large.relative_rmse
        );
        // Absolute error, by contrast, scales with the data.
        assert!(e_large.max_abs_error > e_small.max_abs_error * 100.0);
    }

    #[test]
    fn quant_widths_report_their_levels() {
        assert_eq!(QuantBits::Int8.bits(), 8);
        assert_eq!(QuantBits::Int8.levels(), 256);
        assert_eq!(QuantBits::Int8.max_value(), 255);
        assert_eq!(QuantBits::Int4.bits(), 4);
        assert_eq!(QuantBits::Int4.levels(), 16);
        assert_eq!(QuantBits::Int4.max_value(), 15);
    }
}
