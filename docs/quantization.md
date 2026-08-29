# Quantization

## Status

**Implemented and tested:** INT8 and INT4 quantize/dequantize with group-wise
asymmetric scaling, and the error measurement that judges it. Real code, real
tests, on CPU.

**Not implemented:** quantized matrix multiplication, loading a pre-quantized
checkpoint (GPTQ, AWQ), and serving a quantized model. Nothing in the engine
currently runs a quantized forward pass.

The split is deliberate. The numerics can be verified now, without a GPU, and
they are the part where mistakes are silent: a quantized matmul that is merely
slow announces itself, while one that is subtly wrong produces plausible text
and is very hard to notice.

## Why quantize

Weights dominate memory at rest, and **memory bandwidth dominates decode time**.
Llama-3-8B in FP16 is roughly 16 GB of weights, and every decode step reads all
of them to produce one token.

INT8 halves that traffic; INT4 quarters it. Quantization is therefore a
*latency* optimization at least as much as a capacity one — the opposite of the
usual intuition that it is about fitting bigger models.

## Group-wise scaling

A single scale for a whole weight matrix is too coarse. One outlier channel
forces a scale that crushes everything else into a handful of levels, and
transformer weights reliably have outliers.

Scales are per-group along the input dimension, with `GROUP_SIZE = 128`:

- small enough to track local dynamic range
- large enough that metadata stays negligible (one `f32` and one `i32` per 128
  values)
- the value most quantized checkpoint formats already use

`group_wise_scaling_survives_an_outlier` pins the property: a 1000x outlier in
one group must not degrade precision in the others.

## Asymmetric quantization

Each group gets both a scale and a zero-point, mapping `[min, max]` onto the
full integer range.

Symmetric quantization — zero-point fixed at the midpoint — wastes half the
range when a group's values are one-sided, which weight distributions frequently
are after activations like SiLU.
`asymmetric_quantization_uses_the_full_range_on_one_sided_data` pins this.

## Honest compression ratios

`QuantizedTensor::compression_ratio` includes scale metadata, so it is always
below the naive `32 / bits`:

| Width | Naive | Measured (4096 elements) |
|---|---|---|
| INT8 | 4.0x | ~3.9x |
| INT4 | 8.0x | ~7.3x |

Quoting 4x and 8x would overstate the saving. A test asserts the real figures.

## Error characteristics

Measured on a weight-like distribution, 1024 elements:

| Width | Relative RMSE |
|---|---|
| INT8 | < 1% |
| INT4 | < 15% |

`relative_rmse` rather than raw RMSE, because it is scale-invariant: the same
threshold applies to weight matrices of very different magnitudes.
`error_measurement_is_scale_invariant` pins that.

**What these numbers do not tell you:** whether the model's *output quality*
holds up. Weight reconstruction error and perplexity are related but not the
same thing, and no perplexity measurement has been made because no quantized
forward pass exists to make it with. Any quality claim here would be fabricated.

## Corrupt weights fail loudly

A non-finite weight is rejected rather than encoded. A NaN quantized into an
integer becomes a plausible-looking value, which is far worse than a load
failure.

This caught a real bug during development. The range check originally used
`f32::min` and `f32::max` over the group, and those return the *other* operand
when one side is NaN — so a NaN passed straight through the fold and left finite
bounds behind. The check is now per element, before the fold.

## Roadmap

**Next:** dequantize-on-the-fly matmul. Weights stay quantized in memory and are
dequantized into registers inside the matmul. This is where the bandwidth saving
actually materializes; storing quantized weights and dequantizing to a full
buffer first would save nothing at all.

**Then:** loading pre-quantized GPTQ and AWQ checkpoints, which are the formats
quantized models are actually distributed in.

**Then:** measuring quality. Perplexity on a held-out set, against the FP16
baseline, at each width. Without this the accuracy tradeoff is unquantified.

**Not planned:** activation quantization. It interacts badly with outliers in
attention, needs calibration data, and the weight side is where the bandwidth
win is.
