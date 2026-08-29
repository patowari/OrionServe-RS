//! Minimal dense tensor primitives for CPU inference.
//!
//! This is deliberately small: only the operations a decoder-only transformer
//! forward pass actually needs, in `f32`, row-major. It is **not** a general
//! tensor library and does not try to be.
//!
//! # Why not use an existing tensor crate
//!
//! The operations that matter for this engine are the ones that touch the
//! paged KV cache — attention has to gather keys and values through a block
//! table rather than read contiguous memory, and no general-purpose tensor
//! library exposes that. Writing the handful of dense kernels here keeps the
//! CPU path a self-contained correctness reference with no version-pinning
//! against a fast-moving dependency.
//!
//! Performance is a non-goal for this module. It is the reference
//! implementation that CUDA kernels will be validated against, and being
//! obviously correct matters more than being fast.

use orion_core::EngineError;

/// A row-major 2-D matrix of `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Creates a matrix from row-major data.
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self, EngineError> {
        if data.len() != rows * cols {
            return Err(EngineError::Internal(format!(
                "matrix data has {} elements, expected {rows}x{cols} = {}",
                data.len(),
                rows * cols
            )));
        }
        Ok(Self { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Borrows row `i`.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        if i >= self.rows {
            return None;
        }
        Some(&self.data[i * self.cols..(i + 1) * self.cols])
    }

    /// Mutably borrows row `i`.
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [f32]> {
        if i >= self.rows {
            return None;
        }
        Some(&mut self.data[i * self.cols..(i + 1) * self.cols])
    }
}

/// Computes `out = x · Wᵀ`, the layout HF checkpoints store linear weights in.
///
/// `weight` is `[out_features, in_features]` — the transpose of the
/// mathematical convention — because that is how PyTorch's `nn.Linear` stores
/// it, and transposing multi-gigabyte weight matrices at load time to satisfy a
/// convention would be pure waste.
///
/// `x` is `[num_tokens, in_features]`, output is `[num_tokens, out_features]`.
pub fn linear(x: &Matrix, weight: &Matrix, bias: Option<&[f32]>) -> Result<Matrix, EngineError> {
    let (n, k) = (x.rows(), x.cols());
    let (out_features, in_features) = (weight.rows(), weight.cols());

    if k != in_features {
        return Err(EngineError::Internal(format!(
            "linear shape mismatch: input has {k} features, weight expects {in_features}"
        )));
    }
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(EngineError::Internal(format!(
                "bias has {} elements, expected {out_features}",
                b.len()
            )));
        }
    }

    let mut out = Matrix::zeros(n, out_features);
    for i in 0..n {
        let xr = &x.data[i * k..(i + 1) * k];
        let orow = &mut out.data[i * out_features..(i + 1) * out_features];
        for (j, o) in orow.iter_mut().enumerate() {
            let wr = &weight.data[j * in_features..(j + 1) * in_features];
            // Plain dot product. The compiler autovectorizes this well enough
            // for a reference implementation.
            let mut acc = 0.0f32;
            for (a, b) in xr.iter().zip(wr.iter()) {
                acc += a * b;
            }
            *o = acc + bias.map_or(0.0, |b| b[j]);
        }
    }
    Ok(out)
}

/// Root-mean-square layer normalization, applied per row in place.
///
/// RMSNorm differs from LayerNorm by omitting mean subtraction:
///
/// ```text
/// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
/// RMSNorm:    x / sqrt(mean(x²) + eps) * gamma
/// ```
///
/// Dropping the mean and the bias costs little quality and saves a pass over
/// the data, which is why every recent architecture uses it.
pub fn rms_norm(x: &mut Matrix, weight: &[f32], eps: f32) -> Result<(), EngineError> {
    let cols = x.cols();
    if weight.len() != cols {
        return Err(EngineError::Internal(format!(
            "rms_norm weight has {} elements, expected {cols}",
            weight.len()
        )));
    }

    for row in x.data.chunks_mut(cols) {
        // Accumulate in f64: for a 8192-wide hidden state, f32 accumulation of
        // squares loses enough precision to shift the normalization visibly.
        let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = ((sum_sq / cols as f64) + eps as f64).sqrt();
        let scale = (1.0 / rms) as f32;
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * scale * w;
        }
    }
    Ok(())
}

/// SiLU (swish) activation: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The SwiGLU feed-forward activation: `SiLU(gate) * up`, elementwise.
///
/// Fusing the two is worth doing even on CPU: it halves the passes over what
/// is, at 4x hidden size, the largest intermediate in the layer.
pub fn swiglu(gate: &mut Matrix, up: &Matrix) -> Result<(), EngineError> {
    if gate.rows() != up.rows() || gate.cols() != up.cols() {
        return Err(EngineError::Internal(format!(
            "swiglu shape mismatch: gate is {}x{}, up is {}x{}",
            gate.rows(),
            gate.cols(),
            up.rows(),
            up.cols()
        )));
    }
    for (g, u) in gate.data.iter_mut().zip(up.data.iter()) {
        *g = silu(*g) * u;
    }
    Ok(())
}

/// Numerically stable in-place softmax over a slice.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = 1.0 / x.len() as f32;
        x.fill(uniform);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// Precomputed rotary position embedding tables.
///
/// The cos/sin values depend only on position and head dimension, never on the
/// data, so they are computed once at load time rather than per token. For a
/// 8192-position, 128-dim model this is 4 MB — trivial next to the weights, and
/// it removes two transcendental functions per element from the hot path.
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// `[max_positions, head_dim/2]`
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
    max_positions: usize,
}

impl RopeTable {
    /// Builds tables for `max_positions` positions.
    pub fn new(head_dim: usize, max_positions: usize, theta: f32) -> Result<Self, EngineError> {
        if !head_dim.is_multiple_of(2) {
            return Err(EngineError::Internal(format!(
                "RoPE requires an even head_dim, got {head_dim}"
            )));
        }
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_positions * half);
        let mut sin = Vec::with_capacity(max_positions * half);

        for pos in 0..max_positions {
            for i in 0..half {
                // Frequency for dimension pair i, following the RoPE paper.
                let inv_freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * inv_freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self {
            cos,
            sin,
            head_dim,
            max_positions,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// The cos/sin rows for a given position.
    pub fn at(&self, pos: usize) -> Option<(&[f32], &[f32])> {
        if pos >= self.max_positions {
            return None;
        }
        let half = self.head_dim / 2;
        Some((
            &self.cos[pos * half..(pos + 1) * half],
            &self.sin[pos * half..(pos + 1) * half],
        ))
    }
}

/// Applies rotary position embedding to one head vector, in place.
///
/// Uses the "rotate-half" convention that HF checkpoints are trained with:
/// dimension `i` is paired with dimension `i + head_dim/2`, **not** with
/// `i + 1`. Getting this wrong produces a model that still generates fluent
/// text but attends to the wrong positions — a failure that no shape check
/// catches, which is why it is stated explicitly here and pinned by a test.
pub fn apply_rope(vec: &mut [f32], cos: &[f32], sin: &[f32]) -> Result<(), EngineError> {
    let half = vec.len() / 2;
    if cos.len() != half || sin.len() != half {
        return Err(EngineError::Internal(format!(
            "RoPE table width {} does not match half of head_dim {half}",
            cos.len()
        )));
    }
    for i in 0..half {
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos[i] - x1 * sin[i];
        vec[i + half] = x0 * sin[i] + x1 * cos[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn matrix_construction_validates_dimensions() {
        assert!(Matrix::new(vec![1.0; 6], 2, 3).is_ok());
        assert!(Matrix::new(vec![1.0; 5], 2, 3).is_err());
    }

    #[test]
    fn linear_computes_x_times_w_transposed() {
        // x is 1x2, weight is 3x2 (out=3, in=2), so output is 1x3.
        let x = Matrix::new(vec![1.0, 2.0], 1, 2).unwrap();
        let w = Matrix::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
        let out = linear(&x, &w, None).unwrap();

        assert_eq!(out.rows(), 1);
        assert_eq!(out.cols(), 3);
        // Rows of W dotted with x: [1,0]·[1,2]=1, [0,1]·[1,2]=2, [1,1]·[1,2]=3
        assert_eq!(out.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_applies_bias() {
        let x = Matrix::new(vec![1.0, 1.0], 1, 2).unwrap();
        let w = Matrix::new(vec![1.0, 1.0], 1, 2).unwrap();
        let out = linear(&x, &w, Some(&[10.0])).unwrap();
        assert_eq!(out.data(), &[12.0]);
    }

    #[test]
    fn linear_handles_multiple_tokens() {
        let x = Matrix::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let w = Matrix::new(vec![2.0, 3.0], 1, 2).unwrap();
        let out = linear(&x, &w, None).unwrap();
        assert_eq!(out.rows(), 2);
        assert_eq!(out.data(), &[2.0, 3.0]);
    }

    #[test]
    fn linear_rejects_mismatched_shapes() {
        let x = Matrix::new(vec![1.0; 3], 1, 3).unwrap();
        let w = Matrix::new(vec![1.0; 4], 2, 2).unwrap();
        assert!(linear(&x, &w, None).is_err());

        let w2 = Matrix::new(vec![1.0; 3], 1, 3).unwrap();
        assert!(
            linear(&x, &w2, Some(&[1.0, 2.0])).is_err(),
            "bad bias length"
        );
    }

    #[test]
    fn rms_norm_normalizes_to_unit_rms() {
        let mut x = Matrix::new(vec![3.0, 4.0], 1, 2).unwrap();
        let ones = vec![1.0, 1.0];
        rms_norm(&mut x, &ones, 0.0).unwrap();

        // rms = sqrt((9+16)/2) = sqrt(12.5) ~= 3.5355
        let expected_rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert!(approx(x.data()[0], 3.0 / expected_rms, 1e-5));
        assert!(approx(x.data()[1], 4.0 / expected_rms, 1e-5));

        // The normalized row should itself have unit RMS.
        let rms: f32 = (x.data().iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!(approx(rms, 1.0, 1e-5));
    }

    #[test]
    fn rms_norm_applies_the_gain_vector() {
        let mut x = Matrix::new(vec![1.0, 1.0], 1, 2).unwrap();
        rms_norm(&mut x, &[2.0, 3.0], 0.0).unwrap();
        // rms of [1,1] is 1, so the result is just the gains.
        assert!(approx(x.data()[0], 2.0, 1e-6));
        assert!(approx(x.data()[1], 3.0, 1e-6));
    }

    #[test]
    fn rms_norm_normalizes_each_row_independently() {
        let mut x = Matrix::new(vec![1.0, 1.0, 100.0, 100.0], 2, 2).unwrap();
        rms_norm(&mut x, &[1.0, 1.0], 0.0).unwrap();
        // Both rows normalize to the same values despite differing scale.
        assert!(approx(x.data()[0], x.data()[2], 1e-4));
    }

    #[test]
    fn rms_norm_survives_an_all_zero_row() {
        let mut x = Matrix::new(vec![0.0, 0.0], 1, 2).unwrap();
        rms_norm(&mut x, &[1.0, 1.0], 1e-5).unwrap();
        assert!(
            x.data().iter().all(|v| v.is_finite()),
            "eps must prevent NaN"
        );
    }

    #[test]
    fn rms_norm_rejects_a_mismatched_weight() {
        let mut x = Matrix::new(vec![1.0; 4], 1, 4).unwrap();
        assert!(rms_norm(&mut x, &[1.0, 1.0], 1e-5).is_err());
    }

    #[test]
    fn silu_matches_known_values() {
        assert!(approx(silu(0.0), 0.0, 1e-6));
        // silu(1) = 1 * sigmoid(1) = 0.7310586
        assert!(approx(silu(1.0), 0.7310586, 1e-5));
        // Negative inputs stay negative but small in magnitude.
        assert!(silu(-1.0) < 0.0 && silu(-1.0) > -0.3);
        // Large positive approaches identity.
        assert!(approx(silu(20.0), 20.0, 1e-3));
    }

    #[test]
    fn swiglu_multiplies_activated_gate_by_up() {
        let mut gate = Matrix::new(vec![1.0, 0.0], 1, 2).unwrap();
        let up = Matrix::new(vec![2.0, 5.0], 1, 2).unwrap();
        swiglu(&mut gate, &up).unwrap();

        assert!(approx(gate.data()[0], silu(1.0) * 2.0, 1e-6));
        assert!(approx(gate.data()[1], 0.0, 1e-6), "silu(0) is 0");
    }

    #[test]
    fn swiglu_rejects_mismatched_shapes() {
        let mut gate = Matrix::zeros(1, 2);
        let up = Matrix::zeros(1, 3);
        assert!(swiglu(&mut gate, &up).is_err());
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        assert!(approx(x.iter().sum::<f32>(), 1.0, 1e-5));
        assert!(x[2] > x[1] && x[1] > x[0]);

        let mut big = vec![1000.0, 1001.0];
        softmax_inplace(&mut big);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!(approx(big.iter().sum::<f32>(), 1.0, 1e-5));
    }

    #[test]
    fn rope_tables_have_the_expected_shape() {
        let t = RopeTable::new(4, 8, 10000.0).unwrap();
        assert_eq!(t.head_dim(), 4);
        let (cos, sin) = t.at(0).unwrap();
        assert_eq!(cos.len(), 2);
        assert_eq!(sin.len(), 2);
        // Position zero has zero angle everywhere.
        assert!(cos.iter().all(|&c| approx(c, 1.0, 1e-6)));
        assert!(sin.iter().all(|&s| approx(s, 0.0, 1e-6)));
        assert!(t.at(8).is_none(), "out of range");
    }

    #[test]
    fn rope_requires_an_even_head_dim() {
        assert!(RopeTable::new(3, 8, 10000.0).is_err());
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let t = RopeTable::new(4, 4, 10000.0).unwrap();
        let (cos, sin) = t.at(0).unwrap();
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let before = v.clone();
        apply_rope(&mut v, cos, sin).unwrap();
        for (a, b) in v.iter().zip(before.iter()) {
            assert!(approx(*a, *b, 1e-6));
        }
    }

    #[test]
    fn rope_pairs_i_with_i_plus_half_not_i_plus_one() {
        // The convention that HF checkpoints are trained with. Getting this
        // wrong yields a model that generates fluent text while attending to
        // the wrong positions, which no shape check would catch.
        let cos = [0.0, 0.0];
        let sin = [1.0, 1.0];
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut v, &cos, &sin).unwrap();

        // With cos=0, sin=1: out[i] = -x[i+half], out[i+half] = x[i]
        // Pairing (0,2) and (1,3):
        assert!(approx(v[0], -3.0, 1e-6), "v[0] pairs with v[2]");
        assert!(approx(v[1], -4.0, 1e-6), "v[1] pairs with v[3]");
        assert!(approx(v[2], 1.0, 1e-6));
        assert!(approx(v[3], 2.0, 1e-6));
    }

    #[test]
    fn rope_preserves_vector_norm() {
        // Rotation is orthogonal, so it must not change magnitude.
        let t = RopeTable::new(8, 16, 10000.0).unwrap();
        let (cos, sin) = t.at(7).unwrap();
        let mut v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        apply_rope(&mut v, cos, sin).unwrap();
        let after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx(before, after, 1e-4), "{before} vs {after}");
    }

    #[test]
    fn rope_encodes_relative_position() {
        // The defining property: the dot product of two RoPE'd vectors depends
        // only on the difference between their positions.
        let t = RopeTable::new(8, 64, 10000.0).unwrap();
        let base: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();

        let rotate = |pos: usize| {
            let (c, s) = t.at(pos).unwrap();
            let mut v = base.clone();
            apply_rope(&mut v, c, s).unwrap();
            v
        };
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();

        // Positions 5 and 8 differ by 3; so do 20 and 23.
        let d1 = dot(&rotate(5), &rotate(8));
        let d2 = dot(&rotate(20), &rotate(23));
        assert!(
            approx(d1, d2, 1e-3),
            "RoPE should be relative: {d1} vs {d2}"
        );
    }

    #[test]
    fn rope_rejects_a_mismatched_table_width() {
        let mut v = vec![1.0; 4];
        assert!(apply_rope(&mut v, &[1.0], &[0.0]).is_err());
    }
}
