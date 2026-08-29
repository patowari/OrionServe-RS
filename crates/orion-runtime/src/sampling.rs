//! The sampling engine: logits in, one token out.
//!
//! Sampling is kept entirely separate from model execution. That separation is
//! what makes it testable — every function here can be driven against a
//! hand-written logits row with a known answer, with no model, no cache and no
//! GPU involved.
//!
//! # Filter order
//!
//! The order the filters are applied in is not arbitrary:
//!
//! ```text
//! logits
//!   └─► repetition penalty   (operates on raw logits, sign-aware)
//!       └─► temperature      (scales before any probability is formed)
//!           └─► top-k        (cheap, shrinks the candidate set)
//!               └─► top-p    (needs a normalized distribution)
//!                   └─► sample
//! ```
//!
//! Repetition penalty comes first because it is defined on raw logits, and
//! applying it after temperature would make its strength depend on temperature.
//! Top-k runs before top-p because it is O(n) to bound and cheaply shrinks the
//! set that top-p must sort.

use std::collections::HashSet;

use orion_core::{EngineError, Sampler, SamplingMode, SamplingParams, TokenId};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Sentinel used to mask out filtered tokens.
///
/// `f32::NEG_INFINITY` would be the obvious choice, but it produces `NaN` from
/// `exp(-inf - -inf)` if every candidate is masked. A large finite negative
/// underflows to exactly zero probability while staying arithmetically safe.
const MASKED: f32 = -1.0e30;

/// Applies a repetition penalty to tokens already present in the context.
///
/// Follows the CTRL formulation: positive logits are *divided* by the penalty
/// and negative logits are *multiplied* by it. Both directions move the logit
/// down, which a naive "subtract a constant" would not — for a negative logit,
/// dividing would make it *larger*.
pub fn apply_repetition_penalty(logits: &mut [f32], previous: &[TokenId], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    // Deduplicate: the penalty is applied once per distinct token, not once per
    // occurrence, so a long repeated context does not drive a logit to zero.
    let seen: HashSet<TokenId> = previous.iter().copied().collect();
    for token in seen {
        let idx = token as usize;
        if idx >= logits.len() {
            continue;
        }
        let l = logits[idx];
        logits[idx] = if l > 0.0 { l / penalty } else { l * penalty };
    }
}

/// Scales logits by `1 / temperature`.
///
/// Higher temperature flattens the distribution; lower sharpens it. A
/// temperature of zero means greedy and is handled before this is called, so
/// there is no division by zero here.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

/// Masks all but the `k` highest logits.
///
/// `k == 0` or `k >= len` disables the filter. Uses `select_nth_unstable` for
/// O(n) selection rather than a full O(n log n) sort — over a 128k vocabulary
/// on every token of every sequence, that difference is worth having.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    // Partition so index k-1 holds the k-th largest value.
    sorted.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = sorted[k - 1];

    for l in logits.iter_mut() {
        if *l < threshold {
            *l = MASKED;
        }
    }
}

/// Masks the tail of the distribution beyond cumulative probability `p`.
///
/// Operates on probabilities, so it softmaxes internally. Tokens are kept in
/// descending probability order until the cumulative mass reaches `p`; the
/// token that crosses the threshold is kept, so the retained mass is always
/// at least `p` and the candidate set is never empty.
///
/// Only *unmasked* entries are collected and sorted. After top-k over a 128k
/// vocabulary the vast majority of entries are at [`MASKED`] and contribute
/// exactly zero probability, so they can never be inside the nucleus; sorting
/// them was measurable waste. See `docs/performance-journal.md`.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax(logits);

    // Collect only live candidates. A masked logit has effectively zero
    // probability, so it can never be inside the nucleus and never needs
    // sorting.
    let mut candidates: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|&(i, &pr)| pr > 0.0 && logits[i] > MASKED)
        .map(|(i, &pr)| (i, pr))
        .collect();

    if candidates.is_empty() {
        return;
    }

    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut cutoff = candidates.len();
    for (rank, &(_, pr)) in candidates.iter().enumerate() {
        cumulative += pr;
        if cumulative >= p {
            // Keep the token that crossed the threshold, so the retained mass
            // is always at least `p` and the set is never empty.
            cutoff = rank + 1;
            break;
        }
    }

    for &(idx, _) in &candidates[cutoff..] {
        logits[idx] = MASKED;
    }
}

/// Numerically stable softmax.
///
/// Subtracts the maximum before exponentiating; without that, a logit of 100
/// overflows `f32::exp` to infinity and the whole distribution becomes `NaN`.
///
/// Masked entries are skipped rather than exponentiated. After top-k over a
/// 128k vocabulary, all but `k` entries are at [`MASKED`], and `exp` on them
/// costs a transcendental call to produce a number that underflows to zero
/// anyway. Skipping them was the single largest cost in the decode-step
/// sampler — see `docs/performance-journal.md`.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    if !max.is_finite() || max <= MASKED {
        // Every logit was masked or non-finite. Fall back to uniform rather
        // than producing NaN.
        return vec![1.0 / logits.len() as f32; logits.len()];
    }

    let mut out = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        if l <= MASKED {
            continue; // already zero
        }
        let e = (l - max).exp();
        *o = e;
        sum += e;
    }

    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    } else {
        out.fill(1.0 / logits.len() as f32);
    }
    out
}

/// Returns the index of the largest logit.
///
/// Ties resolve to the lowest index, which makes greedy decoding fully
/// deterministic rather than dependent on iteration order.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    let mut best = None;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_val {
            best_val = l;
            best = Some(i);
        }
    }
    best
}

/// Draws an index from a probability distribution.
fn sample_from(probs: &[f32], rng: &mut ChaCha8Rng) -> usize {
    let r: f32 = rng.random::<f32>();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    // Floating-point accumulation can leave `r` fractionally above the total.
    // Fall back to the last non-zero entry rather than returning out of range.
    probs
        .iter()
        .rposition(|&p| p > 0.0)
        .unwrap_or(probs.len().saturating_sub(1))
}

/// The default sampler: repetition penalty, temperature, top-k, top-p.
///
/// Holds its own RNG so that a seeded request is reproducible independently of
/// how many other requests are in flight — a shared global RNG would make
/// output depend on interleaving, which would make seeded runs unreproducible
/// under concurrency.
#[derive(Debug)]
pub struct DefaultSampler {
    rng: ChaCha8Rng,
}

impl DefaultSampler {
    /// Creates a sampler seeded from system entropy.
    pub fn new() -> Self {
        Self {
            rng: ChaCha8Rng::from_rng(&mut rand::rng()),
        }
    }

    /// Creates a sampler with a fixed seed, for reproducible generation.
    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Creates a sampler honouring `params.seed`, falling back to entropy.
    pub fn for_params(params: &SamplingParams) -> Self {
        match params.seed {
            Some(seed) => Self::seeded(seed),
            None => Self::new(),
        }
    }
}

impl Default for DefaultSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for DefaultSampler {
    fn sample(
        &mut self,
        logits: &mut [f32],
        previous_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<TokenId, EngineError> {
        if logits.is_empty() {
            return Err(EngineError::Internal(
                "sampler received an empty logits row".into(),
            ));
        }

        apply_repetition_penalty(logits, previous_tokens, params.repetition_penalty);

        if params.mode() == SamplingMode::Greedy {
            let idx = argmax(logits)
                .ok_or_else(|| EngineError::Internal("argmax over empty logits".into()))?;
            return Ok(idx as TokenId);
        }

        apply_temperature(logits, params.temperature);
        apply_top_k(logits, params.top_k);
        apply_top_p(logits, params.top_p);

        let probs = softmax(logits);
        Ok(sample_from(&probs, &mut self.rng) as TokenId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SamplingParams {
        SamplingParams::default()
    }

    #[test]
    fn argmax_picks_the_largest_and_breaks_ties_low() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), Some(1));
        assert_eq!(argmax(&[5.0, 5.0, 1.0]), Some(0), "ties go to lowest index");
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn greedy_decoding_is_deterministic() {
        let mut s = DefaultSampler::seeded(1);
        let p = SamplingParams {
            temperature: 0.0,
            ..params()
        };
        for _ in 0..20 {
            let mut logits = vec![0.1, 0.9, 0.3, 0.2];
            assert_eq!(s.sample(&mut logits, &[], &p).unwrap(), 1);
        }
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable_for_large_logits() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);

        // Without max-subtraction this overflows to NaN.
        let big = softmax(&[1000.0, 1001.0, 999.0]);
        assert!(big.iter().all(|p| p.is_finite()), "{big:?}");
        let sum: f32 = big.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_of_fully_masked_logits_does_not_produce_nan() {
        let probs = softmax(&[MASKED, MASKED, MASKED]);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let base = [1.0f32, 2.0, 3.0];

        let mut hot = base;
        apply_temperature(&mut hot, 2.0);
        let hot_p = softmax(&hot);

        let mut cold = base;
        apply_temperature(&mut cold, 0.5);
        let cold_p = softmax(&cold);

        let flat = softmax(&base);
        assert!(
            cold_p[2] > flat[2] && flat[2] > hot_p[2],
            "low temperature must concentrate mass on the top token"
        );
    }

    #[test]
    fn temperature_of_one_is_a_no_op() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let before = logits.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn top_k_keeps_exactly_k_candidates() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        let kept: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, &l)| l > MASKED)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kept, vec![1, 4], "the two largest are 5.0 and 4.0");
    }

    #[test]
    fn top_k_is_disabled_at_zero_or_beyond_the_vocabulary() {
        let original = vec![1.0, 5.0, 3.0];
        for k in [0, 3, 100] {
            let mut logits = original.clone();
            apply_top_k(&mut logits, k);
            assert_eq!(logits, original, "k={k} should be a no-op");
        }
    }

    #[test]
    fn top_k_never_masks_everything() {
        let mut logits = vec![2.0, 2.0, 2.0, 2.0];
        apply_top_k(&mut logits, 1);
        assert!(
            logits.iter().any(|&l| l > MASKED),
            "at least one candidate must survive"
        );
    }

    #[test]
    fn top_p_keeps_the_nucleus_and_masks_the_tail() {
        // Probabilities approximately 0.87 / 0.12 / 0.006 / 0.002.
        let mut logits = vec![5.0, 3.0, 0.0, -1.0];
        apply_top_p(&mut logits, 0.9);

        assert!(logits[0] > MASKED, "the top token must always survive");
        assert!(logits[2] <= MASKED, "the tail must be masked");
        assert!(logits[3] <= MASKED);
    }

    #[test]
    fn top_p_of_one_is_a_no_op() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn top_p_always_keeps_at_least_one_token() {
        // Even a tiny p must not produce an empty candidate set.
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        apply_top_p(&mut logits, 0.01);
        assert_eq!(
            logits.iter().filter(|&&l| l > MASKED).count(),
            1,
            "exactly the single most probable token"
        );
    }

    #[test]
    fn repetition_penalty_lowers_logits_in_both_directions() {
        // The sign-aware part: a naive subtract would raise the negative logit.
        let mut logits = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);

        assert_eq!(logits[0], 1.0, "positive logit divided");
        assert_eq!(logits[1], -4.0, "negative logit multiplied");
        assert_eq!(logits[2], 1.0, "untouched token unchanged");
    }

    #[test]
    fn repetition_penalty_applies_once_per_distinct_token() {
        let mut logits = vec![4.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 0, 0, 0], 2.0);
        assert_eq!(logits[0], 2.0, "four occurrences, one application");
    }

    #[test]
    fn repetition_penalty_of_one_is_a_no_op() {
        let mut logits = vec![2.0, -2.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
        assert_eq!(logits, vec![2.0, -2.0]);
    }

    #[test]
    fn repetition_penalty_ignores_out_of_range_tokens() {
        let mut logits = vec![1.0, 2.0];
        apply_repetition_penalty(&mut logits, &[99], 2.0);
        assert_eq!(logits, vec![1.0, 2.0], "must not panic or corrupt");
    }

    #[test]
    fn the_same_seed_produces_the_same_sequence() {
        let p = SamplingParams {
            temperature: 1.0,
            ..params()
        };
        let run = |seed: u64| {
            let mut s = DefaultSampler::seeded(seed);
            (0..30)
                .map(|_| {
                    let mut logits = vec![1.0, 2.0, 1.5, 0.5, 3.0];
                    s.sample(&mut logits, &[], &p).unwrap()
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(run(42), run(42), "seeded sampling must be reproducible");
        assert_ne!(run(42), run(43), "different seeds should diverge");
    }

    #[test]
    fn sampling_always_returns_a_valid_token_id() {
        let mut s = DefaultSampler::seeded(7);
        let p = SamplingParams {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ..params()
        };
        for _ in 0..200 {
            let mut logits = vec![1.0, 2.0, 3.0, 0.5, -1.0, 4.0, 0.0, 2.5];
            let t = s.sample(&mut logits, &[1, 2], &p).unwrap();
            assert!((t as usize) < 8, "token {t} out of vocabulary range");
        }
    }

    #[test]
    fn an_empty_logits_row_is_an_internal_error_not_a_panic() {
        let mut s = DefaultSampler::seeded(1);
        let err = s.sample(&mut [], &[], &params()).unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    #[test]
    fn stochastic_sampling_respects_a_top_k_of_one() {
        // With k=1 the distribution collapses, so sampling must equal greedy.
        let mut s = DefaultSampler::seeded(3);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..params()
        };
        for _ in 0..50 {
            let mut logits = vec![1.0, 9.0, 2.0];
            assert_eq!(s.sample(&mut logits, &[], &p).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_reflects_the_distribution() {
        // A statistical check: a strongly favoured token should dominate.
        let mut s = DefaultSampler::seeded(11);
        let p = SamplingParams {
            temperature: 1.0,
            ..params()
        };
        let mut counts = [0usize; 3];
        for _ in 0..3000 {
            let mut logits = vec![0.0, 4.0, 0.0];
            counts[s.sample(&mut logits, &[], &p).unwrap() as usize] += 1;
        }
        // softmax([0,4,0]) puts roughly 96% of mass on index 1.
        assert!(
            counts[1] > 2500,
            "expected the dominant token to win most draws, got {counts:?}"
        );
        assert!(
            counts[0] > 0 && counts[2] > 0,
            "sampling should not be greedy"
        );
    }
}
