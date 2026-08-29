//! Microbenchmarks for the engine's own hot paths.
//!
//! These measure OrionServe's data structures — the block allocator, the
//! scheduler, prefix hashing, sampling — rather than model execution. They are
//! meaningful on any machine, including one with no GPU, because none of these
//! paths touch a device.
//!
//! What they are *for*: catching regressions in the code this project owns. A
//! scheduler pass that becomes 10x slower is a real problem even if the forward
//! pass still dominates the wall clock, because it means an algorithmic mistake
//! has crept in.
//!
//! What they are *not*: an inference benchmark. Tokens per second is a
//! whole-system property measured by `orion-bench` against a running server.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use orion_core::{SamplingParams, SchedulerConfig, Sequence, TokenId};
use orion_kv_cache::{hash_block, KvCacheManager};
use orion_scheduler::Scheduler;

/// Block allocation and release, the KV cache hot path.
fn bench_block_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");

    for &prompt_len in &[128usize, 1024, 4096] {
        group.bench_with_input(
            BenchmarkId::new("allocate_and_free", prompt_len),
            &prompt_len,
            |b, &len| {
                let prompt: Vec<TokenId> = (0..len as u32).collect();
                b.iter_batched(
                    || KvCacheManager::new(8192, 16, false),
                    |mut cache| {
                        let seq = orion_core::SequenceId::next();
                        cache.allocate(seq, black_box(&prompt)).unwrap();
                        cache.free(seq);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // The decode-step path: one token appended to an existing sequence. This
    // runs once per sequence per step, so it is the most frequently executed
    // cache operation by a wide margin.
    group.bench_function("append_token", |b| {
        let mut cache = KvCacheManager::new(65536, 16, false);
        let seq = orion_core::SequenceId::next();
        cache.allocate(seq, &vec![1u32; 128]).unwrap();
        b.iter(|| {
            cache.append_token(black_box(seq)).unwrap();
        });
    });

    group.finish();
}

/// Prefix hashing, which runs over every block of every admitted prompt.
fn bench_prefix_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache");

    for &block_size in &[16usize, 64, 256] {
        let tokens: Vec<TokenId> = (0..block_size as u32).collect();
        group.bench_with_input(
            BenchmarkId::new("hash_block", block_size),
            &tokens,
            |b, toks| {
                b.iter(|| hash_block(black_box(None), black_box(toks)));
            },
        );
    }

    // A full prompt's worth of chained hashing, which is what admission
    // actually pays.
    group.bench_function("hash_2048_token_prompt", |b| {
        let tokens: Vec<TokenId> = (0..2048).collect();
        b.iter(|| {
            let mut chain = None;
            for chunk in tokens.chunks(16) {
                chain = Some(hash_block(chain, black_box(chunk)));
            }
            chain
        });
    });

    // Reuse: the second identical prompt should be cheaper overall because it
    // skips prefill entirely. This measures the lookup side of that.
    group.bench_function("allocate_with_cache_hit", |b| {
        let prompt: Vec<TokenId> = (0..512).collect();
        b.iter_batched(
            || {
                let mut cache = KvCacheManager::new(8192, 16, true);
                let warm = orion_core::SequenceId::next();
                cache.allocate(warm, &prompt).unwrap();
                cache.commit_prefill(warm, &prompt);
                cache.free(warm);
                cache
            },
            |mut cache| {
                let seq = orion_core::SequenceId::next();
                let out = cache.allocate(seq, black_box(&prompt)).unwrap();
                debug_assert!(out.reused > 0, "expected a cache hit");
                cache.free(seq);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// A scheduling pass, which runs once per engine step.
fn bench_scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler");

    for &num_seqs in &[8usize, 64, 256] {
        group.bench_with_input(
            BenchmarkId::new("decode_step", num_seqs),
            &num_seqs,
            |b, &n| {
                b.iter_batched(
                    || {
                        let cache = KvCacheManager::new(65536, 16, false);
                        let mut sched = Scheduler::new(
                            SchedulerConfig {
                                max_num_seqs: n,
                                max_num_batched_tokens: 8192,
                                max_model_len: Some(4096),
                                request_timeout_secs: None,
                                ..Default::default()
                            },
                            cache,
                        );
                        // Bring every sequence to the decoding state so the
                        // measured pass is a pure decode step.
                        for _ in 0..n {
                            let seq = Sequence::new(
                                vec![1u32; 64],
                                SamplingParams::default().with_max_tokens(1024),
                            );
                            sched.add_request(seq).unwrap();
                        }
                        sched.schedule();
                        let ids: Vec<_> = (0..n)
                            .filter_map(|_| None::<orion_core::SequenceId>)
                            .collect();
                        let _ = ids;
                        sched
                    },
                    |mut sched| {
                        black_box(sched.schedule());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Admission, which every request pays once.
    group.bench_function("add_request", |b| {
        b.iter_batched(
            || {
                Scheduler::new(
                    SchedulerConfig {
                        max_waiting_requests: 100_000,
                        max_model_len: Some(4096),
                        request_timeout_secs: None,
                        ..Default::default()
                    },
                    KvCacheManager::new(65536, 16, false),
                )
            },
            |mut sched| {
                let seq = Sequence::new(vec![1u32; 128], SamplingParams::default());
                black_box(sched.add_request(seq).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Sampling, which runs once per sequence per step over a vocabulary-sized row.
fn bench_sampling(c: &mut Criterion) {
    use orion_core::Sampler;
    use orion_runtime::DefaultSampler;

    let mut group = c.benchmark_group("sampling");

    // Realistic vocabulary sizes. The 128k case is Llama 3.
    for &vocab in &[32_000usize, 128_256] {
        let logits: Vec<f32> = (0..vocab)
            .map(|i| ((i as f32 * 0.001).sin()) * 5.0)
            .collect();
        let context: Vec<TokenId> = (0..256).collect();

        group.bench_with_input(BenchmarkId::new("greedy", vocab), &logits, |b, l| {
            let mut sampler = DefaultSampler::seeded(1);
            let params = SamplingParams {
                temperature: 0.0,
                ..Default::default()
            };
            b.iter_batched(
                || l.clone(),
                |mut scratch| {
                    black_box(
                        sampler
                            .sample(&mut scratch, black_box(&context), &params)
                            .unwrap(),
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("top_k_top_p", vocab), &logits, |b, l| {
            let mut sampler = DefaultSampler::seeded(1);
            let params = SamplingParams {
                temperature: 0.8,
                top_k: 50,
                top_p: 0.95,
                repetition_penalty: 1.1,
                ..Default::default()
            };
            b.iter_batched(
                || l.clone(),
                |mut scratch| {
                    black_box(
                        sampler
                            .sample(&mut scratch, black_box(&context), &params)
                            .unwrap(),
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// CPU tensor primitives. These are the reference implementation, so the
/// numbers matter mainly as a baseline for future CUDA comparison.
fn bench_tensor_ops(c: &mut Criterion) {
    use orion_models::tensor::{linear, rms_norm, swiglu};
    use orion_models::Matrix;

    let mut group = c.benchmark_group("tensor_cpu");
    // Deliberately small: this is an unoptimized reference implementation, and
    // benchmarking it at production dimensions would take minutes per sample.
    let hidden = 512;
    let tokens = 8;

    group.bench_function("rms_norm", |b| {
        let weight = vec![1.0f32; hidden];
        b.iter_batched(
            || Matrix::new(vec![0.5f32; tokens * hidden], tokens, hidden).unwrap(),
            |mut x| {
                rms_norm(&mut x, black_box(&weight), 1e-5).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("linear", |b| {
        let x = Matrix::new(vec![0.5f32; tokens * hidden], tokens, hidden).unwrap();
        let w = Matrix::new(vec![0.1f32; hidden * hidden], hidden, hidden).unwrap();
        b.iter(|| black_box(linear(black_box(&x), black_box(&w), None).unwrap()));
    });

    group.bench_function("swiglu", |b| {
        let up = Matrix::new(vec![0.5f32; tokens * hidden], tokens, hidden).unwrap();
        b.iter_batched(
            || Matrix::new(vec![0.3f32; tokens * hidden], tokens, hidden).unwrap(),
            |mut gate| {
                swiglu(&mut gate, black_box(&up)).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_block_allocation,
    bench_prefix_hash,
    bench_scheduler,
    bench_sampling,
    bench_tensor_ops
);
criterion_main!(benches);
