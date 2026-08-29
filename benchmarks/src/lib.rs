//! Benchmark harness for OrionServe-RS.
//!
//! # What this measures, and what it does not
//!
//! This drives a running server over HTTP and records what a client actually
//! experiences: time to first token, time per output token, end-to-end latency,
//! and throughput. It deliberately measures from *outside* the server, because
//! internal timings can be made to look good in ways a client never sees.
//!
//! Every result carries the hardware it was produced on. A tokens/second figure
//! without a GPU model, a batch size and a prompt distribution is not a
//! measurement, it is a decoration.
//!
//! # Honesty rules this harness enforces
//!
//! * Warm-up requests are run and discarded, so JIT-like effects (page faults,
//!   cache warming, allocator growth) do not flatter the first measurement.
//! * Percentiles come from the full sample, never from a mean plus an assumed
//!   distribution.
//! * A failed request is recorded as a failure, not dropped. Silently ignoring
//!   errors is the easiest way to publish a fast-looking benchmark of a broken
//!   server.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Hardware and configuration a result was produced on.
///
/// Recorded automatically where possible. Fields that cannot be detected are
/// left for the operator to fill in rather than guessed at.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub timestamp_utc: String,
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    /// GPU name, if one was detected. `None` means the run was CPU-only, and
    /// results must not be presented as GPU numbers.
    pub gpu: Option<String>,
    pub cuda_version: Option<String>,
    pub driver_version: Option<String>,
    pub orion_version: String,
    pub model: String,
    pub precision: String,
    pub notes: Option<String>,
}

impl RunMetadata {
    /// Collects what can be detected from the host.
    pub fn detect(model: String, precision: String) -> Self {
        Self {
            timestamp_utc: format_now(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0),
            gpu: detect_gpu(),
            cuda_version: detect_cuda(),
            driver_version: detect_driver(),
            orion_version: env!("CARGO_PKG_VERSION").to_string(),
            model,
            precision,
            notes: None,
        }
    }

    /// Whether this run had GPU acceleration available.
    ///
    /// Results from a run where this is false must never be described as GPU
    /// performance.
    pub fn is_gpu_run(&self) -> bool {
        self.gpu.is_some()
    }
}

fn format_now() -> String {
    // Seconds since epoch, formatted plainly. A date library would be a
    // dependency for one field.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

fn run_tool(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn detect_gpu() -> Option<String> {
    run_tool(
        "nvidia-smi",
        &["--query-gpu=name", "--format=csv,noheader"],
    )
    .map(|s| s.lines().next().unwrap_or_default().trim().to_string())
    .filter(|s| !s.is_empty())
}

fn detect_driver() -> Option<String> {
    run_tool(
        "nvidia-smi",
        &["--query-gpu=driver_version", "--format=csv,noheader"],
    )
    .map(|s| s.lines().next().unwrap_or_default().trim().to_string())
    .filter(|s| !s.is_empty())
}

fn detect_cuda() -> Option<String> {
    let out = run_tool("nvcc", &["--version"])?;
    out.lines()
        .find(|l| l.contains("release"))
        .and_then(|l| l.split("release").nth(1))
        .map(|s| s.trim().trim_end_matches(',').to_string())
}

/// The shape of a workload.
///
/// Prompt and output lengths interact very differently with the scheduler:
/// long prompts stress prefill and chunking, long outputs stress the KV cache
/// and decode batching. Reporting one number across a mixed workload hides
/// both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadShape {
    /// Short prompt, short output. Chat-like.
    ShortShort,
    /// Short prompt, long output. Generation-heavy.
    ShortLong,
    /// Long prompt, short output. Summarization / RAG-like.
    LongShort,
    /// Long prompt, long output. The most demanding on cache.
    LongLong,
}

impl WorkloadShape {
    pub fn prompt_tokens(self) -> usize {
        match self {
            WorkloadShape::ShortShort | WorkloadShape::ShortLong => 128,
            WorkloadShape::LongShort | WorkloadShape::LongLong => 2048,
        }
    }

    pub fn output_tokens(self) -> usize {
        match self {
            WorkloadShape::ShortShort | WorkloadShape::LongShort => 128,
            WorkloadShape::ShortLong | WorkloadShape::LongLong => 1024,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            WorkloadShape::ShortShort => "short_prompt_short_output",
            WorkloadShape::ShortLong => "short_prompt_long_output",
            WorkloadShape::LongShort => "long_prompt_short_output",
            WorkloadShape::LongLong => "long_prompt_long_output",
        }
    }

    pub fn all() -> [WorkloadShape; 4] {
        [
            WorkloadShape::ShortShort,
            WorkloadShape::ShortLong,
            WorkloadShape::LongShort,
            WorkloadShape::LongLong,
        ]
    }
}

/// One request's measured timings.
#[derive(Debug, Clone)]
pub struct RequestSample {
    pub ttft: Option<Duration>,
    pub total: Duration,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub success: bool,
}

impl RequestSample {
    /// Mean time per output token, excluding the first.
    ///
    /// `None` below two tokens: TPOT is undefined there rather than zero, and
    /// reporting zero would drag an average down misleadingly.
    pub fn tpot(&self) -> Option<Duration> {
        let ttft = self.ttft?;
        if self.completion_tokens < 2 {
            return None;
        }
        let decode_time = self.total.checked_sub(ttft)?;
        Some(decode_time / (self.completion_tokens as u32 - 1))
    }

    pub fn failed(prompt_tokens: usize) -> Self {
        Self {
            ttft: None,
            total: Duration::ZERO,
            prompt_tokens,
            completion_tokens: 0,
            success: false,
        }
    }
}

/// Aggregated results for one configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub metadata: RunMetadata,
    pub workload: String,
    pub concurrency: usize,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub duration_secs: f64,

    pub request_throughput: f64,
    pub output_token_throughput: f64,
    pub total_token_throughput: f64,

    pub ttft_ms: Option<Percentiles>,
    pub tpot_ms: Option<Percentiles>,
    pub latency_ms: Percentiles,

    pub total_prompt_tokens: usize,
    pub total_completion_tokens: usize,
}

/// Distribution summary. Percentiles come from the sorted sample, not from an
/// assumed distribution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Percentiles {
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

impl Percentiles {
    /// Computes percentiles from raw values in milliseconds.
    ///
    /// Returns `None` for an empty sample rather than zeros: "no data" and
    /// "zero latency" must not look the same in a results table.
    pub fn from_millis(mut values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Some(Self {
            mean: values.iter().sum::<f64>() / values.len() as f64,
            p50: percentile(&values, 0.50),
            p90: percentile(&values, 0.90),
            p95: percentile(&values, 0.95),
            p99: percentile(&values, 0.99),
            min: values[0],
            max: values[values.len() - 1],
        })
    }
}

/// Nearest-rank percentile of a sorted slice.
///
/// Nearest-rank rather than interpolated: with the sample sizes a load test
/// produces, an interpolated p99 invents a value that was never observed.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Aggregates samples into a result.
pub fn aggregate(
    metadata: RunMetadata,
    workload: &str,
    concurrency: usize,
    samples: &[RequestSample],
    wall_clock: Duration,
) -> BenchmarkResult {
    let successful: Vec<&RequestSample> = samples.iter().filter(|s| s.success).collect();
    let secs = wall_clock.as_secs_f64().max(f64::EPSILON);

    let total_completion: usize = successful.iter().map(|s| s.completion_tokens).sum();
    let total_prompt: usize = successful.iter().map(|s| s.prompt_tokens).sum();

    let ttft_ms: Vec<f64> = successful
        .iter()
        .filter_map(|s| s.ttft)
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    let tpot_ms: Vec<f64> = successful
        .iter()
        .filter_map(|s| s.tpot())
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    let latency_ms: Vec<f64> = successful
        .iter()
        .map(|s| s.total.as_secs_f64() * 1000.0)
        .collect();

    BenchmarkResult {
        metadata,
        workload: workload.to_string(),
        concurrency,
        total_requests: samples.len(),
        successful_requests: successful.len(),
        failed_requests: samples.len() - successful.len(),
        duration_secs: secs,
        request_throughput: successful.len() as f64 / secs,
        output_token_throughput: total_completion as f64 / secs,
        total_token_throughput: (total_completion + total_prompt) as f64 / secs,
        ttft_ms: Percentiles::from_millis(ttft_ms),
        tpot_ms: Percentiles::from_millis(tpot_ms),
        latency_ms: Percentiles::from_millis(latency_ms).unwrap_or(Percentiles {
            mean: 0.0,
            p50: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
        }),
        total_prompt_tokens: total_prompt,
        total_completion_tokens: total_completion,
    }
}

impl BenchmarkResult {
    /// Renders a human-readable summary.
    ///
    /// Always states the hardware, and says plainly when a run was CPU-only, so
    /// a figure cannot be quoted as a GPU result by accident.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("workload:        {}\n", self.workload));
        out.push_str(&format!("concurrency:     {}\n", self.concurrency));
        out.push_str(&format!(
            "requests:        {} ok, {} failed\n",
            self.successful_requests, self.failed_requests
        ));
        out.push_str(&format!("duration:        {:.2}s\n", self.duration_secs));
        out.push('\n');
        out.push_str(&format!(
            "throughput:      {:.2} req/s, {:.1} output tok/s\n",
            self.request_throughput, self.output_token_throughput
        ));

        if let Some(p) = self.ttft_ms {
            out.push_str(&format!(
                "TTFT (ms):       p50 {:.1}  p90 {:.1}  p99 {:.1}\n",
                p.p50, p.p90, p.p99
            ));
        }
        if let Some(p) = self.tpot_ms {
            out.push_str(&format!(
                "TPOT (ms):       p50 {:.2}  p90 {:.2}  p99 {:.2}\n",
                p.p50, p.p90, p.p99
            ));
        }
        out.push_str(&format!(
            "latency (ms):    p50 {:.1}  p90 {:.1}  p99 {:.1}\n",
            self.latency_ms.p50, self.latency_ms.p90, self.latency_ms.p99
        ));

        out.push('\n');
        out.push_str(&format!(
            "hardware:        {} {} ({} cores)\n",
            self.metadata.os, self.metadata.arch, self.metadata.cpu_cores
        ));
        match &self.metadata.gpu {
            Some(gpu) => out.push_str(&format!("GPU:             {gpu}\n")),
            None => out.push_str(
                "GPU:             none detected - THIS IS A CPU-ONLY RESULT\n",
            ),
        }
        out.push_str(&format!("model:           {}\n", self.metadata.model));
        out.push_str(&format!("precision:       {}\n", self.metadata.precision));
        out
    }

    /// One CSV row, for spreadsheets and trend tracking.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{}",
            self.workload,
            self.concurrency,
            self.successful_requests,
            self.failed_requests,
            self.total_completion_tokens,
            self.request_throughput,
            self.output_token_throughput,
            self.ttft_ms.map(|p| p.p50).unwrap_or(f64::NAN),
            self.ttft_ms.map(|p| p.p99).unwrap_or(f64::NAN),
            self.tpot_ms.map(|p| p.p50).unwrap_or(f64::NAN),
            self.latency_ms.p50,
            self.latency_ms.p99,
            self.metadata.gpu.as_deref().unwrap_or("cpu-only"),
        )
    }

    pub fn csv_header() -> &'static str {
        "workload,concurrency,ok,failed,output_tokens,req_per_s,out_tok_per_s,\
         ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,latency_p50_ms,latency_p99_ms,gpu"
    }
}

/// Measures one request's timings from a stream of token arrival instants.
pub fn measure(
    start: Instant,
    token_times: &[Instant],
    prompt_tokens: usize,
) -> RequestSample {
    let end = token_times.last().copied().unwrap_or(start);
    RequestSample {
        ttft: token_times.first().map(|t| *t - start),
        total: end - start,
        prompt_tokens,
        completion_tokens: token_times.len(),
        success: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ttft_ms: u64, total_ms: u64, tokens: usize) -> RequestSample {
        RequestSample {
            ttft: Some(Duration::from_millis(ttft_ms)),
            total: Duration::from_millis(total_ms),
            prompt_tokens: 100,
            completion_tokens: tokens,
            success: true,
        }
    }

    #[test]
    fn percentiles_come_from_the_observed_sample() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p = Percentiles::from_millis(values).unwrap();

        assert_eq!(p.min, 1.0);
        assert_eq!(p.max, 100.0);
        assert_eq!(p.p50, 50.0);
        assert_eq!(p.p90, 90.0);
        assert_eq!(p.p99, 99.0);
        assert!((p.mean - 50.5).abs() < 1e-9);
    }

    #[test]
    fn percentiles_are_none_for_an_empty_sample() {
        // "No data" and "zero latency" must not look the same.
        assert!(Percentiles::from_millis(vec![]).is_none());
    }

    #[test]
    fn a_single_observation_is_every_percentile() {
        let p = Percentiles::from_millis(vec![42.0]).unwrap();
        assert_eq!(p.p50, 42.0);
        assert_eq!(p.p99, 42.0);
        assert_eq!(p.min, 42.0);
        assert_eq!(p.max, 42.0);
    }

    #[test]
    fn percentiles_are_order_independent() {
        let ascending = Percentiles::from_millis(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let shuffled = Percentiles::from_millis(vec![3.0, 1.0, 4.0, 2.0]).unwrap();
        assert_eq!(ascending.p50, shuffled.p50);
        assert_eq!(ascending.max, shuffled.max);
    }

    #[test]
    fn nearest_rank_never_invents_an_unobserved_value() {
        // An interpolated p99 over a small sample would return a number that
        // was never measured.
        let values = vec![10.0, 20.0, 30.0];
        let p = Percentiles::from_millis(values.clone()).unwrap();
        for q in [p.p50, p.p90, p.p99] {
            assert!(values.contains(&q), "{q} was never observed");
        }
    }

    #[test]
    fn tpot_excludes_the_first_token() {
        // 100ms to first token, 1100ms total, 11 tokens.
        // Decode time is 1000ms over 10 inter-token gaps: 100ms each.
        let s = sample(100, 1100, 11);
        assert_eq!(s.tpot(), Some(Duration::from_millis(100)));
    }

    #[test]
    fn tpot_is_undefined_below_two_tokens() {
        // Reporting zero would drag an average down misleadingly.
        assert!(sample(100, 100, 1).tpot().is_none());
        assert!(sample(100, 100, 0).tpot().is_none());
        assert!(sample(100, 200, 2).tpot().is_some());
    }

    #[test]
    fn failed_requests_are_counted_not_dropped() {
        let samples = vec![
            sample(50, 500, 10),
            RequestSample::failed(100),
            sample(60, 600, 10),
        ];
        let r = aggregate(
            RunMetadata::detect("test".into(), "f32".into()),
            "test",
            2,
            &samples,
            Duration::from_secs(1),
        );

        assert_eq!(r.total_requests, 3);
        assert_eq!(r.successful_requests, 2);
        assert_eq!(r.failed_requests, 1, "failures must be visible");
        // Throughput counts only successes: a fast-failing server is not fast.
        assert!((r.request_throughput - 2.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_is_computed_over_wall_clock() {
        let samples: Vec<_> = (0..10).map(|_| sample(50, 500, 20)).collect();
        let r = aggregate(
            RunMetadata::detect("m".into(), "f32".into()),
            "w",
            4,
            &samples,
            Duration::from_secs(2),
        );

        assert!((r.request_throughput - 5.0).abs() < 1e-9, "10 reqs / 2s");
        assert!(
            (r.output_token_throughput - 100.0).abs() < 1e-9,
            "200 tokens / 2s"
        );
        assert_eq!(r.total_completion_tokens, 200);
    }

    #[test]
    fn an_all_failure_run_reports_zero_throughput_not_a_crash() {
        let samples: Vec<_> = (0..5).map(|_| RequestSample::failed(10)).collect();
        let r = aggregate(
            RunMetadata::detect("m".into(), "f32".into()),
            "w",
            1,
            &samples,
            Duration::from_secs(1),
        );
        assert_eq!(r.successful_requests, 0);
        assert_eq!(r.request_throughput, 0.0);
        assert!(r.ttft_ms.is_none());
    }

    #[test]
    fn workload_shapes_are_distinct_and_named() {
        let shapes = WorkloadShape::all();
        let names: std::collections::HashSet<_> = shapes.iter().map(|s| s.as_str()).collect();
        assert_eq!(names.len(), 4, "shape names must be distinct");

        assert!(WorkloadShape::LongShort.prompt_tokens() > WorkloadShape::ShortShort.prompt_tokens());
        assert!(WorkloadShape::ShortLong.output_tokens() > WorkloadShape::ShortShort.output_tokens());
    }

    #[test]
    fn a_cpu_only_run_says_so_loudly() {
        // The rule this harness exists to enforce: a CPU number must never be
        // quotable as a GPU number.
        let mut meta = RunMetadata::detect("m".into(), "f32".into());
        meta.gpu = None;

        let r = aggregate(meta, "w", 1, &[sample(10, 100, 5)], Duration::from_secs(1));
        let text = r.render();
        assert!(
            text.contains("CPU-ONLY RESULT"),
            "a CPU run must be labelled: {text}"
        );
        assert!(!r.metadata.is_gpu_run());
    }

    #[test]
    fn a_gpu_run_records_the_device() {
        let mut meta = RunMetadata::detect("m".into(), "f16".into());
        meta.gpu = Some("NVIDIA A100-SXM4-80GB".into());

        let r = aggregate(meta, "w", 1, &[sample(10, 100, 5)], Duration::from_secs(1));
        assert!(r.render().contains("A100"));
        assert!(r.metadata.is_gpu_run());
    }

    #[test]
    fn results_round_trip_through_json() {
        let r = aggregate(
            RunMetadata::detect("m".into(), "f32".into()),
            "w",
            8,
            &[sample(10, 100, 5)],
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&r).unwrap();
        let back: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.concurrency, 8);
        assert_eq!(back.successful_requests, 1);
    }

    #[test]
    fn csv_rows_match_the_header_width() {
        let r = aggregate(
            RunMetadata::detect("m".into(), "f32".into()),
            "w",
            1,
            &[sample(10, 100, 5)],
            Duration::from_secs(1),
        );
        let header_cols = BenchmarkResult::csv_header().split(',').count();
        let row_cols = r.to_csv_row().split(',').count();
        assert_eq!(header_cols, row_cols, "CSV header and row must agree");
    }

    #[test]
    fn metadata_records_the_host_it_ran_on() {
        let m = RunMetadata::detect("llama".into(), "f16".into());
        assert!(!m.os.is_empty());
        assert!(!m.arch.is_empty());
        assert!(m.cpu_cores > 0);
        assert_eq!(m.model, "llama");
    }

    #[test]
    fn measure_derives_timings_from_token_arrivals() {
        let start = Instant::now();
        let times: Vec<Instant> = (1..=3)
            .map(|i| start + Duration::from_millis(i * 10))
            .collect();

        let s = measure(start, &times, 42);
        assert_eq!(s.completion_tokens, 3);
        assert_eq!(s.prompt_tokens, 42);
        assert_eq!(s.ttft, Some(Duration::from_millis(10)));
        assert_eq!(s.total, Duration::from_millis(30));
        assert!(s.success);
    }

    #[test]
    fn a_request_that_produced_nothing_has_no_ttft() {
        let start = Instant::now();
        let s = measure(start, &[], 10);
        assert!(s.ttft.is_none());
        assert_eq!(s.completion_tokens, 0);
    }
}
