//! Metrics, tracing and structured logging.
//!
//! # What is instrumented and why
//!
//! The metrics here are chosen to answer the questions an operator actually
//! asks during an incident, not to be exhaustive:
//!
//! * *Is it up and serving?* — request counts and errors by code.
//! * *Is it slow, and where?* — TTFT and TPOT separately, because they have
//!   different causes: TTFT is dominated by queueing and prefill, TPOT by batch
//!   size and cache pressure.
//! * *Is it about to fall over?* — queue depth and KV cache utilization. Cache
//!   utilization is the leading indicator: preemption starts when it saturates,
//!   and throughput collapses shortly after.
//! * *Is the batching working?* — batch size and token counts per step. A batch
//!   size stuck at 1 under load means the scheduler is not doing its job.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Metric names, defined once so the exporter and the recording sites cannot
/// drift apart.
pub mod names {
    // Counters
    pub const REQUESTS_TOTAL: &str = "orion_requests_total";
    pub const REQUEST_ERRORS_TOTAL: &str = "orion_request_errors_total";
    pub const TOKENS_GENERATED_TOTAL: &str = "orion_tokens_generated_total";
    pub const PROMPT_TOKENS_TOTAL: &str = "orion_prompt_tokens_total";
    pub const COMPLETION_TOKENS_TOTAL: &str = "orion_completion_tokens_total";
    pub const PREEMPTIONS_TOTAL: &str = "orion_preemptions_total";
    pub const PREFIX_CACHE_HITS_TOTAL: &str = "orion_prefix_cache_hits_total";
    pub const PREFIX_CACHE_MISSES_TOTAL: &str = "orion_prefix_cache_misses_total";

    // Gauges
    pub const REQUESTS_RUNNING: &str = "orion_requests_running";
    pub const REQUESTS_WAITING: &str = "orion_requests_waiting";
    pub const KV_CACHE_USAGE_RATIO: &str = "orion_kv_cache_usage_ratio";
    pub const KV_CACHE_BLOCKS_USED: &str = "orion_kv_cache_blocks_used";
    pub const KV_CACHE_BLOCKS_FREE: &str = "orion_kv_cache_blocks_free";

    // Histograms
    pub const TIME_TO_FIRST_TOKEN: &str = "orion_time_to_first_token_seconds";
    pub const TIME_PER_OUTPUT_TOKEN: &str = "orion_time_per_output_token_seconds";
    pub const REQUEST_LATENCY: &str = "orion_request_latency_seconds";
    pub const QUEUE_TIME: &str = "orion_queue_time_seconds";
    pub const BATCH_SIZE: &str = "orion_batch_size";
    pub const BATCH_TOKENS: &str = "orion_batch_tokens";
    pub const STEP_DURATION: &str = "orion_step_duration_seconds";
}

/// Latency histogram buckets, in seconds.
///
/// Spans 1ms to 60s. The dense region between 10ms and 1s is where inference
/// latencies actually sit, and uniform buckets would put almost every
/// observation in one bin, making percentiles meaningless.
const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Per-token latency buckets, in seconds.
///
/// Much tighter: a healthy TPOT is single-digit milliseconds, and the
/// difference between 5ms and 50ms is the difference between a good and an
/// unusable experience.
const TOKEN_LATENCY_BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0,
];

/// Batch size buckets. Powers of two, since batch sizes are configured that way
/// and the interesting question is the order of magnitude.
const BATCH_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];

/// Token-count buckets for a batch.
const TOKEN_COUNT_BUCKETS: &[f64] = &[
    1.0, 8.0, 32.0, 128.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
];

/// Installs the Prometheus recorder and returns a handle that renders the
/// exposition format.
///
/// Bucket boundaries are attached here rather than at each recording site so
/// that a metric's resolution is a property of the metric, not of whichever
/// code path happened to record it first.
pub fn install_metrics() -> Result<PrometheusHandle, String> {
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::TIME_TO_FIRST_TOKEN.into()),
            LATENCY_BUCKETS,
        )
        .map_err(|e| e.to_string())?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::REQUEST_LATENCY.into()),
            LATENCY_BUCKETS,
        )
        .map_err(|e| e.to_string())?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::QUEUE_TIME.into()),
            LATENCY_BUCKETS,
        )
        .map_err(|e| e.to_string())?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::TIME_PER_OUTPUT_TOKEN.into()),
            TOKEN_LATENCY_BUCKETS,
        )
        .map_err(|e| e.to_string())?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::STEP_DURATION.into()),
            TOKEN_LATENCY_BUCKETS,
        )
        .map_err(|e| e.to_string())?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::BATCH_SIZE.into()),
            BATCH_BUCKETS,
        )
        .map_err(|e| e.to_string())?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(names::BATCH_TOKENS.into()),
            TOKEN_COUNT_BUCKETS,
        )
        .map_err(|e| e.to_string())?;

    builder.install_recorder().map_err(|e| e.to_string())
}

/// How logs are formatted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable, for a terminal.
    Pretty,
    /// One JSON object per line, for log aggregation.
    Json,
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "pretty" | "text" => Ok(LogFormat::Pretty),
            "json" => Ok(LogFormat::Json),
            other => Err(format!(
                "unknown log format `{other}`, expected `pretty` or `json`"
            )),
        }
    }
}

/// Initializes tracing.
///
/// The filter comes from `RUST_LOG` when set, falling back to `default_level`.
/// Honouring the environment matters: raising log level during an incident
/// should not require a config change and a restart with a new file.
pub fn init_tracing(format: LogFormat, default_level: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

    let registry = tracing_subscriber::registry().with(filter);
    match format {
        LogFormat::Json => {
            registry
                .with(tracing_subscriber::fmt::layer().json().with_target(true))
                .init();
        }
        LogFormat::Pretty => {
            registry
                .with(tracing_subscriber::fmt::layer().with_target(false))
                .init();
        }
    }
}

/// Records the metrics for a completed request.
pub fn record_request_complete(
    outcome: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
    latency: Duration,
    ttft: Option<Duration>,
    tpot: Option<Duration>,
) {
    metrics::counter!(names::REQUESTS_TOTAL, "outcome" => outcome).increment(1);
    metrics::counter!(names::PROMPT_TOKENS_TOTAL).increment(prompt_tokens as u64);
    metrics::counter!(names::COMPLETION_TOKENS_TOTAL).increment(completion_tokens as u64);
    metrics::counter!(names::TOKENS_GENERATED_TOTAL).increment(completion_tokens as u64);
    metrics::histogram!(names::REQUEST_LATENCY).record(latency.as_secs_f64());

    if let Some(t) = ttft {
        metrics::histogram!(names::TIME_TO_FIRST_TOKEN).record(t.as_secs_f64());
    }
    if let Some(t) = tpot {
        metrics::histogram!(names::TIME_PER_OUTPUT_TOKEN).record(t.as_secs_f64());
    }
}

/// Records a failed request, labelled by its stable error code.
pub fn record_request_error(code: &'static str) {
    metrics::counter!(names::REQUESTS_TOTAL, "outcome" => "error").increment(1);
    metrics::counter!(names::REQUEST_ERRORS_TOTAL, "code" => code).increment(1);
}

/// Records the shape of one engine step.
pub fn record_step(num_sequences: usize, num_tokens: usize, duration: Duration, preempted: usize) {
    metrics::histogram!(names::BATCH_SIZE).record(num_sequences as f64);
    metrics::histogram!(names::BATCH_TOKENS).record(num_tokens as f64);
    metrics::histogram!(names::STEP_DURATION).record(duration.as_secs_f64());
    if preempted > 0 {
        metrics::counter!(names::PREEMPTIONS_TOTAL).increment(preempted as u64);
    }
}

/// Publishes engine state as gauges.
///
/// Called once per step rather than on every state change: gauges are sampled
/// by the scrape, so updating them more often than the scrape interval is
/// wasted work.
pub fn record_engine_state(running: usize, waiting: usize, blocks_used: usize, blocks_free: usize) {
    metrics::gauge!(names::REQUESTS_RUNNING).set(running as f64);
    metrics::gauge!(names::REQUESTS_WAITING).set(waiting as f64);
    metrics::gauge!(names::KV_CACHE_BLOCKS_USED).set(blocks_used as f64);
    metrics::gauge!(names::KV_CACHE_BLOCKS_FREE).set(blocks_free as f64);

    let total = blocks_used + blocks_free;
    let ratio = if total == 0 {
        0.0
    } else {
        blocks_used as f64 / total as f64
    };
    metrics::gauge!(names::KV_CACHE_USAGE_RATIO).set(ratio);
}

/// Records prefix cache activity.
pub fn record_prefix_cache(hits: u64, misses: u64) {
    metrics::counter!(names::PREFIX_CACHE_HITS_TOTAL).absolute(hits);
    metrics::counter!(names::PREFIX_CACHE_MISSES_TOTAL).absolute(misses);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_formats_parse_from_strings() {
        assert_eq!("json".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert_eq!("pretty".parse::<LogFormat>().unwrap(), LogFormat::Pretty);
        assert_eq!("TEXT".parse::<LogFormat>().unwrap(), LogFormat::Pretty);
        let err = "yaml".parse::<LogFormat>().unwrap_err();
        assert!(err.contains("yaml"), "the error should name the bad value");
    }

    #[test]
    fn metric_names_are_unique_and_prefixed() {
        let all = [
            names::REQUESTS_TOTAL,
            names::REQUEST_ERRORS_TOTAL,
            names::TOKENS_GENERATED_TOTAL,
            names::PROMPT_TOKENS_TOTAL,
            names::COMPLETION_TOKENS_TOTAL,
            names::PREEMPTIONS_TOTAL,
            names::PREFIX_CACHE_HITS_TOTAL,
            names::PREFIX_CACHE_MISSES_TOTAL,
            names::REQUESTS_RUNNING,
            names::REQUESTS_WAITING,
            names::KV_CACHE_USAGE_RATIO,
            names::KV_CACHE_BLOCKS_USED,
            names::KV_CACHE_BLOCKS_FREE,
            names::TIME_TO_FIRST_TOKEN,
            names::TIME_PER_OUTPUT_TOKEN,
            names::REQUEST_LATENCY,
            names::QUEUE_TIME,
            names::BATCH_SIZE,
            names::BATCH_TOKENS,
            names::STEP_DURATION,
        ];

        let unique: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), all.len(), "duplicate metric name");

        for n in all {
            assert!(n.starts_with("orion_"), "{n} is missing the namespace");
        }
    }

    #[test]
    fn counter_metrics_follow_prometheus_naming() {
        // Counters end in _total; that is what Prometheus tooling expects.
        for n in [
            names::REQUESTS_TOTAL,
            names::REQUEST_ERRORS_TOTAL,
            names::TOKENS_GENERATED_TOTAL,
            names::PREEMPTIONS_TOTAL,
        ] {
            assert!(n.ends_with("_total"), "{n} should end in _total");
        }
    }

    #[test]
    fn latency_metrics_carry_their_unit() {
        for n in [
            names::TIME_TO_FIRST_TOKEN,
            names::TIME_PER_OUTPUT_TOKEN,
            names::REQUEST_LATENCY,
            names::QUEUE_TIME,
            names::STEP_DURATION,
        ] {
            assert!(n.ends_with("_seconds"), "{n} should state its unit");
        }
    }

    #[test]
    fn latency_buckets_are_ascending_and_cover_the_useful_range() {
        for buckets in [LATENCY_BUCKETS, TOKEN_LATENCY_BUCKETS, BATCH_BUCKETS] {
            assert!(
                buckets.windows(2).all(|w| w[0] < w[1]),
                "buckets must be strictly ascending"
            );
        }
        // TPOT resolution has to reach into single-digit milliseconds.
        assert!(TOKEN_LATENCY_BUCKETS[0] <= 0.001);
        // Request latency has to reach a realistic timeout.
        assert!(*LATENCY_BUCKETS.last().unwrap() >= 30.0);
    }

    #[test]
    fn engine_state_ratio_handles_an_empty_cache() {
        // Recording must not divide by zero before the cache is sized.
        record_engine_state(0, 0, 0, 0);
    }

    #[test]
    fn recording_without_an_installed_recorder_is_a_no_op() {
        // Metrics calls must never panic when no exporter is configured, since
        // tests and CLI subcommands run without one.
        record_request_complete(
            "stop",
            10,
            5,
            Duration::from_millis(100),
            Some(Duration::from_millis(20)),
            Some(Duration::from_millis(5)),
        );
        record_request_error("queue_full");
        record_step(4, 128, Duration::from_millis(10), 1);
        record_prefix_cache(10, 3);
    }
}
