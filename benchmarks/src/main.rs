//! `orion-bench` — a load generator for a running OrionServe-RS server.
//!
//! Drives real HTTP requests and measures what a client experiences. Run the
//! server first, then point this at it:
//!
//! ```bash
//! orion serve --model /models/qwen &
//! orion-bench --url http://127.0.0.1:8000 --concurrency 1,10,50 --requests 100
//! ```
//!
//! Results are written as JSON and CSV, each carrying the hardware they were
//! produced on.

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use orion_bench::{
    aggregate, BenchmarkResult, RequestSample, RunMetadata, WorkloadShape,
};

#[derive(Parser, Debug)]
#[command(name = "orion-bench", about = "Load generator for OrionServe-RS")]
struct Args {
    /// Base URL of a running server.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,

    /// Model name to send in requests.
    #[arg(long, default_value = "orion")]
    model: String,

    /// Concurrency levels to sweep, comma-separated.
    #[arg(long, default_value = "1,10,50", value_delimiter = ',')]
    concurrency: Vec<usize>,

    /// Requests per concurrency level.
    #[arg(long, default_value_t = 50)]
    requests: usize,

    /// Warm-up requests, discarded before measuring.
    ///
    /// Without these the first measurement absorbs page faults, allocator
    /// growth and a cold prefix cache, which flatters or penalizes the run
    /// depending on where it lands.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Workload shapes to run. Defaults to all four.
    #[arg(long, value_delimiter = ',')]
    workload: Option<Vec<String>>,

    /// Where to write results.
    #[arg(long, default_value = "benchmarks/results")]
    output: String,

    /// Precision label recorded in the metadata. Not detected: the harness
    /// cannot know what the server loaded, so the operator states it.
    #[arg(long, default_value = "f32")]
    precision: String,

    /// Free-text note recorded with the results.
    #[arg(long)]
    notes: Option<String>,

    /// Per-request timeout.
    #[arg(long, default_value_t = 300)]
    timeout_secs: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let client = Arc::new(
        reqwest::Client::builder()
            .timeout(Duration::from_secs(args.timeout_secs))
            .build()?,
    );

    // Fail early with a clear message rather than reporting a run of zeros.
    let health = format!("{}/health", args.url.trim_end_matches('/'));
    if let Err(e) = client.get(&health).send().await {
        eprintln!("cannot reach server at {}: {e}", args.url);
        eprintln!("start it first: orion serve --model <path>");
        return Err(e.into());
    }

    let shapes = select_shapes(args.workload.as_deref())?;
    let mut metadata = RunMetadata::detect(args.model.clone(), args.precision.clone());
    metadata.notes = args.notes.clone();

    if !metadata.is_gpu_run() {
        eprintln!(
            "warning: no GPU detected. These are CPU-only numbers and must not \
             be reported as GPU performance."
        );
    }

    std::fs::create_dir_all(&args.output)?;
    let mut all_results = Vec::new();

    for shape in shapes {
        for &concurrency in &args.concurrency {
            eprintln!(
                "running {} at concurrency {concurrency}...",
                shape.as_str()
            );

            // Warm-up, discarded.
            if args.warmup > 0 {
                run_batch(&client, &args, shape, concurrency.min(args.warmup), args.warmup)
                    .await;
            }

            let start = Instant::now();
            let samples =
                run_batch(&client, &args, shape, concurrency, args.requests).await;
            let elapsed = start.elapsed();

            let result = aggregate(
                metadata.clone(),
                shape.as_str(),
                concurrency,
                &samples,
                elapsed,
            );
            println!("\n{}", result.render());
            all_results.push(result);
        }
    }

    write_results(&args.output, &all_results)?;
    Ok(())
}

fn select_shapes(names: Option<&[String]>) -> Result<Vec<WorkloadShape>, String> {
    let Some(names) = names else {
        return Ok(WorkloadShape::all().to_vec());
    };
    names
        .iter()
        .map(|n| {
            WorkloadShape::all()
                .into_iter()
                .find(|s| s.as_str() == n)
                .ok_or_else(|| format!("unknown workload `{n}`"))
        })
        .collect()
}

/// Issues `total` requests with at most `concurrency` in flight.
async fn run_batch(
    client: &Arc<reqwest::Client>,
    args: &Args,
    shape: WorkloadShape,
    concurrency: usize,
    total: usize,
) -> Vec<RequestSample> {
    use futures::stream::StreamExt;

    let url = format!("{}/v1/chat/completions", args.url.trim_end_matches('/'));
    // Roughly four characters per token; the exact ratio does not matter, only
    // that the shapes differ from each other consistently.
    let prompt = "word ".repeat(shape.prompt_tokens());

    futures::stream::iter(0..total)
        .map(|i| {
            let client = Arc::clone(client);
            let url = url.clone();
            let model = args.model.clone();
            // Vary the prompt slightly so every request does not trivially hit
            // the prefix cache, which would measure the cache rather than the
            // engine.
            let prompt = format!("[{i}] {prompt}");
            let max_tokens = shape.output_tokens();

            async move { one_request(&client, &url, &model, &prompt, max_tokens).await }
        })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await
}

/// Issues one streaming request and measures token arrival times.
async fn one_request(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> RequestSample {
    use futures::StreamExt;

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
    });

    let start = Instant::now();
    let resp = match client.post(url).json(&body).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            eprintln!("request failed with status {}", r.status());
            return RequestSample::failed(0);
        }
        Err(e) => {
            eprintln!("request error: {e}");
            return RequestSample::failed(0);
        }
    };

    let mut first_token: Option<Instant> = None;
    let mut tokens = 0usize;
    let mut prompt_tokens = 0usize;
    let mut buffer = String::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let Ok(bytes) = chunk else {
            return RequestSample::failed(prompt_tokens);
        };
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // SSE frames are newline-delimited; a chunk may split one, so only
        // complete lines are parsed and the remainder is carried forward.
        while let Some(newline) = buffer.find('\n') {
            let line = buffer[..newline].trim().to_string();
            buffer.drain(..=newline);

            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };

            if v["choices"][0]["delta"]["content"].is_string() {
                if first_token.is_none() {
                    first_token = Some(Instant::now());
                }
                tokens += 1;
            }
            if let Some(p) = v["usage"]["prompt_tokens"].as_u64() {
                prompt_tokens = p as usize;
            }
        }
    }

    RequestSample {
        ttft: first_token.map(|t| t - start),
        total: start.elapsed(),
        prompt_tokens,
        completion_tokens: tokens,
        success: tokens > 0,
    }
}

fn write_results(dir: &str, results: &[BenchmarkResult]) -> std::io::Result<()> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let json_path = format!("{dir}/results-{stamp}.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(results)?)?;

    let csv_path = format!("{dir}/results-{stamp}.csv");
    let mut csv = String::from(BenchmarkResult::csv_header());
    csv.push('\n');
    for r in results {
        csv.push_str(&r.to_csv_row());
        csv.push('\n');
    }
    std::fs::write(&csv_path, csv)?;

    eprintln!("\nwrote {json_path}");
    eprintln!("wrote {csv_path}");
    Ok(())
}
