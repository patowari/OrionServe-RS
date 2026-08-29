//! The `orion` command-line interface.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use orion_api::{router, AppState};
use orion_core::{CacheConfig, EngineConfig, LanguageModel, SchedulerConfig, ServerConfig};
use orion_kv_cache::KvCacheManager;
use orion_models::{CpuBackend, TransformerModel};
use orion_observability::{init_tracing, install_metrics, LogFormat};
use orion_scheduler::Scheduler;
use orion_tokenizer::{ChatTemplate, Tokenizer};

#[derive(Parser, Debug)]
#[command(
    name = "orion",
    version,
    about = "OrionServe-RS: a high-performance LLM inference engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Serve a model over an OpenAI-compatible HTTP API.
    Serve(Box<ServeArgs>),
    /// Print a model's configuration without serving it.
    Inspect {
        /// Path to a Hugging Face model directory.
        #[arg(long)]
        model: PathBuf,
    },
}

#[derive(Parser, Debug)]
struct ServeArgs {
    /// Path to a Hugging Face model directory.
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// KV cache blocks to allocate.
    ///
    /// Required because the CPU backend cannot report a device memory budget to
    /// size the pool from. A GPU backend will make this optional.
    #[arg(long, default_value_t = 4096)]
    num_blocks: usize,

    /// Tokens per KV block. Must be a power of two in 4..=256.
    #[arg(long, default_value_t = 16)]
    block_size: usize,

    /// Maximum sequences resident in the running batch.
    #[arg(long, default_value_t = 64)]
    max_num_seqs: usize,

    /// Token budget for one engine step.
    #[arg(long, default_value_t = 4096)]
    max_num_batched_tokens: usize,

    /// Longest prompt plus output accepted. Defaults to the model's own limit.
    #[arg(long)]
    max_model_len: Option<usize>,

    /// `max_tokens` applied to requests that do not specify one.
    #[arg(long, default_value_t = 128)]
    default_max_tokens: usize,

    /// Disable automatic prefix caching.
    #[arg(long)]
    no_prefix_caching: bool,

    /// Disable chunked prefill.
    #[arg(long)]
    no_chunked_prefill: bool,

    #[arg(long, default_value = "pretty")]
    log_format: LogFormat,

    #[arg(long, default_value = "info", env = "ORION_LOG_LEVEL")]
    log_level: String,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // Startup failures print plainly: tracing may not be initialized
            // yet, and an operator running this in a terminal needs the reason
            // rather than a structured log line.
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Inspect { model } => inspect(&model),
        Command::Serve(args) => serve(*args),
    }
}

fn inspect(dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let config = orion_models::HfConfig::load(dir)?;
    let arch = orion_models::Architecture::detect(&config)?;
    let name = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let meta = config.to_metadata(name)?;

    println!(
        "architecture:      {} ({})",
        meta.architecture,
        arch.as_str()
    );
    println!("hidden size:       {}", meta.hidden_size);
    println!("layers:            {}", meta.num_layers);
    println!("attention heads:   {}", meta.num_attention_heads);
    println!(
        "kv heads:          {} ({})",
        meta.num_kv_heads,
        if meta.uses_gqa() {
            format!("GQA, {}x grouping", meta.gqa_group_size())
        } else {
            "MHA".to_string()
        }
    );
    println!("head dim:          {}", meta.head_dim);
    println!("vocab size:        {}", meta.vocab_size);
    println!("max positions:     {}", meta.max_position_embeddings);
    println!("dtype:             {}", meta.dtype.as_str());
    println!("eos token ids:     {:?}", meta.eos_token_ids);
    println!();

    let per_token = meta.kv_bytes_per_token();
    println!("KV cache per token:      {} bytes", per_token);
    println!(
        "KV cache per 1k tokens:  {:.2} MiB",
        (per_token * 1024) as f64 / (1024.0 * 1024.0)
    );
    println!(
        "KV cache at full context: {:.2} GiB",
        (per_token * meta.max_position_embeddings) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing(args.log_format, &args.log_level);

    // Validate configuration before touching the model: a bad flag should fail
    // in milliseconds, not after loading gigabytes of weights.
    let config = EngineConfig {
        cache: CacheConfig {
            block_size: args.block_size,
            num_blocks: Some(args.num_blocks),
            enable_prefix_caching: !args.no_prefix_caching,
            ..Default::default()
        },
        scheduler: SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
            max_num_batched_tokens: args.max_num_batched_tokens,
            max_model_len: args.max_model_len,
            enable_chunked_prefill: !args.no_chunked_prefill,
            ..Default::default()
        },
        server: ServerConfig {
            host: args.host.clone(),
            port: args.port,
            ..Default::default()
        },
    };
    config.validate()?;

    let metrics_handle = install_metrics().map_err(|e| format!("metrics setup failed: {e}"))?;

    tracing::info!(model = %args.model.display(), "loading model");
    let model = TransformerModel::from_directory(
        &args.model,
        args.num_blocks,
        args.block_size,
        Box::new(CpuBackend::new()),
    )?;
    let metadata = model.metadata().clone();

    let tokenizer = Tokenizer::from_directory(&args.model)?
        .with_special_tokens(metadata.bos_token_id, metadata.eos_token_ids.clone());

    // The model's own context length bounds requests unless the operator set a
    // smaller one explicitly.
    let mut scheduler_config = config.scheduler.clone();
    scheduler_config.max_model_len = Some(
        args.max_model_len
            .unwrap_or(metadata.max_position_embeddings),
    );
    scheduler_config.validate()?;

    let cache = KvCacheManager::new(
        args.num_blocks,
        args.block_size,
        config.cache.enable_prefix_caching,
    );
    let scheduler = Scheduler::new(scheduler_config, cache);

    tracing::info!(
        blocks = args.num_blocks,
        block_size = args.block_size,
        capacity_tokens = args.num_blocks * args.block_size,
        "KV cache configured"
    );

    let (engine, engine_thread) = orion_runtime::spawn(
        scheduler,
        Arc::new(model),
        config.server.max_concurrent_requests,
    );

    let template = ChatTemplate::for_architecture(&metadata.architecture);
    let state = AppState {
        engine: engine.clone(),
        tokenizer: Arc::new(tokenizer),
        metadata: Arc::new(metadata),
        template,
        default_max_tokens: args.default_max_tokens,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let result = runtime.block_on(async move {
        let app = router(state).route(
            "/metrics",
            axum::routing::get(move || {
                let handle = metrics_handle.clone();
                async move { handle.render() }
            }),
        );

        let addr = format!("{}:{}", config.server.host, config.server.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!(address = %addr, "OrionServe-RS listening");

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
    });

    // Stop the engine thread and wait for it, so in-flight cache blocks are
    // released and the process exits cleanly rather than being killed.
    runtime.block_on(engine.shutdown());
    let _ = engine_thread.join();
    tracing::info!("shutdown complete");

    result.map_err(Into::into)
}

/// Resolves on SIGINT, or SIGTERM on Unix.
///
/// SIGTERM matters in a container: an orchestrator sends it on rollout, and a
/// server that only handles Ctrl-C would be SIGKILLed after the grace period,
/// dropping every in-flight request.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => tracing::warn!(error = %e, "cannot listen for SIGTERM"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received interrupt, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_cli_definition_is_valid() {
        // Catches conflicting flags and bad defaults at test time rather than
        // on first run.
        Cli::command().debug_assert();
    }

    #[test]
    fn serve_parses_with_only_a_model_path() {
        let cli = Cli::try_parse_from(["orion", "serve", "--model", "/models/llama"]).unwrap();
        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.model, PathBuf::from("/models/llama"));
                assert_eq!(args.port, 8000);
                assert_eq!(args.block_size, 16);
                assert!(!args.no_prefix_caching, "prefix caching on by default");
                assert!(!args.no_chunked_prefill, "chunked prefill on by default");
            }
            other => panic!("expected serve, got {other:?}"),
        }
    }

    #[test]
    fn serve_accepts_the_documented_flags() {
        let cli = Cli::try_parse_from([
            "orion",
            "serve",
            "--model",
            "/m",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--block-size",
            "32",
            "--max-num-seqs",
            "128",
            "--no-prefix-caching",
            "--log-format",
            "json",
        ])
        .unwrap();

        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.host, "0.0.0.0");
                assert_eq!(args.port, 9000);
                assert_eq!(args.block_size, 32);
                assert_eq!(args.max_num_seqs, 128);
                assert!(args.no_prefix_caching);
                assert_eq!(args.log_format, LogFormat::Json);
            }
            other => panic!("expected serve, got {other:?}"),
        }
    }

    #[test]
    fn serve_requires_a_model() {
        assert!(Cli::try_parse_from(["orion", "serve"]).is_err());
    }

    #[test]
    fn an_invalid_log_format_is_rejected_at_parse_time() {
        assert!(
            Cli::try_parse_from(["orion", "serve", "--model", "/m", "--log-format", "xml"])
                .is_err()
        );
    }

    #[test]
    fn inspect_parses() {
        let cli = Cli::try_parse_from(["orion", "inspect", "--model", "/m"]).unwrap();
        assert!(matches!(cli.command, Command::Inspect { .. }));
    }

    #[test]
    fn a_bad_block_size_is_caught_by_config_validation() {
        // 12 is not a power of two; validation must reject it before any model
        // is loaded.
        let config = EngineConfig {
            cache: CacheConfig {
                block_size: 12,
                num_blocks: Some(64),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
