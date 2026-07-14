pub mod api;
pub mod metrics;
pub mod naive;
pub mod sse;

use anyhow::{bail, Context, Result};
use axum::routing::{get, post};
use axum::Router;
use clap::{Args, ValueEnum};
use hotbatch_core::model::normalize_model_name;
use hotbatch_core::{
    ModelOptions, RequestQueue, Scheduler, SchedulerConfig, SchedulerMetrics, SlabKvCache,
    SmallTransformer, TokenizerBundle,
};
use naive::NaiveEngine;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, ValueEnum)]
pub enum ServeMode {
    Naive,
    #[default]
    Continuous,
}

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long, value_enum, default_value_t = ServeMode::Continuous)]
    pub mode: ServeMode,
    #[arg(long, default_value = "gpt2")]
    pub model: String,
    #[arg(long, default_value = "cpu")]
    pub device: String,
    #[arg(long, default_value_t = 32)]
    pub max_running_seqs: usize,
    #[arg(long, default_value_t = 1024)]
    pub max_queue_depth: usize,
    #[arg(long, default_value_t = 512)]
    pub max_seq_len: usize,
    #[arg(long, default_value_t = 64)]
    pub max_new_tokens: usize,
}

impl ServeArgs {
    /// Validate all inexpensive startup configuration before loading model files or
    /// opening a listening socket.
    pub fn validate(&self) -> Result<()> {
        if self.max_running_seqs == 0 {
            bail!("--max-running-seqs must be greater than zero");
        }
        if self.max_queue_depth == 0 {
            bail!("--max-queue-depth must be greater than zero");
        }
        if self.max_seq_len == 0 {
            bail!("--max-seq-len must be greater than zero");
        }
        if self.max_new_tokens == 0 {
            bail!("--max-new-tokens must be greater than zero");
        }
        if self.max_new_tokens >= self.max_seq_len {
            bail!("--max-new-tokens must be less than --max-seq-len");
        }
        if self.max_seq_len > 1_024 {
            bail!("--max-seq-len cannot exceed GPT-2's 1024-token context window");
        }
        let model_name = normalize_model_name(&self.model)?;
        if model_name == "scripted" {
            bail!("unsupported model 'scripted'; hotbatch serves GPT-2 models only");
        }
        if self.device != "cpu" {
            bail!(
                "unsupported device '{}'; hotbatch supports cpu only",
                self.device
            );
        }
        self.host
            .parse::<IpAddr>()
            .with_context(|| format!("invalid --host '{}': expected an IP address", self.host))?;
        Ok(())
    }

    fn bind_addr(&self) -> Result<SocketAddr> {
        let host = self
            .host
            .parse::<IpAddr>()
            .with_context(|| format!("invalid --host '{}': expected an IP address", self.host))?;
        Ok(SocketAddr::new(host, self.port))
    }
}

#[derive(Clone)]
pub enum Engine {
    Continuous { queue: RequestQueue },
    Naive(NaiveEngine),
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Engine,
    pub tokenizer: TokenizerBundle,
    pub model_name: String,
    pub metrics: SchedulerMetrics,
    pub alive: Arc<AtomicBool>,
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
}

pub struct BuiltApp {
    pub router: Router,
    pub shutdown: CancellationToken,
    pub metrics: SchedulerMetrics,
}

pub struct RunningServer {
    pub addr: SocketAddr,
    pub shutdown: CancellationToken,
    join: tokio::task::JoinHandle<Result<()>>,
}

impl RunningServer {
    pub async fn stop(self) -> Result<()> {
        self.shutdown.cancel();
        self.join.await.context("server task join failed")?
    }
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .json()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

pub async fn build_app(args: ServeArgs) -> Result<BuiltApp> {
    args.validate()?;
    let metrics = SchedulerMetrics::new()?;
    let model_options = ModelOptions {
        model: args.model.clone(),
        device: args.device.clone(),
        ..ModelOptions::default()
    };
    let model = SmallTransformer::load(model_options).await?;
    if args.max_seq_len > model.max_positions() {
        bail!(
            "--max-seq-len cannot exceed the loaded model's {}-token context window",
            model.max_positions()
        );
    }
    let tokenizer = model.tokenizer();
    let model_name = tokenizer.model_name().to_string();
    let shutdown = CancellationToken::new();
    let alive = Arc::new(AtomicBool::new(true));

    let engine = match args.mode {
        ServeMode::Continuous => {
            let queue = RequestQueue::new(args.max_queue_depth, metrics.clone());
            let (num_layers, n_heads, head_dim) = model.kv_shape();
            let kv_cache = SlabKvCache::new(
                args.max_running_seqs,
                args.max_seq_len,
                num_layers,
                n_heads,
                head_dim,
            );
            let scheduler_config = SchedulerConfig {
                max_running_seqs: args.max_running_seqs,
                max_new_tokens: args.max_new_tokens,
                max_seq_len: args.max_seq_len,
                max_queue_depth: args.max_queue_depth,
                idle_sleep: std::time::Duration::from_millis(1),
            };
            let mut scheduler = Scheduler::new(
                scheduler_config,
                queue.clone(),
                kv_cache,
                model,
                metrics.clone(),
            );
            let scheduler_shutdown = shutdown.clone();
            let scheduler_alive = alive.clone();
            tokio::spawn(async move {
                if let Err(err) = scheduler.run(scheduler_shutdown).await {
                    error!(error = %err, "scheduler stopped with error");
                }
                scheduler_alive.store(false, Ordering::SeqCst);
            });
            Engine::Continuous { queue }
        }
        ServeMode::Naive => {
            let engine = NaiveEngine::new(
                model,
                metrics.clone(),
                shutdown.clone(),
                args.max_queue_depth,
            );
            let naive_shutdown = shutdown.clone();
            let naive_alive = alive.clone();
            tokio::spawn(async move {
                naive_shutdown.cancelled().await;
                naive_alive.store(false, Ordering::SeqCst);
            });
            Engine::Naive(engine)
        }
    };

    let state = AppState {
        engine,
        tokenizer,
        model_name,
        metrics: metrics.clone(),
        alive,
        max_new_tokens: args.max_new_tokens,
        max_seq_len: args.max_seq_len,
    };

    let router = Router::new()
        .route("/healthz", get(api::healthz))
        .route("/metrics", get(metrics::metrics))
        .route("/v1/models", get(api::models))
        .route("/v1/completions", post(api::completions))
        .route("/v1/chat/completions", post(api::chat_completions))
        .with_state(state);

    Ok(BuiltApp {
        router,
        shutdown,
        metrics,
    })
}

pub async fn spawn_server(args: ServeArgs) -> Result<RunningServer> {
    args.validate()?;
    let bind = args.bind_addr()?;
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding {bind}"))?;
    spawn_on_listener(args, listener).await
}

pub async fn spawn_on_listener(args: ServeArgs, listener: TcpListener) -> Result<RunningServer> {
    args.validate()?;
    let addr = listener.local_addr().context("reading listener address")?;
    let built = build_app(args).await?;
    let shutdown = built.shutdown.clone();
    let server_shutdown = built.shutdown.clone();
    let router = built.router;
    let join = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                server_shutdown.cancelled().await;
            })
            .await
            .context("serving HTTP")
    });
    Ok(RunningServer {
        addr,
        shutdown,
        join,
    })
}

pub async fn serve(args: ServeArgs) -> Result<()> {
    args.validate()?;
    let bind = args.bind_addr()?;
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding {bind}"))?;
    let addr = listener.local_addr().context("reading listener address")?;
    let built = build_app(args).await?;
    let shutdown = built.shutdown.clone();
    let ctrl_c_shutdown = built.shutdown.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            ctrl_c_shutdown.cancel();
        }
    });
    info!(%addr, "hotbatch server listening");
    axum::serve(listener, built.router)
        .with_graceful_shutdown(async move {
            shutdown.cancelled().await;
        })
        .await
        .context("serving HTTP")
}
