use crate::kv_cache::{KvCache, SlabKvCache};
use crate::model::{DecodeBatch, DecodeInput, SmallTransformer};
use crate::sequence::{GenerationRequest, Sequence, StreamItem};
use anyhow::{anyhow, Result};
use prometheus::{
    Encoder, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry, TextEncoder,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_running_seqs: usize,
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
    pub max_queue_depth: usize,
    pub idle_sleep: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_running_seqs: 32,
            max_new_tokens: 64,
            max_seq_len: 512,
            max_queue_depth: 1_024,
            idle_sleep: Duration::from_millis(1),
        }
    }
}

#[derive(Debug, Error)]
#[error("request queue is full")]
pub struct QueueFull;

#[derive(Clone)]
pub struct RequestQueue {
    inner: Arc<Mutex<QueueState>>,
    notify: Arc<Notify>,
    metrics: SchedulerMetrics,
}

struct QueueState {
    requests: VecDeque<GenerationRequest>,
    max_depth: usize,
}

impl RequestQueue {
    pub fn new(max_depth: usize, metrics: SchedulerMetrics) -> Self {
        Self {
            inner: Arc::new(Mutex::new(QueueState {
                requests: VecDeque::new(),
                max_depth,
            })),
            notify: Arc::new(Notify::new()),
            metrics,
        }
    }

    pub fn submit(&self, req: GenerationRequest) -> std::result::Result<(), QueueFull> {
        let mut state = lock_or_recover(&self.inner);
        if state.requests.len() >= state.max_depth {
            return Err(QueueFull);
        }
        state.requests.push_back(req);
        self.metrics.set_queue_depth(state.requests.len());
        drop(state);
        self.notify.notify_one();
        Ok(())
    }

    pub fn try_pop(&self) -> Option<GenerationRequest> {
        let mut state = lock_or_recover(&self.inner);
        if state.requests.is_empty() {
            self.metrics.set_queue_depth(0);
            return None;
        }
        let mut best_index = 0_usize;
        let mut best_priority = state.requests.front().map(|req| req.priority).unwrap_or(0);
        for (index, req) in state.requests.iter().enumerate().skip(1) {
            if req.priority > best_priority {
                best_index = index;
                best_priority = req.priority;
            }
        }
        let req = state.requests.remove(best_index);
        self.metrics.set_queue_depth(state.requests.len());
        req
    }

    pub fn push_front(&self, req: GenerationRequest) {
        let mut state = lock_or_recover(&self.inner);
        state.requests.push_front(req);
        self.metrics.set_queue_depth(state.requests.len());
        drop(state);
        self.notify.notify_one();
    }

    pub async fn wait_nonempty(&self) {
        loop {
            if !self.is_empty() {
                return;
            }
            self.notify.notified().await;
        }
    }

    pub fn len(&self) -> usize {
        let state = lock_or_recover(&self.inner);
        state.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[derive(Debug)]
pub struct GenerationHandle {
    pub id: Uuid,
    pub receiver: mpsc::Receiver<StreamItem>,
}

pub struct Scheduler {
    pub config: SchedulerConfig,
    pub queue: RequestQueue,
    pub running: Vec<Sequence>,
    pub kv_cache: SlabKvCache,
    pub model: SmallTransformer,
    pub metrics: SchedulerMetrics,
}

impl Scheduler {
    pub fn new(
        config: SchedulerConfig,
        queue: RequestQueue,
        kv_cache: SlabKvCache,
        model: SmallTransformer,
        metrics: SchedulerMetrics,
    ) -> Self {
        Self {
            config,
            queue,
            running: Vec::new(),
            kv_cache,
            model,
            metrics,
        }
    }

    /// Runs continuous batching: unlike static batching, the set of running sequences
    /// can change every decode step, and unlike request-per-batch serving, every
    /// active sequence contributes exactly one next-token input to the same forward
    /// pass. The scheduler is intentionally single-threaded because it owns the
    /// model and KV cache; hardware utilization comes from the batch dimension,
    /// not concurrent mutable access to the model. Prefill processes a new prompt
    /// once to populate KV state, while decode advances already-admitted sequences
    /// one token at a time using the cached state.
    pub async fn run(&mut self, shutdown: CancellationToken) -> Result<()> {
        loop {
            if shutdown.is_cancelled() {
                break;
            }

            while self.running.len() < self.config.max_running_seqs {
                let Some(req) = self.queue.try_pop() else {
                    break;
                };
                if !self
                    .kv_cache
                    .has_room_for(req.prompt_len(), self.config.max_new_tokens)
                {
                    self.queue.push_front(req);
                    break;
                }
                let mut seq = Sequence::new(req, &mut self.kv_cache)?;
                self.prefill(&mut seq)?;
                self.running.push(seq);
            }

            if self.running.is_empty() {
                tokio::select! {
                    _ = self.queue.wait_nonempty() => continue,
                    _ = shutdown.cancelled() => break,
                }
            }

            let batch = self.build_decode_batch();

            let step_start = Instant::now();
            let logits = self.model.forward(&batch, &mut self.kv_cache)?;

            for (i, seq) in self.running.iter_mut().enumerate() {
                let token = seq.sampler.sample(logits.row(i)?);
                let timing = seq.append_token(token);
                self.metrics
                    .record_token(timing.first_latency, timing.inter_token_latency);
                if let Err(err) = seq.sender.try_send(StreamItem::Token(token)) {
                    match err {
                        tokio::sync::mpsc::error::TrySendError::Full(item) => {
                            if seq.sender.send(item).await.is_err() {
                                seq.cancel();
                                self.metrics.record_cancelled();
                            }
                        }
                        tokio::sync::mpsc::error::TrySendError::Closed(_) => {
                            seq.cancel();
                            self.metrics.record_cancelled();
                        }
                    }
                }
                if seq.is_finished() || seq.is_cancelled() {
                    if let Err(tokio::sync::mpsc::error::TrySendError::Full(item)) =
                        seq.sender.try_send(StreamItem::Done)
                    {
                        let _ = seq.sender.send(item).await;
                    }
                }
            }

            let kv_cache = &mut self.kv_cache;
            self.running.retain_mut(|seq| {
                if seq.is_done() {
                    kv_cache.free(seq.kv_handle());
                    false
                } else {
                    true
                }
            });

            self.metrics.record_step(
                batch.dim(0),
                step_start.elapsed().as_micros() as u64,
                self.running.len(),
            );
            self.metrics
                .set_batch_utilization(batch.dim(0), self.config.max_running_seqs);
        }
        Ok(())
    }

    fn prefill(&mut self, seq: &mut Sequence) -> Result<()> {
        self.model
            .prefill(&seq.prompt_tokens, &seq.kv_handle(), &mut self.kv_cache)?;
        seq.mark_running();
        Ok(())
    }

    fn build_decode_batch(&self) -> DecodeBatch {
        let rows = self
            .running
            .iter()
            .map(|seq| DecodeInput {
                seq_id: seq.id,
                kv_handle: seq.kv_handle(),
                last_token: seq.last_token(),
                position: seq.decode_position(),
                prompt_hash: seq.prompt_hash,
                seed: seq.seed(),
            })
            .collect();
        DecodeBatch::new(rows)
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    registry: Registry,
    tokens_generated: IntCounter,
    scheduler_steps: IntCounter,
    shared_decode_steps: IntCounter,
    cancelled_sequences: IntCounter,
    first_token_latency: Histogram,
    inter_token_latency: Histogram,
    batch_size: Gauge,
    running_sequences: IntGauge,
    queue_depth: IntGauge,
    batch_utilization: Gauge,
}

impl SchedulerMetrics {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        let tokens_generated = IntCounter::with_opts(Opts::new(
            "hotbatch_tokens_generated_total",
            "Total generated tokens streamed by Hotbatch.",
        ))?;
        let scheduler_steps = IntCounter::with_opts(Opts::new(
            "hotbatch_scheduler_steps_total",
            "Total scheduler decode steps.",
        ))?;
        let shared_decode_steps = IntCounter::with_opts(Opts::new(
            "hotbatch_shared_decode_steps_total",
            "Decode steps where at least two sequences advanced in the same model forward pass.",
        ))?;
        let cancelled_sequences = IntCounter::with_opts(Opts::new(
            "hotbatch_cancelled_sequences_total",
            "Sequences evicted after the client disconnected.",
        ))?;
        let first_token_latency = Histogram::with_opts(HistogramOpts::new(
            "hotbatch_first_token_latency_seconds",
            "First token latency in seconds.",
        ))?;
        let inter_token_latency = Histogram::with_opts(HistogramOpts::new(
            "hotbatch_inter_token_latency_seconds",
            "Inter-token latency in seconds.",
        ))?;
        let batch_size = Gauge::with_opts(Opts::new(
            "hotbatch_batch_size",
            "Batch size used for the most recent decode step.",
        ))?;
        let running_sequences = IntGauge::with_opts(Opts::new(
            "hotbatch_running_sequences",
            "Running sequences after the most recent scheduler step.",
        ))?;
        let queue_depth = IntGauge::with_opts(Opts::new(
            "hotbatch_queue_depth",
            "Queued requests waiting for admission.",
        ))?;
        let batch_utilization = Gauge::with_opts(Opts::new(
            "hotbatch_batch_utilization",
            "Most recent batch size divided by configured max running sequences.",
        ))?;

        registry.register(Box::new(tokens_generated.clone()))?;
        registry.register(Box::new(scheduler_steps.clone()))?;
        registry.register(Box::new(shared_decode_steps.clone()))?;
        registry.register(Box::new(cancelled_sequences.clone()))?;
        registry.register(Box::new(first_token_latency.clone()))?;
        registry.register(Box::new(inter_token_latency.clone()))?;
        registry.register(Box::new(batch_size.clone()))?;
        registry.register(Box::new(running_sequences.clone()))?;
        registry.register(Box::new(queue_depth.clone()))?;
        registry.register(Box::new(batch_utilization.clone()))?;

        Ok(Self {
            registry,
            tokens_generated,
            scheduler_steps,
            shared_decode_steps,
            cancelled_sequences,
            first_token_latency,
            inter_token_latency,
            batch_size,
            running_sequences,
            queue_depth,
            batch_utilization,
        })
    }

    pub fn record_step(&self, batch_size: usize, _step_us: u64, running_after: usize) {
        self.scheduler_steps.inc();
        if batch_size > 1 {
            self.shared_decode_steps.inc();
        }
        self.batch_size.set(batch_size as f64);
        self.running_sequences.set(running_after as i64);
    }

    pub fn set_max_running(&self, max_running: usize) {
        let current = self.batch_size.get();
        if max_running > 0 {
            self.batch_utilization.set(current / max_running as f64);
        }
    }

    pub fn set_batch_utilization(&self, batch_size: usize, max_running: usize) {
        if max_running > 0 {
            self.batch_utilization
                .set(batch_size as f64 / max_running as f64);
        }
    }

    pub fn record_token(
        &self,
        first_latency: Option<Duration>,
        inter_token_latency: Option<Duration>,
    ) {
        self.tokens_generated.inc();
        if let Some(latency) = first_latency {
            self.first_token_latency.observe(latency.as_secs_f64());
        }
        if let Some(latency) = inter_token_latency {
            self.inter_token_latency.observe(latency.as_secs_f64());
        }
    }

    pub fn record_cancelled(&self) {
        self.cancelled_sequences.inc();
    }

    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.set(depth as i64);
    }

    pub fn set_running(&self, running: usize) {
        self.running_sequences.set(running as i64);
    }

    pub fn gather(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    pub fn counter_value(&self, name: &str) -> Result<u64> {
        let text = self.gather()?;
        for line in text.lines() {
            if line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(metric_name) = parts.next() else {
                continue;
            };
            let Some(metric_value) = parts.next() else {
                continue;
            };
            if metric_name == name {
                let parsed = metric_value
                    .parse::<f64>()
                    .map_err(|err| anyhow!("failed to parse metric {name}: {err}"))?;
                return Ok(parsed as u64);
            }
        }
        Ok(0)
    }
}
