use crate::kv_cache::{KvCache, SlabKvCache};
use crate::model::{DecodeBatch, DecodeInput, SmallTransformer};
use crate::sequence::{try_send_tokens, GenerationRequest, Sequence, StreamItem};
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
    closed: bool,
}

impl RequestQueue {
    pub fn new(max_depth: usize, metrics: SchedulerMetrics) -> Self {
        Self {
            inner: Arc::new(Mutex::new(QueueState {
                requests: VecDeque::new(),
                max_depth,
                closed: false,
            })),
            notify: Arc::new(Notify::new()),
            metrics,
        }
    }

    pub fn submit(&self, req: GenerationRequest) -> std::result::Result<(), QueueFull> {
        let mut state = lock_or_recover(&self.inner);
        let before = state.requests.len();
        state.requests.retain(|pending| !pending.sender.is_closed());
        for _ in state.requests.len()..before {
            self.metrics.record_cancelled();
        }
        self.metrics.set_queue_depth(state.requests.len());
        if state.closed || state.requests.len() >= state.max_depth {
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
        let before = state.requests.len();
        state.requests.retain(|pending| !pending.sender.is_closed());
        for _ in state.requests.len()..before {
            self.metrics.record_cancelled();
        }
        self.metrics.set_queue_depth(state.requests.len());
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

    pub fn is_closed(&self) -> bool {
        let state = lock_or_recover(&self.inner);
        state.closed
    }

    fn close_and_drain(&self) -> Vec<GenerationRequest> {
        let mut state = lock_or_recover(&self.inner);
        state.closed = true;
        let requests = state.requests.drain(..).collect();
        self.metrics.set_queue_depth(0);
        drop(state);
        self.notify.notify_waiters();
        requests
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
    pub response_done: CancellationToken,
}

impl GenerationHandle {
    pub fn stop_generation(&self) {
        self.response_done.cancel();
    }
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

            self.remove_closed_sequences();

            while self.running.len() < self.config.max_running_seqs {
                if self.kv_cache.allocated_slots() >= self.kv_cache.max_sequences() {
                    break;
                }
                let Some(req) = self.queue.try_pop() else {
                    break;
                };

                if req.response_done.is_cancelled() {
                    continue;
                }
                if req.sender.is_closed() {
                    self.metrics.record_cancelled();
                    continue;
                }

                if let Err(message) = self.validate_request(&req) {
                    try_send_error(&req.sender, message);
                    continue;
                }

                let sender = req.sender.clone();
                let request_id = req.id;
                let mut seq = match Sequence::new(req, &mut self.kv_cache) {
                    Ok(seq) => seq,
                    Err(err) => {
                        try_send_error(
                            &sender,
                            format!("failed to admit request {request_id}: {err}"),
                        );
                        continue;
                    }
                };

                if seq.response_is_done() {
                    seq.cancel();
                    self.kv_cache.free(seq.kv_handle());
                    continue;
                }
                if seq.sender.is_closed() {
                    seq.cancel();
                    self.metrics.record_cancelled();
                    self.kv_cache.free(seq.kv_handle());
                    continue;
                }

                if let Err(err) = self.prefill(&mut seq) {
                    try_send_error(
                        &seq.sender,
                        format!("failed to prefill request {}: {err}", seq.id),
                    );
                    seq.fail();
                    self.kv_cache.free(seq.kv_handle());
                    continue;
                }

                if seq.response_is_done() {
                    seq.cancel();
                    self.kv_cache.free(seq.kv_handle());
                    continue;
                }
                if seq.sender.is_closed() {
                    seq.cancel();
                    self.metrics.record_cancelled();
                    self.kv_cache.free(seq.kv_handle());
                    continue;
                }

                self.running.push(seq);
            }
            self.metrics.set_running(self.running.len());

            self.remove_sequences_past_model_capacity();

            if self.running.is_empty() {
                tokio::select! {
                    _ = self.queue.wait_nonempty() => continue,
                    _ = shutdown.cancelled() => break,
                }
            }

            let batch = self.build_decode_batch();
            let batch_size = batch.dim(0);

            let step_start = Instant::now();
            let logits = match self.model.forward(&batch, &mut self.kv_cache) {
                Ok(logits) => logits,
                Err(err) => {
                    let message = format!("decode batch failed: {err}");
                    for seq in &mut self.running {
                        try_send_error(&seq.sender, message.clone());
                        seq.fail();
                    }
                    self.reap_done_sequences();
                    self.metrics.record_step(
                        batch_size,
                        step_start.elapsed().as_micros() as u64,
                        self.running.len(),
                    );
                    self.metrics
                        .set_batch_utilization(batch_size, self.config.max_running_seqs);
                    continue;
                }
            };

            for (i, seq) in self.running.iter_mut().enumerate() {
                if seq.response_is_done() {
                    seq.cancel();
                    continue;
                }
                if seq.sender.is_closed() {
                    seq.cancel();
                    self.metrics.record_cancelled();
                    continue;
                }

                let row = match logits.row(i) {
                    Ok(row) => row,
                    Err(err) => {
                        try_send_error(
                            &seq.sender,
                            format!("decode row failed for request {}: {err}", seq.id),
                        );
                        seq.fail();
                        continue;
                    }
                };
                let token = seq.sampler.sample(row);
                let timing = seq.append_token(token);
                self.metrics
                    .record_token(timing.first_latency, timing.inter_token_latency);

                let finish_reason = seq.finish_reason();
                let output_tokens = seq.take_emittable_tokens();
                if !try_send_tokens(&seq.sender, &output_tokens, finish_reason) {
                    seq.cancel();
                    self.metrics.record_cancelled();
                }
            }

            self.reap_done_sequences();

            self.metrics.record_step(
                batch_size,
                step_start.elapsed().as_micros() as u64,
                self.running.len(),
            );
            self.metrics
                .set_batch_utilization(batch_size, self.config.max_running_seqs);
            // Model execution is synchronous. Yield between decode steps so a
            // healthy response consumer can drain its bounded channel before
            // the next nonblocking delivery attempt.
            tokio::task::yield_now().await;
        }

        let shutdown_message = "generation scheduler is shutting down".to_string();
        for request in self.queue.close_and_drain() {
            try_send_error(&request.sender, shutdown_message.clone());
        }
        for seq in self.running.drain(..) {
            try_send_error(&seq.sender, shutdown_message.clone());
            self.kv_cache.free(seq.kv_handle());
        }
        self.metrics.set_running(0);
        Ok(())
    }

    fn validate_request(&self, req: &GenerationRequest) -> std::result::Result<(), String> {
        if req.prompt_tokens.is_empty() {
            return Err("request prompt must contain at least one token".to_string());
        }
        if req.max_new_tokens() == 0 {
            return Err("request max_new_tokens must be greater than zero".to_string());
        }
        if req.max_new_tokens() > self.config.max_new_tokens {
            return Err(format!(
                "request max_new_tokens={} exceeds scheduler limit {}",
                req.max_new_tokens(),
                self.config.max_new_tokens
            ));
        }

        let max_seq_len = self
            .config
            .max_seq_len
            .min(self.kv_cache.max_sequence_len())
            .min(self.model.max_positions());
        let Some(required_tokens) = req.prompt_len().checked_add(req.max_new_tokens()) else {
            return Err("request sequence length overflowed capacity accounting".to_string());
        };
        if required_tokens > max_seq_len {
            return Err(format!(
                "request requires {required_tokens} tokens but sequence capacity is {max_seq_len}"
            ));
        }
        Ok(())
    }

    fn remove_closed_sequences(&mut self) {
        for seq in &mut self.running {
            if seq.response_is_done() {
                seq.cancel();
            } else if seq.sender.is_closed() {
                seq.cancel();
                self.metrics.record_cancelled();
            }
        }
        self.reap_done_sequences();
    }

    fn remove_sequences_past_model_capacity(&mut self) {
        let max_positions = self.model.max_positions();
        for seq in &mut self.running {
            if seq.decode_position() >= max_positions {
                try_send_error(
                    &seq.sender,
                    format!(
                        "decode position {} exceeds model capacity {} for request {}",
                        seq.decode_position(),
                        max_positions,
                        seq.id
                    ),
                );
                seq.fail();
            }
        }
        self.reap_done_sequences();
    }

    fn reap_done_sequences(&mut self) {
        let kv_cache = &mut self.kv_cache;
        self.running.retain_mut(|seq| {
            if seq.is_done() {
                kv_cache.free(seq.kv_handle());
                false
            } else {
                true
            }
        });
        self.metrics.set_running(self.running.len());
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

fn try_send_error(sender: &mpsc::Sender<StreamItem>, message: String) -> bool {
    sender.try_send(StreamItem::Error(message)).is_ok()
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
            "Total tokens sampled by Hotbatch.",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelOptions;
    use crate::sampler::SamplerConfig;
    use crate::sequence::FinishReason;

    fn request(
        id: Uuid,
        priority: u8,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        channel_capacity: usize,
    ) -> (GenerationRequest, mpsc::Receiver<StreamItem>) {
        let (sender, receiver) = mpsc::channel(channel_capacity);
        (
            GenerationRequest {
                id,
                prompt_hash: 0,
                prompt_tokens,
                sampler_config: SamplerConfig {
                    max_new_tokens,
                    eos_token: u32::MAX,
                    ..SamplerConfig::default()
                },
                sender,
                priority,
                created_at: Instant::now(),
                response_done: CancellationToken::new(),
            },
            receiver,
        )
    }

    #[test]
    fn queue_prioritizes_requests_and_preserves_fifo_within_priority() {
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(4, metrics);
        let low = Uuid::new_v4();
        let high_first = Uuid::new_v4();
        let high_second = Uuid::new_v4();
        let (low_request, _low_receiver) = request(low, 1, vec![1], 1, 1);
        let (high_first_request, _high_first_receiver) = request(high_first, 9, vec![1], 1, 1);
        let (high_second_request, _high_second_receiver) = request(high_second, 9, vec![1], 1, 1);

        queue.submit(low_request).expect("queue low");
        queue.submit(high_first_request).expect("queue first high");
        queue
            .submit(high_second_request)
            .expect("queue second high");

        assert_eq!(queue.try_pop().expect("first request").id, high_first);
        assert_eq!(queue.try_pop().expect("second request").id, high_second);
        assert_eq!(queue.try_pop().expect("third request").id, low);
        assert!(queue.try_pop().is_none());
    }

    #[test]
    fn queue_enforces_exact_capacity_and_recovers_after_pop() {
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(2, metrics);
        let (first, _first_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        let (second, _second_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        let (rejected, _rejected_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        let (replacement, _replacement_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);

        queue.submit(first).expect("first request should fit");
        queue.submit(second).expect("second request should fit");
        assert_eq!(queue.len(), 2);
        assert!(queue.submit(rejected).is_err());
        assert_eq!(queue.len(), 2);

        queue.try_pop().expect("pop should free queue capacity");
        queue
            .submit(replacement)
            .expect("replacement request should fit");
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn submitting_prunes_disconnected_queued_requests() {
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(1, metrics.clone());
        let (disconnected, disconnected_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        queue
            .submit(disconnected)
            .expect("queue disconnected request");
        drop(disconnected_receiver);

        let (replacement, _replacement_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        queue
            .submit(replacement)
            .expect("closed request should not retain capacity");

        assert_eq!(queue.len(), 1);
        assert_eq!(
            metrics
                .counter_value("hotbatch_cancelled_sequences_total")
                .expect("cancelled metric"),
            1
        );
    }

    #[test]
    fn closed_queue_rejects_new_work_and_drains_pending_requests() {
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(2, metrics);
        let (pending, _pending_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        queue.submit(pending).expect("pending request");

        let drained = queue.close_and_drain();

        assert_eq!(drained.len(), 1);
        assert!(queue.is_closed());
        assert!(queue.is_empty());
        let (late, _late_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        assert!(queue.submit(late).is_err());
    }

    #[test]
    fn terminal_delivery_reserves_token_and_finish_together() {
        let (sender, mut receiver) = mpsc::channel(2);

        assert!(try_send_tokens(&sender, &[42], Some(FinishReason::Length)));
        assert_eq!(receiver.try_recv(), Ok(StreamItem::Token(42)));
        assert_eq!(
            receiver.try_recv(),
            Ok(StreamItem::Finished(FinishReason::Length))
        );
    }

    #[test]
    fn terminal_delivery_never_partially_fills_a_small_channel() {
        let (sender, mut receiver) = mpsc::channel(1);

        assert!(!try_send_tokens(&sender, &[42], Some(FinishReason::Stop)));
        assert_eq!(receiver.try_recv(), Err(mpsc::error::TryRecvError::Empty));
    }

    #[tokio::test]
    async fn oversized_request_is_rejected_without_blocking_later_work() {
        let config = SchedulerConfig {
            max_running_seqs: 1,
            max_new_tokens: 4,
            max_seq_len: 4,
            max_queue_depth: 4,
            idle_sleep: Duration::from_millis(1),
        };
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(config.max_queue_depth, metrics.clone());
        let (oversized, mut oversized_receiver) =
            request(Uuid::new_v4(), 0, vec![1, 2, 3, 4], 1, 1);
        let (valid, mut valid_receiver) = request(Uuid::new_v4(), 0, vec![1, 2, 3], 1, 2);
        queue.submit(oversized).expect("queue oversized request");
        queue.submit(valid).expect("queue valid request");

        let model = SmallTransformer::load(ModelOptions {
            model: "scripted".to_string(),
            device: "cpu".to_string(),
            ..ModelOptions::default()
        })
        .await
        .expect("scripted model");
        let kv_cache = SlabKvCache::new(1, 4, 12, 12, 64);
        let mut scheduler = Scheduler::new(config, queue, kv_cache, model, metrics);
        let shutdown = CancellationToken::new();
        let observer_shutdown = shutdown.clone();

        let observer = tokio::spawn(async move {
            let rejected = tokio::time::timeout(Duration::from_secs(5), oversized_receiver.recv())
                .await
                .expect("oversized response timeout")
                .expect("oversized response");
            assert!(matches!(rejected, StreamItem::Error(message) if message.contains("capacity")));

            let token = tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                .await
                .expect("valid token timeout")
                .expect("valid token");
            assert!(matches!(token, StreamItem::Token(_)));
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                    .await
                    .expect("valid finish timeout"),
                Some(StreamItem::Finished(FinishReason::Length))
            );
            observer_shutdown.cancel();
        });

        tokio::time::timeout(Duration::from_secs(10), scheduler.run(shutdown))
            .await
            .expect("scheduler shutdown timeout")
            .expect("scheduler should survive rejection");
        observer.await.expect("observer task");

        assert_eq!(scheduler.kv_cache.allocated_slots(), 0);
        assert!(scheduler.queue.is_empty());
    }

    #[tokio::test]
    async fn prefill_failure_releases_slot_and_scheduler_continues() {
        let config = SchedulerConfig {
            max_running_seqs: 1,
            max_new_tokens: 1,
            max_seq_len: 8,
            max_queue_depth: 2,
            idle_sleep: Duration::from_millis(1),
        };
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(config.max_queue_depth, metrics.clone());
        let (prefill_failure, mut failure_receiver) =
            request(Uuid::new_v4(), 0, vec![u32::MAX], 1, 1);
        let (valid, mut valid_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 2);
        queue
            .submit(prefill_failure)
            .expect("queue request that fails prefill");
        queue.submit(valid).expect("queue valid request");

        let model = SmallTransformer::load(ModelOptions {
            model: "scripted".to_string(),
            device: "cpu".to_string(),
            ..ModelOptions::default()
        })
        .await
        .expect("scripted model");
        let kv_cache = SlabKvCache::new(1, 8, 12, 12, 64);
        let mut scheduler = Scheduler::new(config, queue, kv_cache, model, metrics);
        let shutdown = CancellationToken::new();
        let observer_shutdown = shutdown.clone();

        let observer = tokio::spawn(async move {
            let error = tokio::time::timeout(Duration::from_secs(5), failure_receiver.recv())
                .await
                .expect("prefill error timeout")
                .expect("prefill error");
            assert!(
                matches!(error, StreamItem::Error(message) if message.contains("prefill") && message.contains("scripted prefill failure"))
            );

            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                    .await
                    .expect("valid token timeout"),
                Some(StreamItem::Token(_))
            ));
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                    .await
                    .expect("valid finish timeout"),
                Some(StreamItem::Finished(FinishReason::Length))
            );
            observer_shutdown.cancel();
        });

        tokio::time::timeout(Duration::from_secs(10), scheduler.run(shutdown))
            .await
            .expect("scheduler shutdown timeout")
            .expect("scheduler should survive prefill failure");
        observer.await.expect("observer task");

        assert_eq!(scheduler.kv_cache.allocated_slots(), 0);
        assert!(scheduler.queue.is_empty());
    }

    #[tokio::test]
    async fn model_over_capacity_request_does_not_stop_healthy_work() {
        let config = SchedulerConfig {
            max_running_seqs: 2,
            max_new_tokens: 2,
            max_seq_len: 1_026,
            max_queue_depth: 2,
            idle_sleep: Duration::from_millis(1),
        };
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(config.max_queue_depth, metrics.clone());
        let (decode_failure, mut failure_receiver) =
            request(Uuid::new_v4(), 0, vec![1; 1_024], 2, 4);
        let (valid, mut valid_receiver) = request(Uuid::new_v4(), 0, vec![1], 2, 3);
        queue
            .submit(decode_failure)
            .expect("queue request that fails decode");
        queue.submit(valid).expect("queue valid request");

        let model = SmallTransformer::load(ModelOptions {
            model: "scripted".to_string(),
            device: "cpu".to_string(),
            ..ModelOptions::default()
        })
        .await
        .expect("scripted model");
        let kv_cache = SlabKvCache::new(1, 1_026, 12, 12, 64);
        let mut scheduler = Scheduler::new(config, queue, kv_cache, model, metrics);
        let shutdown = CancellationToken::new();
        let observer_shutdown = shutdown.clone();

        let observer = tokio::spawn(async move {
            let error = tokio::time::timeout(Duration::from_secs(5), failure_receiver.recv())
                .await
                .expect("capacity error timeout")
                .expect("capacity error");
            assert!(matches!(error, StreamItem::Error(message) if message.contains("capacity")));

            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                    .await
                    .expect("first valid token timeout"),
                Some(StreamItem::Token(_))
            ));
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                    .await
                    .expect("second valid token timeout"),
                Some(StreamItem::Token(_))
            ));
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(5), valid_receiver.recv())
                    .await
                    .expect("valid finish timeout"),
                Some(StreamItem::Finished(FinishReason::Length))
            );
            observer_shutdown.cancel();
        });

        tokio::time::timeout(Duration::from_secs(10), scheduler.run(shutdown))
            .await
            .expect("scheduler shutdown timeout")
            .expect("scheduler should survive decode failure");
        observer.await.expect("observer task");

        assert_eq!(scheduler.kv_cache.allocated_slots(), 0);
        assert!(scheduler.queue.is_empty());
    }

    #[tokio::test]
    async fn shutdown_errors_queued_requests_and_closes_the_queue() {
        let config = SchedulerConfig {
            max_running_seqs: 1,
            max_new_tokens: 2,
            max_seq_len: 8,
            max_queue_depth: 2,
            idle_sleep: Duration::from_millis(1),
        };
        let metrics = SchedulerMetrics::new().expect("metrics");
        let queue = RequestQueue::new(config.max_queue_depth, metrics.clone());
        let (pending, mut pending_receiver) = request(Uuid::new_v4(), 0, vec![1], 1, 1);
        queue.submit(pending).expect("queue pending request");

        let model = SmallTransformer::load(ModelOptions {
            model: "scripted".to_string(),
            device: "cpu".to_string(),
            ..ModelOptions::default()
        })
        .await
        .expect("scripted model");
        let kv_cache = SlabKvCache::new(1, 8, 12, 12, 64);
        let mut scheduler = Scheduler::new(config, queue.clone(), kv_cache, model, metrics);
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        scheduler.run(shutdown).await.expect("scheduler shutdown");

        assert!(matches!(
            pending_receiver.recv().await,
            Some(StreamItem::Error(message)) if message.contains("shutting down")
        ));
        assert!(queue.is_closed());
        assert!(queue.is_empty());
        assert_eq!(scheduler.kv_cache.allocated_slots(), 0);
    }
}
