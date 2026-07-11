use anyhow::Result;
use hotbatch_core::{
    sequence::try_send_tokens, DecodeBatch, DecodeInput, GenerationHandle, GenerationRequest,
    KvCache, SchedulerMetrics, Sequence, SlabKvCache, SmallTransformer, StreamItem,
};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NaiveSubmitError {
    QueueFull,
    Unavailable,
}

struct AdmissionState {
    accepted: usize,
}

struct AdmissionControl {
    state: StdMutex<AdmissionState>,
    max_queue_depth: usize,
    metrics: SchedulerMetrics,
}

impl AdmissionControl {
    fn try_acquire(self: &Arc<Self>) -> Option<AdmissionPermit> {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };

        // One request may own the model while at most max_queue_depth requests
        // wait behind it, matching the queue-depth semantics of continuous mode.
        if state.accepted > self.max_queue_depth {
            return None;
        }
        state.accepted = state.accepted.checked_add(1)?;
        self.metrics
            .set_queue_depth(state.accepted.saturating_sub(1));
        Some(AdmissionPermit {
            control: self.clone(),
        })
    }
}

struct AdmissionPermit {
    control: Arc<AdmissionControl>,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        let mut state = match self.control.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.accepted = state.accepted.saturating_sub(1);
        self.control
            .metrics
            .set_queue_depth(state.accepted.saturating_sub(1));
    }
}

#[derive(Clone)]
pub struct NaiveEngine {
    model: Arc<Mutex<SmallTransformer>>,
    metrics: SchedulerMetrics,
    shutdown: CancellationToken,
    admission: Arc<AdmissionControl>,
}

impl NaiveEngine {
    pub fn new(
        model: SmallTransformer,
        metrics: SchedulerMetrics,
        shutdown: CancellationToken,
        max_queue_depth: usize,
    ) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            metrics: metrics.clone(),
            shutdown,
            admission: Arc::new(AdmissionControl {
                state: StdMutex::new(AdmissionState { accepted: 0 }),
                max_queue_depth,
                metrics,
            }),
        }
    }

    pub fn submit(
        &self,
        req: GenerationRequest,
        receiver: tokio::sync::mpsc::Receiver<StreamItem>,
    ) -> std::result::Result<GenerationHandle, NaiveSubmitError> {
        if self.shutdown.is_cancelled() {
            return Err(NaiveSubmitError::Unavailable);
        }
        let permit = self
            .admission
            .try_acquire()
            .ok_or(NaiveSubmitError::QueueFull)?;
        if self.shutdown.is_cancelled() {
            return Err(NaiveSubmitError::Unavailable);
        }

        let id = req.id;
        let response_done = req.response_done.clone();
        let handle = GenerationHandle {
            id,
            receiver,
            response_done: response_done.clone(),
        };
        let model = self.model.clone();
        let metrics = self.metrics.clone();
        let shutdown = self.shutdown.clone();
        let error_sender = req.sender.clone();
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(err) = run_naive_request(model, metrics, req, shutdown.clone()).await {
                if shutdown.is_cancelled() {
                    tracing::debug!(error = %err, "naive request cancelled during shutdown");
                } else {
                    tracing::warn!(error = %err, "naive request failed");
                }
                let _ = error_sender.try_send(StreamItem::Error(err.to_string()));
            }
        });
        Ok(handle)
    }
}

async fn run_naive_request(
    model: Arc<Mutex<SmallTransformer>>,
    metrics: SchedulerMetrics,
    req: GenerationRequest,
    shutdown: CancellationToken,
) -> Result<()> {
    if shutdown.is_cancelled() {
        anyhow::bail!("generation scheduler is shutting down");
    }
    let response_done = req.response_done.clone();
    if response_done.is_cancelled() {
        return Ok(());
    }
    if req.sender.is_closed() {
        metrics.record_cancelled();
        return Ok(());
    }

    let mut model = tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            anyhow::bail!("generation scheduler is shutting down");
        }
        _ = response_done.cancelled() => {
            return Ok(());
        }
        _ = req.sender.closed() => {
            metrics.record_cancelled();
            return Ok(());
        }
        model = model.lock() => model,
    };

    if shutdown.is_cancelled() {
        anyhow::bail!("generation scheduler is shutting down");
    }
    if response_done.is_cancelled() {
        return Ok(());
    }
    if req.sender.is_closed() {
        metrics.record_cancelled();
        return Ok(());
    }
    let (num_layers, n_heads, head_dim) = model.kv_shape();
    let max_seq_len = req.prompt_len().saturating_add(req.max_new_tokens()).max(1);
    let mut kv_cache = SlabKvCache::new(1, max_seq_len, num_layers, n_heads, head_dim);
    let mut seq = Sequence::new(req, &mut kv_cache)?;
    metrics.set_running(1);

    let result = async {
        if shutdown.is_cancelled() {
            anyhow::bail!("generation scheduler is shutting down");
        }
        if seq.response_is_done() {
            return Ok(());
        }
        model.prefill(&seq.prompt_tokens, &seq.kv_handle(), &mut kv_cache)?;
        if shutdown.is_cancelled() {
            anyhow::bail!("generation scheduler is shutting down");
        }
        if seq.response_is_done() {
            return Ok(());
        }
        seq.mark_running();

        while !seq.is_done() {
            if shutdown.is_cancelled() {
                anyhow::bail!("generation scheduler is shutting down");
            }
            if seq.response_is_done() {
                seq.cancel();
                break;
            }
            if seq.sender.is_closed() {
                seq.cancel();
                metrics.record_cancelled();
                break;
            }
            let batch = DecodeBatch::new(vec![DecodeInput {
                seq_id: seq.id,
                kv_handle: seq.kv_handle(),
                last_token: seq.last_token(),
                position: seq.decode_position(),
                prompt_hash: seq.prompt_hash,
                seed: seq.seed(),
            }]);
            let step_start = Instant::now();
            let logits = model.forward(&batch, &mut kv_cache)?;
            if shutdown.is_cancelled() {
                anyhow::bail!("generation scheduler is shutting down");
            }
            if seq.response_is_done() {
                seq.cancel();
                break;
            }
            let token = seq.sampler.sample(logits.row(0)?);
            let timing = seq.append_token(token);
            metrics.record_token(timing.first_latency, timing.inter_token_latency);
            let finish_reason = seq.finish_reason();
            let output_tokens = seq.take_emittable_tokens();
            if !try_send_tokens(&seq.sender, &output_tokens, finish_reason) {
                seq.cancel();
                metrics.record_cancelled();
            }
            metrics.record_step(
                1,
                step_start.elapsed().as_micros() as u64,
                usize::from(!seq.is_done()),
            );
            metrics.set_batch_utilization(1, 1);
            // The model forward pass is synchronous. Give the HTTP response
            // task a chance to drain its bounded channel between steps.
            tokio::task::yield_now().await;
        }

        Ok(())
    }
    .await;

    kv_cache.free(seq.kv_handle());
    metrics.set_running(0);
    result
}
