use anyhow::Result;
use hotbatch_core::{
    DecodeBatch, DecodeInput, GenerationHandle, GenerationRequest, KvCache, SchedulerMetrics,
    Sequence, SlabKvCache, SmallTransformer, StreamItem,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct NaiveEngine {
    model: Arc<Mutex<SmallTransformer>>,
    metrics: SchedulerMetrics,
}

impl NaiveEngine {
    pub fn new(model: SmallTransformer, metrics: SchedulerMetrics) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            metrics,
        }
    }

    pub fn submit(
        &self,
        req: GenerationRequest,
        receiver: tokio::sync::mpsc::Receiver<StreamItem>,
    ) -> Result<GenerationHandle> {
        let id = req.id;
        let handle = GenerationHandle { id, receiver };
        let model = self.model.clone();
        let metrics = self.metrics.clone();
        tokio::spawn(async move {
            if let Err(err) = run_naive_request(model, metrics, req).await {
                tracing::warn!(error = %err, "naive request failed");
            }
        });
        Ok(handle)
    }
}

async fn run_naive_request(
    model: Arc<Mutex<SmallTransformer>>,
    metrics: SchedulerMetrics,
    req: GenerationRequest,
) -> Result<()> {
    let mut model = model.lock().await;
    let (num_layers, n_heads, head_dim) = model.kv_shape();
    let max_seq_len = req.prompt_len().saturating_add(req.max_new_tokens()).max(1);
    let mut kv_cache = SlabKvCache::new(1, max_seq_len, num_layers, n_heads, head_dim);
    let mut seq = Sequence::new(req, &mut kv_cache)?;
    metrics.set_running(1);

    model.prefill(&seq.prompt_tokens, &seq.kv_handle(), &mut kv_cache)?;
    seq.mark_running();

    while !seq.is_done() {
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
        let token = seq.sampler.sample(logits.row(0)?);
        let timing = seq.append_token(token);
        metrics.record_token(timing.first_latency, timing.inter_token_latency);
        if let Err(err) = seq.sender.try_send(StreamItem::Token(token)) {
            match err {
                tokio::sync::mpsc::error::TrySendError::Full(item) => {
                    if seq.sender.send(item).await.is_err() {
                        seq.cancel();
                        metrics.record_cancelled();
                    }
                }
                tokio::sync::mpsc::error::TrySendError::Closed(_) => {
                    seq.cancel();
                    metrics.record_cancelled();
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
        metrics.record_step(
            1,
            step_start.elapsed().as_micros() as u64,
            usize::from(!seq.is_done()),
        );
        metrics.set_batch_utilization(1, 1);
    }

    kv_cache.free(seq.kv_handle());
    metrics.set_running(0);
    Ok(())
}
