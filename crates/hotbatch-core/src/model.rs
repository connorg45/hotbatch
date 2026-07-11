use crate::kv_cache::{KvCache, KvHandle};
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder};
use hf_hub::HFClientSync;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;

pub const GPT2_REVISION: &str = "607a30d783dfa663caf39e06633721c8d4cfcd7e";
const TINY_GPT2_REVISION: &str = "71034c5d8bde858ff824298bdedc65515b97d2b9";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptions {
    pub model: String,
    pub device: String,
    #[serde(default)]
    pub scripted_timing: ScriptedTiming,
}

/// Optional timing controls for the deterministic scripted test backend.
///
/// GPT-2 backends never consult these values, so production and benchmark
/// latency always reflects model execution rather than synthetic sleeps.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScriptedTiming {
    pub forward_base_us: u64,
    pub forward_per_seq_us: u64,
    pub prefill_per_token_us: u64,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            model: "gpt2".to_string(),
            device: "cpu".to_string(),
            scripted_timing: ScriptedTiming::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodeInput {
    pub seq_id: uuid::Uuid,
    pub kv_handle: KvHandle,
    pub last_token: u32,
    pub position: usize,
    pub prompt_hash: u64,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct DecodeBatch {
    rows: Vec<DecodeInput>,
}

impl DecodeBatch {
    pub fn new(rows: Vec<DecodeInput>) -> Self {
        Self { rows }
    }

    pub fn dim(&self, axis: usize) -> usize {
        if axis == 0 {
            self.rows.len()
        } else {
            1
        }
    }

    pub fn rows(&self) -> &[DecodeInput] {
        &self.rows
    }
}

#[derive(Debug, Clone)]
pub struct DecodeLogits {
    rows: Vec<Vec<f32>>,
}

impl DecodeLogits {
    pub fn row(&self, index: usize) -> Result<LogitRow<'_>> {
        let Some(row) = self.rows.get(index) else {
            return Err(anyhow!("logit row {index} out of range"));
        };
        Ok(LogitRow { values: row })
    }
}

#[derive(Debug, Copy, Clone)]
pub struct LogitRow<'a> {
    values: &'a [f32],
}

impl<'a> LogitRow<'a> {
    pub fn iter(&self) -> std::slice::Iter<'a, f32> {
        self.values.iter()
    }

    pub fn argmax(&self) -> u32 {
        let mut best_index = 0_usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in self.values.iter().enumerate() {
            if *value > best_value {
                best_index = index;
                best_value = *value;
            }
        }
        best_index as u32
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerBundle {
    tokenizer: Option<Arc<Tokenizer>>,
    fallback: Arc<FallbackTokenizer>,
    model_name: String,
    eos_token: u32,
    vocab_size: usize,
}

impl TokenizerBundle {
    pub async fn load(model: &str) -> Result<Self> {
        let model_name = normalize_model_name(model)?;
        Self::load_normalized(model_name).await
    }

    async fn load_normalized(model_name: &str) -> Result<Self> {
        if model_name == "scripted" {
            let fallback = FallbackTokenizer::new();
            return Ok(Self {
                tokenizer: None,
                fallback: Arc::new(fallback),
                model_name: model_name.to_string(),
                eos_token: 0,
                vocab_size: 512,
            });
        }

        let downloaded = tokio::task::spawn_blocking({
            let model_name = model_name.to_string();
            move || download_tokenizer(&model_name)
        })
        .await
        .context("tokenizer download task failed")??;
        let tokenizer = Tokenizer::from_file(&downloaded)
            .map_err(|err| anyhow!("failed to load tokenizer {}: {err}", downloaded.display()))?;
        let vocab_size = tokenizer.get_vocab_size(false);
        let eos_token = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(50_256);
        Ok(Self {
            tokenizer: Some(Arc::new(tokenizer)),
            fallback: Arc::new(FallbackTokenizer::new()),
            model_name: model_name.to_string(),
            eos_token,
            vocab_size,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(tokenizer) = &self.tokenizer {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|err| anyhow!("tokenization failed: {err}"))?;
            return Ok(encoding.get_ids().to_vec());
        }
        Ok(self.fallback.encode(text))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        if let Some(tokenizer) = &self.tokenizer {
            return tokenizer
                .decode(tokens, true)
                .map_err(|err| anyhow!("detokenization failed: {err}"));
        }
        Ok(self.fallback.decode(tokens))
    }

    pub fn token_text(&self, token: u32) -> String {
        self.decode(&[token]).unwrap_or_else(|_| String::new())
    }

    pub fn chat_template(&self, messages: &[ChatMessage]) -> String {
        let mut rendered = String::new();
        for message in messages {
            rendered.push_str(&message.role);
            rendered.push_str(": ");
            rendered.push_str(&message.content);
            rendered.push('\n');
        }
        rendered.push_str("assistant: ");
        rendered
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct SmallTransformer {
    tokenizer: TokenizerBundle,
    backend: ModelBackend,
}

#[derive(Debug)]
enum ModelBackend {
    Gpt2(Box<Gpt2Model>),
    Scripted(ScriptedModel),
}

impl SmallTransformer {
    pub async fn load(options: ModelOptions) -> Result<Self> {
        let model_name = normalize_model_name(&options.model)?;
        let device = candle_device(&options.device)?;
        let tokenizer = TokenizerBundle::load_normalized(model_name).await?;
        let backend = match model_name {
            "gpt2" | "tiny-gpt2" => {
                let files = tokio::task::spawn_blocking({
                    let model_name = model_name.to_string();
                    move || download_gpt2_files(&model_name)
                })
                .await
                .context("model download task failed")??;
                ModelBackend::Gpt2(Box::new(Gpt2Model::load(files, device)?))
            }
            "scripted" => ModelBackend::Scripted(ScriptedModel::new(tokenizer.clone(), options)?),
            other => return Err(anyhow!("unsupported model '{other}'")),
        };
        Ok(Self { tokenizer, backend })
    }

    pub fn tokenizer(&self) -> TokenizerBundle {
        self.tokenizer.clone()
    }

    pub fn kv_shape(&self) -> (usize, usize, usize) {
        match &self.backend {
            ModelBackend::Gpt2(model) => {
                (model.config.n_layer, model.config.n_head, model.head_dim)
            }
            ModelBackend::Scripted(model) => model.kv_shape(),
        }
    }

    pub fn max_positions(&self) -> usize {
        match &self.backend {
            ModelBackend::Gpt2(model) => model.config.max_positions(),
            ModelBackend::Scripted(model) => model.max_positions(),
        }
    }

    pub fn prefill(
        &mut self,
        prompt_tokens: &[u32],
        handle: &KvHandle,
        kv_cache: &mut dyn KvCache,
    ) -> Result<()> {
        if prompt_tokens.len() > self.max_positions() {
            return Err(anyhow!(
                "prompt length {} exceeds model capacity {}",
                prompt_tokens.len(),
                self.max_positions()
            ));
        }
        match &mut self.backend {
            ModelBackend::Gpt2(model) => model.prefill(prompt_tokens, handle, kv_cache),
            ModelBackend::Scripted(model) => model.prefill(prompt_tokens, handle, kv_cache),
        }
    }

    pub fn forward(
        &mut self,
        batch: &DecodeBatch,
        kv_cache: &mut dyn KvCache,
    ) -> Result<DecodeLogits> {
        let max_positions = self.max_positions();
        if let Some(input) = batch
            .rows()
            .iter()
            .find(|input| input.position >= max_positions)
        {
            return Err(anyhow!(
                "decode position {} exceeds model capacity {} for sequence {}",
                input.position,
                max_positions,
                input.seq_id
            ));
        }
        match &mut self.backend {
            ModelBackend::Gpt2(model) => model.forward(batch, kv_cache),
            ModelBackend::Scripted(model) => model.forward(batch, kv_cache),
        }
    }
}

#[derive(Debug)]
struct ScriptedModel {
    tokenizer: TokenizerBundle,
    timing: ScriptedTiming,
    script_tokens: Vec<u32>,
}

impl ScriptedModel {
    fn new(tokenizer: TokenizerBundle, options: ModelOptions) -> Result<Self> {
        let script = " France, known for Paris, art, science, and careful engineering.";
        let mut script_tokens = tokenizer.encode(script)?;
        if script_tokens.is_empty() {
            script_tokens.push(tokenizer.eos_token());
        }
        Ok(Self {
            tokenizer,
            timing: options.scripted_timing,
            script_tokens,
        })
    }

    fn kv_shape(&self) -> (usize, usize, usize) {
        (12, 12, 64)
    }

    fn max_positions(&self) -> usize {
        1024
    }

    fn prefill(
        &mut self,
        prompt_tokens: &[u32],
        handle: &KvHandle,
        kv_cache: &mut dyn KvCache,
    ) -> Result<()> {
        #[cfg(test)]
        if prompt_tokens == [u32::MAX] {
            return Err(anyhow!("scripted prefill failure"));
        }

        let sleep_us = self
            .timing
            .prefill_per_token_us
            .saturating_mul(prompt_tokens.len() as u64);
        if sleep_us > 0 {
            std::thread::sleep(Duration::from_micros(sleep_us));
        }
        let (_layers, heads, head_dim) = self.kv_shape();
        let cached_tokens = prompt_tokens.len().saturating_sub(1);
        let marker = Tensor::zeros((heads, cached_tokens, head_dim), DType::F32, &Device::Cpu)?;
        kv_cache.write(handle, 0, &marker, &marker)?;
        Ok(())
    }

    fn forward(&mut self, batch: &DecodeBatch, kv_cache: &mut dyn KvCache) -> Result<DecodeLogits> {
        let batch_size = batch.dim(0);
        let sleep_us = self.timing.forward_base_us.saturating_add(
            self.timing
                .forward_per_seq_us
                .saturating_mul(batch_size as u64),
        );
        if sleep_us > 0 {
            std::thread::sleep(Duration::from_micros(sleep_us));
        }

        let vocab_size = self.tokenizer.vocab_size().max(1);
        let mut rows = Vec::with_capacity(batch_size);
        for input in batch.rows() {
            let mut logits = vec![-12.0_f32; vocab_size];
            let offset = ((input.prompt_hash ^ input.seed) as usize) % self.script_tokens.len();
            let target =
                self.script_tokens[(input.position + offset) % self.script_tokens.len()] as usize;
            let target = target.min(vocab_size.saturating_sub(1));
            logits[target] = 12.0;

            let alternate = self.script_tokens
                [(input.position + offset + 1) % self.script_tokens.len()]
                as usize;
            let alternate = alternate.min(vocab_size.saturating_sub(1));
            logits[alternate] = 8.0;

            let jitter = ((input.last_token as usize)
                .wrapping_add(input.position)
                .wrapping_add(17))
                % vocab_size;
            logits[jitter] = 3.0;

            let (_layers, heads, head_dim) = self.kv_shape();
            let tokens = input.position.saturating_add(1);
            let marker = Tensor::zeros((heads, tokens, head_dim), DType::F32, &Device::Cpu)?;
            kv_cache.write(&input.kv_handle, 0, &marker, &marker)?;
            rows.push(logits);
        }
        Ok(DecodeLogits { rows })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Gpt2Config {
    vocab_size: usize,
    n_positions: Option<usize>,
    n_ctx: Option<usize>,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    layer_norm_epsilon: f64,
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

impl Gpt2Config {
    fn max_positions(&self) -> usize {
        self.n_positions.or(self.n_ctx).unwrap_or(1024)
    }
}

#[derive(Debug)]
struct Gpt2Model {
    config: Gpt2Config,
    head_dim: usize,
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Gpt2Block>,
    ln_f: LayerNorm,
    device: Device,
}

impl Gpt2Model {
    fn load(files: Gpt2Files, device: Device) -> Result<Self> {
        let config_bytes = fs::read(&files.config)
            .with_context(|| format!("reading GPT-2 config {}", files.config.display()))?;
        let config: Gpt2Config =
            serde_json::from_slice(&config_bytes).context("parsing GPT-2 config")?;
        if config.vocab_size == 0
            || config.n_embd == 0
            || config.n_layer == 0
            || config.n_head == 0
            || config.max_positions() == 0
        {
            return Err(anyhow!(
                "invalid GPT-2 config: vocab_size={}, n_embd={}, n_layer={}, n_head={}, max_positions={}",
                config.vocab_size,
                config.n_embd,
                config.n_layer,
                config.n_head,
                config.max_positions()
            ));
        }
        if !config.n_embd.is_multiple_of(config.n_head) {
            return Err(anyhow!(
                "invalid GPT-2 config: n_embd={} is not divisible by n_head={}",
                config.n_embd,
                config.n_head
            ));
        }
        let weights = fs::read(&files.weights)
            .with_context(|| format!("reading GPT-2 weights {}", files.weights.display()))?;
        let vb_root = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
            .context("loading GPT-2 safetensors")?;
        let vb = if vb_root.contains_tensor("transformer.wte.weight") {
            vb_root.pp("transformer")
        } else {
            vb_root
        };
        let wte = Embedding::new(
            vb.pp("wte")
                .get((config.vocab_size, config.n_embd), "weight")?,
            config.n_embd,
        );
        let wpe = Embedding::new(
            vb.pp("wpe")
                .get((config.max_positions(), config.n_embd), "weight")?,
            config.n_embd,
        );
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            blocks.push(Gpt2Block::new(&config, layer, vb.pp("h").pp(layer))?);
        }
        let ln_f = layer_norm(vb.pp("ln_f"), config.n_embd, config.layer_norm_epsilon)?;
        let head_dim = config.n_embd / config.n_head;
        Ok(Self {
            config,
            head_dim,
            wte,
            wpe,
            blocks,
            ln_f,
            device,
        })
    }

    fn prefill(
        &mut self,
        prompt_tokens: &[u32],
        handle: &KvHandle,
        kv_cache: &mut dyn KvCache,
    ) -> Result<()> {
        let cached_len = prompt_tokens.len().saturating_sub(1);
        if cached_len == 0 {
            return Ok(());
        }
        let tokens = vec![prompt_tokens[..cached_len].to_vec()];
        let positions = vec![(0..cached_len as u32).collect::<Vec<u32>>()];
        let input_ids = Tensor::new(tokens, &self.device)?;
        let position_ids = Tensor::new(positions, &self.device)?;
        let _ = self.forward_t(
            input_ids,
            position_ids,
            std::slice::from_ref(handle),
            true,
            kv_cache,
        )?;
        Ok(())
    }

    fn forward(&mut self, batch: &DecodeBatch, kv_cache: &mut dyn KvCache) -> Result<DecodeLogits> {
        if batch.rows().is_empty() {
            return Ok(DecodeLogits { rows: Vec::new() });
        }
        let token_rows: Vec<Vec<u32>> = batch
            .rows()
            .iter()
            .map(|row| vec![row.last_token])
            .collect();
        let position_rows: Vec<Vec<u32>> = batch
            .rows()
            .iter()
            .map(|row| vec![row.position as u32])
            .collect();
        let handles = batch
            .rows()
            .iter()
            .map(|row| row.kv_handle.clone())
            .collect::<Vec<_>>();
        let input_ids = Tensor::new(token_rows, &self.device)?;
        let position_ids = Tensor::new(position_rows, &self.device)?;
        let logits = self.forward_t(input_ids, position_ids, &handles, false, kv_cache)?;
        Ok(DecodeLogits {
            rows: logits.to_vec2::<f32>()?,
        })
    }

    fn forward_t(
        &mut self,
        input_ids: Tensor,
        position_ids: Tensor,
        handles: &[KvHandle],
        prefill: bool,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let token_emb = self.wte.forward(&input_ids)?;
        let pos_emb = self.wpe.forward(&position_ids)?;
        let mut xs = token_emb.broadcast_add(&pos_emb)?;
        for (layer_idx, block) in self.blocks.iter_mut().enumerate() {
            xs = block.forward(&xs, layer_idx, handles, prefill, kv_cache)?;
        }
        let xs = self.ln_f.forward(&xs)?;
        let (batch, seq_len, hidden) = xs.dims3()?;
        let logits = xs
            .reshape((batch * seq_len, hidden))?
            .matmul(&self.wte.embeddings().t()?)?
            .reshape((batch, seq_len, self.config.vocab_size))?;
        if prefill {
            Ok(logits)
        } else {
            Ok(logits.squeeze(1)?)
        }
    }
}

#[derive(Debug)]
struct Gpt2Block {
    ln_1: LayerNorm,
    attn: Gpt2Attention,
    ln_2: LayerNorm,
    mlp: Gpt2Mlp,
}

impl Gpt2Block {
    fn new(config: &Gpt2Config, layer: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: layer_norm(vb.pp("ln_1"), config.n_embd, config.layer_norm_epsilon)
                .with_context(|| format!("loading layer {layer} ln_1"))?,
            attn: Gpt2Attention::new(config, vb.pp("attn"))
                .with_context(|| format!("loading layer {layer} attention"))?,
            ln_2: layer_norm(vb.pp("ln_2"), config.n_embd, config.layer_norm_epsilon)
                .with_context(|| format!("loading layer {layer} ln_2"))?,
            mlp: Gpt2Mlp::new(config, vb.pp("mlp"))
                .with_context(|| format!("loading layer {layer} mlp"))?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        layer_idx: usize,
        handles: &[KvHandle],
        prefill: bool,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let attn = self.attn.forward(
            &self.ln_1.forward(xs)?,
            layer_idx,
            handles,
            prefill,
            kv_cache,
        )?;
        let xs = (attn + residual)?;
        let residual = &xs;
        let mlp = self.mlp.forward(&self.ln_2.forward(&xs)?)?;
        Ok((mlp + residual)?)
    }
}

#[derive(Debug)]
struct Gpt2Attention {
    c_attn: Conv1D,
    c_proj: Conv1D,
    n_head: usize,
    head_dim: usize,
    scale: f64,
}

impl Gpt2Attention {
    fn new(config: &Gpt2Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.n_embd / config.n_head;
        Ok(Self {
            c_attn: Conv1D::new(config.n_embd, 3 * config.n_embd, vb.pp("c_attn"))?,
            c_proj: Conv1D::new(config.n_embd, config.n_embd, vb.pp("c_proj"))?,
            n_head: config.n_head,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        layer_idx: usize,
        handles: &[KvHandle],
        prefill: bool,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;
        let qkv = self.c_attn.forward(xs)?;
        let query = qkv.narrow(2, 0, hidden)?;
        let key = qkv.narrow(2, hidden, hidden)?;
        let value = qkv.narrow(2, 2 * hidden, hidden)?;
        let query = query
            .reshape((batch, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = key
            .reshape((batch, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = value
            .reshape((batch, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let context = if prefill {
            self.prefill_attention(&query, &key, &value, layer_idx, handles, kv_cache)?
        } else {
            self.decode_attention(&query, &key, &value, layer_idx, handles, kv_cache)?
        };
        self.c_proj.forward(&context)
    }

    fn prefill_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        handles: &[KvHandle],
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        if handles.len() != 1 {
            return Err(anyhow!(
                "prefill expects one sequence, got {}",
                handles.len()
            ));
        }
        let (_, _, seq_len, _) = query.dims4()?;
        let key_t = key.transpose(2, 3)?;
        let mut scores = (query.matmul(&key_t)? * self.scale)?;
        let mask = causal_mask(seq_len, query.device())?;
        let mask = mask.broadcast_as(scores.shape())?;
        scores = masked_fill(&scores, &mask, f32::NEG_INFINITY)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = weights
            .matmul(value)?
            .transpose(1, 2)?
            .flatten_from(2)?
            .contiguous()?;
        let k_store = key.i(0)?.contiguous()?;
        let v_store = value.i(0)?.contiguous()?;
        kv_cache.write(&handles[0], layer_idx, &k_store, &v_store)?;
        Ok(context)
    }

    fn decode_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        handles: &[KvHandle],
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let (batch, _heads, seq_len, _head_dim) = query.dims4()?;
        if seq_len != 1 {
            return Err(anyhow!("decode expects sequence length 1, got {seq_len}"));
        }
        if batch != handles.len() {
            return Err(anyhow!(
                "decode batch/handle mismatch: batch={batch}, handles={}",
                handles.len()
            ));
        }
        let mut contexts = Vec::with_capacity(batch);
        for (row, handle) in handles.iter().enumerate() {
            let q = query.i(row)?.contiguous()?;
            let k_current = key.i(row)?.contiguous()?;
            let v_current = value.i(row)?.contiguous()?;
            let (k_total, v_total) = {
                let (past_k, past_v) = kv_cache.read(handle, layer_idx)?;
                if past_k.dim(1)? == 0 {
                    (k_current, v_current)
                } else {
                    (
                        Tensor::cat(&[&past_k, &k_current], 1)?,
                        Tensor::cat(&[&past_v, &v_current], 1)?,
                    )
                }
            };
            kv_cache.write(handle, layer_idx, &k_total, &v_total)?;
            let scores = (q.matmul(&k_total.transpose(1, 2)?)? * self.scale)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let context = weights
                .matmul(&v_total)?
                .transpose(0, 1)?
                .flatten_from(1)?
                .contiguous()?;
            contexts.push(context);
        }
        Ok(Tensor::cat(&contexts.iter().collect::<Vec<_>>(), 0)?.reshape((batch, 1, ()))?)
    }
}

#[derive(Debug)]
struct Gpt2Mlp {
    c_fc: Conv1D,
    c_proj: Conv1D,
}

impl Gpt2Mlp {
    fn new(config: &Gpt2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_fc: Conv1D::new(config.n_embd, 4 * config.n_embd, vb.pp("c_fc"))?,
            c_proj: Conv1D::new(4 * config.n_embd, config.n_embd, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(xs)?.gelu()?)
    }
}

#[derive(Debug)]
struct Conv1D {
    weight: Tensor,
    bias: Tensor,
    layout: ConvLayout,
    out_dim: usize,
}

#[derive(Debug, Copy, Clone)]
enum ConvLayout {
    InOut,
    OutIn,
}

impl Conv1D {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked("weight")?;
        let bias = vb.get(out_dim, "bias")?;
        let dims = weight.dims();
        let layout = match dims {
            [i, o] if *i == in_dim && *o == out_dim => ConvLayout::InOut,
            [o, i] if *i == in_dim && *o == out_dim => ConvLayout::OutIn,
            _ => {
                return Err(anyhow!(
                    "unexpected Conv1D weight shape {dims:?}, expected [{in_dim}, {out_dim}]"
                ));
            }
        };
        Ok(Self {
            weight,
            bias,
            layout,
            out_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let projected = match xs.dims() {
            [batch, seq_len, in_dim] => {
                let flat = xs.reshape((batch * seq_len, *in_dim))?;
                let projected = self.matmul2d(&flat)?;
                projected.reshape((*batch, *seq_len, self.out_dim))?
            }
            _ => self.matmul2d(xs)?,
        };
        Ok(projected.broadcast_add(&self.bias)?)
    }

    fn matmul2d(&self, xs: &Tensor) -> Result<Tensor> {
        match self.layout {
            ConvLayout::InOut => Ok(xs.matmul(&self.weight)?),
            ConvLayout::OutIn => Ok(xs.matmul(&self.weight.t()?)?),
        }
    }
}

fn layer_norm(vb: VarBuilder, hidden: usize, eps: f64) -> Result<LayerNorm> {
    let weight = vb.get(hidden, "weight")?;
    let bias = vb.get(hidden, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn causal_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect::<Vec<_>>();
    Ok(Tensor::from_slice(&mask, (1, 1, size, size), device)?)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    Ok(mask.where_cond(&on_true, on_false)?)
}

#[derive(Debug)]
struct FallbackTokenizer;

impl FallbackTokenizer {
    fn new() -> Self {
        Self
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|byte| byte as u32 + 1).collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|token| {
                token
                    .checked_sub(1)
                    .and_then(|value| u8::try_from(value).ok())
            })
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[derive(Debug)]
struct Gpt2Files {
    config: PathBuf,
    weights: PathBuf,
}

fn candle_device(requested: &str) -> Result<Device> {
    match requested {
        "cpu" => Ok(Device::Cpu),
        other => Err(anyhow!("unsupported device '{other}', expected cpu")),
    }
}

pub fn normalize_model_name(model: &str) -> Result<&'static str> {
    match model {
        "gpt2" | "openai-community/gpt2" => Ok("gpt2"),
        "tiny-gpt2"
        | "tiny-random-gpt2"
        | "sshleifer/tiny-gpt2"
        | "hf-internal-testing/tiny-random-gpt2" => Ok("tiny-gpt2"),
        "scripted" => Ok("scripted"),
        other => Err(anyhow!(
            "unsupported model '{other}'; expected gpt2, openai-community/gpt2, or a tiny GPT-2 alias"
        )),
    }
}

fn hf_repo(model_name: &str) -> Result<(&'static str, &'static str, &'static str)> {
    match model_name {
        "gpt2" => Ok(("openai-community", "gpt2", GPT2_REVISION)),
        "tiny-gpt2" => Ok((
            "hf-internal-testing",
            "tiny-random-gpt2",
            TINY_GPT2_REVISION,
        )),
        other => Err(anyhow!("no Hugging Face repository for model '{other}'")),
    }
}

fn download_tokenizer(model_name: &str) -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating hf-hub client")?;
    let (owner, repo, revision) = hf_repo(model_name)?;
    client
        .model(owner, repo)
        .download_file()
        .filename("tokenizer.json")
        .revision(revision)
        .send()
        .with_context(|| format!("downloading tokenizer for {owner}/{repo}@{revision}"))
}

fn download_gpt2_files(model_name: &str) -> Result<Gpt2Files> {
    let client = HFClientSync::new().context("creating hf-hub client")?;
    let (owner, repo, revision) = hf_repo(model_name)?;
    let model = client.model(owner, repo);
    let config = model
        .download_file()
        .filename("config.json")
        .revision(revision)
        .send()
        .with_context(|| format!("downloading config for {owner}/{repo}@{revision}"))?;
    let weights = model
        .download_file()
        .filename("model.safetensors")
        .revision(revision)
        .send()
        .with_context(|| format!("downloading model.safetensors for {owner}/{repo}@{revision}"))?;
    Ok(Gpt2Files { config, weights })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_supported_gpt2_names() {
        for name in ["gpt2", "openai-community/gpt2"] {
            assert_eq!(normalize_model_name(name).unwrap(), "gpt2");
        }
        for name in [
            "tiny-gpt2",
            "tiny-random-gpt2",
            "sshleifer/tiny-gpt2",
            "hf-internal-testing/tiny-random-gpt2",
        ] {
            assert_eq!(normalize_model_name(name).unwrap(), "tiny-gpt2");
        }
        assert_eq!(normalize_model_name("scripted").unwrap(), "scripted");
    }

    #[test]
    fn rejects_unknown_models_and_non_cpu_devices() {
        assert!(normalize_model_name("unknown/model").is_err());
        assert!(candle_device("cuda").is_err());
        assert!(candle_device("metal").is_err());
        assert!(candle_device("cpu").is_ok());
    }

    #[tokio::test]
    async fn scripted_tokenizer_uses_offline_fallback() {
        let tokenizer = TokenizerBundle::load("scripted").await.unwrap();
        assert_eq!(tokenizer.model_name(), "scripted");
        assert_eq!(
            tokenizer
                .decode(&tokenizer.encode("hello").unwrap())
                .unwrap(),
            "hello"
        );
    }
}
