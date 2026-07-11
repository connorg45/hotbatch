use crate::kv_cache::{KvCache, KvHandle};
use crate::sampler::{Sampler, SamplerConfig};
use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub enum StreamItem {
    Token(u32),
    Done,
}

#[derive(Debug)]
pub struct GenerationRequest {
    pub id: Uuid,
    pub prompt_tokens: Vec<u32>,
    pub sampler_config: SamplerConfig,
    pub sender: mpsc::Sender<StreamItem>,
    pub priority: u8,
    pub created_at: Instant,
    pub prompt_hash: u64,
}

impl GenerationRequest {
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    pub fn max_new_tokens(&self) -> usize {
        self.sampler_config.max_new_tokens
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SequenceState {
    Prefilling,
    Running,
    Finished,
    Cancelled,
}

#[derive(Debug)]
pub struct Sequence {
    pub id: Uuid,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    kv_handle: KvHandle,
    pub sampler: Sampler,
    pub sender: mpsc::Sender<StreamItem>,
    state: SequenceState,
    pub created_at: Instant,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
    pub prompt_hash: u64,
}

impl Sequence {
    pub fn new(req: GenerationRequest, kv_cache: &mut dyn KvCache) -> Result<Self> {
        let kv_handle = kv_cache.allocate(req.prompt_len(), req.max_new_tokens())?;
        Ok(Self {
            id: req.id,
            prompt_tokens: req.prompt_tokens,
            generated_tokens: Vec::new(),
            kv_handle,
            sampler: Sampler::new(req.sampler_config),
            sender: req.sender,
            state: SequenceState::Prefilling,
            created_at: req.created_at,
            first_token_at: None,
            last_token_at: None,
            prompt_hash: req.prompt_hash,
        })
    }

    pub fn mark_running(&mut self) {
        if self.state == SequenceState::Prefilling {
            self.state = SequenceState::Running;
        }
    }

    pub fn append_token(&mut self, token: u32) -> TokenTiming {
        self.generated_tokens.push(token);
        let now = Instant::now();
        let first_latency = if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
            Some(now.saturating_duration_since(self.created_at))
        } else {
            None
        };
        let inter_token_latency = self
            .last_token_at
            .map(|last| now.saturating_duration_since(last));
        self.last_token_at = Some(now);

        if self.generated_tokens.len() >= self.sampler.max_new_tokens()
            || token == self.sampler.eos_token()
            || self.matches_stop_sequence()
        {
            self.state = SequenceState::Finished;
        }

        TokenTiming {
            first_latency,
            inter_token_latency,
        }
    }

    pub fn cancel(&mut self) {
        self.state = SequenceState::Cancelled;
    }

    pub fn is_finished(&self) -> bool {
        self.state == SequenceState::Finished
    }

    pub fn is_cancelled(&self) -> bool {
        self.state == SequenceState::Cancelled
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.state,
            SequenceState::Finished | SequenceState::Cancelled
        )
    }

    pub fn state(&self) -> SequenceState {
        self.state
    }

    pub fn kv_handle(&self) -> KvHandle {
        self.kv_handle.clone()
    }

    pub fn last_token(&self) -> u32 {
        self.generated_tokens
            .last()
            .copied()
            .or_else(|| self.prompt_tokens.last().copied())
            .unwrap_or(0)
    }

    pub fn decode_position(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_sub(1)
            .saturating_add(self.generated_tokens.len())
    }

    pub fn seed(&self) -> u64 {
        self.sampler.config().seed
    }

    fn matches_stop_sequence(&self) -> bool {
        self.sampler.stop_sequences().iter().any(|stop| {
            !stop.is_empty()
                && self.generated_tokens.len() >= stop.len()
                && self.generated_tokens[self.generated_tokens.len() - stop.len()..] == stop[..]
        })
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TokenTiming {
    pub first_latency: Option<Duration>,
    pub inter_token_latency: Option<Duration>,
}
