use crate::kv_cache::{KvCache, KvHandle};
use crate::sampler::{Sampler, SamplerConfig};
use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, mpsc::error::TrySendError};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamItem {
    Token(u32),
    Finished(FinishReason),
    Error(String),
}

/// Atomically reserves room for all newly visible tokens and an optional
/// terminal event. A slow or disconnected consumer never receives a partial
/// stop-safe update.
pub fn try_send_tokens(
    sender: &mpsc::Sender<StreamItem>,
    tokens: &[u32],
    finish_reason: Option<FinishReason>,
) -> bool {
    let message_count = tokens.len() + usize::from(finish_reason.is_some());
    if message_count == 0 {
        return true;
    }
    let mut permits = match sender.try_reserve_many(message_count) {
        Ok(permits) => permits,
        Err(TrySendError::Full(())) | Err(TrySendError::Closed(())) => return false,
    };

    for token in tokens {
        permits
            .next()
            .expect("a channel permit was reserved for each token")
            .send(StreamItem::Token(*token));
    }
    if let Some(reason) = finish_reason {
        permits
            .next()
            .expect("terminal channel permit was reserved")
            .send(StreamItem::Finished(reason));
    }
    true
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
    pub response_done: CancellationToken,
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
    Failed,
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
    finish_reason: Option<FinishReason>,
    emitted_tokens: usize,
    terminal_trim_tokens: usize,
    pub created_at: Instant,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
    pub prompt_hash: u64,
    response_done: CancellationToken,
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
            finish_reason: None,
            emitted_tokens: 0,
            terminal_trim_tokens: 0,
            created_at: req.created_at,
            first_token_at: None,
            last_token_at: None,
            prompt_hash: req.prompt_hash,
            response_done: req.response_done,
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

        if self.finish_reason.is_none() {
            let matched_stop_len = self.matched_stop_len();
            let reason = if token == self.sampler.eos_token() || matched_stop_len.is_some() {
                self.terminal_trim_tokens = matched_stop_len.unwrap_or(1);
                Some(FinishReason::Stop)
            } else if self.generated_tokens.len() >= self.sampler.max_new_tokens() {
                Some(FinishReason::Length)
            } else {
                None
            };
            if let Some(reason) = reason {
                self.state = SequenceState::Finished;
                self.finish_reason = Some(reason);
            }
        }

        TokenTiming {
            first_latency,
            inter_token_latency,
        }
    }

    pub fn cancel(&mut self) {
        if !self.is_done() {
            self.state = SequenceState::Cancelled;
        }
    }

    pub fn fail(&mut self) {
        if !self.is_done() {
            self.state = SequenceState::Failed;
        }
    }

    pub fn is_finished(&self) -> bool {
        self.state == SequenceState::Finished
    }

    pub fn is_cancelled(&self) -> bool {
        self.state == SequenceState::Cancelled
    }

    pub fn is_failed(&self) -> bool {
        self.state == SequenceState::Failed
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.state,
            SequenceState::Finished | SequenceState::Cancelled | SequenceState::Failed
        )
    }

    pub fn state(&self) -> SequenceState {
        self.state
    }

    pub fn response_is_done(&self) -> bool {
        self.response_done.is_cancelled()
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    pub fn take_emittable_tokens(&mut self) -> Vec<u32> {
        let safe_end = match self.finish_reason {
            Some(FinishReason::Stop) => self
                .generated_tokens
                .len()
                .saturating_sub(self.terminal_trim_tokens),
            Some(FinishReason::Length) => self.generated_tokens.len(),
            None => self
                .generated_tokens
                .len()
                .saturating_sub(self.pending_stop_prefix_len()),
        };
        let tokens = self.generated_tokens[self.emitted_tokens..safe_end].to_vec();
        self.emitted_tokens = safe_end;
        tokens
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

    fn matched_stop_len(&self) -> Option<usize> {
        self.sampler
            .stop_sequences()
            .iter()
            .filter(|stop| {
                !stop.is_empty()
                    && self.generated_tokens.len() >= stop.len()
                    && self.generated_tokens[self.generated_tokens.len() - stop.len()..] == stop[..]
            })
            .map(Vec::len)
            .max()
    }

    fn pending_stop_prefix_len(&self) -> usize {
        self.sampler
            .stop_sequences()
            .iter()
            .filter(|stop| !stop.is_empty())
            .map(|stop| {
                let max_prefix = stop.len().min(self.generated_tokens.len());
                (1..=max_prefix)
                    .rev()
                    .find(|prefix_len| {
                        self.generated_tokens[self.generated_tokens.len() - prefix_len..]
                            == stop[..*prefix_len]
                    })
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TokenTiming {
    pub first_latency: Option<Duration>,
    pub inter_token_latency: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::SlabKvCache;

    fn request(config: SamplerConfig) -> (GenerationRequest, mpsc::Receiver<StreamItem>) {
        let (sender, receiver) = mpsc::channel(8);
        (
            GenerationRequest {
                id: Uuid::new_v4(),
                prompt_tokens: vec![11, 12],
                sampler_config: config,
                sender,
                priority: 0,
                created_at: Instant::now(),
                prompt_hash: 7,
                response_done: CancellationToken::new(),
            },
            receiver,
        )
    }

    fn sequence(config: SamplerConfig) -> (Sequence, mpsc::Receiver<StreamItem>) {
        let (request, receiver) = request(config);
        let mut cache = SlabKvCache::new(1, 32, 1, 1, 1);
        let sequence = Sequence::new(request, &mut cache).expect("sequence should allocate");
        (sequence, receiver)
    }

    #[test]
    fn reports_length_when_token_limit_is_reached() {
        let config = SamplerConfig {
            max_new_tokens: 2,
            eos_token: 99,
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(1);
        assert_eq!(sequence.finish_reason(), None);
        sequence.append_token(2);

        assert!(sequence.is_finished());
        assert_eq!(sequence.finish_reason(), Some(FinishReason::Length));
    }

    #[test]
    fn eos_takes_precedence_over_length() {
        let config = SamplerConfig {
            max_new_tokens: 1,
            eos_token: 7,
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(7);

        assert_eq!(sequence.finish_reason(), Some(FinishReason::Stop));
    }

    #[test]
    fn stop_sequence_takes_precedence_over_length() {
        let config = SamplerConfig {
            max_new_tokens: 2,
            eos_token: 99,
            stop_sequences: vec![vec![4, 5]],
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(4);
        sequence.append_token(5);

        assert_eq!(sequence.finish_reason(), Some(FinishReason::Stop));
    }

    #[test]
    fn matched_stop_tokens_are_not_emitted() {
        let config = SamplerConfig {
            max_new_tokens: 4,
            eos_token: 99,
            stop_sequences: vec![vec![4, 5]],
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(4);
        assert!(sequence.take_emittable_tokens().is_empty());
        sequence.append_token(5);

        assert_eq!(sequence.finish_reason(), Some(FinishReason::Stop));
        assert!(sequence.take_emittable_tokens().is_empty());
    }

    #[test]
    fn mismatched_stop_prefix_is_released() {
        let config = SamplerConfig {
            max_new_tokens: 4,
            eos_token: 99,
            stop_sequences: vec![vec![4, 5]],
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(4);
        assert!(sequence.take_emittable_tokens().is_empty());
        sequence.append_token(6);

        assert_eq!(sequence.take_emittable_tokens(), vec![4, 6]);
    }

    #[test]
    fn length_finish_releases_an_incomplete_stop_prefix() {
        let config = SamplerConfig {
            max_new_tokens: 1,
            eos_token: 99,
            stop_sequences: vec![vec![4, 5]],
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(4);

        assert_eq!(sequence.finish_reason(), Some(FinishReason::Length));
        assert_eq!(sequence.take_emittable_tokens(), vec![4]);
    }

    #[test]
    fn eos_token_is_not_emitted() {
        let config = SamplerConfig {
            max_new_tokens: 2,
            eos_token: 7,
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(7);

        assert_eq!(sequence.finish_reason(), Some(FinishReason::Stop));
        assert!(sequence.take_emittable_tokens().is_empty());
    }

    #[test]
    fn terminal_reason_is_preserved() {
        let config = SamplerConfig {
            max_new_tokens: 1,
            eos_token: 99,
            ..SamplerConfig::default()
        };
        let (mut sequence, _receiver) = sequence(config);

        sequence.append_token(3);
        sequence.cancel();
        sequence.append_token(99);

        assert_eq!(sequence.state(), SequenceState::Finished);
        assert_eq!(sequence.finish_reason(), Some(FinishReason::Length));
    }

    #[test]
    fn cancellation_has_no_finish_reason() {
        let (mut sequence, _receiver) = sequence(SamplerConfig::default());

        sequence.cancel();

        assert!(sequence.is_cancelled());
        assert_eq!(sequence.finish_reason(), None);
    }
}
