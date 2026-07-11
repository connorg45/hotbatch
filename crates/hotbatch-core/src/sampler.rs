use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub stop_sequences: Vec<Vec<u32>>,
    pub max_new_tokens: usize,
    pub eos_token: u32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            stop_sequences: Vec::new(),
            max_new_tokens: 16,
            eos_token: 50_256,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng_state = config.seed ^ 0x9e37_79b9_7f4a_7c15;
        Self { config, rng_state }
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    pub fn max_new_tokens(&self) -> usize {
        self.config.max_new_tokens
    }

    pub fn eos_token(&self) -> u32 {
        self.config.eos_token
    }

    pub fn stop_sequences(&self) -> &[Vec<u32>] {
        &self.config.stop_sequences
    }

    pub fn sample(&mut self, logits: crate::model::LogitRow<'_>) -> u32 {
        if self.config.temperature <= f32::EPSILON {
            return logits.argmax();
        }

        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        if let Some(top_k) = self.config.top_k {
            candidates.truncate(top_k.max(1));
        }

        let temperature = self.config.temperature.max(0.001);
        let max_logit = candidates.first().map(|(_, value)| *value).unwrap_or(0.0);
        let mut weighted = Vec::with_capacity(candidates.len());
        let mut total = 0.0_f64;
        for (token, logit) in candidates {
            let weight = (((logit - max_logit) / temperature) as f64).exp();
            total += weight;
            weighted.push((token, weight));
        }

        weighted.sort_by(|a, b| b.1.total_cmp(&a.1));
        if self.config.top_p < 0.999 {
            let mut kept = Vec::new();
            let mut cumulative = 0.0_f64;
            let threshold = self.config.top_p.clamp(0.001, 1.0) as f64;
            for (token, weight) in weighted {
                let probability = if total > 0.0 { weight / total } else { 0.0 };
                cumulative += probability;
                kept.push((token, weight));
                if cumulative >= threshold {
                    break;
                }
            }
            weighted = kept;
            total = weighted.iter().map(|(_, weight)| *weight).sum();
        }

        if weighted.is_empty() || total <= 0.0 {
            return logits.argmax();
        }

        let mut target = self.next_f64() * total;
        for (token, weight) in weighted {
            if target <= weight {
                return token as u32;
            }
            target -= weight;
        }
        logits.argmax()
    }

    fn next_f64(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let value = self.rng_state >> 11;
        (value as f64) / ((1_u64 << 53) as f64)
    }
}
