use anyhow::Result;
use clap::Parser;
use hotbatch_core::{
    DecodeBatch, DecodeInput, KvCache, ModelOptions, Sampler, SamplerConfig, SlabKvCache,
    SmallTransformer,
};

#[derive(Debug, Parser)]
#[command(name = "candle-gpt2-reference")]
struct Args {
    #[arg(long, default_value = "gpt2")]
    model: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let text = generate(args).await?;
    print!("{text}");
    Ok(())
}

async fn generate(args: Args) -> Result<String> {
    let mut model = SmallTransformer::load(ModelOptions {
        model: args.model,
        ..ModelOptions::default()
    })
    .await?;
    let tokenizer = model.tokenizer();
    let prompt_tokens = tokenizer.encode(&args.prompt)?;
    let (num_layers, n_heads, head_dim) = model.kv_shape();
    let max_seq_len = prompt_tokens.len().saturating_add(args.max_tokens).max(1);
    let mut kv_cache = SlabKvCache::new(1, max_seq_len, num_layers, n_heads, head_dim);
    let handle = kv_cache.allocate(prompt_tokens.len(), args.max_tokens)?;
    model.prefill(&prompt_tokens, &handle, &mut kv_cache)?;
    let prompt_hash = hash_tokens(&prompt_tokens);
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        stop_sequences: Vec::new(),
        max_new_tokens: args.max_tokens,
        eos_token: tokenizer.eos_token(),
        seed: args.seed,
    });
    let mut generated = Vec::new();
    for position in 0..args.max_tokens {
        let last_token = generated
            .last()
            .copied()
            .or_else(|| prompt_tokens.last().copied())
            .unwrap_or(0);
        let batch = DecodeBatch::new(vec![DecodeInput {
            seq_id: uuid::Uuid::new_v4(),
            kv_handle: handle.clone(),
            last_token,
            position: prompt_tokens
                .len()
                .saturating_sub(1)
                .saturating_add(position),
            prompt_hash,
            seed: args.seed,
        }]);
        let logits = model.forward(&batch, &mut kv_cache)?;
        let token = sampler.sample(logits.row(0)?);
        generated.push(token);
        if token == tokenizer.eos_token() {
            break;
        }
    }
    tokenizer.decode(&generated)
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        hash ^= *token as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
