mod common;

use anyhow::Result;
use hotbatch_core::TokenizerBundle;
use hotbatch_server::ServeMode;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    model: String,
    revision: String,
    weights_sha256: String,
    generation: GoldenGeneration,
    cases: Vec<GoldenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenGeneration {
    do_sample: bool,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    prompt: String,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
    expected_token_ids: Vec<u32>,
    expected_text: String,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn greedy_gpt2_matches_hugging_face_golden_outputs() -> Result<()> {
    let fixture: GoldenFixture = serde_json::from_str(include_str!("fixtures/gpt2-greedy.json"))?;
    assert_eq!(fixture.model, "openai-community/gpt2");
    assert_eq!(fixture.revision, "607a30d783dfa663caf39e06633721c8d4cfcd7e");
    assert_eq!(
        fixture.weights_sha256,
        "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
    );
    assert!(!fixture.generation.do_sample);
    assert_eq!(fixture.generation.temperature, 0.0);

    let tokenizer = TokenizerBundle::load(&fixture.model).await?;
    let max_tokens = fixture
        .cases
        .iter()
        .map(|case| case.max_new_tokens)
        .max()
        .unwrap_or(1);
    let server =
        common::spawn_with_model(ServeMode::Continuous, max_tokens, fixture.model.clone()).await?;
    let base_url = format!("http://{}", server.addr);

    for case in fixture.cases {
        assert_eq!(tokenizer.encode(&case.prompt)?, case.prompt_token_ids);
        assert_eq!(
            tokenizer.encode(&case.expected_text)?,
            case.expected_token_ids
        );
        let served =
            common::non_stream_completion(&base_url, &case.prompt, case.max_new_tokens, 42).await?;
        assert_eq!(served, case.expected_text, "prompt: {}", case.prompt);
    }

    server.stop().await
}
