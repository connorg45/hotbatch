mod common;

use anyhow::Result;
use hotbatch_server::ServeMode;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn temperature_zero_is_byte_deterministic() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 24, 1_000).await?;
    let base_url = format!("http://{}", server.addr);

    let mut outputs = Vec::new();
    for _ in 0..10 {
        outputs.push(
            common::non_stream_completion(&base_url, "The capital of France is", 16, 42).await?,
        );
    }
    for output in outputs.iter().skip(1) {
        assert_eq!(output, &outputs[0]);
    }

    server.stop().await
}
