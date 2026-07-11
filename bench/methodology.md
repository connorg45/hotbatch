# Benchmark Methodology

`cargo run --release --bin hotbatch-bench` starts the naive server and the continuous server on ephemeral loopback ports, warms each mode with one streaming request, then sweeps concurrency `[1, 2, 4, 8, 16, 32]` on the same `openai-community/gpt2` Candle model and the same CPU. Model assets are pinned to Hugging Face revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

The server does not inject synthetic prefill or decode delays into GPT-2 execution. Reported latency and throughput therefore measure the model, scheduler, HTTP stack, and load generator as they run on the benchmark host.

Every measured request uses:

- prompt: `The capital of France is a useful benchmark prompt because`
- `max_tokens`: 64
- `temperature`: 0
- `stream`: true

The load generator sends the selected model identifier in every request and counts every non-terminal SSE choice chunk as one generated token, including tokens that decode to an empty text fragment. It records first-token latency from request start to first token frame and inter-token latency between subsequent token frames. HTTP connections have a 10-second connection timeout and each streaming request has a five-minute overall timeout. A stream that closes without `[DONE]` fails the run.

One warm-up request per mode is excluded. The full benchmark writes the markdown table and PNG plot to `bench/results.md` and `bench/results.png`. The CI smoke run uses the pinned tiny GPT-2 model, concurrency `[1, 2]`, eight tokens per request, and writes disposable output under `target/bench-smoke`.
