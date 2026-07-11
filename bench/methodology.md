# Benchmark Methodology

`cargo run --release --bin hotbatch-bench` starts the naive server and the continuous server on ephemeral loopback ports, warms each mode with one streaming request, then sweeps concurrency `[1, 2, 4, 8, 16, 32]` on the same `openai-community/gpt2` Candle model and the same CPU.

Every measured request uses:

- prompt: `The capital of France is a useful benchmark prompt because`
- `max_tokens`: 64
- `temperature`: 0
- `stream`: true

The load generator counts SSE token frames, records first-token latency from request start to first token frame, and records inter-token latency between subsequent token frames. One warm-up request per mode is excluded. The markdown table and PNG plot are written to `bench/results.md` and `bench/results.png`.
