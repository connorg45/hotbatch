# Hotbatch Benchmark Results

Hardware: Apple M4, 16.0 GiB RAM

Model: `gpt2`

Prompt: `The capital of France is a useful benchmark prompt because`

Max tokens: `64`. Warm-up request excluded. Each measured client uses `stream=true` and records first-token and inter-token latency from SSE frame arrival times.

| mode | concurrency | agg tok/s | first p50 ms | first p95 ms | inter p50 ms | inter p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 1 | 81.39 | 34.23 | 34.23 | 11.73 | 13.23 |
| Naive | 2 | 80.71 | 841.55 | 841.55 | 11.81 | 12.62 |
| Naive | 4 | 82.90 | 1582.85 | 2356.45 | 11.66 | 12.08 |
| Naive | 8 | 82.79 | 3130.01 | 5449.73 | 11.68 | 12.13 |
| Naive | 16 | 82.53 | 6270.97 | 10904.36 | 11.68 | 12.16 |
| Naive | 32 | 81.11 | 12369.95 | 22822.21 | 11.68 | 12.82 |
| Continuous | 1 | 82.12 | 34.88 | 34.88 | 11.65 | 12.05 |
| Continuous | 2 | 131.29 | 68.92 | 68.92 | 14.41 | 15.24 |
| Continuous | 4 | 215.51 | 112.96 | 112.98 | 17.11 | 17.86 |
| Continuous | 8 | 295.70 | 209.21 | 209.28 | 24.47 | 25.48 |
| Continuous | 16 | 361.41 | 417.58 | 417.78 | 37.56 | 41.62 |
| Continuous | 32 | 420.28 | 864.59 | 864.88 | 63.26 | 70.28 |
