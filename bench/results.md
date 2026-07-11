# Hotbatch Benchmark Results

Hardware: Apple M4, 16.0 GiB RAM

Prompt: `The capital of France is a useful benchmark prompt because`

Max tokens: `64`. Warm-up request excluded. Each measured client uses `stream=true` and records first-token and inter-token latency from SSE frame arrival times.

| mode | concurrency | agg tok/s | first p50 ms | first p95 ms | inter p50 ms | inter p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 1 | 71.74 | 474.35 | 474.35 | 0.01 | 13.59 |
| Naive | 2 | 72.39 | 1347.68 | 1347.68 | 0.00 | 13.64 |
| Naive | 4 | 71.54 | 2222.87 | 3160.15 | 0.00 | 13.54 |
| Naive | 8 | 72.11 | 4028.14 | 6682.74 | 0.00 | 13.53 |
| Naive | 16 | 72.42 | 7507.86 | 12839.88 | 0.00 | 13.55 |
| Naive | 32 | 72.51 | 14568.56 | 26057.93 | 0.00 | 13.55 |
| Continuous | 1 | 72.67 | 462.82 | 462.82 | 0.00 | 13.58 |
| Continuous | 2 | 115.73 | 79.03 | 79.03 | 16.48 | 34.86 |
| Continuous | 4 | 190.74 | 114.73 | 132.83 | 19.55 | 20.37 |
| Continuous | 8 | 271.46 | 214.42 | 239.70 | 26.58 | 27.93 |
| Continuous | 16 | 336.97 | 412.43 | 412.55 | 40.82 | 46.27 |
| Continuous | 32 | 400.49 | 827.55 | 827.97 | 68.39 | 73.46 |
