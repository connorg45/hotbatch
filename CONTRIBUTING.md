# Contributing

## Development setup

Install Rust 1.89 or newer and a C/C++ build toolchain, then clone the repository. Model-backed tests download pinned GPT-2 assets from Hugging Face into the local cache.

Before opening a pull request, run:

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --release --all
cargo run --locked --release --bin hotbatch-bench -- --smoke --model tiny-gpt2 --output-dir target/bench-smoke
```

Keep changes focused, add tests for behavior changes, and update documentation when commands or externally visible behavior change. Do not commit model weights, local caches, credentials, benchmark scratch output, or generated build artifacts.

For vulnerability reports, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
