FROM rust:1.89-slim-bookworm AS build

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake clang g++ pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
COPY src ./src
RUN cargo build --release -p hotbatch-server --bin hotbatch-server

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    RUST_LOG=info
RUN mkdir -p /root/.cache/huggingface
EXPOSE 8080
COPY --from=build /app/target/release/hotbatch-server /usr/local/bin/hotbatch-server
HEALTHCHECK --interval=15s --timeout=5s --start-period=10m --retries=3 \
    CMD curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8080/healthz >/dev/null \
        && curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8080/v1/models >/dev/null \
        || exit 1
STOPSIGNAL SIGINT
CMD ["hotbatch-server", "serve", "--host", "0.0.0.0", "--port", "8080", "--mode", "continuous"]
