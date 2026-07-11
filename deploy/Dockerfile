FROM rust:1.95-slim-bookworm AS build

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
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV RUST_LOG=info
EXPOSE 8080
COPY --from=build /app/target/release/hotbatch-server /usr/local/bin/hotbatch-server
CMD ["hotbatch-server", "serve", "--host", "0.0.0.0", "--port", "8080", "--mode", "continuous"]
