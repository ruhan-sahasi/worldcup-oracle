# ---- Build stage ----
FROM rust:1-bookworm AS builder
WORKDIR /app

# Cache dependencies first: copy manifests, then sources.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --release --bin oracle-server

# ---- Runtime stage ----
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/oracle-server /usr/local/bin/oracle-server

# Run as a non-root user.
RUN useradd --system --uid 10001 oracle
USER oracle

ENV ORACLE_ADDR=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["oracle-server"]
