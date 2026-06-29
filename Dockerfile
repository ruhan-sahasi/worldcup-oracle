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
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/oracle-server /usr/local/bin/oracle-server

# Run as a non-root user.
RUN useradd --system --uid 10001 oracle
USER oracle

ENV ORACLE_ADDR=0.0.0.0:8080
EXPOSE 8080
# `/health` is up as soon as the server binds; the model explorer fits in the background and its
# `/api/*` endpoints return 503 until ready, so liveness never waits on the fit.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["oracle-server"]
