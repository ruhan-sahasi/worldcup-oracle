//! `oracle-server` - standalone binary that boots the engine and serves the API.
//!
//! Picks the live football-data.org feed when `FOOTBALL_DATA_API_KEY` is set,
//! otherwise runs the deterministic simulation. Listen address comes from `$PORT` (the PaaS
//! convention), else `$ORACLE_ADDR`, else `0.0.0.0:8080`.

use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let addr = oracle_api::resolve_listen_addr(
        std::env::var("PORT").ok(),
        std::env::var("ORACLE_ADDR").ok(),
    )?;

    let cancel = CancellationToken::new();
    let (engine, engine_join) = oracle_engine::spawn(
        oracle_engine::presets::auto(),
        oracle_engine::EngineConfig::default(),
        cancel.clone(),
    )
    .await?;

    // The on-demand explorer backs /explore and the /api/* queries; fit it in the background so
    // the server (live dashboard, engine endpoints, health) is responsive immediately.
    let explorer = oracle_api::spawn_explorer();

    let shutdown_cancel = cancel.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
        shutdown_cancel.cancel();
    };

    oracle_api::serve(engine, explorer, addr, shutdown).await?;

    cancel.cancel();
    let _ = engine_join.await;
    Ok(())
}
