//! Ready-made [`EngineDeps`] for the common composition roots, so the server binary
//! and the CLI wire the engine up identically.

use crate::EngineDeps;
use oracle_domain::Tournament;
use oracle_ingest::{data, FootballDataProvider, ReplayProvider, SimProvider};
use std::sync::Arc;
use std::time::Duration;

/// Deterministic simulation over the embedded 2026 World Cup, with the baseline
/// goal model and strength-seeded Elo. No network, no keys.
pub fn simulated() -> EngineDeps {
    EngineDeps::new(Arc::new(SimProvider::new()))
        .with_model(data::fit_baseline_model(7))
        .with_elo_seeds(data::team_strengths())
}

/// Like [`simulated`] but with a custom match clock speed.
pub fn simulated_with_speed(minute_delay: Duration) -> EngineDeps {
    EngineDeps::new(Arc::new(SimProvider::new().with_minute_delay(minute_delay)))
        .with_model(data::fit_baseline_model(7))
        .with_elo_seeds(data::team_strengths())
}

/// Replay a completed tournament event-by-event.
pub fn replay(tournament: Tournament) -> EngineDeps {
    EngineDeps::new(Arc::new(ReplayProvider::new(tournament)))
        .with_model(data::fit_baseline_model(7))
        .with_elo_seeds(data::team_strengths())
}

/// Use the live football-data.org feed when `FOOTBALL_DATA_API_KEY` is set; fall
/// back to [`simulated`] otherwise. The fallback means a demo never hard-fails on a
/// missing key.
pub fn auto() -> EngineDeps {
    match FootballDataProvider::from_env() {
        Ok(provider) => {
            tracing::info!("FOOTBALL_DATA_API_KEY found — using the live feed");
            EngineDeps::new(Arc::new(provider))
        }
        Err(_) => {
            tracing::info!("no API key — using the deterministic simulation feed");
            simulated()
        }
    }
}
