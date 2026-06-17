//! The `DataProvider` abstraction - the seam that makes the engine source-agnostic.

use crate::error::Result;
use async_trait::async_trait;
use oracle_domain::{MatchEvent, Tournament};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

/// A source of tournament data and a live event stream.
///
/// Everything downstream of this trait - the engine, the API, the CLI - is written
/// against `DataProvider` and never against a concrete source. Swapping a simulated
/// feed for the live football-data.org API is a one-line change at the composition
/// root. This is dependency inversion applied at the data boundary.
///
/// Implementors:
/// - [`crate::SimProvider`] - a deterministic synthetic feed (no network, no keys)
/// - [`crate::ReplayProvider`] - replays a completed tournament event-by-event
/// - [`crate::FootballDataProvider`] - the live football-data.org adapter
#[async_trait]
pub trait DataProvider: Send + Sync {
    /// A short, human-readable name (shown in logs and the CLI).
    fn name(&self) -> &'static str;

    /// Load the static tournament structure: teams, groups, and the fixture list
    /// (including any already-played results).
    async fn load_tournament(&self) -> Result<Tournament>;

    /// Drive the live event stream into `tx` until the source is exhausted or
    /// `cancel` is triggered. Returning `Ok(())` means the feed ended cleanly.
    async fn run(&self, tx: Sender<MatchEvent>, cancel: CancellationToken) -> Result<()>;
}
