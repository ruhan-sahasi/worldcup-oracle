//! The immutable, serializable view the engine publishes on every update.
//!
//! A [`Snapshot`] is a complete, self-contained picture of the engine's beliefs at
//! one instant: every match's current probabilities, the tournament forecast, and
//! the live ratings. The engine swaps a fresh `Arc<Snapshot>` into an [`arc_swap`]
//! cell (lock-free reads for the REST layer) and broadcasts it to WebSocket
//! subscribers. Because it's a value type, consumers never touch engine internals.

use chrono::{DateTime, Utc};
use oracle_domain::{
    MatchId, MatchStatus, Probabilities, ScoreGrid, Scoreline, Stage, TeamId, TournamentForecast,
};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// The engine's current prediction for a single match.
#[derive(Debug, Clone, Serialize)]
pub struct MatchPrediction {
    pub match_id: MatchId,
    pub home: TeamId,
    pub away: TeamId,
    pub home_name: String,
    pub away_name: String,
    pub stage: Stage,
    pub status: MatchStatus,
    pub score: Scoreline,
    pub minute: u16,
    /// Current win/draw/win - live (Bayesian) if in play, else the pre-match ensemble.
    pub probabilities: Probabilities,
    /// The exact-score distribution (omitted once the match is finished).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_grid: Option<ScoreGrid>,
    /// Modal scoreline `(home, away, probability)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub most_likely_score: Option<(u8, u8, f64)>,
}

/// A team's current Elo rating (for display).
#[derive(Debug, Clone, Serialize)]
pub struct RatingEntry {
    pub team: TeamId,
    pub name: String,
    pub rating: f64,
}

/// A finished match the model got surprised by: the pre-match favourite failed to win. Ranked by
/// `shock` (how little probability the model gave the outcome that actually happened).
#[derive(Debug, Clone, Serialize)]
pub struct Upset {
    pub match_id: MatchId,
    pub home_name: String,
    pub away_name: String,
    pub stage: Stage,
    pub score: Scoreline,
    /// The pre-match favourite (the side the model gave the higher win probability).
    pub favorite_name: String,
    /// The probability the model gave that favourite pre-match.
    pub favorite_prob: f64,
    /// Shock magnitude in `[0, 1]`: `1 - P(actual outcome)` from the pre-match forecast.
    pub shock: f64,
}

/// The engine's in-tournament self-recalibration state, surfaced for observability. As results
/// arrive the live features adjust these away from their neutral values, so a viewer can watch the
/// model recalibrate itself against the tournament.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AdaptiveState {
    /// Finished matches folded into the live calibration so far.
    pub results_seen: usize,
    /// Temperature applied to the remaining pre-match forecasts (1.0 = no correction; >1 sharpens).
    pub calibration_temperature: f64,
    /// Aggregate gain on the context effects, recalibrated against results (1.0 = the reasoned prior).
    pub context_gain: f64,
}

/// A complete published view of engine state.
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub generated_at: DateTime<Utc>,
    pub tournament: String,
    pub provider: String,
    /// Whether the data feed is currently healthy. `false` ⇒ the figures may be stale.
    pub source_healthy: bool,
    /// When the engine last processed an event from the feed (staleness signal).
    pub last_update: DateTime<Utc>,
    pub matches: Vec<MatchPrediction>,
    pub forecast: TournamentForecast,
    pub ratings: Vec<RatingEntry>,
    /// The engine's live self-recalibration state (temperature, context gain, results seen).
    pub adaptive: AdaptiveState,
    /// Finished matches the model was most surprised by (favourite failed to win), ranked by shock.
    pub shocks: Vec<Upset>,
}

impl Snapshot {
    /// Look up the prediction for a specific match.
    pub fn match_prediction(&self, id: MatchId) -> Option<&MatchPrediction> {
        self.matches.iter().find(|m| m.match_id == id)
    }

    /// Matches currently in play.
    pub fn live_matches(&self) -> impl Iterator<Item = &MatchPrediction> {
        self.matches
            .iter()
            .filter(|m| matches!(m.status, MatchStatus::Live { .. }))
    }
}

/// Lightweight runtime counters, exported on `/metrics`.
#[derive(Debug, Default)]
pub struct Metrics {
    pub events_processed: AtomicU64,
    pub goals_seen: AtomicU64,
    pub forecasts_computed: AtomicU64,
    pub snapshots_published: AtomicU64,
}

impl Metrics {
    pub fn incr(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Render the counters in Prometheus text exposition format.
    pub fn prometheus(&self, subscribers: usize) -> String {
        let g = |name: &str, help: &str, val: u64| {
            format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {val}\n")
        };
        let mut out = String::new();
        out.push_str(&g(
            "oracle_events_processed_total",
            "Match events consumed by the engine.",
            self.events_processed.load(Ordering::Relaxed),
        ));
        out.push_str(&g(
            "oracle_goals_seen_total",
            "Goals observed across all matches.",
            self.goals_seen.load(Ordering::Relaxed),
        ));
        out.push_str(&g(
            "oracle_forecasts_computed_total",
            "Monte-Carlo tournament forecasts computed.",
            self.forecasts_computed.load(Ordering::Relaxed),
        ));
        out.push_str(&g(
            "oracle_snapshots_published_total",
            "Snapshots published to subscribers.",
            self.snapshots_published.load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP oracle_subscribers Current WebSocket subscribers.\n\
             # TYPE oracle_subscribers gauge\n",
        );
        out.push_str(&format!("oracle_subscribers {subscribers}\n"));
        out
    }
}
