//! The live event stream.
//!
//! Data providers (live API, replay, or simulation) all emit a uniform stream of
//! [`MatchEvent`]s. The engine consumes these to mutate match state and trigger a
//! recomputation of probabilities. Keeping a single normalized event type means a
//! new data source only has to translate *into* this shape - nothing downstream
//! changes.

use crate::fixture::{MatchId, Scoreline};
use crate::team::TeamId;
use serde::{Deserialize, Serialize};

/// A timestamped thing that happened in a match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchEvent {
    pub match_id: MatchId,
    /// Match minute the event occurred at (0 for pre-kickoff lineup events).
    pub minute: u16,
    pub kind: EventKind,
}

impl MatchEvent {
    pub fn new(match_id: MatchId, minute: u16, kind: EventKind) -> Self {
        Self {
            match_id,
            minute,
            kind,
        }
    }
}

/// What kind of event occurred. This is intentionally a closed set: every variant
/// maps to a concrete state transition or model input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    /// The match started.
    KickOff,
    /// A clock progression with no incident. Carries no data of its own - the
    /// containing [`MatchEvent::minute`] is the signal - but it lets the engine
    /// advance the live model's time (win probabilities drift as the clock runs).
    Tick,
    /// A goal was scored by `team`.
    Goal {
        team: TeamId,
        scorer: Option<String>,
    },
    /// A red card was shown - reduces that team's scoring intensity in the live model.
    RedCard { team: TeamId },
    /// A yellow card (carried for completeness / future discipline modelling).
    YellowCard { team: TeamId },
    /// Half-time.
    HalfTime,
    /// The match finished with a final `score`.
    FullTime { score: Scoreline },
    /// An **authoritative** scoreline from the data source. The engine *sets* (rather
    /// than increments) the live score, so the running tally self-heals even if a goal
    /// event was dropped, duplicated, or arrived out of order from a polled feed.
    ScoreSync { score: Scoreline },
    /// Health of the data feed. Lets the engine surface staleness instead of serving a
    /// stale snapshot that looks fresh. Not tied to a specific match.
    SourceStatus { healthy: bool },
    /// Confirmed starting line-ups - a hook for per-player strength adjustments.
    Lineup {
        home: Vec<String>,
        away: Vec<String>,
    },
}

impl EventKind {
    /// Events that change the believed score (and so should trigger a forecast
    /// recompute): goals, red cards, and authoritative score corrections.
    pub fn is_material(&self) -> bool {
        matches!(
            self,
            EventKind::Goal { .. } | EventKind::RedCard { .. } | EventKind::ScoreSync { .. }
        )
    }
}
