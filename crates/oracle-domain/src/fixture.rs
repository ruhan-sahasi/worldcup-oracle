//! Matches ("fixtures"): identifiers, scorelines, stages, and live status.

use crate::probability::Outcome;
use crate::team::TeamId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable identifier for a single match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MatchId(pub u32);

impl fmt::Display for MatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M{}", self.0)
    }
}

/// The current goal tally of a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Scoreline {
    pub home: u8,
    pub away: u8,
}

impl Scoreline {
    pub const fn new(home: u8, away: u8) -> Self {
        Self { home, away }
    }

    /// The win/draw/win result implied by this scoreline.
    pub fn outcome(&self) -> Outcome {
        match self.home.cmp(&self.away) {
            std::cmp::Ordering::Greater => Outcome::HomeWin,
            std::cmp::Ordering::Equal => Outcome::Draw,
            std::cmp::Ordering::Less => Outcome::AwayWin,
        }
    }

    /// Total goals in the match (for over/under markets).
    pub fn total(&self) -> u8 {
        self.home + self.away
    }
}

impl fmt::Display for Scoreline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.home, self.away)
    }
}

/// Where in the tournament a match sits. The 2026 format has 12 groups (A–L)
/// feeding a 32-team knockout bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    Group(char),
    RoundOf32,
    RoundOf16,
    QuarterFinal,
    SemiFinal,
    ThirdPlace,
    Final,
}

impl Stage {
    /// Knockout matches cannot end in a draw (penalties decide a winner).
    pub fn is_knockout(&self) -> bool {
        !matches!(self, Stage::Group(_))
    }

    /// Ordinal used to order stages chronologically.
    pub fn order(&self) -> u8 {
        match self {
            Stage::Group(_) => 0,
            Stage::RoundOf32 => 1,
            Stage::RoundOf16 => 2,
            Stage::QuarterFinal => 3,
            Stage::SemiFinal => 4,
            Stage::ThirdPlace => 5,
            Stage::Final => 6,
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage::Group(g) => write!(f, "Group {g}"),
            Stage::RoundOf32 => f.write_str("Round of 32"),
            Stage::RoundOf16 => f.write_str("Round of 16"),
            Stage::QuarterFinal => f.write_str("Quarter-final"),
            Stage::SemiFinal => f.write_str("Semi-final"),
            Stage::ThirdPlace => f.write_str("Third-place play-off"),
            Stage::Final => f.write_str("Final"),
        }
    }
}

/// Live status of a match. `Live { minute }` is what makes the engine "live":
/// the Bayesian in-match model reads the elapsed minute to re-derive odds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum MatchStatus {
    Scheduled,
    Live { minute: u16 },
    Finished,
    Postponed,
}

/// A single match between two teams.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Match {
    pub id: MatchId,
    pub home: TeamId,
    pub away: TeamId,
    pub stage: Stage,
    pub kickoff: DateTime<Utc>,
    pub status: MatchStatus,
    pub score: Scoreline,
}

/// A regulation match is 90 minutes; we treat anything past as stoppage time.
pub const REGULATION_MINUTES: u16 = 90;

impl Match {
    pub fn is_finished(&self) -> bool {
        matches!(self.status, MatchStatus::Finished)
    }

    pub fn is_live(&self) -> bool {
        matches!(self.status, MatchStatus::Live { .. })
    }

    /// Minutes of regulation time remaining (0 once finished or past 90').
    pub fn minutes_remaining(&self) -> u16 {
        match self.status {
            MatchStatus::Live { minute } => REGULATION_MINUTES.saturating_sub(minute),
            MatchStatus::Scheduled => REGULATION_MINUTES,
            _ => 0,
        }
    }
}
