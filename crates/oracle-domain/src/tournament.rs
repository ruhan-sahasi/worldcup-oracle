//! The tournament container and forecast outputs.

use crate::fixture::Match;
use crate::team::{Team, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A first-round group: a named bucket of (usually four) teams that play a
/// single round-robin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub name: char,
    pub teams: Vec<TeamId>,
}

/// The whole competition: its teams, group structure, and full fixture list
/// (played and upcoming). This is the read model the engine, simulator, and API
/// all share.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tournament {
    pub name: String,
    pub teams: Vec<Team>,
    pub groups: Vec<Group>,
    pub matches: Vec<Match>,
}

impl Tournament {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            teams: Vec::new(),
            groups: Vec::new(),
            matches: Vec::new(),
        }
    }

    pub fn team(&self, id: TeamId) -> Option<&Team> {
        self.teams.iter().find(|t| t.id == id)
    }

    /// Build a `TeamId -> &Team` lookup for hot paths (the simulator calls this once).
    pub fn team_index(&self) -> HashMap<TeamId, &Team> {
        self.teams.iter().map(|t| (t.id, t)).collect()
    }

    /// Which group, if any, a team belongs to.
    pub fn group_of(&self, id: TeamId) -> Option<char> {
        self.groups
            .iter()
            .find(|g| g.teams.contains(&id))
            .map(|g| g.name)
    }

    /// Matches that have not yet finished - the set the simulator must resolve.
    pub fn remaining_matches(&self) -> impl Iterator<Item = &Match> {
        self.matches.iter().filter(|m| !m.is_finished())
    }
}

/// Per-team probabilities of reaching each stage, produced by the Monte-Carlo
/// simulator. These are the headline "champion odds" numbers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TeamForecast {
    pub team: TeamId,
    pub p_advance_group: f64,
    pub p_round_of_16: f64,
    pub p_quarter_final: f64,
    pub p_semi_final: f64,
    pub p_final: f64,
    pub p_champion: f64,
}

impl TeamForecast {
    pub fn zeroed(team: TeamId) -> Self {
        Self {
            team,
            p_advance_group: 0.0,
            p_round_of_16: 0.0,
            p_quarter_final: 0.0,
            p_semi_final: 0.0,
            p_final: 0.0,
            p_champion: 0.0,
        }
    }
}

/// The full tournament forecast: one [`TeamForecast`] per team plus the number of
/// Monte-Carlo iterations behind it (so callers can reason about Monte-Carlo error).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TournamentForecast {
    pub iterations: u64,
    pub teams: Vec<TeamForecast>,
}

impl TournamentForecast {
    /// Teams sorted by championship probability, descending.
    pub fn ranked(&self) -> Vec<TeamForecast> {
        let mut v = self.teams.clone();
        v.sort_by(|a, b| {
            b.p_champion
                .partial_cmp(&a.p_champion)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}
