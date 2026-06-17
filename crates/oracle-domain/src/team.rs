//! Teams, their confederations, and stable identifiers.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A stable, compact identifier for a national team.
///
/// A newtype around `u32` rather than a bare integer so the type system prevents
/// us from accidentally mixing a [`TeamId`] with a [`crate::MatchId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TeamId(pub u32);

impl fmt::Display for TeamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

/// The six FIFA confederations. Used both for grouping and as a small prior
/// signal (confederation strength) when seeding ratings for unseen teams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Confederation {
    /// Europe
    Uefa,
    /// South America
    Conmebol,
    /// North/Central America & Caribbean
    Concacaf,
    /// Africa
    Caf,
    /// Asia
    Afc,
    /// Oceania
    Ofc,
}

impl Confederation {
    /// A rough historical strength prior (Elo points added at seeding time).
    /// Reflects long-run World Cup performance by confederation, not a hard rule.
    pub fn strength_prior(self) -> f64 {
        match self {
            Confederation::Conmebol => 120.0,
            Confederation::Uefa => 110.0,
            Confederation::Concacaf => -30.0,
            Confederation::Afc => -40.0,
            Confederation::Caf => -20.0,
            Confederation::Ofc => -120.0,
        }
    }
}

impl fmt::Display for Confederation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Confederation::Uefa => "UEFA",
            Confederation::Conmebol => "CONMEBOL",
            Confederation::Concacaf => "CONCACAF",
            Confederation::Caf => "CAF",
            Confederation::Afc => "AFC",
            Confederation::Ofc => "OFC",
        };
        f.write_str(s)
    }
}

/// A national team competing in the tournament.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Team {
    pub id: TeamId,
    /// Full display name, e.g. "Brazil".
    pub name: String,
    /// FIFA three-letter code, e.g. "BRA".
    pub code: String,
    pub confederation: Confederation,
}

impl Team {
    pub fn new(
        id: u32,
        name: impl Into<String>,
        code: impl Into<String>,
        confederation: Confederation,
    ) -> Self {
        Self {
            id: TeamId(id),
            name: name.into(),
            code: code.into(),
            confederation,
        }
    }
}
