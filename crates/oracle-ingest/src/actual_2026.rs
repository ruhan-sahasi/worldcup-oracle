//! The **real** 2026 FIFA World Cup knockout results, from the Round of 16 onward.
//!
//! The rest of this crate ships a *synthetic* offline tournament (see [`crate::data`]): a
//! representative field, a snake-seeded draw, and simulated results. That is what the engine
//! predicts over by default. This module is different - it encodes what *actually happened* in the
//! 2026 tournament, so the app can answer "given the real teams still alive at stage X, who does
//! the model make champion?" (the stage-conditioned forecast).
//!
//! ## Scope and provenance (read this)
//!
//! Coverage is the **Round of 16 through the Final** only. The group stage and the Round of 32 are
//! deliberately **not** encoded: their full match-by-match detail could not be verified from public
//! reporting to the standard the later rounds could, and the real early-round field includes teams
//! outside this project's rating universe. Encoding unverified scorelines and labelling them "real"
//! would be worse than omitting them, so the stage selector starts at the Round of 16.
//!
//! Every result below is corroborated across multiple public match reports (FIFA match centre,
//! ESPN, and major outlets) and the bracket is internally consistent - each round's participants are
//! exactly the previous round's winners (asserted by a test). Confidence is **high** for all encoded
//! rounds. One participant, **Cape Verde** (lost to Argentina in the Round of 16), is not in the
//! synthetic rating table; it is added here with a stable id and given a weak rating by the engine.
//!
//! Because World Cup knockout ties are played at neutral venues, the `home`/`away` fields are just
//! bracket slots (they carry no home-advantage meaning); scores are stored winner-first for display.

use crate::data;
use oracle_domain::{Confederation, MatchId, MatchStatus, Scoreline, Stage, Team, Tournament};

/// The stable id given to Cape Verde, which is absent from the synthetic [`crate::data`] table
/// (48 teams, ids `0..=47`). Kept just past that range so it never collides.
pub const CAPE_VERDE_ID: u32 = 48;

/// One real knockout result. `home`/`away` are FIFA three-letter codes (bracket slots, winner
/// first); `note` records extra time or a shootout for display.
#[derive(Debug, Clone, Copy)]
pub struct KnockoutResult {
    pub home: &'static str,
    pub away: &'static str,
    pub home_score: u8,
    pub away_score: u8,
    pub note: &'static str,
}

impl KnockoutResult {
    const fn new(
        home: &'static str,
        away: &'static str,
        hs: u8,
        aw: u8,
        note: &'static str,
    ) -> Self {
        Self {
            home,
            away,
            home_score: hs,
            away_score: aw,
            note,
        }
    }

    /// The FIFA code of the side that advanced. Scores are stored winner-first, so the home side
    /// won unless the away side has the higher score.
    pub fn winner(&self) -> &'static str {
        if self.away_score > self.home_score {
            self.away
        } else {
            self.home
        }
    }
}

/// Round of 16 (8 ties), in bracket order: consecutive pairs feed one quarter-final, so the fold
/// [`R16[0]`/`R16[1]`] -> QF, etc. reproduces the real tree.
const R16: [KnockoutResult; 8] = [
    KnockoutResult::new("ESP", "POR", 1, 0, ""),
    KnockoutResult::new("BEL", "USA", 4, 1, ""),
    KnockoutResult::new("FRA", "PAR", 1, 0, ""),
    KnockoutResult::new("MAR", "CAN", 3, 0, ""),
    KnockoutResult::new("ARG", "CPV", 3, 2, ""),
    KnockoutResult::new("SUI", "ALG", 2, 0, ""),
    KnockoutResult::new("ENG", "MEX", 3, 2, ""),
    KnockoutResult::new("NOR", "BRA", 2, 0, ""),
];

/// Quarter-finals (4 ties), in bracket order.
const QF: [KnockoutResult; 4] = [
    KnockoutResult::new("ESP", "BEL", 2, 1, ""),
    KnockoutResult::new("FRA", "MAR", 2, 0, ""),
    KnockoutResult::new("ARG", "SUI", 3, 1, "a.e.t."),
    KnockoutResult::new("ENG", "NOR", 2, 1, ""),
];

/// Semi-finals (2 ties), in bracket order.
const SF: [KnockoutResult; 2] = [
    KnockoutResult::new("ESP", "FRA", 2, 0, ""),
    KnockoutResult::new("ARG", "ENG", 2, 1, ""),
];

/// The Final.
const FINAL: [KnockoutResult; 1] = [KnockoutResult::new("ESP", "ARG", 1, 0, "a.e.t.")];

/// The third-place play-off (for the "what actually happened" reveal; not a forecastable stage).
const THIRD: [KnockoutResult; 1] = [KnockoutResult::new("ENG", "FRA", 6, 4, "")];

/// The stages the selector offers, earliest first.
pub const STAGES: [Stage; 4] = [
    Stage::RoundOf16,
    Stage::QuarterFinal,
    Stage::SemiFinal,
    Stage::Final,
];

/// The real ties at a forecastable stage, in bracket order. `None` for stages this dataset does not
/// cover (group stage, Round of 32) or the third-place play-off.
pub fn stage_results(stage: Stage) -> Option<&'static [KnockoutResult]> {
    match stage {
        Stage::RoundOf16 => Some(&R16),
        Stage::QuarterFinal => Some(&QF),
        Stage::SemiFinal => Some(&SF),
        Stage::Final => Some(&FINAL),
        _ => None,
    }
}

/// The third-place play-off result.
pub fn third_place() -> KnockoutResult {
    THIRD[0]
}

/// A short, URL-friendly slug for a stage (`"round-of-16"`, `"quarter-final"`, ...).
pub fn stage_slug(stage: Stage) -> &'static str {
    match stage {
        Stage::RoundOf16 => "round-of-16",
        Stage::QuarterFinal => "quarter-final",
        Stage::SemiFinal => "semi-final",
        Stage::Final => "final",
        Stage::RoundOf32 => "round-of-32",
        Stage::ThirdPlace => "third-place",
        Stage::Group(_) => "group",
    }
}

/// Parse a stage from a slug or common alias. Accepts `r16`, `qf`, `sf`, `final`, and the full
/// slugs. Returns `None` for anything not offered by the selector.
pub fn parse_stage(s: &str) -> Option<Stage> {
    match s.trim().to_ascii_lowercase().as_str() {
        "round-of-16" | "round_of_16" | "roundof16" | "r16" | "16" | "last-16" => {
            Some(Stage::RoundOf16)
        }
        "quarter-final" | "quarter-finals" | "quarterfinal" | "qf" | "quarters" | "8" => {
            Some(Stage::QuarterFinal)
        }
        "semi-final" | "semi-finals" | "semifinal" | "sf" | "semis" | "4" => Some(Stage::SemiFinal),
        "final" | "f" | "2" => Some(Stage::Final),
        _ => None,
    }
}

/// The full 48-team synthetic table extended with Cape Verde, so any real 2026 team resolves to a
/// stable [`Team`]. Real teams keep their synthetic-table id (and thus the fitted model's
/// coefficients); Cape Verde takes [`CAPE_VERDE_ID`].
fn actual_team_table() -> Vec<Team> {
    let mut teams = data::teams();
    teams.push(Team::new(
        CAPE_VERDE_ID,
        "Cape Verde",
        "CPV",
        Confederation::Caf,
    ));
    teams
}

/// Resolve a FIFA code to its [`Team`] in the extended table (real teams keep their synthetic-table
/// id; Cape Verde uses [`CAPE_VERDE_ID`]). `None` for a code not in the table.
pub fn resolve_code(code: &str) -> Option<Team> {
    actual_team_table().into_iter().find(|t| t.code == code)
}

/// Resolve a FIFA code, panicking on an unknown one - a bug in the const tables above (guarded by
/// [`tests`]).
fn team_by_code(code: &str) -> Team {
    resolve_code(code).unwrap_or_else(|| panic!("unknown FIFA code in the 2026 dataset: {code}"))
}

/// The teams still alive at a stage, in bracket order (each consecutive pair is one tie). `None` for
/// stages the dataset does not cover.
pub fn teams_alive(stage: Stage) -> Option<Vec<Team>> {
    let results = stage_results(stage)?;
    let mut teams = Vec::with_capacity(results.len() * 2);
    for r in results {
        teams.push(team_by_code(r.home));
        teams.push(team_by_code(r.away));
    }
    Some(teams)
}

/// Build a knockout-only [`Tournament`] positioned at `stage`: its teams are exactly the real field
/// still alive, and its fixtures are that stage's ties in bracket order. The fixtures are labelled
/// [`Stage::RoundOf32`] because that is the simulator's generic "knockout entry" layer - the
/// bracket depth (and therefore which forecast column means Semi-final, Final, Champion) is derived
/// from the number of ties, so an 8-tie entry reads as the Round of 16, a 4-tie entry as the
/// quarter-finals, and so on. Every tie starts `Scheduled`, so the Monte-Carlo plays the whole
/// remaining bracket forward over just these teams. Returns `None` for uncovered stages.
pub fn stage_tournament(stage: Stage) -> Option<Tournament> {
    let results = stage_results(stage)?;
    let mut t = Tournament::new(format!("FIFA World Cup 2026 - {stage}"));
    // Teams in bracket order, de-duplicated while preserving first-seen order.
    for r in results {
        for code in [r.home, r.away] {
            let team = team_by_code(code);
            if !t.teams.iter().any(|x| x.id == team.id) {
                t.teams.push(team);
            }
        }
    }
    for (i, r) in results.iter().enumerate() {
        t.matches.push(oracle_domain::Match {
            id: MatchId(1 + i as u32),
            home: team_by_code(r.home).id,
            away: team_by_code(r.away).id,
            stage: Stage::RoundOf32,
            kickoff: base_kickoff(),
            status: MatchStatus::Scheduled,
            score: Scoreline::new(0, 0),
        });
    }
    Some(t)
}

/// Any real team whose id is outside the synthetic table (`0..=47`) - i.e. Cape Verde - so the
/// engine can seed a rating for it before forecasting.
pub fn teams_outside_model() -> Vec<Team> {
    actual_team_table()
        .into_iter()
        .filter(|t| t.id.0 as usize >= data::teams().len())
        .collect()
}

/// How the tournament actually finished, for the "what happened" reveal.
#[derive(Debug, Clone, Copy)]
pub struct ActualOutcome {
    pub champion: &'static str,
    pub runner_up: &'static str,
    pub third: &'static str,
    pub fourth: &'static str,
}

/// The real final four, derived from the encoded Final and third-place results.
pub fn actual_outcome() -> ActualOutcome {
    let f = FINAL[0];
    let t = THIRD[0];
    ActualOutcome {
        champion: f.winner(),
        runner_up: if f.winner() == f.home { f.away } else { f.home },
        third: t.winner(),
        fourth: if t.winner() == t.home { t.away } else { t.home },
    }
}

fn base_kickoff() -> chrono::DateTime<chrono::Utc> {
    use chrono::TimeZone;
    chrono::Utc.with_ymd_and_hms(2026, 7, 4, 16, 0, 0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_code_resolves() {
        for stage in STAGES {
            for r in stage_results(stage).unwrap() {
                let _ = team_by_code(r.home);
                let _ = team_by_code(r.away);
            }
        }
        let _ = team_by_code(THIRD[0].home);
        let _ = team_by_code(THIRD[0].away);
    }

    #[test]
    fn bracket_is_internally_consistent() {
        // Each round's participants must be exactly the previous round's winners, in bracket order:
        // winners of ties 2k and 2k+1 meet in tie k of the next round.
        let rounds: [&[KnockoutResult]; 3] = [&R16, &QF, &SF];
        let next: [&[KnockoutResult]; 3] = [&QF, &SF, &FINAL];
        for (round, nxt) in rounds.iter().zip(next.iter()) {
            for (k, tie) in nxt.iter().enumerate() {
                let left = round[2 * k].winner();
                let right = round[2 * k + 1].winner();
                let pair: HashSet<&str> = [left, right].into_iter().collect();
                let got: HashSet<&str> = [tie.home, tie.away].into_iter().collect();
                assert_eq!(
                    pair, got,
                    "next-round tie {k} should be the two prior winners {left} & {right}"
                );
            }
        }
        // Third place is contested by the two beaten semi-finalists.
        let sf_losers: HashSet<&str> = SF
            .iter()
            .map(|t| if t.winner() == t.home { t.away } else { t.home })
            .collect();
        let third: HashSet<&str> = [THIRD[0].home, THIRD[0].away].into_iter().collect();
        assert_eq!(
            sf_losers, third,
            "third-place play-off is the two SF losers"
        );
    }

    #[test]
    fn stage_tournament_has_expected_field_size() {
        assert_eq!(stage_tournament(Stage::RoundOf16).unwrap().teams.len(), 16);
        assert_eq!(
            stage_tournament(Stage::QuarterFinal).unwrap().teams.len(),
            8
        );
        assert_eq!(stage_tournament(Stage::SemiFinal).unwrap().teams.len(), 4);
        assert_eq!(stage_tournament(Stage::Final).unwrap().teams.len(), 2);
        assert!(stage_tournament(Stage::RoundOf32).is_none());
    }

    #[test]
    fn outcome_matches_reporting() {
        let o = actual_outcome();
        assert_eq!(o.champion, "ESP");
        assert_eq!(o.runner_up, "ARG");
        assert_eq!(o.third, "ENG");
        assert_eq!(o.fourth, "FRA");
    }

    #[test]
    fn slugs_round_trip() {
        for stage in STAGES {
            assert_eq!(parse_stage(stage_slug(stage)), Some(stage));
        }
    }
}
