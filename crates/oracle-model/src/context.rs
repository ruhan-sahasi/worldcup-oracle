//! Venue and travel context adjustments.
//!
//! A World Cup is not played at neutral sites in any meaningful sense: hosts and their
//! diaspora pack the stands, Mexico City sits at ~2240 m, and 2026 spans a continent so
//! rest and travel differ sharply between teams. This module turns that context into the
//! same per-team `(attack_delta, defense_delta)` log-space adjustments used for lineups,
//! so it flows through [`crate::GoalModel::expected_goals_adjusted`] unchanged.

/// Which side, if any, is effectively at home (host nation or heavy crowd support).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
    HomeTeam,
    AwayTeam,
    Neutral,
}

/// The non-strength context of a match.
#[derive(Debug, Clone, Copy)]
pub struct MatchContext {
    pub host: Host,
    pub altitude_m: f64,
    pub home_rest_days: u8,
    pub away_rest_days: u8,
}

impl Default for MatchContext {
    fn default() -> Self {
        Self {
            host: Host::Neutral,
            altitude_m: 0.0,
            home_rest_days: 4,
            away_rest_days: 4,
        }
    }
}

// Tuning constants (log space; modest, clamped).
const HOST_ATTACK: f64 = 0.18;
const HOST_DEFENSE: f64 = 0.10;
const HOST_VISITOR_PENALTY: f64 = 0.05;
const ALTITUDE_THRESHOLD_M: f64 = 1500.0;
const ALTITUDE_TEMPO: f64 = 0.08; // both teams score a little less up high
const ALTITUDE_VISITOR: f64 = 0.10; // the non-acclimatized side suffers more
const REST_PER_DAY: f64 = 0.02;
const REST_BASELINE_DAYS: f64 = 4.0;

/// Per-team `((home_attack, home_defense), (away_attack, away_defense))` log-space deltas
/// implied by the match context (positive = stronger). Combines additively with lineup
/// adjustments. All terms are modest and the result is clamped.
pub fn context_adjustment(ctx: &MatchContext) -> ((f64, f64), (f64, f64)) {
    let mut home = (0.0, 0.0);
    let mut away = (0.0, 0.0);

    // Host advantage: crowd + familiarity lift the host side and dent the visitor.
    match ctx.host {
        Host::HomeTeam => {
            home.0 += HOST_ATTACK;
            home.1 += HOST_DEFENSE;
            away.0 -= HOST_VISITOR_PENALTY;
        }
        Host::AwayTeam => {
            away.0 += HOST_ATTACK;
            away.1 += HOST_DEFENSE;
            home.0 -= HOST_VISITOR_PENALTY;
        }
        Host::Neutral => {}
    }

    // Altitude: thin air lowers tempo for everyone, and hurts the non-host side more.
    if ctx.altitude_m > ALTITUDE_THRESHOLD_M {
        let a = ((ctx.altitude_m - ALTITUDE_THRESHOLD_M) / 1000.0).min(1.0);
        home.0 -= ALTITUDE_TEMPO * a;
        away.0 -= ALTITUDE_TEMPO * a;
        match ctx.host {
            Host::HomeTeam => away.0 -= ALTITUDE_VISITOR * a,
            Host::AwayTeam => home.0 -= ALTITUDE_VISITOR * a,
            Host::Neutral => {}
        }
    }

    // Rest: fresher legs help; fewer rest days than the norm fatigue a side's attack.
    let rest = |days: u8| REST_PER_DAY * (f64::from(days) - REST_BASELINE_DAYS).clamp(-3.0, 3.0);
    home.0 += rest(ctx.home_rest_days);
    away.0 += rest(ctx.away_rest_days);

    let clamp = |x: f64| x.clamp(-0.5, 0.5);
    (
        (clamp(home.0), clamp(home.1)),
        (clamp(away.0), clamp(away.1)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_is_boosted_visitor_dented() {
        let (home, away) = context_adjustment(&MatchContext {
            host: Host::HomeTeam,
            ..Default::default()
        });
        assert!(home.0 > 0.0 && home.1 > 0.0, "host attack and defense rise");
        assert!(away.0 < 0.0, "visitor attack dips");
    }

    #[test]
    fn altitude_lowers_tempo_for_both() {
        let (home, away) = context_adjustment(&MatchContext {
            altitude_m: 2240.0,
            ..Default::default()
        });
        assert!(home.0 < 0.0 && away.0 < 0.0, "thin air lowers both attacks");
    }

    #[test]
    fn more_rest_helps_relative_to_less() {
        let (home, away) = context_adjustment(&MatchContext {
            home_rest_days: 6,
            away_rest_days: 2,
            ..Default::default()
        });
        assert!(home.0 > away.0, "the better-rested side is favoured");
    }

    #[test]
    fn neutral_default_is_zero() {
        let (home, away) = context_adjustment(&MatchContext::default());
        assert_eq!(home, (0.0, 0.0));
        assert_eq!(away, (0.0, 0.0));
    }
}
