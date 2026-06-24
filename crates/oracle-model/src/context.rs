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
    /// Net crowd partisanship in `[-1, 1]`: positive favors the home side, negative the away
    /// side. Captures *crowd composition* (diaspora, traveling fans) beyond literal host status -
    /// e.g. Mexico drawing a near-home crowd across US venues, or an Argentina-heavy stand in
    /// Miami. Distinct from [`Host`], which is the host nation's familiarity/logistics edge.
    pub crowd_support: f64,
    /// Kilometres each side travelled from its previous fixture's venue to this one (a
    /// continent-spanning 2026 makes this material). 0 = no prior fixture / same venue.
    pub home_travel_km: f64,
    pub away_travel_km: f64,
    /// Signed time-zone change (hours) since each side's previous fixture: positive = eastward
    /// (a phase advance, the harder direction to adjust to), negative = westward.
    pub home_tz_shift: f64,
    pub away_tz_shift: f64,
    /// Match-time temperature in degrees Celsius (venue climate + kickoff hour). Above a comfort
    /// threshold it suppresses tempo for both sides - a real factor for the midday/afternoon
    /// kickoffs of a North American summer (Dallas, Monterrey, Miami). 0 is treated as benign.
    pub temperature_c: f64,
}

impl Default for MatchContext {
    fn default() -> Self {
        Self {
            host: Host::Neutral,
            altitude_m: 0.0,
            home_rest_days: 4,
            away_rest_days: 4,
            crowd_support: 0.0,
            home_travel_km: 0.0,
            away_travel_km: 0.0,
            home_tz_shift: 0.0,
            away_tz_shift: 0.0,
            temperature_c: 0.0,
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
// Crowd composition: a fully partisan crowd (|support| = 1) is worth a bit less than the host
// nation's own familiarity edge, and it lifts the favored side while denting the other.
const CROWD_ATTACK: f64 = 0.12;
const CROWD_DEFENSE: f64 = 0.06;
// Travel & circadian load (continent-spanning 2026). A coast-to-coast hop saturates the distance
// term; a large eastward time-zone shift saturates the circadian one. East ("phase advance") is
// harder to adjust to than west.
const TRAVEL_DISTANCE: f64 = 0.06; // attack lost for a full coast-to-coast trip
const TRAVEL_CIRCADIAN: f64 = 0.05; // attack lost for a fully disruptive time-zone shift
const TRAVEL_SATURATION_KM: f64 = 4000.0;
const TZ_EAST_WEIGHT: f64 = 0.8; // eastward shift, per hour, toward saturation
const TZ_WEST_WEIGHT: f64 = 0.4; // westward shift is gentler
const TZ_SATURATION_HOURS: f64 = 3.0;
// Heat: comfortable below the threshold; above it, every degree saps tempo, capped for extremes.
const HEAT_COMFORT_C: f64 = 24.0;
const HEAT_PER_DEGREE: f64 = 0.012;
const HEAT_CAP: f64 = 0.12;

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

    // Crowd composition: a partisan crowd lifts the favored side and dents the other,
    // continuously (not just for literal hosts). `crowd_support` > 0 favors home.
    let c = ctx.crowd_support.clamp(-1.0, 1.0);
    home.0 += CROWD_ATTACK * c;
    home.1 += CROWD_DEFENSE * c;
    away.0 -= CROWD_ATTACK * c;
    away.1 -= CROWD_DEFENSE * c;

    // Travel & circadian load: distance and (especially eastward) time-zone shifts since the
    // last match sap a side's attack. The differential between the two sides is what tilts the
    // match; a team that stayed put is fresher than one that crossed the continent.
    let fatigue = |km: f64, tz: f64| {
        let distance = (km / TRAVEL_SATURATION_KM).clamp(0.0, 1.0);
        let circadian_hours = if tz >= 0.0 {
            tz * TZ_EAST_WEIGHT
        } else {
            -tz * TZ_WEST_WEIGHT
        };
        let circadian = (circadian_hours / TZ_SATURATION_HOURS).clamp(0.0, 1.0);
        TRAVEL_DISTANCE * distance + TRAVEL_CIRCADIAN * circadian
    };
    home.0 -= fatigue(ctx.home_travel_km, ctx.home_tz_shift);
    away.0 -= fatigue(ctx.away_travel_km, ctx.away_tz_shift);

    // Heat: above a comfort threshold, high temperatures suppress tempo for *both* sides (less
    // running, more conservative play). Scaling both goal rates down by the same factor also
    // flattens the favourite's edge - fewer goals make the scoreline noisier, so the underdog's
    // chance rises. That leveling falls out of the Poisson variance; we model only the tempo cut.
    let heat = ((ctx.temperature_c - HEAT_COMFORT_C).max(0.0) * HEAT_PER_DEGREE).min(HEAT_CAP);
    home.0 -= heat;
    away.0 -= heat;

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

    #[test]
    fn partisan_crowd_lifts_favored_side_and_dents_the_other() {
        // A home-favoring crowd (a Mexico-in-the-US type stand) on an otherwise neutral match.
        let (home, away) = context_adjustment(&MatchContext {
            crowd_support: 0.6,
            ..Default::default()
        });
        assert!(home.0 > 0.0 && home.1 > 0.0, "favored side lifted");
        assert!(away.0 < 0.0 && away.1 < 0.0, "the other side dented");
        // Symmetric: flipping the crowd flips the signs.
        let (fh, fa) = context_adjustment(&MatchContext {
            crowd_support: -0.6,
            ..Default::default()
        });
        assert!((fh.0 + home.0).abs() < 1e-9 && (fa.0 + away.0).abs() < 1e-9);
    }

    #[test]
    fn travel_saps_the_side_that_moved_more() {
        // One side stayed put; the other crossed the continent (and shifted east).
        let (home, away) = context_adjustment(&MatchContext {
            away_travel_km: 4000.0,
            away_tz_shift: 3.0,
            ..Default::default()
        });
        assert_eq!(home.0, 0.0, "the side that stayed put is unaffected");
        assert!(away.0 < 0.0, "the side that travelled loses attack");
    }

    #[test]
    fn eastward_travel_is_harder_than_westward() {
        let east = context_adjustment(&MatchContext {
            home_tz_shift: 3.0,
            ..Default::default()
        })
        .0
         .0;
        let west = context_adjustment(&MatchContext {
            home_tz_shift: -3.0,
            ..Default::default()
        })
        .0
         .0;
        // Both are penalties (negative); eastward should bite harder (more negative).
        assert!(
            east < west && west < 0.0,
            "east {east} should be worse than west {west}"
        );
    }

    #[test]
    fn heat_suppresses_tempo_for_both_above_comfort() {
        // A comfortable temperature does nothing; a scorching one lowers both attacks equally.
        let mild = context_adjustment(&MatchContext {
            temperature_c: 20.0,
            ..Default::default()
        });
        assert_eq!(mild, ((0.0, 0.0), (0.0, 0.0)));

        let (home, away) = context_adjustment(&MatchContext {
            temperature_c: 36.0,
            ..Default::default()
        });
        assert!(home.0 < 0.0 && away.0 < 0.0, "heat lowers both attacks");
        assert!(
            (home.0 - away.0).abs() < 1e-9,
            "heat hits both sides equally"
        );
        // Defense is untouched: it is a tempo effect, not a strength one.
        assert_eq!(home.1, 0.0);
    }
}
