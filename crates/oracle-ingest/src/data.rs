//! Embedded, offline-friendly tournament data.
//!
//! So the engine runs with **zero external dependencies** - no API key, no network -
//! this module ships:
//!
//! - [`world_cup_2026`] - a 48-team, 12-group tournament in the 2026 format, with a
//!   balanced snake-seeded draw and a generated group-stage fixture list.
//! - [`team_strengths`] - approximate strength ratings used to seed Elo.
//! - [`synthetic_history`] - a reproducible set of plausible past international
//!   results (drawn from the ratings via a Poisson model) so the Dixon-Coles fit and
//!   the backtest have realistic, deterministic training data offline.
//! - [`fit_baseline_model`] - convenience: fit a goal model on the synthetic history.
//!
//! > The roster and draw are a representative sample for demonstration, not the
//! > official FIFA draw. Point the [`crate::FootballDataProvider`] at the live API
//! > for real teams, fixtures, and results.

use chrono::{TimeZone, Utc};
use oracle_domain::{
    Confederation, Group, Match, MatchId, MatchStatus, Scoreline, Stage, Team, TeamId, Tournament,
};
use oracle_model::{DixonColesConfig, Ensemble, GoalModel, Observation};
use oracle_ratings::RatingStore;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};

use Confederation::{Afc, Caf, Concacaf, Conmebol, Ofc, Uefa};

/// `(name, FIFA code, confederation, approximate strength rating)`.
const TEAMS: &[(&str, &str, Confederation, f64)] = &[
    // CONMEBOL
    ("Argentina", "ARG", Conmebol, 2100.0),
    ("Brazil", "BRA", Conmebol, 2050.0),
    ("Uruguay", "URU", Conmebol, 1950.0),
    ("Colombia", "COL", Conmebol, 1930.0),
    ("Ecuador", "ECU", Conmebol, 1830.0),
    ("Paraguay", "PAR", Conmebol, 1760.0),
    // UEFA
    ("France", "FRA", Uefa, 2080.0),
    ("Spain", "ESP", Uefa, 2060.0),
    ("England", "ENG", Uefa, 2040.0),
    ("Portugal", "POR", Uefa, 2010.0),
    ("Netherlands", "NED", Uefa, 1990.0),
    ("Germany", "GER", Uefa, 1980.0),
    ("Italy", "ITA", Uefa, 1960.0),
    ("Croatia", "CRO", Uefa, 1940.0),
    ("Belgium", "BEL", Uefa, 1930.0),
    ("Switzerland", "SUI", Uefa, 1860.0),
    ("Denmark", "DEN", Uefa, 1850.0),
    ("Austria", "AUT", Uefa, 1840.0),
    ("Serbia", "SRB", Uefa, 1800.0),
    ("Turkey", "TUR", Uefa, 1820.0),
    ("Norway", "NOR", Uefa, 1810.0),
    ("Ukraine", "UKR", Uefa, 1790.0),
    ("Poland", "POL", Uefa, 1770.0),
    ("Scotland", "SCO", Uefa, 1740.0),
    // CAF
    ("Morocco", "MAR", Caf, 1900.0),
    ("Senegal", "SEN", Caf, 1850.0),
    ("Nigeria", "NGA", Caf, 1820.0),
    ("Algeria", "ALG", Caf, 1800.0),
    ("Egypt", "EGY", Caf, 1790.0),
    ("Ivory Coast", "CIV", Caf, 1780.0),
    ("Cameroon", "CMR", Caf, 1760.0),
    ("Ghana", "GHA", Caf, 1740.0),
    ("Tunisia", "TUN", Caf, 1730.0),
    // AFC
    ("Japan", "JPN", Afc, 1870.0),
    ("South Korea", "KOR", Afc, 1830.0),
    ("Iran", "IRN", Afc, 1820.0),
    ("Australia", "AUS", Afc, 1790.0),
    ("Saudi Arabia", "KSA", Afc, 1700.0),
    ("Qatar", "QAT", Afc, 1690.0),
    ("Uzbekistan", "UZB", Afc, 1670.0),
    ("Iraq", "IRQ", Afc, 1660.0),
    // CONCACAF (incl. hosts)
    ("Mexico", "MEX", Concacaf, 1820.0),
    ("United States", "USA", Concacaf, 1810.0),
    ("Canada", "CAN", Concacaf, 1790.0),
    ("Costa Rica", "CRC", Concacaf, 1680.0),
    ("Panama", "PAN", Concacaf, 1660.0),
    ("Jamaica", "JAM", Concacaf, 1630.0),
    // OFC
    ("New Zealand", "NZL", Ofc, 1610.0),
];

const NUM_GROUPS: usize = 12;

/// All 48 teams, with stable ids equal to their index in [`TEAMS`].
pub fn teams() -> Vec<Team> {
    TEAMS
        .iter()
        .enumerate()
        .map(|(i, &(name, code, conf, _))| Team::new(i as u32, name, code, conf))
        .collect()
}

/// Approximate strength rating per team, for seeding Elo.
pub fn team_strengths() -> Vec<(TeamId, f64)> {
    TEAMS
        .iter()
        .enumerate()
        .map(|(i, &(_, _, _, rating))| (TeamId(i as u32), rating))
        .collect()
}

/// Build the full 2026-format tournament: balanced draw + group-stage fixtures.
pub fn world_cup_2026() -> Tournament {
    let teams = teams();
    let mut t = Tournament::new("FIFA World Cup 2026");
    t.teams = teams.clone();

    // Snake-seed by rating into 12 balanced groups (pot 1 strongest, etc.).
    let mut by_rating: Vec<usize> = (0..teams.len()).collect();
    by_rating.sort_by(|&a, &b| {
        TEAMS[b]
            .3
            .partial_cmp(&TEAMS[a].3)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut groups: Vec<Vec<TeamId>> = vec![Vec::new(); NUM_GROUPS];
    for (pot, chunk) in by_rating.chunks(NUM_GROUPS).enumerate() {
        for (slot, &team_ix) in chunk.iter().enumerate() {
            // Reverse direction on odd pots so strong/weak teams spread evenly.
            let g = if pot % 2 == 0 {
                slot
            } else {
                NUM_GROUPS - 1 - slot
            };
            groups[g].push(TeamId(team_ix as u32));
        }
    }
    t.groups = groups
        .iter()
        .enumerate()
        .map(|(i, teams)| Group {
            name: (b'A' + i as u8) as char,
            teams: teams.clone(),
        })
        .collect();

    // Group-stage fixtures: a round-robin per group, scheduled across the opening
    // fortnight. Status starts Scheduled; results arrive via the live event stream.
    let base = Utc.with_ymd_and_hms(2026, 6, 11, 16, 0, 0).unwrap();
    let mut next_id = 1u32;
    for group in &t.groups {
        let m = &group.teams;
        if m.len() < 4 {
            continue;
        }
        let pairings = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
        for (matchday, (i, j)) in pairings.into_iter().enumerate() {
            let kickoff = base + chrono::Duration::hours((next_id as i64) * 3);
            t.matches.push(Match {
                id: MatchId(next_id),
                home: m[i],
                away: m[j],
                stage: Stage::Group(group.name),
                kickoff,
                status: MatchStatus::Scheduled,
                score: Scoreline::new(0, 0),
            });
            next_id += 1;
            let _ = matchday;
        }
    }

    t
}

/// Expected goals between two strengths (neutral venue) under a simple log-linear map.
fn rating_to_xg(home: f64, away: f64) -> (f64, f64) {
    let sup = (home - away) / 250.0;
    let lambda = (1.35 * (0.30 * sup).exp()).clamp(0.1, 6.0);
    let mu = (1.35 * (-0.30 * sup).exp()).clamp(0.1, 6.0);
    (lambda, mu)
}

/// Generate `n_matches` reproducible synthetic historical results among the 48 teams,
/// drawn from their ratings via a Poisson goal model. Ages are spread over the past
/// ~3 years so the Dixon-Coles time-decay has something to bite on.
pub fn synthetic_history(n_matches: usize, seed: u64) -> Vec<Observation> {
    let strengths = team_strengths();
    let n = strengths.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n_matches);

    let sample = |rng: &mut StdRng, lambda: f64| -> u8 {
        Poisson::new(lambda)
            .map(|d| (d.sample(rng) as u32).min(12) as u8)
            .unwrap_or(0)
    };

    for k in 0..n_matches {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n);
        while j == i {
            j = rng.gen_range(0..n);
        }
        let (lambda, mu) = rating_to_xg(strengths[i].1, strengths[j].1);
        let score = Scoreline::new(sample(&mut rng, lambda), sample(&mut rng, mu));
        // Spread matches over ~1100 days, oldest first.
        let age_days = 1100.0 * (1.0 - k as f64 / n_matches as f64);
        out.push(Observation::new(
            strengths[i].0,
            strengths[j].0,
            score,
            age_days,
        ));
    }
    out
}

/// The offline-fitted baseline: a goal model, Elo seeds, and a **learned** ensemble.
pub struct Baseline {
    pub model: GoalModel,
    pub elo_seeds: Vec<(TeamId, f64)>,
    pub ensemble: Ensemble,
}

/// Fit the full baseline on synthetic history with a proper train→validation split:
/// Dixon-Coles and Elo are fit on the older 70%, then the ensemble weights +
/// temperature are *learned* on the held-out 30%. This is what makes the shipped
/// ensemble provably no worse than its best member (no more hardcoded weights).
pub fn fit_baseline(seed: u64) -> Baseline {
    let mut history = synthetic_history(4000, seed);
    // Oldest first, so we train on the past and validate on more recent matches.
    history.sort_by(|a, b| {
        b.age_days
            .partial_cmp(&a.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let split = history.len() * 7 / 10;
    let (train, validation) = history.split_at(split);

    let model = GoalModel::fit(train, DixonColesConfig::default());

    let elo_seeds = team_strengths();
    let mut ratings = RatingStore::with_defaults();
    for &(team, rating) in &elo_seeds {
        ratings.seed(team, rating);
    }
    for obs in train {
        ratings.record(obs.home, obs.away, obs.score, true);
    }

    // Member predictions on the validation slice: [Dixon-Coles, Elo].
    let mut member_preds = Vec::with_capacity(validation.len());
    let mut actuals = Vec::with_capacity(validation.len());
    for obs in validation {
        let dc = model.outcome_probabilities(obs.home, obs.away, true);
        let elo = ratings.win_probabilities(obs.home, obs.away, true);
        member_preds.push(vec![dc, elo]);
        actuals.push(obs.score.outcome());
    }
    let ensemble = Ensemble::fit(&member_preds, &actuals, 2);

    Baseline {
        model,
        elo_seeds,
        ensemble,
    }
}

/// Convenience: just the goal model from [`fit_baseline`].
pub fn fit_baseline_model(seed: u64) -> GoalModel {
    fit_baseline(seed).model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tournament_has_48_teams_in_12_full_groups() {
        let t = world_cup_2026();
        assert_eq!(t.teams.len(), 48);
        assert_eq!(t.groups.len(), 12);
        for g in &t.groups {
            assert_eq!(g.teams.len(), 4, "group {} should have 4 teams", g.name);
        }
        // 12 groups × 6 round-robin matches = 72 group fixtures.
        assert_eq!(t.matches.len(), 72);
    }

    #[test]
    fn every_team_is_in_exactly_one_group() {
        let t = world_cup_2026();
        let mut seen = std::collections::HashSet::new();
        for g in &t.groups {
            for &team in &g.teams {
                assert!(seen.insert(team), "{team} appears in two groups");
            }
        }
        assert_eq!(seen.len(), 48);
    }

    #[test]
    fn baseline_model_recovers_argentina_above_new_zealand() {
        let model = fit_baseline_model(1);
        // Argentina (id 0, strongest) vs New Zealand (id 47, weakest).
        let p = model.outcome_probabilities(TeamId(0), TeamId(47), true);
        assert!(p.home_win > 0.6, "fit should rate Argentina well above NZ");
    }
}
