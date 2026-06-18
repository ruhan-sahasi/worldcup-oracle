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

use crate::error::{IngestError, Result};
use chrono::{TimeZone, Utc};
use oracle_domain::{
    Confederation, Group, Match, MatchId, MatchStatus, Probabilities, Scoreline, Stage, Team,
    TeamId, Tournament,
};
use oracle_model::poisson::poisson_pmf;
use oracle_model::{implied_probabilities, DixonColesConfig, Ensemble, GoalModel, Observation};
use oracle_ratings::RatingStore;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};
use std::collections::HashMap;
use std::path::Path;

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

// ----- Squads & lineups (synthetic, offline) -----
//
// Real player rosters are not bundled (the live adapter would supply real lineups). To
// make the lineup-aware feature work and be demonstrable offline, each team gets a
// deterministic synthetic squad scaled to its strength rating, with one clear standout.
// A confirmed lineup is turned into an attack/defense adjustment by comparing the XI on
// the pitch to the team's strongest available XI.

/// A playing position, used to weight a player's attacking vs defensive contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Gk,
    Def,
    Mid,
    Fwd,
}

/// A squad member with attacking and defensive contribution scores (arbitrary units;
/// only relative magnitudes within a squad matter).
#[derive(Debug, Clone)]
pub struct Player {
    pub name: String,
    pub position: Position,
    pub attack: f64,
    pub defense: f64,
}

impl Player {
    fn overall(&self) -> f64 {
        self.attack + self.defense
    }
}

const SURNAMES: &[&str] = &[
    "Silva", "Costa", "Muller", "Kovac", "Diallo", "Tanaka", "Rossi", "Nguyen", "Okafor", "Hassan",
    "Andersen", "Park", "Lopez", "Ferreira", "Schmidt", "Ivanov", "Haddad", "Mensah", "Yilmaz",
    "Novak", "Reyes", "Bauer", "Moreno", "Dubois",
];

const STARTING_XI: usize = 11;

/// Deterministic synthetic squad for a team, scaled to its strength rating. 16 players in
/// a 2 GK / 6 DEF / 5 MID / 3 FWD shape, with the single best player given a star boost.
pub fn squad(team: TeamId) -> Vec<Player> {
    let rating = TEAMS.get(team.0 as usize).map(|t| t.3).unwrap_or(1500.0);
    let base = ((rating - 1500.0) / 400.0).clamp(0.1, 1.6);
    let mut rng = StdRng::seed_from_u64(0xF1FA_2026 ^ u64::from(team.0));

    let mut players: Vec<Player> = (0..16)
        .map(|i| {
            let position = match i {
                0..=1 => Position::Gk,
                2..=7 => Position::Def,
                8..=12 => Position::Mid,
                _ => Position::Fwd,
            };
            let (atk_w, def_w) = match position {
                Position::Gk => (0.05, 0.95),
                Position::Def => (0.30, 0.85),
                Position::Mid => (0.65, 0.55),
                Position::Fwd => (0.95, 0.25),
            };
            let skill = (base + rng.gen_range(-0.30..0.30)).max(0.05);
            let initial = (b'A' + rng.gen_range(0..26u8)) as char;
            let surname = SURNAMES[rng.gen_range(0..SURNAMES.len())];
            Player {
                name: format!("{initial}. {surname}"),
                position,
                attack: skill * atk_w,
                defense: skill * def_w,
            }
        })
        .collect();

    // Give the team a clear talisman: boost the strongest outfield player's attack.
    if let Some(star) = players
        .iter_mut()
        .filter(|p| p.position != Position::Gk)
        .max_by(|a, b| {
            a.overall()
                .partial_cmp(&b.overall())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        star.attack += 0.6;
    }
    players
}

/// Indices (into `squad`) of the strongest available XI: the best keeper plus the ten
/// best of the rest by overall contribution.
fn strongest_xi(squad: &[Player]) -> Vec<usize> {
    let mut keepers: Vec<usize> = (0..squad.len())
        .filter(|&i| squad[i].position == Position::Gk)
        .collect();
    keepers.sort_by(|&a, &b| {
        squad[b]
            .overall()
            .partial_cmp(&squad[a].overall())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut outfield: Vec<usize> = (0..squad.len())
        .filter(|&i| squad[i].position != Position::Gk)
        .collect();
    outfield.sort_by(|&a, &b| {
        squad[b]
            .overall()
            .partial_cmp(&squad[a].overall())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut xi: Vec<usize> = keepers.into_iter().take(1).collect();
    xi.extend(outfield.into_iter().take(STARTING_XI - 1));
    xi
}

/// The names of a team's starting XI. With `drop_star`, the strongest outfield player is
/// benched and replaced by the next best, simulating an injury or rotation.
pub fn starting_lineup(team: TeamId, drop_star: bool) -> Vec<String> {
    let squad = squad(team);
    let mut xi = strongest_xi(&squad);
    if drop_star {
        // Drop the best outfield player in the XI; bring on the best player not selected.
        if let Some((pos_in_xi, _)) = xi
            .iter()
            .enumerate()
            .filter(|(_, &i)| squad[i].position != Position::Gk)
            .max_by(|(_, &a), (_, &b)| {
                squad[a]
                    .overall()
                    .partial_cmp(&squad[b].overall())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            let replacement = (0..squad.len())
                .filter(|i| !xi.contains(i) && squad[*i].position != Position::Gk)
                .max_by(|&a, &b| {
                    squad[a]
                        .overall()
                        .partial_cmp(&squad[b].overall())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(rep) = replacement {
                xi[pos_in_xi] = rep;
            }
        }
    }
    xi.into_iter().map(|i| squad[i].name.clone()).collect()
}

/// Scale from raw contribution-shortfall to a log-space goal-rate adjustment.
const LINEUP_SCALE: f64 = 0.4;

/// Turn a confirmed lineup into a `(attack_delta, defense_delta)` adjustment in log space
/// (positive = stronger), by comparing the present XI to the team's strongest XI. A
/// full-strength XI yields ~0; a missing key player yields a negative attack delta. Unknown
/// or empty lineups yield no adjustment, so the live feed degrades gracefully.
pub fn lineup_adjustment(team: TeamId, present: &[String]) -> (f64, f64) {
    if present.is_empty() {
        return (0.0, 0.0);
    }
    let squad = squad(team);
    let present_lower: Vec<String> = present.iter().map(|n| n.to_lowercase()).collect();

    let strongest = strongest_xi(&squad);
    let strongest_atk: f64 = strongest.iter().map(|&i| squad[i].attack).sum();
    let strongest_def: f64 = strongest.iter().map(|&i| squad[i].defense).sum();

    let mut present_atk = 0.0;
    let mut present_def = 0.0;
    let mut matched = 0;
    for p in &squad {
        if present_lower.contains(&p.name.to_lowercase()) {
            present_atk += p.attack;
            present_def += p.defense;
            matched += 1;
        }
    }
    if matched == 0 {
        return (0.0, 0.0); // none of the names are in our (synthetic) squad
    }

    let atk_delta = ((present_atk - strongest_atk) * LINEUP_SCALE).clamp(-0.5, 0.1);
    let def_delta = ((present_def - strongest_def) * LINEUP_SCALE).clamp(-0.5, 0.1);
    (atk_delta, def_delta)
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

/// A historical match for backtesting: the observation to train/score on, plus the
/// bookmaker's implied probabilities when odds are available.
#[derive(Debug, Clone)]
pub struct MatchRecord {
    pub obs: Observation,
    pub market: Option<Probabilities>,
}

/// Analytic win/draw/win probabilities for two independent Poisson goal rates.
fn outcome_probs_from_rates(lambda: f64, mu: f64) -> Probabilities {
    let (mut h, mut d, mut a) = (0.0, 0.0, 0.0);
    for x in 0..10u32 {
        for y in 0..10u32 {
            let p = poisson_pmf(x, lambda) * poisson_pmf(y, mu);
            match x.cmp(&y) {
                std::cmp::Ordering::Greater => h += p,
                std::cmp::Ordering::Equal => d += p,
                std::cmp::Ordering::Less => a += p,
            }
        }
    }
    Probabilities::new(h, d, a)
}

/// Like [`synthetic_history`] but also attaches a synthetic "bookmaker" line per match:
/// the true outcome probabilities with small noise (a sharp, near-optimal book). This
/// lets the backtest show a market baseline fully offline. Supply a real CSV via
/// [`load_results_csv`] for genuine odds.
pub fn synthetic_history_with_market(n_matches: usize, seed: u64) -> Vec<MatchRecord> {
    let strengths = team_strengths();
    let n = strengths.len();
    let mut rng = StdRng::seed_from_u64(seed ^ 0xB00C);
    let sample = |rng: &mut StdRng, lambda: f64| -> u8 {
        Poisson::new(lambda)
            .map(|d| (d.sample(rng) as u32).min(12) as u8)
            .unwrap_or(0)
    };

    (0..n_matches)
        .map(|k| {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            while j == i {
                j = rng.gen_range(0..n);
            }
            let (lambda, mu) = rating_to_xg(strengths[i].1, strengths[j].1);
            let score = Scoreline::new(sample(&mut rng, lambda), sample(&mut rng, mu));
            let age_days = 1100.0 * (1.0 - k as f64 / n_matches as f64);
            let obs = Observation::new(strengths[i].0, strengths[j].0, score, age_days);

            let truth = outcome_probs_from_rates(lambda, mu);
            let mut noisy = |p: f64| p * (1.0 + rng.gen_range(-0.04..0.04));
            let market = Probabilities::new(
                noisy(truth.home_win),
                noisy(truth.draw),
                noisy(truth.away_win),
            );
            MatchRecord {
                obs,
                market: Some(market),
            }
        })
        .collect()
}

/// Load real historical results from a [football-data.co.uk](https://www.football-data.co.uk)
/// style CSV. Required columns: `HomeTeam, AwayTeam, FTHG, FTAG`. Optional Bet365 odds
/// (`B365H, B365D, B365A`) populate the market line. Team names are interned to ids; rows
/// are assumed oldest-first, so match age is taken from row order for the time decay.
pub fn load_results_csv(path: impl AsRef<Path>) -> Result<Vec<MatchRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|e| IngestError::Data(format!("open CSV: {e}")))?;
    let headers = reader
        .headers()
        .map_err(|e| IngestError::Data(e.to_string()))?
        .clone();
    let col = |name: &str| headers.iter().position(|h| h.eq_ignore_ascii_case(name));
    let (Some(ch), Some(ca), Some(chg), Some(cag)) =
        (col("HomeTeam"), col("AwayTeam"), col("FTHG"), col("FTAG"))
    else {
        return Err(IngestError::Data(
            "CSV needs HomeTeam, AwayTeam, FTHG, FTAG columns".into(),
        ));
    };
    let (oh, od, oa) = (col("B365H"), col("B365D"), col("B365A"));

    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut rows: Vec<(TeamId, TeamId, Scoreline, Option<Probabilities>)> = Vec::new();
    for record in reader.records() {
        let r = record.map_err(|e| IngestError::Data(e.to_string()))?;
        let cell = |i: usize| r.get(i).map(str::trim).filter(|s| !s.is_empty());
        let (Some(home), Some(away)) = (cell(ch), cell(ca)) else {
            continue;
        };
        let (Some(hg), Some(ag)) = (
            cell(chg).and_then(|s| s.parse::<u8>().ok()),
            cell(cag).and_then(|s| s.parse::<u8>().ok()),
        ) else {
            continue;
        };
        let odd = |c: Option<usize>| c.and_then(cell).and_then(|s| s.parse::<f64>().ok());
        let market = match (odd(oh), odd(od), odd(oa)) {
            (Some(h), Some(d), Some(a)) => Some(implied_probabilities(h, d, a)),
            _ => None,
        };
        let mut intern = |name: &str| {
            let next = ids.len() as u32;
            TeamId(*ids.entry(name.to_string()).or_insert(next))
        };
        let (h_id, a_id) = (intern(home), intern(away));
        rows.push((h_id, a_id, Scoreline::new(hg, ag), market));
    }

    let total = rows.len();
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(k, (h, a, score, market))| MatchRecord {
            obs: Observation::new(h, a, score, (total - k) as f64),
            market,
        })
        .collect())
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

    #[test]
    fn squads_are_well_formed_and_deterministic() {
        let a = squad(TeamId(0));
        let b = squad(TeamId(0));
        assert!(a.len() >= STARTING_XI, "squad must field at least an XI");
        assert!(
            a.iter().any(|p| p.position == Position::Gk),
            "needs a keeper"
        );
        // Deterministic for a given team.
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].name, b[0].name);
    }

    #[test]
    fn full_strength_lineup_is_neutral_but_dropping_star_hurts() {
        let team = TeamId(0);
        let full = lineup_adjustment(team, &starting_lineup(team, false));
        // Best available XI should be close to no adjustment.
        assert!(
            full.0.abs() < 0.05 && full.1.abs() < 0.05,
            "full XI ~ neutral: {full:?}"
        );

        let weakened = lineup_adjustment(team, &starting_lineup(team, true));
        assert!(
            weakened.0 < -0.01,
            "benching the star should drop the attack delta: {weakened:?}"
        );
    }

    #[test]
    fn unknown_lineup_yields_no_adjustment() {
        let adj = lineup_adjustment(TeamId(0), &["Nobody Real".to_string()]);
        assert_eq!(adj, (0.0, 0.0));
    }

    #[test]
    fn synthetic_market_is_present_and_normalized() {
        let records = synthetic_history_with_market(100, 3);
        assert_eq!(records.len(), 100);
        assert!(records.iter().all(|r| r.market.is_some()));
        let m = records[0].market.unwrap();
        assert!((m.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn loads_football_data_csv_with_odds() {
        let csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n\
                   01/01/2024,Spain,Malta,5,0,1.10,9.00,21.0\n\
                   02/01/2024,Brazil,Spain,2,2,2.50,3.30,2.80\n";
        let path = std::env::temp_dir().join("oracle_csv_loader_test.csv");
        std::fs::write(&path, csv).unwrap();

        let records = load_results_csv(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].obs.score, Scoreline::new(5, 0));
        let market = records[0].market.expect("odds should be parsed");
        assert!((market.sum() - 1.0).abs() < 1e-9, "vig normalized away");
        assert!(
            market.home_win > market.away_win,
            "Spain are heavy favourites"
        );

        let _ = std::fs::remove_file(&path);
    }
}
