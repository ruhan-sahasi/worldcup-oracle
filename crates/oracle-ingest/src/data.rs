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
use chrono::{TimeZone, Timelike, Utc};
use oracle_domain::bracket::{resolve_slot, FIXED_R32};
use oracle_domain::{
    Confederation, Group, Match, MatchId, MatchStatus, Probabilities, Scoreline, Stage, Team,
    TeamId, Tournament,
};
use oracle_model::poisson::poisson_pmf;
use oracle_model::{
    context_adjustment, implied_probabilities, style_adjustment, BradleyTerry, BradleyTerryConfig,
    DixonColesConfig, Ensemble, GoalModel, Host, MatchContext, Observation, StyleProfile,
};
use oracle_numeric::Rng;
use oracle_ratings::{RatingStore, StateSpaceRatings};
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

/// Each team's confederation, for the goal model's hierarchical (confederation-aware) fit.
pub fn confederations() -> HashMap<TeamId, Confederation> {
    TEAMS
        .iter()
        .enumerate()
        .map(|(i, &(_, _, conf, _))| (TeamId(i as u32), conf))
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
    let mut rng = Rng::new(0xF1FA_2026 ^ u64::from(team.0));

    // Names must be unique within a squad: lineups are matched by name, so a duplicate
    // would let one present player count twice in `lineup_adjustment`.
    let mut used = std::collections::HashSet::new();
    let mut players: Vec<Player> = Vec::with_capacity(16);
    for i in 0..16 {
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
        let skill = (base + rng.range(-0.30, 0.30)).max(0.05);
        let name = loop {
            let initial = (b'A' + rng.index_below(26) as u8) as char;
            let surname = SURNAMES[rng.index_below(SURNAMES.len())];
            let candidate = format!("{initial}. {surname}");
            if used.insert(candidate.clone()) {
                break candidate;
            }
        };
        players.push(Player {
            name,
            position,
            attack: skill * atk_w,
            defense: skill * def_w,
        });
    }

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

/// Build the Round-of-32 fixtures once the group stage is complete.
///
/// When every group match in `tournament` is finished (and no knockout fixtures exist yet), the
/// 32 qualifiers are known: the 12 group winners, 12 runners-up, and 8 best third-placed teams.
/// This slots them into the fixed 2026 bracket ([`oracle_domain::bracket::FIXED_R32`]) and returns
/// the 16 Round-of-32 [`Match`]es (status `Scheduled`, ids following the existing fixtures, a
/// kickoff after the last group match). Returns an empty vec if the group stage is unfinished, the
/// tournament is not the 12-group 2026 shape, or knockout fixtures already exist.
///
/// The offline simulation feed plays only the group stage, so the engine calls this when the last
/// group result lands (and the live football-data.org adapter exposes the real bracket directly);
/// it is therefore exercised by the engine path and by tests rather than by the offline feed. The
/// best-third -> slot assignment is the fixed bracket's deterministic rule, not FIFA's full lookup
/// table (see the bracket module).
pub fn materialize_knockout(tournament: &Tournament) -> Vec<Match> {
    if tournament.groups.len() != NUM_GROUPS {
        return Vec::new();
    }
    // Do not duplicate an already-materialized bracket.
    if tournament.matches.iter().any(|m| m.stage.is_knockout()) {
        return Vec::new();
    }
    let group_matches: Vec<&Match> = tournament
        .matches
        .iter()
        .filter(|m| matches!(m.stage, Stage::Group(_)))
        .collect();
    if group_matches.is_empty() || !group_matches.iter().all(|m| m.is_finished()) {
        return Vec::new();
    }

    // Rank each group from its finished matches: (team, points, goal-diff, goals-for).
    let mut winners: Vec<TeamId> = vec![TeamId(0); NUM_GROUPS];
    let mut runners: Vec<TeamId> = vec![TeamId(0); NUM_GROUPS];
    let mut thirds: Vec<(TeamId, i32, i32, i32)> = Vec::new();

    for (gi, group) in tournament.groups.iter().enumerate() {
        let mut table: Vec<(TeamId, i32, i32, i32)> =
            group.teams.iter().map(|&t| (t, 0, 0, 0)).collect();
        let pos: HashMap<TeamId, usize> = group
            .teams
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, i))
            .collect();
        for m in group_matches
            .iter()
            .filter(|m| matches!(m.stage, Stage::Group(c) if c == group.name))
        {
            let (Some(&hi), Some(&ai)) = (pos.get(&m.home), pos.get(&m.away)) else {
                continue;
            };
            let (gh, ga) = (i32::from(m.score.home), i32::from(m.score.away));
            table[hi].2 += gh - ga;
            table[hi].3 += gh;
            table[ai].2 += ga - gh;
            table[ai].3 += ga;
            match gh.cmp(&ga) {
                std::cmp::Ordering::Greater => table[hi].1 += 3,
                std::cmp::Ordering::Less => table[ai].1 += 3,
                std::cmp::Ordering::Equal => {
                    table[hi].1 += 1;
                    table[ai].1 += 1;
                }
            }
        }
        if table.len() < 3 {
            return Vec::new();
        }
        table.sort_by(cmp_group_row);
        winners[gi] = table[0].0;
        runners[gi] = table[1].0;
        thirds.push(table[2]);
    }

    // The eight best third-placed teams, by the same ranking.
    thirds.sort_by(cmp_group_row);
    let qualified_thirds: Vec<TeamId> = thirds.iter().take(8).map(|t| t.0).collect();
    if qualified_thirds.len() < 8 {
        return Vec::new();
    }

    // Ids continue past the existing fixtures; kickoff sits after the last group match.
    let base_id = tournament.matches.iter().map(|m| m.id.0).max().unwrap_or(0);
    let last_kickoff = tournament
        .matches
        .iter()
        .map(|m| m.kickoff)
        .max()
        .unwrap_or_else(|| Utc.with_ymd_and_hms(2026, 6, 30, 16, 0, 0).unwrap());

    FIXED_R32
        .iter()
        .enumerate()
        .map(|(i, (top, bottom))| {
            let home = resolve_slot(top, &winners, &runners, &qualified_thirds);
            let away = resolve_slot(bottom, &winners, &runners, &qualified_thirds);
            Match {
                id: MatchId(base_id + 1 + i as u32),
                home,
                away,
                stage: Stage::RoundOf32,
                kickoff: last_kickoff + chrono::Duration::hours(24 + (i as i64) * 3),
                status: MatchStatus::Scheduled,
                score: Scoreline::new(0, 0),
            }
        })
        .collect()
}

/// Rank two `(team, points, goal-diff, goals-for)` rows best-first, with team id ascending as a
/// deterministic final tie-break (matching the simulator's group ranking).
fn cmp_group_row(a: &(TeamId, i32, i32, i32), b: &(TeamId, i32, i32, i32)) -> std::cmp::Ordering {
    b.1.cmp(&a.1)
        .then(b.2.cmp(&a.2))
        .then(b.3.cmp(&a.3))
        .then(a.0 .0.cmp(&b.0 .0))
}

/// Match-level overdispersion of the synthetic data: a Gamma-Poisson "form on the day" with this
/// negative-binomial size, matching the mild overdispersion of real football. Mean-preserving, so
/// expected goals are unchanged, but the goal-count *spread* is fatter than Poisson - which lets
/// the fitted goal model actually learn a finite dispersion offline.
const SYNTHETIC_DISPERSION: f64 = 8.0;

/// Sample a goal count from a Gamma-Poisson (negative binomial) with the given mean, capped at 12.
fn sample_goals_overdispersed(rng: &mut Rng, lambda: f64) -> u8 {
    let r = SYNTHETIC_DISPERSION;
    let rate = rng.gamma(r, lambda / r).max(1e-9);
    rng.poisson(rate).min(12) as u8
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
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n_matches);

    for k in 0..n_matches {
        let i = rng.index_below(n);
        let mut j = rng.index_below(n);
        while j == i {
            j = rng.index_below(n);
        }
        let (lambda, mu) = rating_to_xg(strengths[i].1, strengths[j].1);
        let score = Scoreline::new(
            sample_goals_overdispersed(&mut rng, lambda),
            sample_goals_overdispersed(&mut rng, mu),
        );
        // xG is a noisy estimate of the true rate, but much less noisy than the
        // Poisson-sampled scoreline, so fitting on it sharpens the model.
        let home_xg = (lambda * (1.0 + rng.range(-0.15, 0.15))).max(0.05);
        let away_xg = (mu * (1.0 + rng.range(-0.15, 0.15))).max(0.05);
        // Spread matches over ~1100 days, oldest first.
        let age_days = 1100.0 * (1.0 - k as f64 / n_matches as f64);
        out.push(Observation::with_xg(
            strengths[i].0,
            strengths[j].0,
            score,
            home_xg,
            away_xg,
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

/// A synthetic bookmaker line (decimal odds) for a matchup, derived from the teams'
/// strength ratings with a ~6% margin. Used by the simulation feed to emit an `Odds`
/// event so the engine can demonstrate anchoring to the market.
pub fn market_line(home: TeamId, away: TeamId) -> (f64, f64, f64) {
    let strengths = team_strengths();
    let rating = |t: TeamId| {
        strengths
            .iter()
            .find(|(id, _)| *id == t)
            .map_or(1500.0, |(_, r)| *r)
    };
    let (lambda, mu) = rating_to_xg(rating(home), rating(away));
    let p = outcome_probs_from_rates(lambda, mu);
    const MARGIN: f64 = 1.06;
    let odds = |q: f64| (1.0 / (MARGIN * q.max(1e-6))).max(1.01);
    (odds(p.home_win), odds(p.draw), odds(p.away_win))
}

/// Like [`synthetic_history`] but also attaches a synthetic "bookmaker" line per match:
/// the true outcome probabilities with small noise (a sharp, near-optimal book). This
/// lets the backtest show a market baseline fully offline. Supply a real CSV via
/// [`load_results_csv`] for genuine odds.
pub fn synthetic_history_with_market(n_matches: usize, seed: u64) -> Vec<MatchRecord> {
    let strengths = team_strengths();
    let n = strengths.len();
    let mut rng = Rng::new(seed ^ 0xB00C);

    (0..n_matches)
        .map(|k| {
            let i = rng.index_below(n);
            let mut j = rng.index_below(n);
            while j == i {
                j = rng.index_below(n);
            }
            let (lambda, mu) = rating_to_xg(strengths[i].1, strengths[j].1);
            let score = Scoreline::new(
                sample_goals_overdispersed(&mut rng, lambda),
                sample_goals_overdispersed(&mut rng, mu),
            );
            let home_xg = (lambda * (1.0 + rng.range(-0.15, 0.15))).max(0.05);
            let away_xg = (mu * (1.0 + rng.range(-0.15, 0.15))).max(0.05);
            let age_days = 1100.0 * (1.0 - k as f64 / n_matches as f64);
            let obs = Observation::with_xg(
                strengths[i].0,
                strengths[j].0,
                score,
                home_xg,
                away_xg,
                age_days,
            );

            let truth = outcome_probs_from_rates(lambda, mu);
            let mut noisy = |p: f64| p * (1.0 + rng.range(-0.04, 0.04));
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
    // Optional expected-goals columns (no universal standard, so try common spellings).
    let cxgh = col("HxG")
        .or_else(|| col("Home xG"))
        .or_else(|| col("xG_home"));
    let cxga = col("AxG")
        .or_else(|| col("Away xG"))
        .or_else(|| col("xG_away"));

    let mut ids: HashMap<String, u32> = HashMap::new();
    #[allow(clippy::type_complexity)]
    let mut rows: Vec<(
        TeamId,
        TeamId,
        Scoreline,
        Option<(f64, f64)>,
        Option<Probabilities>,
    )> = Vec::new();
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
        let num = |c: Option<usize>| c.and_then(cell).and_then(|s| s.parse::<f64>().ok());
        let market = match (num(oh), num(od), num(oa)) {
            (Some(h), Some(d), Some(a)) => Some(implied_probabilities(h, d, a)),
            _ => None,
        };
        let xg = match (num(cxgh), num(cxga)) {
            (Some(h), Some(a)) => Some((h, a)),
            _ => None,
        };
        let mut intern = |name: &str| {
            let next = ids.len() as u32;
            TeamId(*ids.entry(name.to_string()).or_insert(next))
        };
        let (h_id, a_id) = (intern(home), intern(away));
        rows.push((h_id, a_id, Scoreline::new(hg, ag), xg, market));
    }

    let total = rows.len();
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(k, (h, a, score, xg, market))| {
            // Oldest-first assumption: earlier rows get a larger age.
            let age = (total - k) as f64;
            let obs = match xg {
                Some((hx, ax)) => Observation::with_xg(h, a, score, hx, ax, age),
                None => Observation::new(h, a, score, age),
            };
            MatchRecord { obs, market }
        })
        .collect())
}

// ----- Venue & travel context (representative, offline) -----
//
// Real 2026 venue assignments come from the official schedule; this representative map lets
// the host/altitude/rest feature work offline. Rest days are derived from the real gaps
// between a team's fixtures, so that part is genuine.

/// A 2026 host venue. `utc_offset` is the June/July 2026 offset: US/Canada observe daylight time,
/// while Mexico does not (it abolished DST nationally in 2022), so its venues sit at UTC-6 year
/// round. `summer_high_c` is a typical June/July afternoon high, used with the kickoff hour to
/// model match-time heat.
struct Venue {
    country: &'static str,
    altitude_m: f64,
    lat: f64,
    lon: f64,
    utc_offset: f64,
    summer_high_c: f64,
}

const VENUES: &[Venue] = &[
    Venue {
        // Mexico City
        country: "MEX",
        altitude_m: 2240.0,
        lat: 19.43,
        lon: -99.13,
        utc_offset: -6.0,
        summer_high_c: 24.0,
    },
    Venue {
        // Guadalajara
        country: "MEX",
        altitude_m: 1566.0,
        lat: 20.67,
        lon: -103.35,
        utc_offset: -6.0,
        summer_high_c: 30.0,
    },
    Venue {
        // Monterrey
        country: "MEX",
        altitude_m: 540.0,
        lat: 25.69,
        lon: -100.32,
        utc_offset: -6.0,
        summer_high_c: 35.0,
    },
    Venue {
        // Denver
        country: "USA",
        altitude_m: 1609.0,
        lat: 39.74,
        lon: -104.99,
        utc_offset: -6.0,
        summer_high_c: 31.0,
    },
    Venue {
        // Atlanta
        country: "USA",
        altitude_m: 320.0,
        lat: 33.75,
        lon: -84.39,
        utc_offset: -4.0,
        summer_high_c: 31.0,
    },
    Venue {
        // Dallas
        country: "USA",
        altitude_m: 130.0,
        lat: 32.78,
        lon: -96.80,
        utc_offset: -5.0,
        summer_high_c: 36.0,
    },
    Venue {
        // Kansas City
        country: "USA",
        altitude_m: 270.0,
        lat: 39.10,
        lon: -94.58,
        utc_offset: -5.0,
        summer_high_c: 32.0,
    },
    Venue {
        // Los Angeles
        country: "USA",
        altitude_m: 93.0,
        lat: 34.05,
        lon: -118.24,
        utc_offset: -7.0,
        summer_high_c: 28.0,
    },
    Venue {
        // New York
        country: "USA",
        altitude_m: 10.0,
        lat: 40.71,
        lon: -74.01,
        utc_offset: -4.0,
        summer_high_c: 29.0,
    },
    Venue {
        // Miami
        country: "USA",
        altitude_m: 2.0,
        lat: 25.76,
        lon: -80.19,
        utc_offset: -4.0,
        summer_high_c: 32.0,
    },
    Venue {
        // Toronto
        country: "CAN",
        altitude_m: 76.0,
        lat: 43.65,
        lon: -79.38,
        utc_offset: -4.0,
        summer_high_c: 27.0,
    },
    Venue {
        // Vancouver
        country: "CAN",
        altitude_m: 4.0,
        lat: 49.28,
        lon: -123.12,
        utc_offset: -7.0,
        summer_high_c: 22.0,
    },
];

/// Great-circle distance between two `(lat, lon)` points in kilometres (haversine).
fn haversine_km(a: &Venue, b: &Venue) -> f64 {
    let r = 6371.0_f64;
    let (lat1, lat2) = (a.lat.to_radians(), b.lat.to_radians());
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * h.sqrt().asin()
}

/// How much of the crowd in a venue in `venue_country` is expected to back `team`, in `[0, 1]`.
/// A literal host on home soil packs the stadium; Mexico draws near-home crowds across US venues;
/// otherwise it is a confederation-level diaspora / traveling-fan pull, since 2026 sits in a
/// region with very large Latin American and sizable European communities. Synthetic but reasoned
/// (no real attendance data is bundled).
fn crowd_pull(team: TeamId, venue_country: &str) -> f64 {
    let Some(&(_, code, conf, _)) = TEAMS.get(team.0 as usize) else {
        return 0.0;
    };
    if code == venue_country {
        return 1.0; // literal host, at home
    }
    let host_cross: f64 = match (code, venue_country) {
        ("MEX", "USA") => 0.85,
        ("USA", "CAN") | ("CAN", "USA") => 0.5,
        ("MEX", "CAN") | ("CAN", "MEX") | ("USA", "MEX") => 0.4,
        _ => 0.0,
    };
    let diaspora: f64 = match conf {
        Conmebol => 0.6,
        Concacaf => 0.55,
        Uefa => 0.35,
        Caf => 0.3,
        Afc => 0.3,
        Ofc => 0.15,
    };
    host_cross.max(diaspora)
}

/// Match-time temperature in Celsius: the venue's summer afternoon high, cooled toward mornings
/// and evenings by distance from the ~16:00 local peak. Drives the model's heat tempo suppression.
fn match_temperature(venue: &Venue, kickoff: chrono::DateTime<Utc>) -> f64 {
    let local_hour = (f64::from(kickoff.hour()) + venue.utc_offset).rem_euclid(24.0);
    let cooling = ((local_hour - 16.0).abs() * 1.3).min(12.0);
    venue.summer_high_c - cooling
}

/// A team's synthetic playing-style embedding: confederations sit at different angles on the
/// style circle (regional football cultures), with deterministic per-team jitter so teams within
/// a confederation still differ. On real data these would be fit from match residuals; here they
/// are reasoned-synthetic, like the squads and crowd model.
fn style_profile(team: TeamId) -> StyleProfile {
    let Some(&(_, _, conf, _)) = TEAMS.get(team.0 as usize) else {
        return StyleProfile::neutral();
    };
    let conf_index: f64 = match conf {
        Conmebol => 0.0,
        Uefa => 1.0,
        Caf => 2.0,
        Afc => 3.0,
        Concacaf => 4.0,
        Ofc => 5.0,
    };
    let conf_angle = conf_index * std::f64::consts::TAU / 6.0;
    let mut rng = Rng::new(0x5713_2026 ^ u64::from(team.0));
    let theta = conf_angle + rng.range(-0.6, 0.6);
    StyleProfile::new([theta.cos(), theta.sin()])
}

/// The style-matchup tilt for a matchup: the home side's attacking log-rate advantage from the
/// antisymmetric bilinear "rock-paper-scissors" style edge. Positive favours the home side.
pub fn matchup_style_tilt(home: TeamId, away: TeamId) -> f64 {
    style_adjustment(&style_profile(home), &style_profile(away))
        .0
         .0
}

/// Per-match style matchup adjustments (the bilinear style tilt from `oracle-model`), keyed by
/// match id, in the same log-space `((home_atk, home_def), (away_atk, away_def))` shape as venue.
pub fn style_adjustments(tournament: &Tournament) -> HashMap<MatchId, VenueAdj> {
    tournament
        .matches
        .iter()
        .map(|m| {
            let adj = style_adjustment(&style_profile(m.home), &style_profile(m.away));
            (m.id, adj)
        })
        .collect()
}

/// Synthetic per-team **penalty-shootout skill** (roughly a z-score, mean ~0; positive = better).
/// Shootout outcomes carry a persistent component - kicking technique, the goalkeeper, composure -
/// that open-play strength does not capture, so most models' 50/50 shootout leaves signal on the
/// table. Reasoned-synthetic (on real data: historical shootout conversion and save rate).
pub fn shootout_ratings() -> HashMap<TeamId, f64> {
    (0..TEAMS.len() as u32)
        .map(|i| {
            let mut rng = Rng::new(0x5400_7E07 ^ u64::from(i));
            (TeamId(i), rng.range(-1.0, 1.0))
        })
        .collect()
}

/// Synthetic per-team **knockout pedigree**: how well a side handles single-elimination pressure.
/// It tracks strength only mildly (success breeds experience) but carries substantial independent
/// variation, since temperament is not open-play quality - and 2026's 48-team field brings more
/// debutants than ever. Reasoned-synthetic (on real data: tournament knockout history / debutant
/// status). Applied only to knockout ties, an effect additive strength (which acts everywhere)
/// cannot represent.
pub fn knockout_pedigree() -> HashMap<TeamId, f64> {
    TEAMS
        .iter()
        .enumerate()
        .map(|(i, &(_, _, _, rating))| {
            let strength_part = ((rating - 1800.0) / 350.0).clamp(-1.0, 1.0);
            let mut rng = Rng::new(0x9ED1_6EE5 ^ i as u64);
            let independent = rng.range(-0.8, 0.8);
            (
                TeamId(i as u32),
                (0.4 * strength_part + independent).clamp(-1.2, 1.2),
            )
        })
        .collect()
}

/// Which of the toggleable per-match context/style signals to include when building matchup
/// adjustments. All-true (the [`Default`]) reproduces the full model; flipping one off drops that
/// signal, which is how the `sensitivity` analysis isolates each one's effect. The structural
/// terms (host advantage, altitude, rest) are always applied and are not part of the mask.
#[derive(Clone, Copy, Debug)]
pub struct SignalMask {
    pub crowd: bool,
    pub travel: bool,
    pub heat: bool,
    pub style: bool,
}

impl Default for SignalMask {
    fn default() -> Self {
        Self {
            crowd: true,
            travel: true,
            heat: true,
            style: true,
        }
    }
}

/// Every per-match log-space adjustment that applies to a fixture, with a chosen subset of the
/// toggleable signals enabled. A disabled context signal is zeroed at its [`MatchContext`] source
/// (so the rest of the context still composes correctly); style is simply not added. An all-true
/// mask is identical to [`matchup_adjustments`].
pub fn matchup_adjustments_masked(
    tournament: &Tournament,
    mask: SignalMask,
) -> HashMap<MatchId, VenueAdj> {
    let mut adj: HashMap<MatchId, VenueAdj> = match_contexts(tournament)
        .into_iter()
        .map(|(id, mut ctx)| {
            if !mask.crowd {
                ctx.crowd_support = 0.0;
            }
            if !mask.travel {
                ctx.home_travel_km = 0.0;
                ctx.away_travel_km = 0.0;
                ctx.home_tz_shift = 0.0;
                ctx.away_tz_shift = 0.0;
            }
            if !mask.heat {
                ctx.temperature_c = 0.0;
            }
            (id, context_adjustment(&ctx))
        })
        .collect();
    if mask.style {
        for (id, ((sha, shd), (saa, sad))) in style_adjustments(tournament) {
            let e = adj.entry(id).or_default();
            e.0 .0 += sha;
            e.0 .1 += shd;
            e.1 .0 += saa;
            e.1 .1 += sad;
        }
    }
    adj
}

/// Every per-match log-space adjustment that applies to a fixture: venue/crowd/travel context
/// plus the style matchup, summed componentwise. This is what the engine and CLI feed to the
/// Monte-Carlo (and the engine's single-match prediction), so all context reaches the forecast.
pub fn matchup_adjustments(tournament: &Tournament) -> HashMap<MatchId, VenueAdj> {
    matchup_adjustments_masked(tournament, SignalMask::default())
}

/// A per-team venue/travel adjustment: `((home_attack, home_defense), (away_attack,
/// away_defense))` in log space.
pub type VenueAdj = ((f64, f64), (f64, f64));

/// Host-nation country code for a team, or `None` if it is not a 2026 host.
fn host_country(id: TeamId) -> Option<&'static str> {
    match TEAMS.get(id.0 as usize).map(|t| t.1) {
        Some(code @ ("USA" | "MEX" | "CAN")) => Some(code),
        _ => None,
    }
}

/// Per-match venue/travel context for the tournament: a representative venue assignment, rest
/// days, the continent-spanning travel/time-zone load between each team's fixtures, and the
/// expected crowd partisanship.
pub fn match_contexts(tournament: &Tournament) -> HashMap<MatchId, MatchContext> {
    // Assign a representative venue (index into VENUES) to each match. Hosts play in their own
    // country; other matches round-robin across all venues.
    let venue_of = |m: &Match| -> usize {
        match host_country(m.home).or_else(|| host_country(m.away)) {
            Some(c) => {
                let in_country: Vec<usize> = VENUES
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.country == c)
                    .map(|(i, _)| i)
                    .collect();
                in_country[(m.id.0 as usize) % in_country.len().max(1)]
            }
            None => (m.id.0 as usize) % VENUES.len(),
        }
    };
    let venue_ix: HashMap<MatchId, usize> = tournament
        .matches
        .iter()
        .map(|m| (m.id, venue_of(m)))
        .collect();

    // Per team, the ordered (by kickoff) sequence of fixtures, to derive rest days and the
    // travel distance + time-zone shift between consecutive venues.
    let mut by_team: HashMap<TeamId, Vec<(MatchId, chrono::DateTime<Utc>, usize)>> = HashMap::new();
    for m in &tournament.matches {
        let v = venue_ix[&m.id];
        by_team
            .entry(m.home)
            .or_default()
            .push((m.id, m.kickoff, v));
        by_team
            .entry(m.away)
            .or_default()
            .push((m.id, m.kickoff, v));
    }
    // (team, match) -> (rest_days, travel_km, tz_shift).
    let mut load: HashMap<(TeamId, MatchId), (u8, f64, f64)> = HashMap::new();
    for (team, mut fixtures) in by_team {
        fixtures.sort_by_key(|(_, k, _)| *k);
        let mut prev: Option<(chrono::DateTime<Utc>, usize)> = None;
        for (mid, k, v) in fixtures {
            let entry = match prev {
                Some((pk, pv)) => {
                    let days = (k - pk).num_days().clamp(1, 14) as u8;
                    let km = haversine_km(&VENUES[pv], &VENUES[v]);
                    // Eastward (offset increases) is the harder, phase-advancing direction.
                    let tz = VENUES[v].utc_offset - VENUES[pv].utc_offset;
                    (days, km, tz)
                }
                // Before the first match a side has arrived early and acclimatized.
                None => (4, 0.0, 0.0),
            };
            load.insert((team, mid), entry);
            prev = Some((k, v));
        }
    }

    tournament
        .matches
        .iter()
        .map(|m| {
            let v = &VENUES[venue_ix[&m.id]];
            let host = if host_country(m.home) == Some(v.country) {
                Host::HomeTeam
            } else if host_country(m.away) == Some(v.country) {
                Host::AwayTeam
            } else {
                Host::Neutral
            };
            let (home_rest, home_km, home_tz) =
                load.get(&(m.home, m.id)).copied().unwrap_or((4, 0.0, 0.0));
            let (away_rest, away_km, away_tz) =
                load.get(&(m.away, m.id)).copied().unwrap_or((4, 0.0, 0.0));
            let ctx = MatchContext {
                host,
                altitude_m: v.altitude_m,
                home_rest_days: home_rest,
                away_rest_days: away_rest,
                crowd_support: crowd_pull(m.home, v.country) - crowd_pull(m.away, v.country),
                home_travel_km: home_km,
                away_travel_km: away_km,
                home_tz_shift: home_tz,
                away_tz_shift: away_tz,
                temperature_c: match_temperature(v, m.kickoff),
            };
            (m.id, ctx)
        })
        .collect()
}

/// Per-match venue/travel adjustments: `match_contexts` mapped through the model's
/// `context_adjustment`. Each value is `((home_atk, home_def), (away_atk, away_def))`.
pub fn venue_adjustments(tournament: &Tournament) -> HashMap<MatchId, VenueAdj> {
    match_contexts(tournament)
        .into_iter()
        .map(|(id, ctx)| (id, context_adjustment(&ctx)))
        .collect()
}

/// The offline-fitted baseline: a goal model, Elo seeds, a trained state-space rating, and a
/// **learned** ensemble over all of them.
pub struct Baseline {
    pub model: GoalModel,
    pub elo_seeds: Vec<(TeamId, f64)>,
    pub state_space: StateSpaceRatings,
    pub ensemble: Ensemble,
}

/// Fit the full baseline on synthetic history via **out-of-fold stacking**. The four ensemble
/// members `[Dixon-Coles (confederation-aware, on xG), Elo, State-space, Market]` are evaluated
/// out-of-fold (members fit on K-1 folds, predicting the held-out fold) so the ensemble weights +
/// temperature are learned on leakage-free predictions over the whole dataset; the members are
/// then refit on all the data for deployment. When a match has odds the engine anchors to the
/// market, and degrades gracefully to the three model members when it does not.
pub fn fit_baseline(seed: u64) -> Baseline {
    let mut history = synthetic_history_with_market(4000, seed);
    // Oldest first (chronological).
    history.sort_by(|a, b| {
        b.obs
            .age_days
            .partial_cmp(&a.obs.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let confs = confederations();
    let elo_seeds = team_strengths();

    // Fit the three learnable members (Dixon-Coles, Elo, state-space) on a set of records.
    // Confederation-aware: hierarchical pooling shrinks each team toward its confederation level
    // rather than the global mean, the principled treatment for a sparse, unbalanced field.
    let fit_members = |records: &[MatchRecord]| -> (GoalModel, RatingStore, StateSpaceRatings) {
        let obs: Vec<Observation> = records.iter().map(|r| r.obs).collect();
        let model = GoalModel::fit_with_confederations(&obs, DixonColesConfig::default(), &confs);
        let mut ratings = RatingStore::with_defaults();
        for &(team, rating) in &elo_seeds {
            ratings.seed(team, rating);
        }
        let mut state_space = StateSpaceRatings::with_defaults();
        for r in records {
            ratings.record(r.obs.home, r.obs.away, r.obs.score, true);
            state_space.observe(r.obs.home, r.obs.away, r.obs.score, r.obs.age_days, true);
        }
        (model, ratings, state_space)
    };
    let member_preds =
        |r: &MatchRecord, m: &GoalModel, rt: &RatingStore, ss: &StateSpaceRatings| {
            vec![
                m.outcome_probabilities(r.obs.home, r.obs.away, true),
                rt.win_probabilities(r.obs.home, r.obs.away, true),
                ss.win_probabilities(r.obs.home, r.obs.away, true),
                r.market.unwrap_or_else(Probabilities::uniform),
            ]
        };

    // Out-of-fold **stacking**: with K interleaved folds, each fold's members are fit on the other
    // folds and predict the held-out fold, so the ensemble weights are learned on leakage-free
    // predictions covering the *whole* dataset (not just a held-out tail). Members are [Dixon-Coles,
    // Elo, State-space, Market]; the market needs no fitting so it is the same in or out of fold.
    const K: usize = 5;
    let mut oof_preds = Vec::with_capacity(history.len());
    let mut oof_actuals = Vec::with_capacity(history.len());
    for fold in 0..K {
        let train_subset: Vec<MatchRecord> = history
            .iter()
            .enumerate()
            .filter(|(i, _)| i % K != fold)
            .map(|(_, r)| r.clone())
            .collect();
        let (m, rt, ss) = fit_members(&train_subset);
        for r in history.iter().skip(fold).step_by(K) {
            oof_preds.push(member_preds(r, &m, &rt, &ss));
            oof_actuals.push(r.obs.score.outcome());
        }
    }
    let ensemble = Ensemble::fit(&oof_preds, &oof_actuals, 4);

    // Final members fit on *all* the data, for deployment.
    let (model, _ratings, state_space) = fit_members(&history);

    Baseline {
        model,
        elo_seeds,
        state_space,
        ensemble,
    }
}

/// Convenience: just the goal model from [`fit_baseline`].
pub fn fit_baseline_model(seed: u64) -> GoalModel {
    fit_baseline(seed).model
}

/// Fit the **Bradley-Terry-Davidson** model (the second, outcome-based forecaster) on the same
/// synthetic history. Time-weighted, so real tournament results replayed into it later dominate.
pub fn fit_bradley_terry(seed: u64) -> BradleyTerry {
    let obs: Vec<Observation> = synthetic_history_with_market(4000, seed)
        .into_iter()
        .map(|r| r.obs)
        .collect();
    BradleyTerry::fit(&obs, BradleyTerryConfig::default())
}

/// A goal model fit on the same synthetic history but **without confederation pooling**: plain
/// global ridge shrinkage toward the overall mean and no cross-confederation offset (an empty
/// confederations map reduces exactly to [`GoalModel::fit`]). Used by the `sensitivity` analysis
/// to isolate how much hierarchical pooling reshapes the title picture.
pub fn fit_model_unpooled(seed: u64) -> GoalModel {
    let obs: Vec<Observation> = synthetic_history_with_market(4000, seed)
        .into_iter()
        .map(|r| r.obs)
        .collect();
    GoalModel::fit(&obs, DixonColesConfig::default())
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
    fn full_signal_mask_matches_unmasked_and_disabling_changes_adjustments() {
        let t = world_cup_2026();
        let full = matchup_adjustments(&t);
        let masked_full = matchup_adjustments_masked(&t, SignalMask::default());
        assert_eq!(full.len(), masked_full.len());
        for (id, v) in &full {
            let m = masked_full.get(id).expect("same fixtures");
            assert!((v.0 .0 - m.0 .0).abs() < 1e-12 && (v.1 .1 - m.1 .1).abs() < 1e-12);
        }
        // Turning off the style matchup must change at least one fixture's adjustment.
        let no_style = matchup_adjustments_masked(
            &t,
            SignalMask {
                style: false,
                ..Default::default()
            },
        );
        let changed = full
            .iter()
            .any(|(id, v)| (v.0 .0 - no_style.get(id).unwrap().0 .0).abs() > 1e-9);
        assert!(changed, "disabling style should move some adjustment");
    }

    #[test]
    fn unpooled_fit_still_ranks_argentina_over_new_zealand() {
        let model = fit_model_unpooled(1);
        let p = model.outcome_probabilities(TeamId(0), TeamId(47), true);
        assert!(
            p.home_win > 0.6,
            "even unpooled, the fit should rate Argentina well above NZ"
        );
    }

    #[test]
    fn baseline_is_confederation_aware() {
        // The offline baseline uses the hierarchical (confederation-aware) fit, so it reports a
        // strength level per confederation, and the strong ones outrank the weak ones.
        let model = fit_baseline_model(7);
        let levels = model.confederation_levels();
        assert_eq!(levels.len(), 6, "all six confederations represented");
        // CONMEBOL (Argentina, Brazil, ...) should sit well above OFC (New Zealand).
        assert!(
            levels[&Confederation::Conmebol] > levels[&Confederation::Ofc],
            "CONMEBOL level {:.3} should exceed OFC {:.3}",
            levels[&Confederation::Conmebol],
            levels[&Confederation::Ofc]
        );
    }

    #[test]
    fn baseline_model_learns_overdispersion_offline() {
        // The synthetic history carries mild Gamma-Poisson overdispersion, so the fit should
        // recover a finite negative-binomial dispersion rather than defaulting to Poisson.
        let model = fit_baseline_model(7);
        assert!(
            model.dispersion() > 0.0,
            "offline baseline should learn overdispersion, got {}",
            model.dispersion()
        );
    }

    #[test]
    fn squad_names_are_unique() {
        for id in [0u32, 7, 23, 47] {
            let mut seen = std::collections::HashSet::new();
            for p in squad(TeamId(id)) {
                assert!(
                    seen.insert(p.name.clone()),
                    "duplicate name {} in team {id}",
                    p.name
                );
            }
        }
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
    fn venue_adjustments_cover_all_matches_and_are_bounded() {
        let t = world_cup_2026();
        let adj = venue_adjustments(&t);
        assert_eq!(adj.len(), t.matches.len());
        for &((ha, hd), (aa, ad)) in adj.values() {
            for v in [ha, hd, aa, ad] {
                assert!(
                    v.is_finite() && v.abs() <= 0.5,
                    "adjustment out of range: {v}"
                );
            }
        }
        // At least the host nations exist in the field, so some context is non-neutral.
        let contexts = match_contexts(&t);
        let hosted = contexts
            .values()
            .filter(|c| c.host != Host::Neutral)
            .count();
        assert!(hosted > 0, "expected some host-nation matches");
    }

    #[test]
    fn crowd_pull_ranks_host_above_diaspora_above_distant() {
        let id = |code: &str| TeamId(TEAMS.iter().position(|t| t.1 == code).unwrap() as u32);
        // A literal host on home soil packs the stadium.
        assert!((crowd_pull(id("MEX"), "MEX") - 1.0).abs() < 1e-9);
        // Mexico still draws a near-home crowd across US venues, far more than a distant side.
        assert!(crowd_pull(id("MEX"), "USA") > crowd_pull(id("JPN"), "USA"));
        // Latin American diaspora pull beats an OFC minnow's.
        assert!(crowd_pull(id("ARG"), "USA") > crowd_pull(id("NZL"), "USA"));
    }

    #[test]
    fn contexts_carry_crowd_and_travel_signals() {
        let t = world_cup_2026();
        let ctx = match_contexts(&t);
        assert_eq!(ctx.len(), t.matches.len());

        // Crowd partisanship is a difference of two pulls in [0, 1], so within [-1, 1], and at
        // least some matches are clearly partisan (a host on home soil).
        assert!(ctx.values().all(|c| c.crowd_support.abs() <= 1.0 + 1e-9));
        assert!(
            ctx.values().any(|c| c.crowd_support.abs() > 0.5),
            "expected some strongly partisan matches"
        );

        // Teams travel between fixtures, and continent-spanning trips shift time zones.
        assert!(
            ctx.values()
                .any(|c| c.home_travel_km > 0.0 || c.away_travel_km > 0.0),
            "expected some inter-venue travel"
        );
        assert!(
            ctx.values()
                .any(|c| c.home_tz_shift.abs() > 0.0 || c.away_tz_shift.abs() > 0.0),
            "expected some time-zone shifts"
        );
    }

    #[test]
    fn haversine_matches_known_distance() {
        // New York (idx 8) to Los Angeles (idx 7) is ~3,940 km.
        let km = haversine_km(&VENUES[8], &VENUES[7]);
        assert!((km - 3940.0).abs() < 150.0, "NY-LA distance was {km:.0} km");
    }

    #[test]
    fn match_temperature_peaks_in_the_afternoon_and_tracks_the_venue() {
        let dallas = &VENUES[5]; // hot: 36C high, UTC-5
        let vancouver = &VENUES[11]; // mild: 22C high, UTC-7
                                     // 21:00 UTC = 16:00 local in Dallas -> at the afternoon peak.
        let mid = chrono::DateTime::from_timestamp(0, 0)
            .unwrap()
            .with_hour(21)
            .unwrap();
        let hot = match_temperature(dallas, mid);
        assert!(
            (hot - dallas.summer_high_c).abs() < 0.5,
            "afternoon peak ~ high"
        );
        // A late-evening kickoff is cooler than the afternoon one at the same venue.
        let late = mid.with_hour(3).unwrap(); // 22:00 local previous day
        assert!(match_temperature(dallas, late) < hot, "evening is cooler");
        // The mild venue is cooler than the hot one at the same instant.
        assert!(match_temperature(vancouver, mid) < hot);
    }

    #[test]
    fn style_profiles_are_unit_vectors_and_tilt_is_antisymmetric() {
        let id = |code: &str| TeamId(TEAMS.iter().position(|t| t.1 == code).unwrap() as u32);
        for code in ["ARG", "FRA", "JPN", "MAR", "NZL"] {
            let s = style_profile(id(code));
            let norm = (s.axes[0].powi(2) + s.axes[1].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "{code} style is a unit vector");
        }
        // The matchup adjustment is antisymmetric: home's tilt is the negative of away's.
        let adj =
            oracle_model::style_adjustment(&style_profile(id("ARG")), &style_profile(id("JPN")));
        assert!((adj.0 .0 + adj.1 .0).abs() < 1e-12);
    }

    #[test]
    fn knockout_factors_cover_every_team_in_range_and_are_deterministic() {
        let so = shootout_ratings();
        let ped = knockout_pedigree();
        assert_eq!(so.len(), 48);
        assert_eq!(ped.len(), 48);
        assert!(so.values().all(|v| v.abs() <= 1.0 + 1e-9));
        assert!(ped.values().all(|v| v.abs() <= 1.2 + 1e-9));
        // Deterministic (reproducible across calls).
        assert_eq!(so, shootout_ratings());
        assert_eq!(ped, knockout_pedigree());
    }

    #[test]
    fn matchup_adjustments_add_style_on_top_of_venue() {
        let t = world_cup_2026();
        let venue = venue_adjustments(&t);
        let full = matchup_adjustments(&t);
        assert_eq!(full.len(), t.matches.len());
        // For at least one match the combined adjustment differs from venue alone (style added).
        let changed = t
            .matches
            .iter()
            .any(|m| full[&m.id].0 .0 != venue[&m.id].0 .0);
        assert!(
            changed,
            "style should perturb some matches beyond venue context"
        );
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

    /// Finish every group match with a deterministic score so the qualifiers are well-defined:
    /// the lower team-id (stronger seed) wins, by a margin that separates the table.
    fn play_out_groups(t: &mut Tournament) {
        for m in &mut t.matches {
            if matches!(m.stage, Stage::Group(_)) {
                let (h, a) = (m.home.0, m.away.0);
                let (gh, ga) = if h < a { (2, 0) } else { (0, 2) };
                m.score = Scoreline::new(gh, ga);
                m.status = MatchStatus::Finished;
            }
        }
    }

    #[test]
    fn materialize_knockout_is_empty_until_groups_finish() {
        let t = world_cup_2026();
        assert!(
            materialize_knockout(&t).is_empty(),
            "no bracket while group matches are unplayed"
        );
    }

    #[test]
    fn materialize_knockout_builds_a_valid_round_of_32() {
        let mut t = world_cup_2026();
        play_out_groups(&mut t);
        let ko = materialize_knockout(&t);

        assert_eq!(ko.len(), 16, "16 Round-of-32 fixtures");
        assert!(ko.iter().all(|m| m.stage == Stage::RoundOf32));
        assert!(ko.iter().all(|m| m.status == MatchStatus::Scheduled));

        // 32 distinct qualifiers, none facing itself.
        let mut teams = std::collections::HashSet::new();
        for m in &ko {
            assert_ne!(m.home, m.away);
            teams.insert(m.home);
            teams.insert(m.away);
        }
        assert_eq!(teams.len(), 32, "32 distinct qualifiers");

        // Ids are unique and do not collide with the existing group fixtures.
        let group_ids: std::collections::HashSet<u32> = t.matches.iter().map(|m| m.id.0).collect();
        let mut ko_ids = std::collections::HashSet::new();
        for m in &ko {
            assert!(!group_ids.contains(&m.id.0), "ko id reuses a group id");
            assert!(ko_ids.insert(m.id.0), "duplicate ko id");
        }

        // Calling again once the bracket exists must not duplicate it.
        t.matches.extend(ko);
        assert!(
            materialize_knockout(&t).is_empty(),
            "should not re-materialize an existing bracket"
        );
    }

    #[test]
    fn materialized_qualifiers_are_real_group_finishers() {
        let mut t = world_cup_2026();
        play_out_groups(&mut t);
        let ko = materialize_knockout(&t);
        let qualified: std::collections::HashSet<TeamId> =
            ko.iter().flat_map(|m| [m.home, m.away]).collect();
        // With the stronger seed always winning, every group's bottom team (the highest id in
        // each group) is eliminated, so at least the 12 group last-placed teams are absent.
        let eliminated = 48 - qualified.len();
        assert_eq!(eliminated, 16, "48 teams minus 32 qualifiers");
    }
}
