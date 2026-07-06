//! # oracle-sim
//!
//! A parallel **Monte-Carlo tournament simulator**. Given the current tournament
//! state (finished results stay fixed, everything else is uncertain) and a
//! [`MatchSampler`] that supplies expected goals for any matchup, it plays the rest
//! of the competition out tens of thousands of times and aggregates how often each
//! team reaches each stage - the headline "champion odds".
//!
//! ## Design notes
//! - **Embarrassingly parallel.** Each iteration is independent, so we fan out over
//!   `rayon` and reduce per-thread tallies. Throughput scales with cores.
//! - **Deterministic.** Each iteration's RNG is seeded from `base_seed + i`, so a
//!   given `(seed, iterations)` reproduces exactly - essential for trustworthy demos
//!   and tests.
//! - **Precompute everything invariant.** Expected goals for every ordered team pair
//!   and the base group standings (from already-played matches) are computed once;
//!   each iteration then does nothing but sample and tally.
//!
//! The 2026 format is the default target: 12 groups of four, top two plus the eight
//! best third-placed teams advance to a 32-team single-elimination knockout.
//!
//! In-progress matches - **group or knockout** - are conditioned on their live
//! score/minute/red-cards (see [`simulate_with_live`] / [`InProgress`]) so the tournament
//! forecast tracks live results. A level knockout tie goes to 30' of extra time at a reduced rate
//! and then a near-50/50 penalty shootout (see [`SimConfig::extra_time_fraction`] /
//! [`SimConfig::shootout_skill`]). Knockout ties also carry per-team factors that open-play
//! strength cannot represent (they act only in the knockout stage): a **shootout skill** that
//! tilts the shootout, a **knockout pedigree** that tilts the scoring rates (see
//! [`LiveInputs::shootout_rating`] / [`LiveInputs::knockout_pedigree`]), and a one-round
//! **extra-time fatigue** carry-over - a side whose previous tie needed extra time is dampened in
//! the next round. When the tournament already contains knockout fixtures (the
//! group stage is complete), the simulator plays that **real bracket** - finished ties stay
//! fixed, in-progress ties are conditioned, scheduled ties are sampled - instead of re-deriving a
//! bracket from a fresh group simulation. Until then, when the tournament has the real 2026 shape
//! (12 groups of four, top two plus eight best thirds) the knockout uses the **fixed 2026
//! bracket** template ([`oracle_domain::bracket::FIXED_R32`]); other shapes fall back to generic
//! reflection seeding. Documented modelling choices: the best-third -> slot assignment is a fixed
//! deterministic rule rather than FIFA's full 495-row lookup table, the team draw is synthetic,
//! and a finished knockout tie level on the scoreline (decided on penalties, which the domain does
//! not record) is resolved to the home side.
#![forbid(unsafe_code)]

use oracle_domain::bracket::{resolve_slot, FIXED_R32};
use oracle_domain::{
    MatchId, MatchStatus, Scoreline, Stage, TeamForecast, TeamId, Tournament, TournamentForecast,
};
use oracle_model::{remaining_rates, LiveConfig, LiveState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma, Normal, Poisson};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Live state of an in-progress (or lineup-announced) match, supplied to
/// [`simulate_with_live`] so the simulator samples the *remainder* conditioned on what's
/// already happened instead of replaying from 0-0. The `*_adj` fields are log-space
/// attack/defense adjustments from a confirmed lineup (positive = stronger); a match with
/// an announced lineup but not yet kicked off is passed as `minute 0` with these set.
#[derive(Debug, Clone, Copy, Default)]
pub struct InProgress {
    pub score: Scoreline,
    pub minute: u16,
    pub home_reds: u8,
    pub away_reds: u8,
    pub home_attack_adj: f64,
    pub home_defense_adj: f64,
    pub away_attack_adj: f64,
    pub away_defense_adj: f64,
}

/// A per-team venue/travel adjustment in log space:
/// `((home_attack, home_defense), (away_attack, away_defense))`.
pub type VenueAdj = ((f64, f64), (f64, f64));

/// Extra per-match inputs to the simulation. `live` conditions in-progress (or
/// lineup-announced) matches; `venue` applies host/altitude/rest adjustments to **every**
/// remaining fixture so context reaches the tournament-level forecast.
#[derive(Debug, Clone, Default)]
pub struct LiveInputs {
    pub live: HashMap<MatchId, InProgress>,
    pub venue: HashMap<MatchId, VenueAdj>,
    /// Per-team strength uncertainty (log-rate SD) overriding `MatchSampler::rating_stderr`.
    /// The engine fills this from the dynamic state-space rating; empty falls back to the
    /// sampler's own (static, fit-based) uncertainty.
    pub rating_sigma: HashMap<TeamId, f64>,
    /// Per-team **penalty-shootout skill** (positive = better). Tilts a level knockout tie's
    /// shootout away from the coin flip, on top of the small expected-goal edge. Applies only in
    /// the knockout stage; absent = neutral.
    pub shootout_rating: HashMap<TeamId, f64>,
    /// Per-team **knockout pedigree** (positive = handles single-elimination pressure better).
    /// A log-rate tilt applied only to knockout ties - an effect open-play strength, which acts
    /// everywhere, cannot represent. Absent = neutral.
    pub knockout_pedigree: HashMap<TeamId, f64>,
}

/// Supplies expected goals for a (neutral-venue) matchup. Implemented for the
/// Dixon-Coles [`oracle_model::GoalModel`]; mockable in tests.
pub trait MatchSampler: Sync {
    /// Expected goals `(home_xg, away_xg)` for `home` vs `away` at a neutral site.
    fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64);

    /// Log-space standard deviation of a team's strength, i.e. how uncertain its rating is.
    /// The simulator resamples each team's strength from this once per iteration, so the
    /// forecast reflects parameter uncertainty rather than treating point estimates as
    /// certain. Defaults to 0 (no uncertainty), which reproduces the deterministic forecast.
    fn rating_stderr(&self, _team: TeamId) -> f64 {
        0.0
    }

    /// Negative-binomial dispersion (the NB "size"/`r`) for sampled goal counts. The simulator
    /// draws goals from a Gamma-Poisson mixture with this size, so scorelines have the same fatter
    /// tails as the goal model's grid. Defaults to 0 (Poisson), reproducing the fixed-rate sampler.
    fn dispersion(&self) -> f64 {
        0.0
    }
}

impl MatchSampler for oracle_model::GoalModel {
    fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64) {
        // World Cup matches are played at neutral venues.
        self.expected_goals(home, away, true)
    }

    fn rating_stderr(&self, team: TeamId) -> f64 {
        self.strength_uncertainty(team)
    }

    fn dispersion(&self) -> f64 {
        oracle_model::GoalModel::dispersion(self)
    }
}

/// Simulation parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimConfig {
    pub iterations: u64,
    /// Base RNG seed; the run is fully reproducible for a fixed `(seed, iterations)`.
    pub seed: u64,
    /// Teams advancing directly from each group (2 in the 2026 format).
    pub advance_per_group: usize,
    /// Best third-placed teams that also advance (8 in the 2026 format).
    pub best_thirds: usize,
    /// Extra-time goal rate as a fraction of a full match (30' is 1/3, damped a little since
    /// extra time is cautious). Applied when a knockout tie is level after 90'.
    pub extra_time_fraction: f64,
    /// How much an expected-goal edge tilts a penalty shootout away from 50/50 (small;
    /// shootouts are mostly luck). The shootout win probability is clamped to [0.35, 0.65].
    pub shootout_skill: f64,
    /// Multiplier on each team's strength uncertainty (`MatchSampler::rating_stderr`) when
    /// resampling team strength per iteration. 1.0 uses the fitted uncertainty; 0 disables
    /// parameter uncertainty (a purely deterministic-strength forecast).
    pub rating_uncertainty: f64,
    /// Log-attack penalty carried into a survivor's next knockout tie when its previous tie went
    /// to extra time (a within-tournament fatigue state). 0 disables the carry-over.
    pub ko_fatigue_penalty: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            iterations: 50_000,
            seed: 0,
            advance_per_group: 2,
            best_thirds: 8,
            extra_time_fraction: 0.30,
            shootout_skill: 0.10,
            rating_uncertainty: 1.0,
            ko_fatigue_penalty: KO_FATIGUE_PENALTY,
        }
    }
}

/// Run the Monte-Carlo simulation and return per-team stage probabilities.
///
/// Equivalent to [`simulate_with_live`] with no in-progress matches - every unfinished
/// fixture is played from 0-0.
pub fn simulate<S: MatchSampler>(
    tournament: &Tournament,
    sampler: &S,
    config: SimConfig,
) -> TournamentForecast {
    simulate_with_live(
        tournament,
        sampler,
        config,
        &LiveInputs::default(),
        LiveConfig::default(),
    )
}

/// Like [`simulate`], but applies extra per-match [`LiveInputs`]: in-progress matches are
/// conditioned on their live score (sampling only the remainder), and venue/travel
/// adjustments are applied to every remaining fixture. This keeps the live match odds,
/// host advantage, and the tournament forecast coherent.
pub fn simulate_with_live<S: MatchSampler>(
    tournament: &Tournament,
    sampler: &S,
    config: SimConfig,
    inputs: &LiveInputs,
    live_config: LiveConfig,
) -> TournamentForecast {
    let prep = Prepared::build(tournament, sampler, config, inputs, live_config);
    let n = prep.teams.len();

    // Fan out over iterations; each rayon task folds into its own tally, then we
    // reduce the per-thread tallies into one.
    let tally = (0..config.iterations)
        .into_par_iter()
        .fold(
            || Tally::new(n, prep.required),
            |mut acc, i| {
                let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(i).wrapping_add(1));
                let wins = prep.simulate_once(&mut rng);
                acc.add(&wins);
                acc
            },
        )
        .reduce(|| Tally::new(n, prep.required), Tally::merge);

    tally.into_forecast(&prep.teams, config.iterations)
}

/// The knockout rounds two teams can meet at, in order (the 2026 shape).
const MEETING_ROUNDS: [&str; 5] = [
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final",
];

/// How likely two teams are to meet in the knockouts, overall and by round.
#[derive(Debug, Clone)]
pub struct MeetingForecast {
    pub iterations: u64,
    pub meet_probability: f64,
    /// `(round label, probability of meeting at that round)` for each knockout round.
    pub by_round: Vec<(&'static str, f64)>,
}

/// Estimate the probability that two teams meet in the knockout bracket, and at which round, by
/// Monte-Carlo. Reuses [`simulate_with_live`]'s machinery, observing each iteration's knockout
/// pairings; a meeting is counted at the first round the pair faces off. Returns zeros if either
/// team is unknown or they are the same team.
pub fn meeting_probabilities<S: MatchSampler>(
    tournament: &Tournament,
    sampler: &S,
    config: SimConfig,
    inputs: &LiveInputs,
    live_config: LiveConfig,
    a: TeamId,
    b: TeamId,
) -> MeetingForecast {
    let prep = Prepared::build(tournament, sampler, config, inputs, live_config);
    let (a_ix, b_ix) = match (
        prep.teams.iter().position(|&t| t == a),
        prep.teams.iter().position(|&t| t == b),
    ) {
        (Some(x), Some(y)) if x != y => (x, y),
        _ => {
            return MeetingForecast {
                iterations: config.iterations,
                meet_probability: 0.0,
                by_round: Vec::new(),
            }
        }
    };
    const ROUNDS: usize = MEETING_ROUNDS.len();
    let counts = (0..config.iterations)
        .into_par_iter()
        .fold(
            || [0u64; ROUNDS],
            |mut acc, i| {
                let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(i).wrapping_add(1));
                let mut met: Option<u8> = None;
                prep.simulate_once_with(&mut rng, |round, x, y| {
                    if met.is_none() && ((x == a_ix && y == b_ix) || (x == b_ix && y == a_ix)) {
                        met = Some(round);
                    }
                });
                if let Some(r) = met {
                    if (r as usize) < ROUNDS {
                        acc[r as usize] += 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || [0u64; ROUNDS],
            |mut x, y| {
                for i in 0..ROUNDS {
                    x[i] += y[i];
                }
                x
            },
        );
    let n = config.iterations.max(1) as f64;
    let total: u64 = counts.iter().sum();
    MeetingForecast {
        iterations: config.iterations,
        meet_probability: total as f64 / n,
        by_round: MEETING_ROUNDS
            .iter()
            .enumerate()
            .map(|(i, &label)| (label, counts[i] as f64 / n))
            .collect(),
    }
}

/// Immutable, precomputed state shared (by reference) across all iterations.
struct Prepared {
    teams: Vec<TeamId>,
    /// Expected goals for every ordered pair, indexed `eg[home_ix * n + away_ix]`.
    eg: Vec<(f64, f64)>,
    /// Points/goal-diff/goals-for already secured from finished group matches.
    base: Vec<Standing>,
    groups: Vec<GroupSim>,
    advance_per_group: usize,
    best_thirds: usize,
    n: usize,
    extra_time_fraction: f64,
    shootout_skill: f64,
    /// Negative-binomial dispersion for sampled goal counts (0 = Poisson). From the sampler, so
    /// the simulated scorelines carry the same overdispersion the goal model fit.
    dispersion: f64,
    /// Per-team log-space strength SD (indexed by team-index), resampled each iteration.
    team_sigma: Vec<f64>,
    /// Per-team penalty-shootout skill (indexed by team-index); tilts the shootout coin flip.
    shootout_rating: Vec<f64>,
    /// Per-team knockout pedigree (indexed by team-index); a knockout-only log-rate tilt.
    ko_pedigree: Vec<f64>,
    /// Extra-time fatigue penalty carried to a survivor's next tie (from [`SimConfig`]; 0 = off).
    ko_fatigue_penalty: f64,
    /// Whether any team has non-zero strength uncertainty (skip the draw entirely if not).
    has_uncertainty: bool,
    /// Whether the tournament has the 2026 shape (12 full groups, top-2 + 8 thirds), so the
    /// fixed bracket applies; otherwise the generic reflection seeding is used.
    fixed_bracket: bool,
    /// Round-of-32 pairings (team-index pairs, in bracket order) when the tournament already
    /// contains materialized knockout fixtures. Empty when the bracket is derived from a fresh
    /// group simulation instead.
    ko_r32: Vec<(usize, usize)>,
    /// Per-pairing knockout outcomes (finished or live) for any materialized knockout fixture,
    /// keyed by [`pair_key`]. Consulted by [`Self::play_ko_tie`] before sampling.
    ko_outcomes: HashMap<(usize, usize), KoOutcome>,
    /// Knockout wins required to be credited with reaching each forecast stage:
    /// `[advanced, R16, QF, SF, Final, Champion]`. Derived from the bracket depth so
    /// the stages line up with the real 32-team format and degrade gracefully for
    /// smaller test brackets.
    required: [i64; 6],
}

#[derive(Clone, Copy, Default)]
struct Standing {
    points: i32,
    gd: i32,
    gf: i32,
}

struct GroupSim {
    members: Vec<usize>,
    /// Group matches not yet finished, with sampling rates pre-resolved.
    remaining: Vec<RemainingMatch>,
}

/// A group fixture still to be decided. `rates` are the goal rates to sample (full-match
/// xG for a not-yet-started match, or the *remaining* rates for an in-progress one), and
/// `current` is the score already on the board (0-0 unless in progress).
struct RemainingMatch {
    home_ix: usize,
    away_ix: usize,
    rates: (f64, f64),
    current: Scoreline,
}

impl Prepared {
    fn build<S: MatchSampler>(
        tournament: &Tournament,
        sampler: &S,
        config: SimConfig,
        inputs: &LiveInputs,
        live_config: LiveConfig,
    ) -> Self {
        let teams: Vec<TeamId> = tournament.teams.iter().map(|t| t.id).collect();
        let n = teams.len();
        let index: HashMap<TeamId, usize> =
            teams.iter().enumerate().map(|(i, &t)| (t, i)).collect();

        // Expected-goals matrix (computed once, reused every iteration).
        let mut eg = vec![(0.0, 0.0); n * n];
        for (i, &home) in teams.iter().enumerate() {
            for (j, &away) in teams.iter().enumerate() {
                if i != j {
                    eg[i * n + j] = sampler.xg(home, away);
                }
            }
        }

        // Base standings from already-played group matches.
        let mut base = vec![Standing::default(); n];
        for m in &tournament.matches {
            if !matches!(m.stage, Stage::Group(_)) || !m.is_finished() {
                continue;
            }
            let (Some(&h), Some(&a)) = (index.get(&m.home), index.get(&m.away)) else {
                continue;
            };
            apply_result_at(&mut base, h, a, m.score.home as i32, m.score.away as i32);
        }

        // Group structure + remaining fixtures, derived from the actual fixture list so
        // each match keeps its id (for live lookup) and home/away orientation.
        let groups = tournament
            .groups
            .iter()
            .map(|g| {
                let members: Vec<usize> = g
                    .teams
                    .iter()
                    .filter_map(|t| index.get(t).copied())
                    .collect();
                let remaining: Vec<RemainingMatch> = tournament
                    .matches
                    .iter()
                    .filter(|m| {
                        matches!(m.stage, Stage::Group(c) if c == g.name) && !m.is_finished()
                    })
                    .filter_map(|m| {
                        let h = *index.get(&m.home)?;
                        let a = *index.get(&m.away)?;
                        // Venue/travel context adjusts the base rates of every fixture.
                        let ((vh_atk, vh_def), (va_atk, va_def)) =
                            inputs.venue.get(&m.id).copied().unwrap_or_default();
                        let base_l = eg[h * n + a].0 * (vh_atk - va_def).exp();
                        let base_m = eg[h * n + a].1 * (va_atk - vh_def).exp();
                        // In-progress matches keep their score and sample only the rest;
                        // a confirmed lineup further adjusts the rates.
                        let (rates, current) = match inputs.live.get(&m.id) {
                            Some(ip) => {
                                let adj_lambda =
                                    base_l * (ip.home_attack_adj - ip.away_defense_adj).exp();
                                let adj_mu =
                                    base_m * (ip.away_attack_adj - ip.home_defense_adj).exp();
                                let state = LiveState {
                                    current: ip.score,
                                    minute: ip.minute,
                                    home_red_cards: ip.home_reds,
                                    away_red_cards: ip.away_reds,
                                };
                                (
                                    remaining_rates(adj_lambda, adj_mu, &state, &live_config),
                                    ip.score,
                                )
                            }
                            None => ((base_l, base_m), Scoreline::new(0, 0)),
                        };
                        Some(RemainingMatch {
                            home_ix: h,
                            away_ix: a,
                            rates,
                            current,
                        })
                    })
                    .collect();
                GroupSim { members, remaining }
            })
            .collect::<Vec<GroupSim>>();

        // Materialized knockout fixtures (present once the group stage is complete): record the
        // Round-of-32 pairings in bracket order, and index every finished/in-progress knockout
        // tie so the simulator plays the real bracket rather than re-deriving it from the groups.
        let mut ko_r32: Vec<(usize, usize)> = Vec::new();
        let mut ko_outcomes: HashMap<(usize, usize), KoOutcome> = HashMap::new();
        for m in &tournament.matches {
            if !m.stage.is_knockout() {
                continue;
            }
            let (Some(&h), Some(&a)) = (index.get(&m.home), index.get(&m.away)) else {
                continue;
            };
            if matches!(m.stage, Stage::RoundOf32) {
                ko_r32.push((h, a));
            }
            if m.is_finished() {
                // A knockout match cannot be drawn; the higher score won. A finished tie level on
                // the scoreline was decided on penalties, which the domain does not record, so it
                // is resolved deterministically to the home side (a documented limitation).
                let winner = if m.score.away > m.score.home { a } else { h };
                ko_outcomes.insert(pair_key(h, a), KoOutcome::Finished(winner));
            } else if inputs.live.contains_key(&m.id)
                || matches!(m.status, MatchStatus::Live { .. })
            {
                // In progress: condition on the live score exactly as in-progress group matches
                // are. Prefer the explicit live-inputs entry; fall back to the match's own
                // live status/score.
                let ((vh_atk, vh_def), (va_atk, va_def)) =
                    inputs.venue.get(&m.id).copied().unwrap_or_default();
                let base_l = eg[h * n + a].0 * (vh_atk - va_def).exp();
                let base_m = eg[h * n + a].1 * (va_atk - vh_def).exp();
                let (rates, current) = match inputs.live.get(&m.id) {
                    Some(ip) => {
                        let adj_lambda = base_l * (ip.home_attack_adj - ip.away_defense_adj).exp();
                        let adj_mu = base_m * (ip.away_attack_adj - ip.home_defense_adj).exp();
                        let state = LiveState {
                            current: ip.score,
                            minute: ip.minute,
                            home_red_cards: ip.home_reds,
                            away_red_cards: ip.away_reds,
                        };
                        (
                            remaining_rates(adj_lambda, adj_mu, &state, &live_config),
                            ip.score,
                        )
                    }
                    None => {
                        let minute = match m.status {
                            MatchStatus::Live { minute } => minute,
                            _ => 0,
                        };
                        let state = LiveState {
                            current: m.score,
                            minute,
                            home_red_cards: 0,
                            away_red_cards: 0,
                        };
                        (
                            remaining_rates(base_l, base_m, &state, &live_config),
                            m.score,
                        )
                    }
                };
                ko_outcomes.insert(
                    pair_key(h, a),
                    KoOutcome::Live {
                        home_ix: h,
                        away_ix: a,
                        rates,
                        current,
                    },
                );
            }
        }

        // How many teams qualify (fixed across iterations) → bracket size → depth.
        let groups_with_third = groups
            .iter()
            .filter(|g| g.members.len() > config.advance_per_group)
            .count();
        let qualifier_count: usize = groups
            .iter()
            .map(|g| g.members.len().min(config.advance_per_group))
            .sum::<usize>()
            + config.best_thirds.min(groups_with_third);
        // With a materialized bracket the field size is fixed by the Round-of-32 fixtures;
        // otherwise it follows the group qualifiers.
        let bracket = if ko_r32.is_empty() {
            next_pow2(qualifier_count.max(2))
        } else {
            next_pow2((ko_r32.len() * 2).max(2))
        };
        let total_rounds = bracket.trailing_zeros() as i64;
        // Stage k is reached by winning `total_rounds - (5 - k)` knockout matches,
        // i.e. counting back from the final. Clamped at 0; champion needs them all.
        let required = [
            0,
            (total_rounds - 4).max(0),
            (total_rounds - 3).max(0),
            (total_rounds - 2).max(0),
            (total_rounds - 1).max(0),
            total_rounds,
        ];

        // Per-team strength uncertainty (scaled by the config multiplier), resampled per
        // iteration so the forecast carries parameter uncertainty, not just match variance.
        // A per-team override (e.g. the engine's dynamic state-space SD) takes precedence over
        // the sampler's own (static) uncertainty.
        let team_sigma: Vec<f64> = teams
            .iter()
            .map(|&t| {
                let base = inputs
                    .rating_sigma
                    .get(&t)
                    .copied()
                    .unwrap_or_else(|| sampler.rating_stderr(t));
                (config.rating_uncertainty * base).max(0.0)
            })
            .collect();
        let has_uncertainty = team_sigma.iter().any(|&s| s > 0.0);

        // Per-team knockout factors (neutral 0 when absent): shootout skill and pedigree.
        let shootout_rating: Vec<f64> = teams
            .iter()
            .map(|&t| inputs.shootout_rating.get(&t).copied().unwrap_or(0.0))
            .collect();
        let ko_pedigree: Vec<f64> = teams
            .iter()
            .map(|&t| inputs.knockout_pedigree.get(&t).copied().unwrap_or(0.0))
            .collect();

        // The fixed 2026 bracket applies only to the real shape: 12 groups that each can
        // produce a third-placed team, top two plus the eight best thirds advancing.
        let fixed_bracket = groups.len() == 12
            && config.advance_per_group == 2
            && config.best_thirds == 8
            && groups.iter().all(|g| g.members.len() >= 3);

        Self {
            teams,
            eg,
            base,
            groups,
            advance_per_group: config.advance_per_group,
            best_thirds: config.best_thirds,
            n,
            extra_time_fraction: config.extra_time_fraction,
            shootout_skill: config.shootout_skill,
            dispersion: sampler.dispersion().max(0.0),
            team_sigma,
            shootout_rating,
            ko_pedigree,
            ko_fatigue_penalty: config.ko_fatigue_penalty,
            has_uncertainty,
            fixed_bracket,
            ko_r32,
            ko_outcomes,
            required,
        }
    }

    /// Sample a single goal count. With a positive dispersion the rate is first drawn from a
    /// Gamma (the negative-binomial / Gamma-Poisson mixture), so a team's effective rate varies
    /// match to match and scorelines get the fatter tails real football shows; dispersion 0 is a
    /// plain Poisson(λ).
    fn sample_goals(&self, rng: &mut StdRng, lambda: f64) -> i32 {
        if lambda <= 1e-9 {
            return 0;
        }
        let rate = if self.dispersion > 0.0 {
            let r = self.dispersion;
            Gamma::new(r, lambda / r)
                .map(|g| g.sample(rng))
                .unwrap_or(lambda)
                .max(1e-9)
        } else {
            lambda
        };
        match Poisson::new(rate) {
            Ok(dist) => (dist.sample(rng) as i32).min(20),
            Err(_) => 0,
        }
    }

    /// Sample the winner's team-index of a knockout tie: 90 minutes, then 30 of extra time
    /// at a reduced rate, then a penalty shootout that is close to a coin flip.
    /// Returns `(winner, went_to_extra_time)`. `fatigue` is a log-attack penalty per side carried
    /// from a previous round's extra time.
    fn sample_knockout(
        &self,
        rng: &mut StdRng,
        a: usize,
        b: usize,
        att: &[f64],
        def: &[f64],
        fatigue: (f64, f64),
    ) -> (usize, bool) {
        if a == BYE {
            return (b, false);
        }
        if b == BYE {
            return (a, false);
        }
        // Apply this iteration's resampled team strengths to the base expected goals, plus the
        // knockout-pedigree tilt (this is a knockout tie, where temperament/experience tells) and
        // any carried-over extra-time fatigue.
        let (eg_a, eg_b) = self.eg[a * self.n + b];
        let la =
            eg_a * (att[a] - def[b] + KO_PEDIGREE_SCALE * self.ko_pedigree[a] - fatigue.0).exp();
        let ma =
            eg_b * (att[b] - def[a] + KO_PEDIGREE_SCALE * self.ko_pedigree[b] - fatigue.1).exp();
        let mut gh = self.sample_goals(rng, la);
        let mut ga = self.sample_goals(rng, ma);
        let extra_time = gh == ga;
        if extra_time {
            // Extra time: a third of a match, at lower scoring intensity.
            gh += self.sample_goals(rng, la * self.extra_time_fraction);
            ga += self.sample_goals(rng, ma * self.extra_time_fraction);
        }
        let winner = match gh.cmp(&ga) {
            std::cmp::Ordering::Greater => a,
            std::cmp::Ordering::Less => b,
            std::cmp::Ordering::Equal => {
                // Shootout: near 50/50, tilted by the expected-goal edge and shootout skill.
                let p = (0.5
                    + self.shootout_skill * (la - ma)
                    + SHOOTOUT_RATING_SCALE * (self.shootout_rating[a] - self.shootout_rating[b]))
                    .clamp(0.35, 0.65);
                if rng.gen::<f64>() < p {
                    a
                } else {
                    b
                }
            }
        };
        (winner, extra_time)
    }

    /// Play a knockout tie, respecting a materialized fixture for this exact pairing: a finished
    /// fixture returns its winner, an in-progress fixture is conditioned on the live score, and a
    /// pairing with no fixture (a scheduled or not-yet-materialized tie) is sampled fresh.
    /// Returns `(winner, went_to_extra_time)`. A finished (materialized) tie reports `false` (its
    /// course is unknown), so it carries no fatigue forward.
    fn play_ko_tie(
        &self,
        rng: &mut StdRng,
        a: usize,
        b: usize,
        att: &[f64],
        def: &[f64],
        fatigue: (f64, f64),
    ) -> (usize, bool) {
        if a == BYE {
            return (b, false);
        }
        if b == BYE {
            return (a, false);
        }
        match self.ko_outcomes.get(&pair_key(a, b)) {
            Some(KoOutcome::Finished(w)) => (*w, false),
            Some(KoOutcome::Live {
                home_ix,
                away_ix,
                rates,
                current,
            }) => {
                // `fatigue` is ordered (a, b); re-orient it to the fixture's (home, away).
                let fat = if *home_ix == a {
                    fatigue
                } else {
                    (fatigue.1, fatigue.0)
                };
                self.sample_live_ko(rng, (*home_ix, *away_ix), *rates, *current, (att, def), fat)
            }
            None => self.sample_knockout(rng, a, b, att, def, fatigue),
        }
    }

    /// Sample the winner of an in-progress knockout tie: add the sampled remainder (at the
    /// live-conditioned rates) to the score already on the board, then extra time and a shootout
    /// if still level, mirroring [`Self::sample_knockout`]. `shifts` is this iteration's
    /// `(attack, defense)` strength draw.
    /// Returns `(winner, went_to_extra_time)`. `fatigue` is `(home, away)` log-attack penalties.
    fn sample_live_ko(
        &self,
        rng: &mut StdRng,
        (home_ix, away_ix): (usize, usize),
        rem_rates: (f64, f64),
        current: Scoreline,
        (att, def): (&[f64], &[f64]),
        fatigue: (f64, f64),
    ) -> (usize, bool) {
        // Knockout-pedigree tilt and carried-over fatigue apply here too (this is a knockout tie).
        let ph = KO_PEDIGREE_SCALE * self.ko_pedigree[home_ix] - fatigue.0;
        let pa = KO_PEDIGREE_SCALE * self.ko_pedigree[away_ix] - fatigue.1;
        let rate_h = rem_rates.0 * (att[home_ix] - def[away_ix] + ph).exp();
        let rate_a = rem_rates.1 * (att[away_ix] - def[home_ix] + pa).exp();
        let mut gh = i32::from(current.home) + self.sample_goals(rng, rate_h);
        let mut ga = i32::from(current.away) + self.sample_goals(rng, rate_a);
        // Extra time and the shootout use full-match strengths (the remaining-rate conditioning
        // only governs the rest of regulation).
        let (eg_h, eg_a) = self.eg[home_ix * self.n + away_ix];
        let la = eg_h * (att[home_ix] - def[away_ix] + ph).exp();
        let ma = eg_a * (att[away_ix] - def[home_ix] + pa).exp();
        let extra_time = gh == ga;
        if extra_time {
            gh += self.sample_goals(rng, la * self.extra_time_fraction);
            ga += self.sample_goals(rng, ma * self.extra_time_fraction);
        }
        let winner = match gh.cmp(&ga) {
            std::cmp::Ordering::Greater => home_ix,
            std::cmp::Ordering::Less => away_ix,
            std::cmp::Ordering::Equal => {
                let p = (0.5
                    + self.shootout_skill * (la - ma)
                    + SHOOTOUT_RATING_SCALE
                        * (self.shootout_rating[home_ix] - self.shootout_rating[away_ix]))
                    .clamp(0.35, 0.65);
                if rng.gen::<f64>() < p {
                    home_ix
                } else {
                    away_ix
                }
            }
        };
        (winner, extra_time)
    }

    /// Draw this iteration's per-team log-space `(attack, defense)` strength shifts from each
    /// team's uncertainty. Returns all-zero vectors when no uncertainty is configured.
    fn draw_strength_shifts(&self, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>) {
        if !self.has_uncertainty {
            return (vec![0.0; self.n], vec![0.0; self.n]);
        }
        let std_normal = Normal::new(0.0, 1.0).expect("valid normal");
        let draw = |rng: &mut StdRng| -> Vec<f64> {
            self.team_sigma
                .iter()
                .map(|&s| {
                    if s > 0.0 {
                        s * std_normal.sample(rng)
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        (draw(rng), draw(rng))
    }

    /// Play one full tournament (the normal forecast path).
    fn simulate_once(&self, rng: &mut StdRng) -> Vec<i64> {
        self.simulate_once_with(rng, |_, _, _| {})
    }

    /// Play one full tournament, invoking `on_tie(round, home_ix, away_ix)` for every knockout tie
    /// contested (round 0 = Round of 32, 1 = R16, 2 = QF, 3 = SF, 4 = Final). Returns a per-team
    /// vector of knockout rounds won: `-1` = did not qualify, `0` = qualified but lost the first
    /// knockout match, `total_rounds` = champion. The forecast passes a no-op observer; the
    /// meeting analysis records the pairings.
    fn simulate_once_with<F: FnMut(u8, usize, usize)>(
        &self,
        rng: &mut StdRng,
        mut on_tie: F,
    ) -> Vec<i64> {
        let mut wins = vec![-1i64; self.n];
        // Per-team log-attack penalty carried into a team's *next* knockout tie when its last tie
        // went to extra time (reset to 0 after a tie settled in regulation).
        let mut fatigue = vec![0.0f64; self.n];

        // Resample each team's strength for this simulated universe (log-space attack and
        // defense shifts). Held fixed across all of the team's matches this iteration, so a
        // team that turns out stronger than its point estimate is stronger everywhere. All
        // zero when no uncertainty is configured, reproducing the deterministic forecast.
        let (att, def) = self.draw_strength_shifts(rng);

        // `survivors` holds the 16 first-round winners in bracket order; the fold below plays
        // out R16 -> Final by repeatedly pairing adjacent survivors.
        let mut survivors: Vec<usize> = if !self.ko_r32.is_empty() {
            // ---- Real knockout bracket present (group stage already complete) ----
            // Play the materialized Round of 32 directly: a finished tie keeps its result, an
            // in-progress tie is conditioned on its live score, a scheduled tie is sampled. The
            // group stage is not re-simulated, so finished knockout results stay fixed.
            self.ko_r32
                .iter()
                .map(|&(a, b)| {
                    wins[a] = 0;
                    wins[b] = 0;
                    on_tie(0, a, b);
                    let (w, et) = self.play_ko_tie(rng, a, b, &att, &def, (0.0, 0.0));
                    wins[w] += 1;
                    fatigue[w] = if et { self.ko_fatigue_penalty } else { 0.0 };
                    w
                })
                .collect()
        } else {
            // ---- Group stage ----
            let mut winners = Vec::new();
            let mut runners = Vec::new();
            let mut thirds: Vec<(usize, Standing)> = Vec::new();

            for group in &self.groups {
                let mut table: Vec<(usize, Standing)> =
                    group.members.iter().map(|&m| (m, self.base[m])).collect();
                // Index within `table` by team-ix for quick mutation.
                let pos: HashMap<usize, usize> = table
                    .iter()
                    .enumerate()
                    .map(|(i, (ix, _))| (*ix, i))
                    .collect();
                let mut standings: Vec<Standing> = table.iter().map(|(_, s)| *s).collect();

                for rm in &group.remaining {
                    // Apply this iteration's strength shifts to the (venue/lineup-adjusted) rates.
                    let rate_h = rm.rates.0 * (att[rm.home_ix] - def[rm.away_ix]).exp();
                    let rate_a = rm.rates.1 * (att[rm.away_ix] - def[rm.home_ix]).exp();
                    // Final goals = already scored (0 unless in progress) + sampled remainder.
                    let gh = i32::from(rm.current.home) + self.sample_goals(rng, rate_h);
                    let ga = i32::from(rm.current.away) + self.sample_goals(rng, rate_a);
                    apply_result_at(&mut standings, pos[&rm.home_ix], pos[&rm.away_ix], gh, ga);
                }
                for (i, (_, s)) in table.iter_mut().enumerate() {
                    *s = standings[i];
                }
                rank(&mut table);

                for (rank_ix, (team_ix, standing)) in table.into_iter().enumerate() {
                    if rank_ix < self.advance_per_group {
                        if rank_ix == 0 {
                            winners.push(team_ix);
                        } else {
                            runners.push(team_ix);
                        }
                    } else if rank_ix == self.advance_per_group {
                        thirds.push((team_ix, standing));
                    }
                }
            }

            // Best third-placed teams.
            thirds.sort_by(|a, b| cmp_standing(&b.1, &a.1));
            let qualified_thirds: Vec<usize> = thirds
                .into_iter()
                .take(self.best_thirds)
                .map(|(ix, _)| ix)
                .collect();

            // ---- Round of 32: the fixed 2026 bracket when the shape matches, else reflection --
            if self.fixed_bracket {
                FIXED_R32
                    .iter()
                    .map(|(top, bottom)| {
                        let a = resolve_slot(top, &winners, &runners, &qualified_thirds);
                        let b = resolve_slot(bottom, &winners, &runners, &qualified_thirds);
                        wins[a] = 0;
                        wins[b] = 0;
                        on_tie(0, a, b);
                        let (w, et) = self.sample_knockout(rng, a, b, &att, &def, (0.0, 0.0));
                        wins[w] += 1;
                        fatigue[w] = if et { self.ko_fatigue_penalty } else { 0.0 };
                        w
                    })
                    .collect()
            } else {
                // Fallback for non-2026 shapes (e.g. test tournaments): winners, then best
                // thirds, then runners-up, paired by reflection to keep strong winners apart.
                let mut seed_order: Vec<usize> = Vec::new();
                seed_order.extend(&winners);
                seed_order.extend(&qualified_thirds);
                seed_order.extend(&runners);
                for &ix in &seed_order {
                    wins[ix] = 0;
                }
                let bracket = next_pow2(seed_order.len().max(2));
                seed_order.resize(bracket, BYE);
                (0..bracket / 2)
                    .map(|i| {
                        let (a, b) = (seed_order[i], seed_order[bracket - 1 - i]);
                        if a != BYE && b != BYE {
                            on_tie(0, a, b);
                        }
                        let (w, et) = self.sample_knockout(rng, a, b, &att, &def, (0.0, 0.0));
                        if w != BYE {
                            wins[w] += 1;
                            fatigue[w] = if et { self.ko_fatigue_penalty } else { 0.0 };
                        }
                        w
                    })
                    .collect()
            }
        };
        // ---- Fold R16 -> Final ----. `play_ko_tie` respects any materialized later-round
        // fixtures (finished/in-progress) and samples the rest. A side that needed extra time last
        // round carries a fatigue penalty into this one.
        let mut round: u8 = 1;
        while survivors.len() > 1 {
            let mut next = Vec::with_capacity(survivors.len().div_ceil(2));
            let mut k = 0;
            while k + 1 < survivors.len() {
                let (a, b) = (survivors[k], survivors[k + 1]);
                if a != BYE && b != BYE {
                    on_tie(round, a, b);
                }
                let fat = (
                    if a == BYE { 0.0 } else { fatigue[a] },
                    if b == BYE { 0.0 } else { fatigue[b] },
                );
                let (w, et) = self.play_ko_tie(rng, a, b, &att, &def, fat);
                if w != BYE {
                    wins[w] += 1;
                    fatigue[w] = if et { self.ko_fatigue_penalty } else { 0.0 };
                }
                next.push(w);
                k += 2;
            }
            survivors = next;
            round += 1;
        }

        wins
    }
}

/// Sentinel team-index representing a bracket bye.
const BYE: usize = usize::MAX;

/// Log-rate tilt per unit of knockout pedigree, applied only to knockout ties (a side that
/// handles single-elimination pressure better scores a touch more / concedes a touch less).
const KO_PEDIGREE_SCALE: f64 = 0.10;
/// Shootout win-probability tilt per unit of shootout-skill difference, on top of the
/// expected-goal edge. Kept modest so a shootout stays mostly a coin flip.
const SHOOTOUT_RATING_SCALE: f64 = 0.06;
/// Log-attack penalty carried into the *next* knockout round by a side whose previous tie went to
/// extra time (the extra 30 minutes plus less recovery sap a team). One round of memory: a side
/// whose next tie is settled in 90 is recovered for the one after.
const KO_FATIGUE_PENALTY: f64 = 0.07;

/// A materialized knockout fixture's bearing on the simulation, keyed by its (sorted) team-index
/// pair. A `Finished` tie is settled; a `Live` tie supplies the live-conditioned remaining rates
/// and the score already on the board so the simulator samples only the rest.
enum KoOutcome {
    /// Team-index that already won this tie.
    Finished(usize),
    Live {
        home_ix: usize,
        away_ix: usize,
        /// Remaining-regulation goal rates (venue- and live-conditioned).
        rates: (f64, f64),
        current: Scoreline,
    },
}

/// Order-independent key for a knockout pairing.
fn pair_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn next_pow2(x: usize) -> usize {
    let mut p = 1;
    while p < x {
        p <<= 1;
    }
    p
}

fn apply_result_at(table: &mut [Standing], h: usize, a: usize, gh: i32, ga: i32) {
    table[h].gf += gh;
    table[h].gd += gh - ga;
    table[a].gf += ga;
    table[a].gd += ga - gh;
    match gh.cmp(&ga) {
        std::cmp::Ordering::Greater => table[h].points += 3,
        std::cmp::Ordering::Less => table[a].points += 3,
        std::cmp::Ordering::Equal => {
            table[h].points += 1;
            table[a].points += 1;
        }
    }
}

fn cmp_standing(a: &Standing, b: &Standing) -> std::cmp::Ordering {
    a.points
        .cmp(&b.points)
        .then(a.gd.cmp(&b.gd))
        .then(a.gf.cmp(&b.gf))
}

/// Rank a group table best-first (points, then goal difference, then goals for,
/// with team-index as a deterministic final tie-break).
fn rank(table: &mut [(usize, Standing)]) {
    table.sort_by(|(ia, a), (ib, b)| {
        cmp_standing(b, a).then(ia.cmp(ib)) // descending standing, ascending index
    });
}

/// Per-team cumulative counts of reaching each stage.
struct Tally {
    /// `counts[team_ix]` = [advanced, R16, QF, SF, Final, Champion].
    counts: Vec<[u64; 6]>,
    /// Knockout wins required for each stage (see [`Prepared::required`]).
    required: [i64; 6],
}

impl Tally {
    fn new(n: usize, required: [i64; 6]) -> Self {
        Self {
            counts: vec![[0; 6]; n],
            required,
        }
    }

    fn add(&mut self, wins: &[i64]) {
        for (ix, &w) in wins.iter().enumerate() {
            if w < 0 {
                continue; // did not qualify
            }
            for (k, &need) in self.required.iter().enumerate() {
                if w >= need {
                    self.counts[ix][k] += 1;
                }
            }
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            for k in 0..6 {
                a[k] += b[k];
            }
        }
        self
    }

    fn into_forecast(self, teams: &[TeamId], iterations: u64) -> TournamentForecast {
        let denom = iterations.max(1) as f64;
        let team_forecasts = teams
            .iter()
            .enumerate()
            .map(|(ix, &team)| {
                let c = self.counts[ix];
                TeamForecast {
                    team,
                    p_advance_group: c[0] as f64 / denom,
                    p_round_of_16: c[1] as f64 / denom,
                    p_quarter_final: c[2] as f64 / denom,
                    p_semi_final: c[3] as f64 / denom,
                    p_final: c[4] as f64 / denom,
                    p_champion: c[5] as f64 / denom,
                }
            })
            .collect();
        TournamentForecast {
            iterations,
            teams: team_forecasts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_domain::{Confederation, Group, Match, MatchStatus, Team};

    /// A sampler where lower team-id = stronger (scores more, concedes less).
    struct RankSampler;
    impl MatchSampler for RankSampler {
        fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64) {
            let strength = |t: TeamId| 2.0 - 0.1 * t.0 as f64; // team 0 strongest
            let h = (1.2 + strength(home) - strength(away)).max(0.2);
            let a = (1.2 + strength(away) - strength(home)).max(0.2);
            (h, a)
        }
    }

    /// A sampler giving every matchup the same expected goals (perfectly even teams).
    struct EvenSampler;
    impl MatchSampler for EvenSampler {
        fn xg(&self, _home: TeamId, _away: TeamId) -> (f64, f64) {
            (1.3, 1.3)
        }
    }

    /// Even, very low-scoring teams, so most knockout ties reach a penalty shootout (lets the
    /// shootout-skill effect be isolated).
    struct GoallessSampler;
    impl MatchSampler for GoallessSampler {
        fn xg(&self, _home: TeamId, _away: TeamId) -> (f64, f64) {
            (0.35, 0.35)
        }
    }

    #[test]
    fn shootout_skill_tilts_a_level_tie() {
        // Even, low-scoring teams reach shootouts often; team 0 is far better at them.
        let t = tiny_tournament();
        let inputs = LiveInputs {
            shootout_rating: [(TeamId(0), 4.0)].into_iter().collect(),
            ..Default::default()
        };
        let prep = Prepared::build(
            &t,
            &GoallessSampler,
            SimConfig::default(),
            &inputs,
            LiveConfig::default(),
        );
        let mut rng = StdRng::seed_from_u64(11);
        let z = vec![0.0; prep.n];
        let trials = 30_000;
        let wins = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 0, 1, &z, &z, (0.0, 0.0)).0 == 0)
            .count();
        let rate = wins as f64 / trials as f64;
        assert!(
            rate > 0.54,
            "shootout skill should give a real edge when ties reach shootouts: {rate:.3}"
        );
    }

    #[test]
    fn knockout_pedigree_favours_the_clutch_side() {
        let t = tiny_tournament();
        let inputs = LiveInputs {
            knockout_pedigree: [(TeamId(0), 2.0)].into_iter().collect(),
            ..Default::default()
        };
        let prep = Prepared::build(
            &t,
            &EvenSampler,
            SimConfig::default(),
            &inputs,
            LiveConfig::default(),
        );
        let mut rng = StdRng::seed_from_u64(5);
        let z = vec![0.0; prep.n];
        let trials = 30_000;
        let wins = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 0, 1, &z, &z, (0.0, 0.0)).0 == 0)
            .count();
        let rate = wins as f64 / trials as f64;
        assert!(
            rate > 0.53,
            "knockout pedigree should favour team 0: {rate:.3}"
        );
    }

    #[test]
    fn extra_time_fatigue_handicaps_the_tired_side() {
        let t = tiny_tournament();
        let prep = Prepared::build(
            &t,
            &EvenSampler,
            SimConfig::default(),
            &LiveInputs::default(),
            LiveConfig::default(),
        );
        let z = vec![0.0; prep.n];
        let trials = 40_000;

        // Even ties reach extra time a non-trivial share of the time (so fatigue can be carried).
        let mut rng = StdRng::seed_from_u64(3);
        let et = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 0, 1, &z, &z, (0.0, 0.0)).1)
            .count();
        assert!(
            et > trials / 20,
            "even ties should reach extra time sometimes: {et}/{trials}"
        );

        // A side carrying extra-time fatigue wins fewer ties than when fresh (same RNG stream).
        let win = |fatigue: (f64, f64), seed: u64| -> f64 {
            let mut r = StdRng::seed_from_u64(seed);
            (0..trials)
                .filter(|_| prep.sample_knockout(&mut r, 0, 1, &z, &z, fatigue).0 == 0)
                .count() as f64
                / trials as f64
        };
        let fresh = win((0.0, 0.0), 1);
        let tired = win((KO_FATIGUE_PENALTY, 0.0), 1);
        assert!(
            tired < fresh - 0.01,
            "a fatigued side should win less: tired {tired:.3} vs fresh {fresh:.3}"
        );
    }

    #[test]
    fn knockout_factors_leave_the_group_stage_untouched() {
        // The factors only enter knockout sampling, so per-team group-qualification probabilities
        // must be bit-identical with and without them (each iteration is independently seeded).
        let t = full_tournament();
        let cfg = SimConfig {
            iterations: 2000,
            ..Default::default()
        };
        let base = simulate(&t, &RankSampler, cfg);
        let inputs = LiveInputs {
            knockout_pedigree: (0..48u32)
                .map(|i| (TeamId(i), (i % 5) as f64 - 2.0))
                .collect(),
            shootout_rating: (0..48u32)
                .map(|i| (TeamId(i), (i % 3) as f64 - 1.0))
                .collect(),
            ..Default::default()
        };
        let with_ko = simulate_with_live(&t, &RankSampler, cfg, &inputs, LiveConfig::default());
        for (b, w) in base.teams.iter().zip(&with_ko.teams) {
            assert_eq!(b.team, w.team);
            assert!(
                (b.p_advance_group - w.p_advance_group).abs() < 1e-12,
                "group qualification changed for {:?}",
                b.team
            );
        }
    }

    /// Even teams, with a configurable goal-count dispersion.
    struct DispersedSampler(f64);
    impl MatchSampler for DispersedSampler {
        fn xg(&self, _home: TeamId, _away: TeamId) -> (f64, f64) {
            (1.4, 1.4)
        }
        fn dispersion(&self) -> f64 {
            self.0
        }
    }

    #[test]
    fn overdispersed_sampling_widens_goal_variance() {
        let mean_var = |disp: f64| -> (f64, f64) {
            let prep = Prepared::build(
                &tiny_tournament(),
                &DispersedSampler(disp),
                SimConfig::default(),
                &LiveInputs::default(),
                LiveConfig::default(),
            );
            let mut rng = StdRng::seed_from_u64(1);
            let n = 20_000;
            let xs: Vec<f64> = (0..n)
                .map(|_| prep.sample_goals(&mut rng, 1.4) as f64)
                .collect();
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            (mean, var)
        };
        let (mp, vp) = mean_var(0.0); // Poisson: variance ~ mean
        let (mo, vo) = mean_var(4.0); // negative binomial: variance > mean
        assert!(
            (mp - 1.4).abs() < 0.1 && (mo - 1.4).abs() < 0.1,
            "means preserved"
        );
        assert!(
            vo > vp + 0.2,
            "overdispersed variance {vo} should exceed Poisson {vp}"
        );
    }

    fn epoch() -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::from_timestamp(0, 0).unwrap()
    }

    #[test]
    fn knockout_between_even_teams_is_a_coin_flip() {
        // With identical teams, extra time stays symmetric and the shootout is 50/50, so
        // neither side has an edge over many knockout ties.
        let t = tiny_tournament();
        let prep = Prepared::build(
            &t,
            &EvenSampler,
            SimConfig::default(),
            &LiveInputs::default(),
            LiveConfig::default(),
        );
        let mut rng = StdRng::seed_from_u64(99);
        let z = vec![0.0; prep.n];
        let trials = 20_000;
        let a_wins = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 0, 1, &z, &z, (0.0, 0.0)).0 == 0)
            .count();
        let rate = a_wins as f64 / trials as f64;
        assert!(
            (rate - 0.5).abs() < 0.03,
            "even knockout should be ~50/50, got {rate:.3}"
        );
    }

    #[test]
    fn knockout_favours_the_stronger_team_but_not_overwhelmingly() {
        // Close teams (ids 3 and 4): the favourite wins more than half, but extra time and a
        // near-coin-flip shootout keep it well short of certainty.
        let t = tiny_tournament();
        let prep = Prepared::build(
            &t,
            &RankSampler,
            SimConfig::default(),
            &LiveInputs::default(),
            LiveConfig::default(),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let z = vec![0.0; prep.n];
        let trials = 20_000;
        let strong_wins = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 3, 4, &z, &z, (0.0, 0.0)).0 == 3)
            .count();
        let rate = strong_wins as f64 / trials as f64;
        assert!(
            rate > 0.52 && rate < 0.85,
            "stronger side favoured, not certain: {rate:.3}"
        );
    }

    fn tiny_tournament() -> Tournament {
        // 8 teams, 2 groups of 4, with a full round-robin of scheduled fixtures.
        let mut t = Tournament::new("Test Cup");
        for i in 0..8u32 {
            t.teams.push(Team::new(
                i,
                format!("T{i}"),
                format!("T{i}"),
                Confederation::Uefa,
            ));
        }
        t.groups.push(Group {
            name: 'A',
            teams: vec![TeamId(0), TeamId(1), TeamId(2), TeamId(3)],
        });
        t.groups.push(Group {
            name: 'B',
            teams: vec![TeamId(4), TeamId(5), TeamId(6), TeamId(7)],
        });
        let pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
        let mut id = 1u32;
        for g in &t.groups.clone() {
            for (i, j) in pairs {
                t.matches.push(Match {
                    id: MatchId(id),
                    home: g.teams[i],
                    away: g.teams[j],
                    stage: Stage::Group(g.name),
                    kickoff: epoch(),
                    status: MatchStatus::Scheduled,
                    score: Scoreline::new(0, 0),
                });
                id += 1;
            }
        }
        t
    }

    /// A 48-team, 12-group (A..L) tournament with a full round-robin, matching the 2026 shape
    /// so the fixed bracket applies.
    fn full_tournament() -> Tournament {
        let mut t = Tournament::new("Full Cup");
        for i in 0..48u32 {
            t.teams.push(Team::new(
                i,
                format!("T{i}"),
                format!("{i:03}"),
                Confederation::Uefa,
            ));
        }
        for g in 0..12u32 {
            let base = g * 4;
            t.groups.push(Group {
                name: (b'A' + g as u8) as char,
                teams: (base..base + 4).map(TeamId).collect(),
            });
        }
        let pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
        let mut id = 1u32;
        for g in &t.groups.clone() {
            for (i, j) in pairs {
                t.matches.push(Match {
                    id: MatchId(id),
                    home: g.teams[i],
                    away: g.teams[j],
                    stage: Stage::Group(g.name),
                    kickoff: epoch(),
                    status: MatchStatus::Scheduled,
                    score: Scoreline::new(0, 0),
                });
                id += 1;
            }
        }
        t
    }

    #[test]
    fn fixed_bracket_forecast_is_coherent() {
        let t = full_tournament();
        let f = simulate(
            &t,
            &RankSampler,
            SimConfig {
                iterations: 4000,
                ..Default::default()
            },
        );
        assert_eq!(f.teams.len(), 48);
        let total: f64 = f.teams.iter().map(|tf| tf.p_champion).sum();
        assert!((total - 1.0).abs() < 0.02, "champion mass = {total}");
        // Strongest team (id 0) should out-title the weakest (id 47) under the fixed bracket.
        let champ = |team: u32| {
            f.teams
                .iter()
                .find(|x| x.team == TeamId(team))
                .unwrap()
                .p_champion
        };
        assert!(
            champ(0) > champ(47),
            "stronger team should win more under the fixed bracket"
        );
    }

    /// 32 teams with 16 materialized Round-of-32 fixtures (pairing 2i vs 2i+1), no group stage,
    /// so the simulator plays the real bracket instead of deriving one.
    fn bracket_tournament() -> Tournament {
        let mut t = Tournament::new("KO Cup");
        for i in 0..32u32 {
            t.teams.push(Team::new(
                i,
                format!("T{i}"),
                format!("{i:03}"),
                Confederation::Uefa,
            ));
        }
        for i in 0..16u32 {
            t.matches.push(Match {
                id: MatchId(1 + i),
                home: TeamId(2 * i),
                away: TeamId(2 * i + 1),
                stage: Stage::RoundOf32,
                kickoff: epoch(),
                status: MatchStatus::Scheduled,
                score: Scoreline::new(0, 0),
            });
        }
        t
    }

    fn reach_r16(f: &TournamentForecast, team: u32) -> f64 {
        f.teams
            .iter()
            .find(|tf| tf.team == TeamId(team))
            .unwrap()
            .p_round_of_16
    }

    #[test]
    fn live_knockout_match_conditions_the_forecast() {
        let t = bracket_tournament();
        let cfg = SimConfig {
            iterations: 8000,
            ..Default::default()
        };

        // Fresh: the away side of R32 match 0 (team 1) is the underdog against team 0.
        let fresh = simulate(&t, &RankSampler, cfg);
        let fresh_p = reach_r16(&fresh, 1);

        // In progress: team 1 leads 2-0 at minute 85, so its chance of advancing should jump.
        let mut live = HashMap::new();
        live.insert(
            MatchId(1),
            InProgress {
                score: Scoreline::new(0, 2),
                minute: 85,
                ..Default::default()
            },
        );
        let inputs = LiveInputs {
            live,
            ..Default::default()
        };
        let conditioned = simulate_with_live(&t, &RankSampler, cfg, &inputs, LiveConfig::default());
        let cond_p = reach_r16(&conditioned, 1);

        assert!(
            cond_p > fresh_p + 0.2,
            "a live lead should lift the advance probability: {fresh_p:.3} -> {cond_p:.3}"
        );
        assert!(
            cond_p > 0.7,
            "a two-goal lead at 85' should usually hold: {cond_p:.3}"
        );
    }

    #[test]
    fn finished_knockout_result_is_respected() {
        let mut t = bracket_tournament();
        // R32 match 1 (id 2): team 2 vs team 3, finished 0-1 (the weaker side, team 3, won).
        let m = t.matches.iter_mut().find(|m| m.id == MatchId(2)).unwrap();
        m.status = MatchStatus::Finished;
        m.score = Scoreline::new(0, 1);

        let f = simulate(
            &t,
            &RankSampler,
            SimConfig {
                iterations: 4000,
                ..Default::default()
            },
        );
        // The actual winner reaches the Round of 16 with certainty; the loser never does.
        assert!(
            reach_r16(&f, 3) > 0.999,
            "the actual winner always advances: {:.3}",
            reach_r16(&f, 3)
        );
        assert!(
            reach_r16(&f, 2) < 0.001,
            "the actual loser never advances: {:.3}",
            reach_r16(&f, 2)
        );
    }

    fn champion_prob(f: &TournamentForecast, team: u32) -> f64 {
        f.teams
            .iter()
            .find(|tf| tf.team == TeamId(team))
            .map(|tf| tf.p_champion)
            .unwrap()
    }

    #[test]
    fn probabilities_are_coherent() {
        let t = tiny_tournament();
        let f = simulate(
            &t,
            &RankSampler,
            SimConfig {
                iterations: 4000,
                ..Default::default()
            },
        );

        // Champion probabilities across all teams sum to ~1.
        let total: f64 = f.teams.iter().map(|tf| tf.p_champion).sum();
        assert!((total - 1.0).abs() < 0.02, "champion mass = {total}");

        for tf in &f.teams {
            // Monotonic nesting: champion ⊆ final ⊆ ... ⊆ advanced.
            assert!(tf.p_champion <= tf.p_final + 1e-9);
            assert!(tf.p_final <= tf.p_semi_final + 1e-9);
            assert!(tf.p_semi_final <= tf.p_quarter_final + 1e-9);
            assert!(tf.p_quarter_final <= tf.p_round_of_16 + 1e-9);
            assert!(tf.p_round_of_16 <= tf.p_advance_group + 1e-9);
            assert!(tf.p_advance_group <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn stronger_team_wins_more_often() {
        let t = tiny_tournament();
        let f = simulate(
            &t,
            &RankSampler,
            SimConfig {
                iterations: 4000,
                ..Default::default()
            },
        );
        assert!(
            champion_prob(&f, 0) > champion_prob(&f, 7),
            "strongest team should out-win the weakest"
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let t = tiny_tournament();
        let cfg = SimConfig {
            iterations: 2000,
            seed: 42,
            ..Default::default()
        };
        let a = simulate(&t, &RankSampler, cfg);
        let b = simulate(&t, &RankSampler, cfg);
        assert_eq!(a.teams, b.teams, "same seed ⇒ identical forecast");
    }

    #[test]
    fn finished_results_are_respected() {
        // Mark team 3's (normally weak) group-A matches as big wins so it tops the group.
        let mut t = tiny_tournament();
        for m in &mut t.matches {
            if matches!(m.stage, Stage::Group('A')) && (m.home == TeamId(3) || m.away == TeamId(3))
            {
                m.status = MatchStatus::Finished;
                m.score = if m.home == TeamId(3) {
                    Scoreline::new(5, 0)
                } else {
                    Scoreline::new(0, 5)
                };
            }
        }
        let f = simulate(
            &t,
            &RankSampler,
            SimConfig {
                iterations: 4000,
                ..Default::default()
            },
        );
        let p3 = f.teams.iter().find(|x| x.team == TeamId(3)).unwrap();
        assert!(p3.p_advance_group > 0.9, "team 3 already has 9 points");
    }

    #[test]
    fn in_progress_lead_lifts_advance_probability() {
        let t = tiny_tournament();
        let cfg = SimConfig {
            iterations: 6000,
            seed: 7,
            ..Default::default()
        };
        let fresh = simulate(&t, &RankSampler, cfg);

        // Weak team 3 is away and leading 0-3 at the 85th minute of its match vs team 0.
        let m = t
            .matches
            .iter()
            .find(|m| m.home == TeamId(0) && m.away == TeamId(3))
            .unwrap();
        let mut inputs = LiveInputs::default();
        inputs.live.insert(
            m.id,
            InProgress {
                score: Scoreline::new(0, 3),
                minute: 85,
                ..Default::default()
            },
        );
        let conditioned = simulate_with_live(&t, &RankSampler, cfg, &inputs, LiveConfig::default());

        let advance = |f: &TournamentForecast, team: u32| {
            f.teams
                .iter()
                .find(|x| x.team == TeamId(team))
                .unwrap()
                .p_advance_group
        };
        assert!(
            advance(&conditioned, 3) > advance(&fresh, 3) + 0.05,
            "a 0-3 lead at 85' should lift team 3's advance odds (fresh {:.3} -> live {:.3})",
            advance(&fresh, 3),
            advance(&conditioned, 3),
        );
    }

    #[test]
    fn venue_advantage_lifts_champion_probability() {
        let t = tiny_tournament();
        let cfg = SimConfig {
            iterations: 6000,
            seed: 11,
            ..Default::default()
        };
        let neutral = simulate(&t, &RankSampler, cfg);

        // Give the weakest team (7) a strong home boost in every one of its fixtures.
        let mut inputs = LiveInputs::default();
        for m in &t.matches {
            if m.home == TeamId(7) {
                inputs.venue.insert(m.id, ((0.4, 0.2), (0.0, 0.0)));
            } else if m.away == TeamId(7) {
                inputs.venue.insert(m.id, ((0.0, 0.0), (0.4, 0.2)));
            }
        }
        let hosted = simulate_with_live(&t, &RankSampler, cfg, &inputs, LiveConfig::default());

        let champ = |f: &TournamentForecast, team: u32| {
            f.teams
                .iter()
                .find(|x| x.team == TeamId(team))
                .unwrap()
                .p_champion
        };
        assert!(
            champ(&hosted, 7) > champ(&neutral, 7),
            "home advantage should raise team 7's title odds ({:.3} -> {:.3})",
            champ(&neutral, 7),
            champ(&hosted, 7),
        );
    }

    /// A sampler with a strength gradient AND a fixed per-team rating uncertainty.
    struct NoisyRankSampler;
    impl MatchSampler for NoisyRankSampler {
        fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64) {
            RankSampler.xg(home, away)
        }
        fn rating_stderr(&self, _team: TeamId) -> f64 {
            0.30
        }
    }

    #[test]
    fn parameter_uncertainty_spreads_out_the_champion_odds() {
        let t = tiny_tournament();
        // Herfindahl concentration of the champion distribution: lower = less concentrated.
        let herfindahl = |rating_uncertainty: f64| -> f64 {
            let cfg = SimConfig {
                iterations: 8000,
                seed: 5,
                rating_uncertainty,
                ..Default::default()
            };
            simulate(&t, &NoisyRankSampler, cfg)
                .teams
                .iter()
                .map(|tf| tf.p_champion * tf.p_champion)
                .sum()
        };
        let concentrated = herfindahl(0.0); // point estimates treated as certain
        let spread = herfindahl(1.0); // resample team strength each iteration
        assert!(
            spread < concentrated,
            "parameter uncertainty should de-concentrate the title odds ({concentrated:.4} -> {spread:.4})"
        );
    }
}
