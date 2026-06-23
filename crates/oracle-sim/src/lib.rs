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
//! In-progress **group** matches are conditioned on their live score/minute/red-cards
//! (see [`simulate_with_live`] / [`InProgress`]) so the tournament forecast tracks live
//! results. A level knockout tie goes to 30' of extra time at a reduced rate and then a
//! near-50/50 penalty shootout (see [`SimConfig::extra_time_fraction`] /
//! [`SimConfig::shootout_skill`]). Documented modelling choices: the knockout bracket uses a
//! standard seeded single-elimination template (winners spread against runners-up/best-thirds)
//! rather than FIFA's exact slotting, and in-progress *knockout* matches are still built fresh.
#![forbid(unsafe_code)]

use oracle_domain::{
    MatchId, Scoreline, Stage, TeamForecast, TeamId, Tournament, TournamentForecast,
};
use oracle_model::{remaining_rates, LiveConfig, LiveState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};
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
}

/// Supplies expected goals for a (neutral-venue) matchup. Implemented for the
/// Dixon-Coles [`oracle_model::GoalModel`]; mockable in tests.
pub trait MatchSampler: Sync {
    /// Expected goals `(home_xg, away_xg)` for `home` vs `away` at a neutral site.
    fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64);
}

impl MatchSampler for oracle_model::GoalModel {
    fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64) {
        // World Cup matches are played at neutral venues.
        self.expected_goals(home, away, true)
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
        let bracket = next_pow2(qualifier_count.max(2));
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
            required,
        }
    }

    /// Sample a single goal count from Poisson(λ).
    fn sample_goals(rng: &mut StdRng, lambda: f64) -> i32 {
        if lambda <= 1e-9 {
            return 0;
        }
        match Poisson::new(lambda) {
            Ok(dist) => (dist.sample(rng) as i32).min(20),
            Err(_) => 0,
        }
    }

    /// Sample the winner's team-index of a knockout tie: 90 minutes, then 30 of extra time
    /// at a reduced rate, then a penalty shootout that is close to a coin flip.
    fn sample_knockout(&self, rng: &mut StdRng, a: usize, b: usize) -> usize {
        if a == BYE {
            return b;
        }
        if b == BYE {
            return a;
        }
        let (la, ma) = self.eg[a * self.n + b];
        let mut gh = Self::sample_goals(rng, la);
        let mut ga = Self::sample_goals(rng, ma);
        if gh == ga {
            // Extra time: a third of a match, at lower scoring intensity.
            gh += Self::sample_goals(rng, la * self.extra_time_fraction);
            ga += Self::sample_goals(rng, ma * self.extra_time_fraction);
        }
        match gh.cmp(&ga) {
            std::cmp::Ordering::Greater => a,
            std::cmp::Ordering::Less => b,
            std::cmp::Ordering::Equal => {
                // Shootout: near 50/50, only slightly tilted by the expected-goal edge.
                let p = (0.5 + self.shootout_skill * (la - ma)).clamp(0.35, 0.65);
                if rng.gen::<f64>() < p {
                    a
                } else {
                    b
                }
            }
        }
    }

    /// Play one full tournament. Returns a per-team vector of knockout rounds won:
    /// `-1` = did not qualify, `0` = qualified but lost first knockout match,
    /// `total_rounds` = champion.
    fn simulate_once(&self, rng: &mut StdRng) -> Vec<i64> {
        let mut wins = vec![-1i64; self.n];

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
                // Final goals = already scored (0 unless in progress) + sampled remainder.
                let gh = i32::from(rm.current.home) + Self::sample_goals(rng, rm.rates.0);
                let ga = i32::from(rm.current.away) + Self::sample_goals(rng, rm.rates.1);
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

        // ---- Seed the knockout bracket ----
        // Order: winners, then best thirds, then runners-up; padded with byes to a
        // power of two. Reflection pairing keeps strong winners away from each other.
        let mut seed_order: Vec<usize> = Vec::new();
        seed_order.extend(&winners);
        seed_order.extend(&qualified_thirds);
        seed_order.extend(&runners);
        for &ix in &seed_order {
            wins[ix] = 0; // qualified for the knockouts
        }
        let bracket = next_pow2(seed_order.len().max(2));
        seed_order.resize(bracket, BYE);

        // ---- Knockout: round 1 by reflection, then fold adjacent winners ----
        let mut survivors: Vec<usize> = Vec::with_capacity(bracket / 2);
        for i in 0..bracket / 2 {
            let w = self.sample_knockout(rng, seed_order[i], seed_order[bracket - 1 - i]);
            if w != BYE {
                wins[w] += 1;
            }
            survivors.push(w);
        }
        while survivors.len() > 1 {
            let mut next = Vec::with_capacity(survivors.len().div_ceil(2));
            let mut k = 0;
            while k + 1 < survivors.len() {
                let w = self.sample_knockout(rng, survivors[k], survivors[k + 1]);
                if w != BYE {
                    wins[w] += 1;
                }
                next.push(w);
                k += 2;
            }
            survivors = next;
        }

        wins
    }
}

/// Sentinel team-index representing a bracket bye.
const BYE: usize = usize::MAX;

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
        let trials = 20_000;
        let a_wins = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 0, 1) == 0)
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
        let trials = 20_000;
        let strong_wins = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 3, 4) == 3)
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
}
