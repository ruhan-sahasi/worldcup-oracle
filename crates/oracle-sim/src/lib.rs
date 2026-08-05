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

use oracle_domain::bracket::{resolve_slot, FIXED_R32};
use oracle_domain::{
    MatchId, MatchStatus, Scoreline, Stage, TeamForecast, TeamId, Tournament, TournamentForecast,
};
use oracle_model::{remaining_rates, LiveConfig, LiveState};
use oracle_numeric::Rng;
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
    /// How the per-iteration team strength shocks are correlated. Defaults to the independence the
    /// simulator has always assumed, so an unconfigured forecast is unchanged.
    pub shocks: ShockModel,
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
            shocks: ShockModel::default(),
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

    tally_iterations(&prep, config.seed, 0..config.iterations, n)
        .into_forecast(&prep.teams, config.iterations)
}

/// Simulate the given half-open range of iteration indices and tally the outcomes.
///
/// Fans out over iterations; each rayon task folds into its own tally and the per-thread tallies
/// reduce into one. Iterations are addressed by their *global* index, so a run assembled from
/// several batches draws exactly the randomness a single run of the same total length would - which
/// is what lets [`simulate_to_precision`] extend a run without invalidating what it already has.
fn tally_iterations(
    prep: &Prepared,
    seed: u64,
    range: std::ops::Range<u64>,
    n_teams: usize,
) -> Tally {
    range
        .into_par_iter()
        .fold(
            || Tally::new(n_teams, prep.required),
            |mut acc, i| {
                acc.add(&prep.simulate_once(&Streams::new(seed, i)));
                acc
            },
        )
        .reduce(|| Tally::new(n_teams, prep.required), Tally::merge)
}

/// How precisely a forecast needs to pin its champion probabilities, instead of how long it should
/// run for.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionTarget {
    /// Stop once no team's champion probability has a standard error above this.
    pub champion_std_error: f64,
    /// Iterations per batch. The target is checked between batches, so this trades how far the run
    /// can overshoot against how often it pays for a check.
    pub batch: u64,
    /// Hard ceiling, so an unreachable target terminates. A target near zero is unreachable at any
    /// finite cost, since the standard error falls only as 1/sqrt(n).
    pub max_iterations: u64,
}

impl Default for PrecisionTarget {
    fn default() -> Self {
        Self {
            champion_std_error: 0.002,
            batch: 5_000,
            max_iterations: 500_000,
        }
    }
}

/// A forecast together with what it actually achieved.
#[derive(Debug, Clone)]
pub struct PreciseForecast {
    pub forecast: TournamentForecast,
    /// The largest champion-probability standard error across teams when the run stopped.
    pub worst_champion_std_error: f64,
    /// Whether the target was reached, as opposed to the iteration ceiling.
    pub target_met: bool,
}

/// Simulate until the champion probabilities are pinned to `target`, rather than for a fixed number
/// of iterations.
///
/// A fixed iteration count is a guess about how hard the question is. It over-spends on a field with
/// a runaway favourite and under-spends on an open one, and it never reports which happened. This
/// runs in batches and stops when the worst team's standard error clears the target, so the caller
/// asks for the precision it needs and finds out what it cost.
///
/// One honest caveat: stopping on an observed standard error is a sequential decision, and the
/// stopped estimate carries a slight optimistic bias - the run is more likely to halt on a batch
/// where the sample happened to look settled. The effect is small next to the target itself at these
/// batch sizes, and reporting the achieved error rather than the target keeps it visible.
pub fn simulate_to_precision<S: MatchSampler>(
    tournament: &Tournament,
    sampler: &S,
    config: SimConfig,
    inputs: &LiveInputs,
    live_config: LiveConfig,
    target: PrecisionTarget,
) -> PreciseForecast {
    let prep = Prepared::build(tournament, sampler, config, inputs, live_config);
    let n = prep.teams.len();
    let batch = target.batch.max(1);
    let ceiling = target.max_iterations.max(batch);

    let mut tally = Tally::new(n, prep.required);
    let mut done = 0u64;
    loop {
        let next = (done + batch).min(ceiling);
        tally = tally.merge(tally_iterations(&prep, config.seed, done..next, n));
        done = next;
        let worst = tally.worst_champion_std_error(done);
        if worst <= target.champion_std_error || done >= ceiling {
            return PreciseForecast {
                forecast: tally.into_forecast(&prep.teams, done),
                worst_champion_std_error: worst,
                target_met: worst <= target.champion_std_error,
            };
        }
    }
}

/// Whether `team` was champion in each simulated tournament, one entry per iteration in iteration
/// order.
///
/// The order is the point. Two calls with the same `config.seed` draw from the same labelled
/// streams, so entry `i` of each describes the *same* simulated universe under two different
/// premises - which is what lets [`PairedDifference`] difference them iteration by iteration
/// instead of comparing two aggregate probabilities.
pub fn champion_indicators<S: MatchSampler>(
    tournament: &Tournament,
    sampler: &S,
    config: SimConfig,
    inputs: &LiveInputs,
    live_config: LiveConfig,
    team: TeamId,
) -> Vec<bool> {
    let prep = Prepared::build(tournament, sampler, config, inputs, live_config);
    let Some(ix) = prep.teams.iter().position(|&t| t == team) else {
        return vec![false; config.iterations as usize];
    };
    let need = prep.required[5];
    (0..config.iterations)
        .into_par_iter()
        .map(|i| prep.simulate_once(&Streams::new(config.seed, i))[ix] >= need)
        .collect()
}

/// A difference between two coupled simulations, with the uncertainty of the *difference*.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PairedDifference {
    /// The mean difference (scenario minus baseline).
    pub mean: f64,
    /// The standard error of that mean, from the spread of the per-iteration differences.
    pub std_error: f64,
    pub iterations: u64,
}

impl PairedDifference {
    /// Difference two per-iteration series that were drawn from the same streams.
    ///
    /// The standard error comes from the variance of the per-iteration differences, which is the
    /// only way to get it right when the two series are correlated. Treating the two aggregate
    /// probabilities as independent and adding their variances would overstate the error wherever
    /// the coupling bites - and understate nothing, but a forecast that reports pessimistic error
    /// bars on a real effect is as misleading as one that reports none.
    ///
    /// Series are compared up to their common length; a mismatch means one run was configured with
    /// fewer iterations, and the extra tail has no partner to pair with.
    pub fn from_indicators(scenario: &[bool], baseline: &[bool]) -> Self {
        let n = scenario.len().min(baseline.len());
        if n == 0 {
            return Self {
                mean: 0.0,
                std_error: 0.0,
                iterations: 0,
            };
        }
        // Each difference is one of -1, 0, +1.
        let diffs = scenario
            .iter()
            .zip(baseline)
            .take(n)
            .map(|(&s, &b)| f64::from(i8::from(s) - i8::from(b)));
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for d in diffs {
            sum += d;
            sum_sq += d * d;
        }
        let nf = n as f64;
        let mean = sum / nf;
        // Sample variance with Bessel's correction, then the standard error of the mean.
        let var = if n > 1 {
            (sum_sq - nf * mean * mean) / (nf - 1.0)
        } else {
            0.0
        };
        Self {
            mean,
            std_error: (var.max(0.0) / nf).sqrt(),
            iterations: n as u64,
        }
    }

    /// The half-width of an approximate 95% interval around [`mean`](Self::mean).
    pub fn margin_95(&self) -> f64 {
        1.96 * self.std_error
    }

    /// Whether the difference is distinguishable from zero at ~95% confidence. A swing that fails
    /// this is a swing the Monte-Carlo cannot actually resolve, however large it looks.
    pub fn is_significant(&self) -> bool {
        self.mean.abs() > self.margin_95()
    }
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
                let streams = Streams::new(config.seed, i);
                let mut met: Option<u8> = None;
                prep.simulate_once_with(&streams, |round, x, y| {
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
    /// How those per-team shocks are correlated with each other and across attack/defence.
    shocks: ShockModel,
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
    /// The fixture's own identity, used to address its goal-draw stream. Keying on the match id
    /// rather than on its position in the fixture list is what makes the stream stable when a
    /// scenario removes some *other* match from the remaining set.
    id: MatchId,
    home_ix: usize,
    away_ix: usize,
    rates: (f64, f64),
    current: Scoreline,
}

/// How a tournament's per-team strength shocks are correlated.
///
/// # The assumption this makes visible
///
/// Each iteration already draws a persistent per-team shock: an attack and a defence offset held
/// fixed across all of that team's matches, so a side that turns out better than its rating is
/// better all tournament. What was never stated is that those draws are **independent** - a team's
/// attack shock uncorrelated with its own defence shock, and every team's shock uncorrelated with
/// every other's.
///
/// Neither is obviously right. A squad better than its rating is usually better at both ends, not
/// at one. And a tournament has conditions of its own - the ball, the refereeing, the heat - that
/// push scoring the same way for everybody. Independence was an implicit modelling choice that no
/// configuration could express and no test could vary.
///
/// This type makes it a parameter. [`Default`] is exact independence, so a forecast is unchanged
/// unless a caller asks for something else.
///
/// # Why there are two knobs and not one
///
/// The rate for team `a` against `b` is `exp(att[a] - def[b])`, which is what makes a single
/// "global quality" factor useless: raising every team's attack and defence together cancels in the
/// subtraction. A shared factor only does anything if it is **antisymmetric** - attack up and
/// defence down - which is exactly a high-scoring tournament. So the two knobs are doing genuinely
/// different work rather than being two dials on the same effect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShockModel {
    /// Correlation between a team's own attack and defence shocks, in `[-1, 1]`.
    ///
    /// Positive means a side better than its rating tends to be better at both ends. Zero is the
    /// independence the simulator has always assumed.
    pub attack_defence: f64,
    /// Share of each team's shock variance carried by a single tournament-wide scoring environment,
    /// in `[0, 1]`.
    ///
    /// The shared draw enters attack positively and defence negatively, so a high draw is a
    /// high-scoring tournament for everyone rather than a uniformly stronger field. Zero is the
    /// per-team independence the simulator has always assumed.
    pub environment: f64,
}

impl Default for ShockModel {
    fn default() -> Self {
        // Exactly the behaviour before this type existed. Defaulting to anything else would change
        // every published champion probability without a commit saying so.
        Self {
            attack_defence: 0.0,
            environment: 0.0,
        }
    }
}

impl ShockModel {
    /// Whether any correlation is configured. `false` means the fast, historical path.
    pub fn is_independent(&self) -> bool {
        self.attack_defence == 0.0 && self.environment == 0.0
    }

    /// The parameters clamped to their valid ranges.
    ///
    /// Clamped rather than rejected because these arrive from a CLI flag or a query string, where a
    /// caller experimenting with values should get the nearest sensible model rather than an error -
    /// and because a correlation outside `[-1, 1]` has no Cholesky factor, so it would otherwise
    /// produce `NaN` shocks and a silently empty forecast.
    pub fn sanitized(&self) -> Self {
        Self {
            attack_defence: if self.attack_defence.is_finite() {
                self.attack_defence.clamp(-1.0, 1.0)
            } else {
                0.0
            },
            environment: if self.environment.is_finite() {
                self.environment.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }
}

/// Which *kind* of draw a substream carries, so streams serving different purposes can never
/// collide even when their entity indices coincide (team 3 and bracket slot 3 are unrelated).
mod stream_kind {
    pub const ITERATION: u32 = 0;
    pub const TEAM_STRENGTH: u32 = 1;
    /// The tournament-wide scoring environment: one draw per iteration, shared by every team.
    pub const ENVIRONMENT: u32 = 4;
    pub const GROUP_MATCH: u32 = 2;
    pub const KO_TIE: u32 = 3;
}

/// The random draws of one simulated tournament, addressed by *what they belong to* rather than by
/// the order in which they happen.
///
/// A sequential generator makes every draw depend on how many draws preceded it, so a change early
/// in a tournament shifts all the randomness after it. Two runs that differ in one match then share
/// no randomness downstream, and a *difference* between them carries the full noise of both. Since
/// the difference is exactly what the kingmaker, collision and sensitivity analyses report, that
/// noise is the thing worth engineering away: giving each entity its own labelled stream means an
/// entity the change does not touch draws identical numbers in both runs.
///
/// Streams are derived, not stored - `Rng::stream` is a few multiplications, so addressing one per
/// team or per match inside the innermost loop is affordable.
#[derive(Clone, Copy)]
struct Streams {
    /// This iteration's root seed. Folding the iteration in once here means every entity stream
    /// below is automatically distinct across iterations without the iteration appearing in each
    /// address.
    iteration_seed: u64,
}

impl Streams {
    fn new(seed: u64, iteration: u64) -> Self {
        Self {
            iteration_seed: Rng::stream(seed, stream_kind::ITERATION, iteration).next_u64(),
        }
    }

    /// The stream for one team's strength perturbation this iteration.
    fn team_strength(&self, team_ix: usize) -> Rng {
        Rng::stream(
            self.iteration_seed,
            stream_kind::TEAM_STRENGTH,
            team_ix as u64,
        )
    }

    /// The stream for this iteration's single shared scoring-environment draw.
    fn environment(&self) -> Rng {
        Rng::stream(self.iteration_seed, stream_kind::ENVIRONMENT, 0)
    }

    /// The stream for one group fixture's goal draws this iteration.
    fn group_match(&self, id: MatchId) -> Rng {
        Rng::stream(
            self.iteration_seed,
            stream_kind::GROUP_MATCH,
            u64::from(id.0),
        )
    }

    /// The stream for the knockout tie at a given bracket position this iteration.
    ///
    /// Addressed by *position*, not by the pair who reach it. Who plays a quarter-final is
    /// precisely what a scenario changes, so keying on the pairing would give the tie fresh
    /// randomness whenever the scenario altered who got there - reintroducing the noise this is
    /// meant to remove. Keying on the slot means the tie's luck (its scorelines, whether it goes to
    /// extra time, which way a shootout falls) is a property of the bracket position, and swapping
    /// the occupants changes the result only through their strengths.
    fn ko_tie(&self, round: u8, slot: usize) -> Rng {
        Rng::stream(
            self.iteration_seed,
            stream_kind::KO_TIE,
            (u64::from(round) << 32) | slot as u64,
        )
    }
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
                            id: m.id,
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
            shocks: config.shocks.sanitized(),
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
    fn sample_goals(&self, rng: &mut Rng, lambda: f64) -> i32 {
        if lambda <= 1e-9 {
            return 0;
        }
        let rate = if self.dispersion > 0.0 {
            let r = self.dispersion;
            rng.gamma(r, lambda / r).max(1e-9)
        } else {
            lambda
        };
        (rng.poisson(rate) as i32).min(20)
    }

    /// Sample the winner's team-index of a knockout tie: 90 minutes, then 30 of extra time
    /// at a reduced rate, then a penalty shootout that is close to a coin flip.
    /// Returns `(winner, went_to_extra_time)`. `fatigue` is a log-attack penalty per side carried
    /// from a previous round's extra time.
    fn sample_knockout(
        &self,
        rng: &mut Rng,
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
                if rng.chance(p) {
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
        rng: &mut Rng,
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
        rng: &mut Rng,
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
                if rng.chance(p) {
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
    /// Each team draws from its own stream, so a team's luck this iteration depends only on the
    /// team and the iteration - not on how many other teams were drawn first. Both of its shifts
    /// come from that one stream, which keeps attack and defence perturbations together where they
    /// belong.
    fn draw_strength_shifts(&self, streams: &Streams) -> (Vec<f64>, Vec<f64>) {
        if !self.has_uncertainty {
            return (vec![0.0; self.n], vec![0.0; self.n]);
        }
        let shocks = self.shocks;
        // The independent path is kept separate rather than falling out of the general formula with
        // zero parameters. Two reasons: it consumes exactly the draws it always did, so a default
        // forecast stays bit-identical to every one published before this; and it skips the shared
        // draw entirely, so the common case pays nothing for a feature it is not using.
        if shocks.is_independent() {
            let mut att = vec![0.0; self.n];
            let mut def = vec![0.0; self.n];
            for (i, &sigma) in self.team_sigma.iter().enumerate() {
                if sigma > 0.0 {
                    let mut rng = streams.team_strength(i);
                    att[i] = sigma * rng.normal();
                    def[i] = sigma * rng.normal();
                }
            }
            return (att, def);
        }

        // ---- The factor model ----
        //
        // Each team's pair of shocks is built from a shared draw and two of its own:
        //
        //     att[i] = sigma_i * (  sqrt(w) * G + sqrt(1 - w) * a_i )
        //     def[i] = sigma_i * ( -sqrt(w) * G + sqrt(1 - w) * d_i )
        //
        // where `G` is one standard normal for the whole tournament, `a_i` and `d_i` are the team's
        // own standard normals correlated at `rho` by a 2x2 Cholesky factor, and `w` is the share of
        // variance the shared factor carries.
        //
        // The `sqrt(w)` / `sqrt(1 - w)` split is what keeps each team's *marginal* shock variance at
        // `sigma_i^2` however the correlation is configured. Without it, adding a factor would
        // silently make every team more uncertain than its fitted rating says - a change to the
        // model's confidence disguised as a change to its correlation structure.
        //
        // `G` enters attack positively and defence negatively because the rate is
        // `exp(att[a] - def[b])`: a factor that raised both would cancel in the subtraction and do
        // nothing at all. Antisymmetric, it reads as a high-scoring tournament for everyone.
        let w = shocks.environment;
        let (shared, own) = (w.sqrt(), (1.0 - w).sqrt());
        let rho = shocks.attack_defence;
        let rho_c = (1.0 - rho * rho).max(0.0).sqrt();
        let g = streams.environment().normal();

        let mut att = vec![0.0; self.n];
        let mut def = vec![0.0; self.n];
        for (i, &sigma) in self.team_sigma.iter().enumerate() {
            if sigma > 0.0 {
                let mut rng = streams.team_strength(i);
                let z1 = rng.normal();
                let z2 = rng.normal();
                // Cholesky of [[1, rho], [rho, 1]]: unit variances, correlation rho.
                let (a_i, d_i) = (z1, rho * z1 + rho_c * z2);
                att[i] = sigma * (shared * g + own * a_i);
                def[i] = sigma * (-shared * g + own * d_i);
            }
        }
        (att, def)
    }

    /// Play one full tournament (the normal forecast path).
    fn simulate_once(&self, streams: &Streams) -> Vec<i64> {
        self.simulate_once_with(streams, |_, _, _| {})
    }

    /// Play one full tournament, invoking `on_tie(round, home_ix, away_ix)` for every knockout tie
    /// contested (round 0 = Round of 32, 1 = R16, 2 = QF, 3 = SF, 4 = Final). Returns a per-team
    /// vector of knockout rounds won: `-1` = did not qualify, `0` = qualified but lost the first
    /// knockout match, `total_rounds` = champion. The forecast passes a no-op observer; the
    /// meeting analysis records the pairings.
    fn simulate_once_with<F: FnMut(u8, usize, usize)>(
        &self,
        streams: &Streams,
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
        let (att, def) = self.draw_strength_shifts(streams);

        // `survivors` holds the 16 first-round winners in bracket order; the fold below plays
        // out R16 -> Final by repeatedly pairing adjacent survivors.
        let mut survivors: Vec<usize> = if !self.ko_r32.is_empty() {
            // ---- Real knockout bracket present (group stage already complete) ----
            // Play the materialized Round of 32 directly: a finished tie keeps its result, an
            // in-progress tie is conditioned on its live score, a scheduled tie is sampled. The
            // group stage is not re-simulated, so finished knockout results stay fixed.
            self.ko_r32
                .iter()
                .enumerate()
                .map(|(slot, &(a, b))| {
                    wins[a] = 0;
                    wins[b] = 0;
                    on_tie(0, a, b);
                    let mut rng = streams.ko_tie(0, slot);
                    let (w, et) = self.play_ko_tie(&mut rng, a, b, &att, &def, (0.0, 0.0));
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
                    // Each fixture draws its goals from its own stream, so conditioning some other
                    // match cannot shift this one's scoreline.
                    let mut mrng = streams.group_match(rm.id);
                    let gh = i32::from(rm.current.home) + self.sample_goals(&mut mrng, rate_h);
                    let ga = i32::from(rm.current.away) + self.sample_goals(&mut mrng, rate_a);
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
                    .enumerate()
                    .map(|(slot, (top, bottom))| {
                        let a = resolve_slot(top, &winners, &runners, &qualified_thirds);
                        let b = resolve_slot(bottom, &winners, &runners, &qualified_thirds);
                        wins[a] = 0;
                        wins[b] = 0;
                        on_tie(0, a, b);
                        let mut rng = streams.ko_tie(0, slot);
                        let (w, et) = self.sample_knockout(&mut rng, a, b, &att, &def, (0.0, 0.0));
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
                        let mut rng = streams.ko_tie(0, i);
                        let (w, et) = self.sample_knockout(&mut rng, a, b, &att, &def, (0.0, 0.0));
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
                let mut rng = streams.ko_tie(round, k / 2);
                let (w, et) = self.play_ko_tie(&mut rng, a, b, &att, &def, fat);
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

    /// The largest standard error on any team's champion probability after `n` iterations.
    ///
    /// Each champion count is a binomial over `n` trials. The obvious estimate of its standard
    /// error, `sqrt(p(1-p)/n)` with `p` the observed share, is unusable as a *stopping rule*: it
    /// returns exactly zero for any team that has not yet won a single iteration. Since most of a
    /// 48-team field never wins one, and after one iteration every team has a share of exactly 0 or
    /// 1, a run could halt claiming perfect precision having learned almost nothing.
    ///
    /// So the share is shrunk by the Agresti-Coull "plus four" adjustment - two notional wins and
    /// two notional losses - before the error is computed. A team on zero wins then carries a small
    /// non-zero error that shrinks properly with `n`, which is the honest statement: never having
    /// seen an event is not the same as knowing it cannot happen.
    ///
    /// The maximum over teams is the run's binding constraint, since a mid-probability contender is
    /// far harder to pin down than a no-hoper.
    fn worst_champion_std_error(&self, n: u64) -> f64 {
        if n == 0 {
            return f64::INFINITY;
        }
        let adjusted = n as f64 + 4.0;
        self.counts
            .iter()
            .map(|c| {
                let p = (c[5] as f64 + 2.0) / adjusted;
                (p * (1.0 - p) / adjusted).sqrt()
            })
            .fold(0.0, f64::max)
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
        let mut rng = Rng::new(11);
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
        let mut rng = Rng::new(5);
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
        let mut rng = Rng::new(3);
        let et = (0..trials)
            .filter(|_| prep.sample_knockout(&mut rng, 0, 1, &z, &z, (0.0, 0.0)).1)
            .count();
        assert!(
            et > trials / 20,
            "even ties should reach extra time sometimes: {et}/{trials}"
        );

        // A side carrying extra-time fatigue wins fewer ties than when fresh (same RNG stream).
        let win = |fatigue: (f64, f64), seed: u64| -> f64 {
            let mut r = Rng::new(seed);
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
            let mut rng = Rng::new(1);
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
        let mut rng = Rng::new(99);
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
        let mut rng = Rng::new(7);
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

    /// Two forecasts of `tournament` at the given seed, differing only in the conditioned result.
    /// Returns the mean absolute difference in group-qualification probability across every team
    /// outside the conditioned group.
    fn qualification_drift(conditioned_seed: u64, baseline_seed: u64) -> f64 {
        let base = full_tournament();
        // Team 0 (the strongest) is handed a heavy win over its group rival team 1. Nothing outside
        // group A (teams 0-3) is touched by this.
        let mut cond = base.clone();
        let m = cond
            .matches
            .iter_mut()
            .find(|m| m.home == TeamId(0) && m.away == TeamId(1))
            .expect("group A fixture 0 v 1");
        m.status = MatchStatus::Finished;
        m.score = Scoreline::new(4, 0);

        let run = |t: &Tournament, seed: u64| {
            simulate(
                t,
                &RankSampler,
                SimConfig {
                    iterations: 3000,
                    seed,
                    ..Default::default()
                },
            )
            .teams
        };
        let (b, c) = (run(&base, baseline_seed), run(&cond, conditioned_seed));
        let p = |v: &[TeamForecast], t: TeamId| {
            v.iter()
                .find(|f| f.team == t)
                .expect("team in forecast")
                .p_advance_group
        };
        let others: Vec<TeamId> = (4..48u32).map(TeamId).collect();
        others
            .iter()
            .map(|&t| (p(&c, t) - p(&b, t)).abs())
            .sum::<f64>()
            / others.len() as f64
    }

    #[test]
    fn batching_a_run_matches_running_it_whole() {
        // The property the precision loop rests on: iterations are addressed by global index, so a
        // run assembled from batches must be bit-identical to one run straight through. If this
        // fails, extending a run silently rerandomises what it already had.
        let t = full_tournament();
        let cfg = SimConfig {
            iterations: 4000,
            seed: 5,
            ..Default::default()
        };
        let whole = simulate(&t, &RankSampler, cfg);

        let prep = Prepared::build(
            &t,
            &RankSampler,
            cfg,
            &LiveInputs::default(),
            LiveConfig::default(),
        );
        let n = prep.teams.len();
        let batched = tally_iterations(&prep, cfg.seed, 0..1500, n)
            .merge(tally_iterations(&prep, cfg.seed, 1500..4000, n))
            .into_forecast(&prep.teams, 4000);
        assert_eq!(whole.teams, batched.teams, "batching changed the forecast");
    }

    #[test]
    fn a_precision_run_reaches_its_target_and_reports_the_cost() {
        let t = full_tournament();
        let target = PrecisionTarget {
            champion_std_error: 0.004,
            batch: 2_000,
            max_iterations: 200_000,
        };
        let out = simulate_to_precision(
            &t,
            &RankSampler,
            SimConfig {
                seed: 9,
                ..Default::default()
            },
            &LiveInputs::default(),
            LiveConfig::default(),
            target,
        );
        assert!(out.target_met, "target should be reachable");
        assert!(
            out.worst_champion_std_error <= 0.004,
            "achieved {}",
            out.worst_champion_std_error
        );
        // It should stop as soon as the target is met, not run to the ceiling.
        assert!(out.forecast.iterations < 200_000);
        assert_eq!(
            out.forecast.iterations % 2_000,
            0,
            "stops on a batch boundary"
        );
        // The forecast itself must still be coherent.
        let total: f64 = out.forecast.teams.iter().map(|f| f.p_champion).sum();
        assert!((total - 1.0).abs() < 1e-9, "champion mass {total}");
    }

    #[test]
    fn a_tighter_target_costs_more_iterations() {
        let t = full_tournament();
        let run = |se: f64| {
            simulate_to_precision(
                &t,
                &RankSampler,
                SimConfig {
                    seed: 3,
                    ..Default::default()
                },
                &LiveInputs::default(),
                LiveConfig::default(),
                PrecisionTarget {
                    champion_std_error: se,
                    batch: 2_000,
                    max_iterations: 200_000,
                },
            )
            .forecast
            .iterations
        };
        // The standard error falls as 1/sqrt(n), so halving it should cost roughly four times as
        // many iterations. Assert only the direction, which is what the feature promises.
        assert!(run(0.003) > run(0.006), "a tighter target must cost more");
    }

    #[test]
    fn an_unreachable_target_stops_at_the_ceiling_and_says_so() {
        let t = full_tournament();
        let out = simulate_to_precision(
            &t,
            &RankSampler,
            SimConfig::default(),
            &LiveInputs::default(),
            LiveConfig::default(),
            PrecisionTarget {
                // No finite run achieves this; the error falls only as 1/sqrt(n).
                champion_std_error: 1e-9,
                batch: 1_000,
                max_iterations: 3_000,
            },
        );
        assert!(!out.target_met, "must not claim an unmet target");
        assert_eq!(out.forecast.iterations, 3_000, "stops at the ceiling");
        assert!(out.worst_champion_std_error > 1e-9);
    }

    #[test]
    fn a_degenerate_precision_target_still_terminates() {
        let t = full_tournament();
        let out = simulate_to_precision(
            &t,
            &RankSampler,
            SimConfig::default(),
            &LiveInputs::default(),
            LiveConfig::default(),
            // A zero batch and a ceiling below it: both are clamped rather than looping forever.
            PrecisionTarget {
                champion_std_error: 0.0,
                batch: 0,
                max_iterations: 0,
            },
        );
        assert_eq!(out.forecast.iterations, 1, "one clamped iteration");
        // A single iteration must not be mistaken for perfect precision. The unshrunk binomial
        // error would be exactly zero here, since every team's share is 0 or 1 after one trial.
        assert!(
            !out.target_met,
            "must not claim a met target after one trial"
        );
        assert!(
            out.worst_champion_std_error > 0.1,
            "one iteration should look very uncertain, got {}",
            out.worst_champion_std_error
        );
    }

    #[test]
    fn a_team_with_no_wins_still_carries_uncertainty() {
        // The stopping statistic must not read zero error off an unobserved event. In this field the
        // weakest teams never win a single iteration, so the unshrunk formula would give them
        // exactly zero - and with a small enough run, drag the worst-case error to zero with them.
        let t = full_tournament();
        let out = simulate_to_precision(
            &t,
            &RankSampler,
            SimConfig {
                seed: 4,
                ..Default::default()
            },
            &LiveInputs::default(),
            LiveConfig::default(),
            PrecisionTarget {
                champion_std_error: 0.01,
                batch: 500,
                max_iterations: 10_000,
            },
        );
        let never_won = out
            .forecast
            .teams
            .iter()
            .filter(|f| f.p_champion == 0.0)
            .count();
        assert!(never_won > 0, "this field should have teams on zero");
        assert!(
            out.worst_champion_std_error > 0.0,
            "a run containing unobserved events cannot have zero error"
        );
    }

    /// Draw `iters` tournaments' worth of shifts for a field of `n` equally uncertain teams, and
    /// return the (att, def) pairs so their moments can be checked.
    fn shift_samples(shocks: ShockModel, n: usize, iters: u64) -> Vec<(Vec<f64>, Vec<f64>)> {
        let t = full_tournament();
        let cfg = SimConfig {
            iterations: iters,
            seed: 4,
            shocks,
            ..Default::default()
        };
        // A sampler with a flat, non-zero per-team uncertainty, so every team's sigma is 1.
        struct UnitSigma;
        impl MatchSampler for UnitSigma {
            fn xg(&self, _h: TeamId, _a: TeamId) -> (f64, f64) {
                (1.3, 1.3)
            }
            fn rating_stderr(&self, _t: TeamId) -> f64 {
                1.0
            }
        }
        let prep = Prepared::build(
            &t,
            &UnitSigma,
            cfg,
            &LiveInputs::default(),
            LiveConfig::default(),
        );
        assert!(prep.has_uncertainty, "the fixture must actually have sigma");
        (0..iters)
            .map(|i| prep.draw_strength_shifts(&Streams::new(cfg.seed, i)))
            .map(|(a, d)| (a[..n].to_vec(), d[..n].to_vec()))
            .collect()
    }

    fn corr(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let cov = xs
            .iter()
            .zip(ys)
            .map(|(x, y)| (x - mx) * (y - my))
            .sum::<f64>()
            / n;
        let sx = (xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>() / n).sqrt();
        let sy = (ys.iter().map(|y| (y - my).powi(2)).sum::<f64>() / n).sqrt();
        cov / (sx * sy)
    }

    fn variance(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let m = xs.iter().sum::<f64>() / n;
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n
    }

    #[test]
    fn a_teams_attack_and_defence_shocks_take_the_configured_correlation() {
        // With no shared factor, the observed within-team correlation is exactly rho.
        for rho in [-0.6, 0.0, 0.4, 0.9] {
            let samples = shift_samples(
                ShockModel {
                    attack_defence: rho,
                    environment: 0.0,
                },
                1,
                8000,
            );
            let att: Vec<f64> = samples.iter().map(|(a, _)| a[0]).collect();
            let def: Vec<f64> = samples.iter().map(|(_, d)| d[0]).collect();
            let got = corr(&att, &def);
            assert!((got - rho).abs() < 0.04, "rho={rho}: recovered {got:.3}");
        }
    }

    #[test]
    fn the_shared_factor_correlates_different_teams() {
        // Two different teams' attack shocks are independent under the historical model and
        // correlated at w once a scoring environment is configured.
        for w in [0.0, 0.3, 0.8] {
            let samples = shift_samples(
                ShockModel {
                    attack_defence: 0.0,
                    environment: w,
                },
                2,
                8000,
            );
            let a0: Vec<f64> = samples.iter().map(|(a, _)| a[0]).collect();
            let a1: Vec<f64> = samples.iter().map(|(a, _)| a[1]).collect();
            let got = corr(&a0, &a1);
            assert!(
                (got - w).abs() < 0.05,
                "w={w}: cross-team attack correlation {got:.3}"
            );

            // And it is antisymmetric: one team's attack against another's defence is -w.
            let d1: Vec<f64> = samples.iter().map(|(_, d)| d[1]).collect();
            let cross = corr(&a0, &d1);
            assert!(
                (cross + w).abs() < 0.05,
                "w={w}: attack vs other defence {cross:.3}, expected {}",
                -w
            );
        }
    }

    #[test]
    fn every_configuration_preserves_each_teams_marginal_variance() {
        // The invariant that makes the factor model honest. A team's shock spread is its *fitted*
        // uncertainty; if adding correlation also inflated it, the change would be to the model's
        // confidence disguised as a change to its correlation structure.
        for (rho, w) in [
            (0.0, 0.0),
            (0.7, 0.0),
            (0.0, 0.6),
            (0.5, 0.4),
            (-0.8, 0.9),
            (1.0, 1.0),
        ] {
            let samples = shift_samples(
                ShockModel {
                    attack_defence: rho,
                    environment: w,
                },
                1,
                8000,
            );
            let att: Vec<f64> = samples.iter().map(|(a, _)| a[0]).collect();
            let def: Vec<f64> = samples.iter().map(|(_, d)| d[0]).collect();
            for (label, v) in [("attack", variance(&att)), ("defence", variance(&def))] {
                assert!(
                    (v - 1.0).abs() < 0.06,
                    "rho={rho} w={w}: {label} variance {v:.3}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn the_two_parameters_combine_as_the_algebra_says() {
        // With both knobs on, the within-team correlation is (1 - w) * rho - w: the shared factor
        // pushes attack and defence apart while rho pulls them together. Worth pinning, because a
        // reader setting rho = 0.5 and w = 0.5 would otherwise reasonably expect 0.5, not 0.
        for (rho, w) in [(0.5, 0.5), (0.8, 0.2), (0.0, 0.5)] {
            let samples = shift_samples(
                ShockModel {
                    attack_defence: rho,
                    environment: w,
                },
                1,
                8000,
            );
            let att: Vec<f64> = samples.iter().map(|(a, _)| a[0]).collect();
            let def: Vec<f64> = samples.iter().map(|(_, d)| d[0]).collect();
            let expected = (1.0 - w) * rho - w;
            let got = corr(&att, &def);
            assert!(
                (got - expected).abs() < 0.05,
                "rho={rho} w={w}: got {got:.3}, algebra says {expected:.3}"
            );
        }
    }

    #[test]
    fn the_default_shock_model_consumes_the_historical_draw_sequence() {
        // The guard on every published forecast. The independent path exists so that an unconfigured
        // run draws exactly what it drew before the factor model was added; this pins that by
        // reconstructing the old sequence by hand - two normals per team from the team's own stream,
        // and no shared draw at all - and asserting the shifts match to the last bit.
        //
        // Comparing against a recorded table would catch a change too, but only after someone
        // noticed the table was stale. This fails in the same commit that breaks it.
        let t = full_tournament();
        let cfg = SimConfig {
            iterations: 1,
            seed: 99,
            ..Default::default()
        };
        struct UnitSigma;
        impl MatchSampler for UnitSigma {
            fn xg(&self, _h: TeamId, _a: TeamId) -> (f64, f64) {
                (1.3, 1.3)
            }
            fn rating_stderr(&self, _t: TeamId) -> f64 {
                0.7
            }
        }
        let prep = Prepared::build(
            &t,
            &UnitSigma,
            cfg,
            &LiveInputs::default(),
            LiveConfig::default(),
        );
        assert!(
            prep.shocks.is_independent(),
            "the default must be independent"
        );

        let streams = Streams::new(cfg.seed, 0);
        let (att, def) = prep.draw_strength_shifts(&streams);

        for i in 0..prep.n {
            // The pre-factor-model sequence: sigma * normal() twice, from the team's own stream.
            let mut rng = streams.team_strength(i);
            let want_att = prep.team_sigma[i] * rng.normal();
            let want_def = prep.team_sigma[i] * rng.normal();
            assert_eq!(att[i], want_att, "team {i} attack shift changed");
            assert_eq!(def[i], want_def, "team {i} defence shift changed");
        }
    }

    #[test]
    fn a_default_forecast_is_unchanged_by_the_shock_model_existing() {
        // The same property one level up, where a reader would notice it: two configs that differ
        // only in spelling out the default must give an identical forecast.
        let t = full_tournament();
        let implicit = SimConfig {
            iterations: 3000,
            seed: 8,
            ..Default::default()
        };
        let explicit = SimConfig {
            shocks: ShockModel {
                attack_defence: 0.0,
                environment: 0.0,
            },
            ..implicit
        };
        let a = simulate(&t, &RankSampler, implicit);
        let b = simulate(&t, &RankSampler, explicit);
        assert_eq!(a.teams, b.teams);
    }

    #[test]
    fn a_symmetric_shared_factor_would_cancel_in_the_rate() {
        // The reason the environment factor enters defence with a minus sign, demonstrated rather
        // than asserted in a comment.
        //
        // The rate for `a` against `b` is `eg * exp(att[a] - def[b])`. Take two equally uncertain
        // teams and a shared draw `g`. Symmetric (both signs positive), the exponent is
        //
        //     (sqrt(w)g + own_a) - (sqrt(w)g + own_b) = own_a - own_b
        //
        // with `g` gone entirely - the factor could take any value and change nothing. Antisymmetric,
        // it survives as `2*sqrt(w)*g`, which is what a high-scoring tournament means.
        let (w, sigma) = (0.5f64, 1.0f64);
        let (shared, own_scale) = (w.sqrt(), (1.0f64 - w).sqrt());
        let (own_a, own_b) = (0.3f64, -0.2f64);

        for g in [-2.0f64, -0.5, 0.0, 0.5, 2.0] {
            // What the implementation does: attack + shared, defence - shared.
            let att_a = sigma * (shared * g + own_scale * own_a);
            let def_b = sigma * (-shared * g + own_scale * own_b);
            let antisymmetric = att_a - def_b;

            // The rejected alternative: both signs positive.
            let sym_att_a = sigma * (shared * g + own_scale * own_a);
            let sym_def_b = sigma * (shared * g + own_scale * own_b);
            let symmetric = sym_att_a - sym_def_b;

            // The symmetric version is the same number for every g: the factor has no effect.
            let no_factor = own_scale * (own_a - own_b);
            assert!(
                (symmetric - no_factor).abs() < 1e-12,
                "a symmetric factor should vanish, but g={g} moved the exponent to {symmetric}"
            );
            // The antisymmetric one scales with g, which is the point.
            let expected = own_scale * (own_a - own_b) + 2.0 * shared * g;
            assert!(
                (antisymmetric - expected).abs() < 1e-12,
                "g={g}: exponent {antisymmetric}, expected {expected}"
            );
        }
    }

    #[test]
    fn the_environment_factor_moves_total_goals_not_the_winner() {
        // The behavioural consequence: a shared scoring environment should change how many goals a
        // tournament produces without systematically favouring anyone, since it lifts every team's
        // rate together. Both halves matter - if it moved the champion odds much, it would be acting
        // as a strength factor and the name would be wrong.
        let t = full_tournament();
        let run = |w: f64| {
            simulate(
                &t,
                &RankSampler,
                SimConfig {
                    iterations: 6000,
                    seed: 21,
                    shocks: ShockModel {
                        attack_defence: 0.0,
                        environment: w,
                    },
                    ..Default::default()
                },
            )
        };
        let (base, env) = (run(0.0), run(0.9));

        // The strongest team's title odds should barely move.
        let champ = |f: &TournamentForecast, id: u32| {
            f.teams
                .iter()
                .find(|x| x.team == TeamId(id))
                .expect("team")
                .p_champion
        };
        let moved = (champ(&env, 0) - champ(&base, 0)).abs();
        assert!(
            moved < 0.03,
            "a scoring-environment factor should not act as a strength factor: the favourite moved \
             {moved:.4}"
        );
        // And the whole field still sums to one champion.
        for f in [&base, &env] {
            let total: f64 = f.teams.iter().map(|x| x.p_champion).sum();
            assert!((total - 1.0).abs() < 1e-9, "champion mass {total}");
        }
    }

    #[test]
    fn the_shocks_stay_finite_at_the_extremes() {
        // rho = +-1 makes the Cholesky factor's second term zero, and w = 1 makes the team's own
        // term vanish. Neither may produce a NaN.
        for (rho, w) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let samples = shift_samples(
                ShockModel {
                    attack_defence: rho,
                    environment: w,
                },
                2,
                200,
            );
            for (a, d) in &samples {
                assert!(
                    a.iter().chain(d).all(|x| x.is_finite()),
                    "rho={rho} w={w} produced a non-finite shock"
                );
            }
        }
    }

    #[test]
    fn the_default_shock_model_is_exact_independence() {
        // The whole safety property of this type: an unconfigured forecast is the one the simulator
        // has always produced.
        let d = ShockModel::default();
        assert_eq!(d.attack_defence, 0.0);
        assert_eq!(d.environment, 0.0);
        assert!(d.is_independent());
    }

    #[test]
    fn either_parameter_alone_makes_the_model_non_independent() {
        assert!(!ShockModel {
            attack_defence: 0.3,
            ..Default::default()
        }
        .is_independent());
        assert!(!ShockModel {
            environment: 0.2,
            ..Default::default()
        }
        .is_independent());
    }

    #[test]
    fn parameters_are_clamped_to_their_valid_ranges() {
        let s = ShockModel {
            attack_defence: 1.7,
            environment: 2.5,
        }
        .sanitized();
        assert_eq!(s.attack_defence, 1.0);
        assert_eq!(s.environment, 1.0);

        let s = ShockModel {
            attack_defence: -3.0,
            environment: -0.5,
        }
        .sanitized();
        assert_eq!(s.attack_defence, -1.0);
        assert_eq!(
            s.environment, 0.0,
            "a negative variance share is meaningless"
        );
    }

    #[test]
    fn a_non_finite_parameter_falls_back_to_independence() {
        // These arrive from a CLI flag or a query string. A NaN correlation has no Cholesky factor,
        // so letting one through would produce NaN shocks and a silently empty forecast rather than
        // an obviously wrong one.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let s = ShockModel {
                attack_defence: bad,
                environment: bad,
            }
            .sanitized();
            assert!(s.is_independent(), "{bad} should sanitize to independence");
        }
    }

    #[test]
    fn a_paired_difference_of_identical_series_is_exactly_zero() {
        // The same premise twice must give a zero swing with zero error - not a small number.
        let xs = vec![true, false, true, true, false];
        let d = PairedDifference::from_indicators(&xs, &xs);
        assert_eq!(d.mean, 0.0);
        assert_eq!(d.std_error, 0.0);
        assert_eq!(d.iterations, 5);
        assert!(!d.is_significant(), "a zero swing cannot be significant");
    }

    #[test]
    fn a_paired_difference_reports_a_constant_shift_without_error() {
        // Scenario wins whenever baseline does not: every per-iteration difference is +1, so the
        // mean is 1 and there is no spread to be uncertain about.
        let base = vec![false; 64];
        let scen = vec![true; 64];
        let d = PairedDifference::from_indicators(&scen, &base);
        assert!((d.mean - 1.0).abs() < 1e-12);
        assert_eq!(d.std_error, 0.0);
        assert!(d.is_significant());
    }

    #[test]
    fn a_paired_difference_prices_disagreement_as_uncertainty() {
        // Half the iterations swing +1, half -1: the mean is zero but the spread is maximal, so the
        // estimator must report that it cannot resolve the swing.
        let n = 400;
        let base: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let scen: Vec<bool> = (0..n).map(|i| i % 2 != 0).collect();
        let d = PairedDifference::from_indicators(&scen, &base);
        assert!(d.mean.abs() < 1e-12, "mean {}", d.mean);
        // Per-iteration differences are +-1, so the SD is 1 and the SE is 1/sqrt(n).
        assert!(
            (d.std_error - 1.0 / (n as f64).sqrt()).abs() < 1e-3,
            "std_error {}",
            d.std_error
        );
        assert!(!d.is_significant());
    }

    #[test]
    fn an_empty_or_mismatched_series_pairs_only_what_it_can() {
        assert_eq!(PairedDifference::from_indicators(&[], &[]).iterations, 0);
        let d = PairedDifference::from_indicators(&[true, true, true], &[false]);
        assert_eq!(d.iterations, 1, "pairs up to the common length");
        assert!((d.mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn champion_indicators_agree_with_the_aggregate_forecast() {
        // The per-iteration series must average to the same champion probability the ordinary
        // forecast reports, or the paired estimator is measuring something else.
        let t = full_tournament();
        let cfg = SimConfig {
            iterations: 3000,
            seed: 17,
            ..Default::default()
        };
        let inputs = LiveInputs::default();
        let flags = champion_indicators(
            &t,
            &RankSampler,
            cfg,
            &inputs,
            LiveConfig::default(),
            TeamId(0),
        );
        assert_eq!(flags.len(), 3000);
        let share = flags.iter().filter(|&&f| f).count() as f64 / 3000.0;
        let aggregate = simulate(&t, &RankSampler, cfg)
            .teams
            .iter()
            .find(|f| f.team == TeamId(0))
            .expect("team 0")
            .p_champion;
        assert!(
            (share - aggregate).abs() < 1e-12,
            "indicator share {share} vs forecast {aggregate}"
        );
    }

    #[test]
    fn an_unknown_team_has_no_champion_iterations() {
        let t = full_tournament();
        let flags = champion_indicators(
            &t,
            &RankSampler,
            SimConfig {
                iterations: 50,
                ..Default::default()
            },
            &LiveInputs::default(),
            LiveConfig::default(),
            TeamId(9999),
        );
        assert_eq!(flags.len(), 50);
        assert!(flags.iter().all(|&f| !f));
    }

    #[test]
    fn labelled_streams_couple_the_untouched_fixtures() {
        // Conditioning one group-A result should leave the other eleven groups almost exactly as
        // they were, because every fixture outside group A draws from a stream addressed by its own
        // match id and so sees identical randomness in both runs. Sharing only a base seed - the
        // way a single sequential stream would - does not achieve this: the removed fixture shifts
        // every draw after it.
        //
        // The residual drift is not error. Third place qualifies across groups, so group A's
        // third-placed record genuinely moves who advances elsewhere.
        for seed in [1u64, 2, 3] {
            let coupled = qualification_drift(seed, seed);
            let independent = qualification_drift(seed, seed + 5000);
            assert!(
                coupled < 0.004,
                "seed {seed}: coupled drift {coupled:.5} is larger than the cross-group effect"
            );
            assert!(
                independent > 3.0 * coupled,
                "seed {seed}: coupling should cut the drift several-fold, \
                 got coupled {coupled:.5} vs independent {independent:.5}"
            );
        }
    }

    #[test]
    fn a_streams_draw_depends_on_its_address_and_nothing_else() {
        let s = Streams::new(11, 4);
        let take =
            |mut r: Rng| -> [u64; 4] { [r.next_u64(), r.next_u64(), r.next_u64(), r.next_u64()] };

        // Same address, same draws - and re-deriving the Streams gives the same answer, so nothing
        // is carried in hidden state.
        assert_eq!(
            take(s.group_match(MatchId(7))),
            take(s.group_match(MatchId(7)))
        );
        assert_eq!(
            take(s.group_match(MatchId(7))),
            take(Streams::new(11, 4).group_match(MatchId(7)))
        );

        // Distinct entities, distinct streams.
        assert_ne!(
            take(s.group_match(MatchId(7))),
            take(s.group_match(MatchId(8)))
        );
        assert_ne!(take(s.team_strength(2)), take(s.team_strength(3)));
        assert_ne!(take(s.ko_tie(1, 0)), take(s.ko_tie(2, 0)), "round matters");
        assert_ne!(take(s.ko_tie(1, 0)), take(s.ko_tie(1, 1)), "slot matters");

        // Different kinds must not collide on a shared index: team 7, match 7 and slot 7 are
        // unrelated things.
        assert_ne!(take(s.group_match(MatchId(7))), take(s.team_strength(7)));
        assert_ne!(take(s.team_strength(7)), take(s.ko_tie(0, 7)));

        // A different iteration is a different universe.
        assert_ne!(
            take(s.group_match(MatchId(7))),
            take(Streams::new(11, 5).group_match(MatchId(7)))
        );
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
