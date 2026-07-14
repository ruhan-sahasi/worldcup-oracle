//! The immutable, serializable view the engine publishes on every update.
//!
//! A [`Snapshot`] is a complete, self-contained picture of the engine's beliefs at
//! one instant: every match's current probabilities, the tournament forecast, and
//! the live ratings. The engine swaps a fresh `Arc<Snapshot>` into an [`arc_swap`]
//! cell (lock-free reads for the REST layer) and broadcasts it to WebSocket
//! subscribers. Because it's a value type, consumers never touch engine internals.

use chrono::{DateTime, Utc};
use oracle_domain::{
    MatchId, MatchStatus, Probabilities, ScoreGrid, Scoreline, Stage, TeamId, TournamentForecast,
};
use oracle_model::ReliabilityReport;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// One point on a live match's win-probability timeline (the "drama graph").
#[derive(Debug, Clone, Copy, Serialize)]
pub struct WinProbSample {
    pub minute: u16,
    pub probabilities: Probabilities,
}

/// The engine's current prediction for a single match.
#[derive(Debug, Clone, Serialize)]
pub struct MatchPrediction {
    pub match_id: MatchId,
    pub home: TeamId,
    pub away: TeamId,
    pub home_name: String,
    pub away_name: String,
    pub stage: Stage,
    pub status: MatchStatus,
    pub score: Scoreline,
    pub minute: u16,
    /// Current win/draw/win - live (Bayesian) if in play, else the pre-match ensemble.
    pub probabilities: Probabilities,
    /// The exact-score distribution (omitted once the match is finished).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_grid: Option<ScoreGrid>,
    /// Modal scoreline `(home, away, probability)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub most_likely_score: Option<(u8, u8, f64)>,
    /// Win-probability timeline while the match is live (empty otherwise): the drama graph.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<WinProbSample>,
}

/// A team's current Elo rating (for display).
#[derive(Debug, Clone, Serialize)]
pub struct RatingEntry {
    pub team: TeamId,
    pub name: String,
    /// FIFA code (e.g. `BRA`), so clients can resolve a team without a separate lookup.
    pub code: String,
    pub rating: f64,
}

/// A finished match the model got surprised by: the pre-match favourite failed to win. Ranked by
/// `shock` (how little probability the model gave the outcome that actually happened).
#[derive(Debug, Clone, Serialize)]
pub struct Upset {
    pub match_id: MatchId,
    pub home_name: String,
    pub away_name: String,
    pub stage: Stage,
    pub score: Scoreline,
    /// The pre-match favourite (the side the model gave the higher win probability).
    pub favorite_name: String,
    /// The probability the model gave that favourite pre-match.
    pub favorite_prob: f64,
    /// Shock magnitude in `[0, 1]`: `1 - P(actual outcome)` from the pre-match forecast.
    pub shock: f64,
}

/// One scored pre-match call: what the model predicted for a now-finished match, and whether it
/// came off. `confidence` is the pre-match probability the model gave its most-likely outcome.
#[derive(Debug, Clone, Serialize)]
pub struct Call {
    pub match_id: MatchId,
    pub home_name: String,
    pub away_name: String,
    pub score: Scoreline,
    pub confidence: f64,
    pub correct: bool,
}

/// One model's scorecard on the finished matches, for the head-to-head between the two forecasters.
#[derive(Debug, Clone, Serialize)]
pub struct ModelScore {
    pub model: String,
    pub scored: usize,
    pub accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
}

/// The model's self-scored "report card": how its own pre-match calls have held up so far. Honest
/// accountability - most of the numbers come free from the pre-match forecasts + results already
/// stored. `baseline_brier` is the naive uniform baseline, for context. `head_to_head` scores both
/// forecasters (the Dixon-Coles ensemble and Bradley-Terry) on the same results.
#[derive(Debug, Clone, Serialize)]
pub struct ReportCard {
    pub scored: usize,
    pub winners_called: usize,
    pub accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
    pub baseline_brier: f64,
    /// Its most confident correct calls, and its most confident misses.
    pub best_calls: Vec<Call>,
    pub worst_calls: Vec<Call>,
    /// Both models scored on the same finished matches (ensemble vs Bradley-Terry).
    pub head_to_head: Vec<ModelScore>,
}

/// The engine's in-tournament self-recalibration state, surfaced for observability. As results
/// arrive the live features adjust these away from their neutral values, so a viewer can watch the
/// model recalibrate itself against the tournament.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AdaptiveState {
    /// Finished matches folded into the live calibration so far.
    pub results_seen: usize,
    /// Temperature applied to the remaining pre-match forecasts (1.0 = no correction; >1 sharpens).
    pub calibration_temperature: f64,
    /// Aggregate gain on the context effects, recalibrated against results (1.0 = the reasoned prior).
    pub context_gain: f64,
}

/// A complete published view of engine state.
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub generated_at: DateTime<Utc>,
    pub tournament: String,
    pub provider: String,
    /// Whether the data feed is currently healthy. `false` ⇒ the figures may be stale.
    pub source_healthy: bool,
    /// When the engine last processed an event from the feed (staleness signal).
    pub last_update: DateTime<Utc>,
    pub matches: Vec<MatchPrediction>,
    pub forecast: TournamentForecast,
    pub ratings: Vec<RatingEntry>,
    /// The engine's live self-recalibration state (temperature, context gain, results seen).
    pub adaptive: AdaptiveState,
    /// Finished matches the model was most surprised by (favourite failed to win), ranked by shock.
    pub shocks: Vec<Upset>,
    /// The model's self-scored report card on its own pre-match calls so far.
    pub report_card: ReportCard,
    /// The second model's (Bradley-Terry) live champion odds over the current knockout bracket,
    /// conditioning on knockout results already played. Empty until the bracket is materialized.
    pub bt_champions: Vec<BtChampion>,
    /// A consensus title forecast blending the two models, with a live Jensen-Shannon divergence
    /// measuring how far they disagree. Empty until the bracket is materialized.
    pub consensus: Consensus,
    /// The headline forecaster's live reliability curve over its own leak-free pre-match calls, with
    /// the expected calibration error: is a 70% call right about 70% of the time?
    pub reliability: ReliabilityReport,
    /// A prior-free Massey power ranking over only this tournament's results: who has been strongest
    /// here, strength-of-schedule adjusted. Empty until matches finish.
    pub power_ranking: PowerRanking,
    /// The biggest over- and under-performers versus their pre-tournament seeding (the gap between
    /// the strength prior's ranking and the live power ranking). Empty until matches finish.
    pub form: TournamentForm,
    /// Each surviving contender's road to the final: the expected strength of the opponents it still
    /// has to get through. Empty until the knockout bracket is materialized.
    pub road: RoadBoard,
    /// The model's single most likely completion of the knockout bracket, with the chance that exact
    /// bracket occurs. Empty until the knockout bracket is materialized.
    pub predicted_bracket: PredictedBracket,
    /// The current knockout round's ties ranked by how much each would reshape the title race. Empty
    /// until the knockout bracket is materialized.
    pub leverage: MatchLeverage,
    /// How open the title race is (entropy of the champion odds, effective number of contenders).
    pub openness: Openness,
    /// The championship-odds time series for the current top contenders over the tournament so far.
    pub timeline: ChampionTimeline,
}

/// One team's live champion probability from the second model (Bradley-Terry) over the current
/// knockout bracket.
#[derive(Debug, Clone, Serialize)]
pub struct BtChampion {
    pub team: String,
    pub champion: f64,
}

/// One team's line in the consensus title forecast: what each of the two models gives it, their
/// 50/50 average, and the signed gap between them (`bradley_terry - ensemble`).
#[derive(Debug, Clone, Serialize)]
pub struct ConsensusTeam {
    pub team: String,
    pub ensemble: f64,
    pub bradley_terry: f64,
    pub consensus: f64,
    pub delta: f64,
}

/// A consensus title forecast from the two independent models plus a single measure of how far they
/// disagree. `jsd` is the Jensen-Shannon divergence between the two champion distributions in bits
/// `[0, 1]` (0 = identical, higher = more disagreement). `teams` is the per-team consensus, ranked.
#[derive(Debug, Clone, Serialize)]
pub struct Consensus {
    pub jsd: f64,
    pub teams: Vec<ConsensusTeam>,
}

/// One team's line in the within-tournament Massey power ranking: its least-squares rating and the
/// offense/defense split, in goal-difference units (centered so the field averages zero).
#[derive(Debug, Clone, Serialize)]
pub struct PowerTeam {
    pub team: String,
    pub rating: f64,
    pub offense: f64,
    pub defense: f64,
    pub games: u32,
}

/// A prior-free power ranking built by a Massey least-squares fit over only the matches played in
/// this tournament: who has actually been strongest here, strength-of-schedule adjusted. `matches`
/// is how many results fed the fit. Empty until matches finish.
#[derive(Debug, Clone, Serialize)]
pub struct PowerRanking {
    pub matches: usize,
    pub teams: Vec<PowerTeam>,
}

/// One team's over/under-performance versus its pre-tournament seeding: where the prior ranked it
/// against where the prior-free power ranking now places it. `delta = pre_rank - power_rank`, so a
/// positive delta is a climb (it has beaten expectations).
#[derive(Debug, Clone, Serialize)]
pub struct FormLine {
    pub team: String,
    pub pre_rank: usize,
    pub power_rank: usize,
    pub delta: i32,
    pub rating: f64,
}

/// The tournament's biggest over- and under-performers versus their pre-tournament seeding, by the
/// gap between the strength prior's ranking and the live Massey power ranking. Empty until matches
/// finish.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TournamentForm {
    pub risers: Vec<FormLine>,
    pub fallers: Vec<FormLine>,
}

/// One future round on a team's road to the title: the probability-weighted strength of the
/// opponent it would face there, and that round's single most likely opponent.
#[derive(Debug, Clone, Serialize)]
pub struct RoadRound {
    pub stage: String,
    /// Expected opponent Elo, weighting each possible opponent by its chance of reaching this round.
    pub expected_opponent_elo: f64,
    pub likely_opponent: String,
}

/// One contender's remaining path to the title: its live champion probability, the rounds it has
/// yet to play, and an overall `difficulty` (the mean expected opponent Elo across those rounds).
#[derive(Debug, Clone, Serialize)]
pub struct RoadTeam {
    pub team: String,
    pub champion: f64,
    pub difficulty: f64,
    pub rounds: Vec<RoadRound>,
}

/// Each surviving contender's road to the final, ranked by championship probability: who they would
/// have to get through and how hard that path is. Empty until the knockout bracket is materialized.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RoadBoard {
    pub teams: Vec<RoadTeam>,
}

/// One tie in the model's predicted bracket: the two sides, the favoured winner, and the favourite's
/// win probability. `decided` marks a tie whose result is already in.
#[derive(Debug, Clone, Serialize)]
pub struct BracketTie {
    pub home: String,
    pub away: String,
    pub favorite: String,
    pub favorite_prob: f64,
    pub decided: bool,
}

/// One round of the predicted bracket.
#[derive(Debug, Clone, Serialize)]
pub struct BracketRound {
    pub stage: String,
    pub ties: Vec<BracketTie>,
}

/// The model's single most likely completion of the knockout bracket: every remaining tie resolved
/// to its favourite, up to a projected `champion`. `probability` is the product of the favourites'
/// win probabilities, i.e. the chance this exact bracket occurs. Empty until the bracket is set.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PredictedBracket {
    pub champion: String,
    pub probability: f64,
    pub rounds: Vec<BracketRound>,
}

/// One undecided current-round tie and how much its result reshapes the title race. `swing` is the
/// total-variation distance (in `[0, 1]`) between the champion distributions that follow from each
/// side advancing; `home_advance` is the home side's win probability, for context.
#[derive(Debug, Clone, Serialize)]
pub struct LeverageTie {
    pub home: String,
    pub away: String,
    pub home_advance: f64,
    pub swing: f64,
}

/// The current knockout round's ties ranked by how much each would reshape the title race. Empty
/// until the bracket is materialized (and once every tie in the round is decided).
#[derive(Debug, Clone, Default, Serialize)]
pub struct MatchLeverage {
    pub ties: Vec<LeverageTie>,
}

/// How open the title race is, read off the shape of the champion-odds distribution itself.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Openness {
    /// Shannon entropy of the champion odds, in bits.
    pub entropy_bits: f64,
    /// The entropy ceiling `log2(k)` over the `k` teams with any title chance (all equally likely).
    pub max_entropy_bits: f64,
    /// Effective number of contenders, `2^entropy`: the race is as open as this many equal favourites.
    pub effective_contenders: f64,
    /// Normalized openness `entropy / max_entropy` in `[0, 1]`: 0 = decided, 1 = wide open.
    pub openness: f64,
    /// How many teams still have any championship probability.
    pub contenders_alive: usize,
    pub favorite: String,
    pub favorite_prob: f64,
    /// Combined championship probability of the top four contenders (a concentration measure).
    pub top4_share: f64,
}

/// One team's championship-probability trajectory over the recorded history, aligned to the shared
/// sample axis (zero where the team had no title chance at that point).
#[derive(Debug, Clone, Serialize)]
pub struct TeamTrajectory {
    pub team: String,
    pub points: Vec<f64>,
}

/// The championship-odds time series: the current top contenders' trajectories over the tournament
/// so far, with the timestamp of each recorded sample. Empty until a forecast has been recorded.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChampionTimeline {
    pub samples: usize,
    pub at: Vec<DateTime<Utc>>,
    pub series: Vec<TeamTrajectory>,
}

/// The information-theoretic shape of a probability distribution: its Shannon entropy in bits, the
/// ceiling `log2(k)` over the `k` nonzero outcomes, the effective number of outcomes `2^entropy`,
/// and the normalized openness `entropy / max_entropy` in `[0, 1]`. Assumes `probs` sum to one.
pub(crate) fn entropy_stats(probs: &[f64]) -> (f64, f64, f64, f64) {
    let mut entropy = 0.0;
    let mut nonzero = 0usize;
    for &p in probs {
        if p > 0.0 {
            entropy -= p * p.log2();
            nonzero += 1;
        }
    }
    let max_entropy = if nonzero > 1 {
        (nonzero as f64).log2()
    } else {
        0.0
    };
    let effective = entropy.exp2();
    let openness = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (entropy, max_entropy, effective, openness)
}

impl Snapshot {
    /// Look up the prediction for a specific match.
    pub fn match_prediction(&self, id: MatchId) -> Option<&MatchPrediction> {
        self.matches.iter().find(|m| m.match_id == id)
    }

    /// Matches currently in play.
    pub fn live_matches(&self) -> impl Iterator<Item = &MatchPrediction> {
        self.matches
            .iter()
            .filter(|m| matches!(m.status, MatchStatus::Live { .. }))
    }
}

/// Lightweight runtime counters, exported on `/metrics`.
#[derive(Debug, Default)]
pub struct Metrics {
    pub events_processed: AtomicU64,
    pub goals_seen: AtomicU64,
    pub forecasts_computed: AtomicU64,
    pub snapshots_published: AtomicU64,
}

impl Metrics {
    pub fn incr(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Render the counters in Prometheus text exposition format.
    pub fn prometheus(&self, subscribers: usize) -> String {
        let g = |name: &str, help: &str, val: u64| {
            format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {val}\n")
        };
        let mut out = String::new();
        out.push_str(&g(
            "oracle_events_processed_total",
            "Match events consumed by the engine.",
            self.events_processed.load(Ordering::Relaxed),
        ));
        out.push_str(&g(
            "oracle_goals_seen_total",
            "Goals observed across all matches.",
            self.goals_seen.load(Ordering::Relaxed),
        ));
        out.push_str(&g(
            "oracle_forecasts_computed_total",
            "Monte-Carlo tournament forecasts computed.",
            self.forecasts_computed.load(Ordering::Relaxed),
        ));
        out.push_str(&g(
            "oracle_snapshots_published_total",
            "Snapshots published to subscribers.",
            self.snapshots_published.load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP oracle_subscribers Current WebSocket subscribers.\n\
             # TYPE oracle_subscribers gauge\n",
        );
        out.push_str(&format!("oracle_subscribers {subscribers}\n"));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::entropy_stats;

    #[test]
    fn entropy_of_a_uniform_distribution_is_log2_of_its_size() {
        let (h, max, eff, open) = entropy_stats(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h - 2.0).abs() < 1e-12);
        assert!((max - 2.0).abs() < 1e-12);
        assert!((eff - 4.0).abs() < 1e-9);
        assert!((open - 1.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_of_a_point_mass_is_zero() {
        let (h, max, eff, open) = entropy_stats(&[1.0, 0.0, 0.0]);
        assert_eq!(h, 0.0);
        assert_eq!(max, 0.0);
        assert!((eff - 1.0).abs() < 1e-12);
        assert_eq!(open, 0.0);
    }

    #[test]
    fn entropy_matches_a_hand_computed_skewed_distribution() {
        // H = -(0.5 log2 0.5 + 2 * 0.25 log2 0.25) = 0.5 + 1.0 = 1.5 bits.
        let (h, max, eff, open) = entropy_stats(&[0.5, 0.25, 0.25]);
        assert!((h - 1.5).abs() < 1e-12);
        assert!((max - 3.0_f64.log2()).abs() < 1e-12);
        assert!((eff - 1.5_f64.exp2()).abs() < 1e-9); // 2^1.5 ~= 2.828
        assert!((open - 1.5 / 3.0_f64.log2()).abs() < 1e-12);
    }
}
