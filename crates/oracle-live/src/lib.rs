//! # oracle-live
//!
//! The in-play layer: what the odds do *during* a match, and what a trader can do about it. Every
//! other part of the project prices a match before kickoff; this one follows the score as it
//! changes, re-derives the win probability minute by minute, and simulates **in-play trading** with
//! the tools a real exchange gives you: **hedging** and **cash-out**. The honest question it answers
//! is whether actively trading a live position beats simply holding a pre-match bet.
//!
//! Like the market and players crates it is small and pure: plain calculations, no I/O, every layer
//! unit-tested in isolation. The pre-match goal rates come from the goal model upstream; this crate
//! turns them into a live win-probability path and a settled trading ledger.
//!
//! This first layer is the vocabulary and the Monte-Carlo's random source: a [`MatchState`] (the
//! score at a minute) and [`oracle_numeric`]'s seeded generator, so a simulated match is
//! reproducible from a seed.

use oracle_domain::{Outcome, Probabilities};
use oracle_numeric::poisson_pmf;
use serde::{Deserialize, Serialize};

/// The full 90-minute length used to prorate the remaining goal rate.
const FULL_MATCH: f64 = 90.0;

/// The live win/draw/win probability from a match state and the two sides' **full-match** goal
/// rates. The remaining goals are Poisson with rate prorated by the time left, added to the current
/// score. At kickoff this is the pre-match forecast; at the whistle it is the settled result.
pub fn win_probabilities(state: MatchState, lambda_home: f64, lambda_away: f64) -> Probabilities {
    let remaining = (FULL_MATCH - (state.minute as f64).min(FULL_MATCH)) / FULL_MATCH;
    let lh = lambda_home.max(0.0) * remaining;
    let la = lambda_away.max(0.0) * remaining;
    const K: u32 = 12;
    let (mut home, mut draw, mut away) = (0.0, 0.0, 0.0);
    for x in 0..=K {
        let px = poisson_pmf(x, lh);
        for y in 0..=K {
            let p = px * poisson_pmf(y, la);
            let final_home = state.home as i32 + x as i32;
            let final_away = state.away as i32 + y as i32;
            match final_home.cmp(&final_away) {
                std::cmp::Ordering::Greater => home += p,
                std::cmp::Ordering::Equal => draw += p,
                std::cmp::Ordering::Less => away += p,
            }
        }
    }
    Probabilities::new(home, draw, away)
}

/// The score at a point in a match: minutes elapsed and goals for each side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchState {
    pub minute: u16,
    pub home: u8,
    pub away: u8,
}

impl MatchState {
    /// Kickoff: 0-0 at minute zero.
    pub fn kickoff() -> Self {
        Self {
            minute: 0,
            home: 0,
            away: 0,
        }
    }
}

/// The seeded generator a simulated match draws from, re-exported so callers can build one without
/// naming [`oracle_numeric`] themselves.
pub use oracle_numeric::Rng;

/// Simulate a match's goals from the two full-match rates: draw each side's goal count from a
/// Poisson, scatter the goals across the 90 minutes, and return the score after each one, bracketed
/// by kickoff (0-0) and a final minute-90 state. Reproducible from the generator's seed.
pub fn simulate_match(rng: &mut Rng, lambda_home: f64, lambda_away: f64) -> Vec<MatchState> {
    let mut goals: Vec<(u16, bool)> = Vec::new();
    for (lambda, is_home) in [(lambda_home, true), (lambda_away, false)] {
        for _ in 0..rng.poisson(lambda.max(0.0)) {
            let minute = 1 + (rng.unit() * FULL_MATCH) as u16;
            goals.push((minute.min(90), is_home));
        }
    }
    goals.sort_by_key(|&(minute, _)| minute);

    let mut states = vec![MatchState::kickoff()];
    let (mut home, mut away) = (0u8, 0u8);
    for (minute, is_home) in goals {
        if is_home {
            home += 1;
        } else {
            away += 1;
        }
        states.push(MatchState { minute, home, away });
    }
    if states.last().map_or(true, |s| s.minute < 90) {
        states.push(MatchState {
            minute: 90,
            home,
            away,
        });
    }
    states
}

/// One point on a live win-probability path: the minute and the probability of the tracked outcome.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PathPoint {
    pub minute: u16,
    pub prob: f64,
}

/// The score as of `minute`: the latest timeline state at or before it (kickoff if none).
fn score_at(timeline: &[MatchState], minute: u16) -> MatchState {
    let mut current = timeline
        .first()
        .copied()
        .unwrap_or_else(MatchState::kickoff);
    for &state in timeline {
        if state.minute <= minute {
            current = state;
        } else {
            break;
        }
    }
    MatchState {
        minute,
        home: current.home,
        away: current.away,
    }
}

/// The live probability of `outcome` across a match: at every sampled minute (a `step`-minute grid
/// plus each goal minute), combine the score so far with the live model. This is the drama graph a
/// trader watches, and the price path the trading simulator settles against.
pub fn win_prob_path(
    timeline: &[MatchState],
    lambda_home: f64,
    lambda_away: f64,
    outcome: Outcome,
    step: u16,
) -> Vec<PathPoint> {
    let step = step.max(1);
    let mut minutes: Vec<u16> = (0..=90).step_by(step as usize).collect();
    minutes.extend(timeline.iter().map(|s| s.minute));
    minutes.push(90);
    minutes.sort_unstable();
    minutes.dedup();
    minutes
        .into_iter()
        .map(|minute| PathPoint {
            minute,
            prob: win_probabilities(score_at(timeline, minute), lambda_home, lambda_away)
                .of(outcome),
        })
        .collect()
}

/// The fair decimal odds for a probability, `1 / p` (floored at 1.0, guarded against zero).
pub fn fair_odds(prob: f64) -> f64 {
    (1.0 / prob.clamp(1e-9, 1.0)).max(1.0)
}

/// The lay stake that hedges a back bet to an equal profit whatever the result: `stake * B / L` for
/// back odds `B` and current lay odds `L`.
pub fn hedge_stake(back_stake: f64, back_odds: f64, lay_odds: f64) -> f64 {
    if lay_odds <= 0.0 {
        return 0.0;
    }
    back_stake * back_odds / lay_odds
}

/// The profit locked in by that hedge, the same whichever way the match goes: `stake * (B - L) / L`.
/// Positive when the odds have shortened since the back (the position is in profit), negative when
/// they have drifted out.
pub fn locked_profit(back_stake: f64, back_odds: f64, lay_odds: f64) -> f64 {
    if lay_odds <= 0.0 {
        return 0.0;
    }
    back_stake * (back_odds - lay_odds) / lay_odds
}

/// The cash-out value of a back position: the guaranteed profit from fully hedging now at the
/// current fair probability of the backed outcome. Zero when the outcome is exactly as likely as its
/// back price implied, positive once it has become more likely, negative if less. At certainty it is
/// the full back winnings.
pub fn cash_out_value(back_stake: f64, back_odds: f64, current_prob: f64) -> f64 {
    locked_profit(back_stake, back_odds, fair_odds(current_prob))
}

/// A trading rule for a live position: how much to stake, and the cash-out triggers as fractions of
/// the stake (take profit at `+profit_target`, cut a loss at `-stop_loss`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TradeConfig {
    pub stake: f64,
    pub profit_target: f64,
    pub stop_loss: f64,
}

impl Default for TradeConfig {
    fn default() -> Self {
        Self {
            stake: 10.0,
            profit_target: 0.5,
            stop_loss: 0.5,
        }
    }
}

/// The outcome of trading one match.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TradeResult {
    pub pnl: f64,
    pub cashed_out: bool,
    pub exit_minute: u16,
}

/// Trade a single back position over a win-probability `path`: back the outcome at `back_odds`, and
/// at each in-play point cash out if the position has reached the profit target or the stop. If
/// neither triggers before full time, hold to settlement (`won` decides the payout). The final,
/// settled path point is not a cash-out opportunity, so holding is distinguishable from cashing out.
pub fn trade_match(
    back_odds: f64,
    path: &[PathPoint],
    won: bool,
    config: &TradeConfig,
) -> TradeResult {
    let target = config.profit_target * config.stake;
    let stop = -config.stop_loss.abs() * config.stake;
    for point in path.iter().filter(|p| p.minute < 90) {
        let value = cash_out_value(config.stake, back_odds, point.prob);
        if value >= target || value <= stop {
            return TradeResult {
                pnl: value,
                cashed_out: true,
                exit_minute: point.minute,
            };
        }
    }
    let pnl = if won {
        config.stake * (back_odds - 1.0)
    } else {
        -config.stake
    };
    TradeResult {
        pnl,
        cashed_out: false,
        exit_minute: 90,
    }
}

/// Which outcome a final score settled to.
fn outcome_of(state: MatchState) -> Outcome {
    match state.home.cmp(&state.away) {
        std::cmp::Ordering::Greater => Outcome::HomeWin,
        std::cmp::Ordering::Equal => Outcome::Draw,
        std::cmp::Ordering::Less => Outcome::AwayWin,
    }
}

/// The most likely of the three outcomes under a distribution.
fn favourite(p: Probabilities) -> Outcome {
    if p.home_win >= p.draw && p.home_win >= p.away_win {
        Outcome::HomeWin
    } else if p.away_win >= p.draw {
        Outcome::AwayWin
    } else {
        Outcome::Draw
    }
}

/// Settings for the in-play trading backtest.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InPlayConfig {
    pub trade: TradeConfig,
    pub iters: u32,
    pub seed: u64,
    pub step: u16,
}

impl Default for InPlayConfig {
    fn default() -> Self {
        Self {
            trade: TradeConfig::default(),
            iters: 20_000,
            seed: 42,
            step: 5,
        }
    }
}

/// The result of the in-play trading backtest, with a hold-to-settlement baseline for comparison.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InPlayReport {
    pub trades: u32,
    /// Mean profit per match of the cash-out strategy, and as a fraction of stake.
    pub mean_pnl: f64,
    pub roi: f64,
    /// Fraction of matches that cashed out in play rather than settling.
    pub cash_out_rate: f64,
    /// Fraction of matches that finished in profit.
    pub profitable_rate: f64,
    /// Worst peak-to-trough fall of the cumulative P&L, in stake units.
    pub max_drawdown: f64,
    /// Mean profit per match of simply holding the pre-match bet (never cashing out).
    pub hold_mean_pnl: f64,
}

/// Backtest trading the pre-match favourite in play over many simulated matches. Each match is
/// simulated from the goal rates, the favourite is backed at fair odds, and the position is managed
/// by [`trade_match`]; the same bet held to settlement is tracked alongside as a baseline. Because
/// the odds are fair, neither approach has an edge, so the honest comparison is about the shape of
/// the P&L (drawdown, cash-out frequency), not its mean. Reproducible from the seed.
///
/// # Panics
/// If [`simulate_match`] ever returns an empty timeline. It cannot: every timeline is bracketed by
/// a kickoff state and a minute-90 state, so the final state this reads always exists.
pub fn inplay_backtest(lambda_home: f64, lambda_away: f64, config: &InPlayConfig) -> InPlayReport {
    let pre_match = win_probabilities(MatchState::kickoff(), lambda_home, lambda_away);
    let backed = favourite(pre_match);
    let back_odds = fair_odds(pre_match.of(backed));
    let stake = config.trade.stake;
    let iters = config.iters.max(1);

    let mut rng = Rng::new(config.seed);
    let (mut total, mut hold_total, mut cashed, mut profitable) = (0.0, 0.0, 0u32, 0u32);
    let (mut cumulative, mut peak, mut max_drawdown) = (0.0f64, 0.0f64, 0.0f64);

    for _ in 0..iters {
        let timeline = simulate_match(&mut rng, lambda_home, lambda_away);
        let won = outcome_of(*timeline.last().unwrap()) == backed;
        let path = win_prob_path(&timeline, lambda_home, lambda_away, backed, config.step);
        let result = trade_match(back_odds, &path, won, &config.trade);

        total += result.pnl;
        hold_total += if won {
            stake * (back_odds - 1.0)
        } else {
            -stake
        };
        if result.cashed_out {
            cashed += 1;
        }
        if result.pnl > 0.0 {
            profitable += 1;
        }
        cumulative += result.pnl;
        peak = peak.max(cumulative);
        max_drawdown = max_drawdown.max(peak - cumulative);
    }

    let n = iters as f64;
    InPlayReport {
        trades: iters,
        mean_pnl: total / n,
        roi: total / n / stake,
        cash_out_rate: cashed as f64 / n,
        profitable_rate: profitable as f64 / n,
        max_drawdown: max_drawdown / stake,
        hold_mean_pnl: hold_total / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(minute: u16, home: u8, away: u8) -> MatchState {
        MatchState { minute, home, away }
    }

    #[test]
    fn full_time_win_probability_is_settled_by_the_score() {
        let full = win_probabilities(st(90, 2, 0), 1.5, 1.2);
        assert!((full.home_win - 1.0).abs() < 1e-9);
        let level = win_probabilities(st(90, 1, 1), 1.5, 1.2);
        assert!((level.draw - 1.0).abs() < 1e-9);
    }

    #[test]
    fn kickoff_probability_tracks_the_goal_rates() {
        // Equal rates: symmetric, with a real draw chance.
        let even = win_probabilities(st(0, 0, 0), 1.3, 1.3);
        assert!((even.home_win - even.away_win).abs() < 1e-9);
        assert!(even.draw > 0.15);
        // A stronger home rate favours the home side.
        let favoured = win_probabilities(st(0, 0, 0), 2.0, 0.8);
        assert!(favoured.home_win > favoured.away_win);
    }

    #[test]
    fn a_lead_is_worth_more_as_time_runs_out() {
        let early = win_probabilities(st(10, 1, 0), 1.4, 1.4);
        let late = win_probabilities(st(85, 1, 0), 1.4, 1.4);
        assert!(late.home_win > early.home_win);
        // And the probabilities always normalize.
        assert!((late.home_win + late.draw + late.away_win - 1.0).abs() < 1e-9);
    }

    #[test]
    fn simulated_match_is_a_coherent_reproducible_timeline() {
        let mut a = Rng::new(5);
        let mut b = Rng::new(5);
        let ta = simulate_match(&mut a, 1.6, 1.1);
        let tb = simulate_match(&mut b, 1.6, 1.1);
        assert_eq!(ta, tb, "same seed, same timeline");

        // Starts at kickoff, ends at minute 90, minutes and scores never go backwards, and each
        // step adds exactly one goal.
        assert_eq!(ta[0], MatchState::kickoff());
        assert_eq!(ta.last().unwrap().minute, 90);
        for w in ta.windows(2) {
            assert!(w[1].minute >= w[0].minute);
            let added = (w[1].home - w[0].home) + (w[1].away - w[0].away);
            assert!(added <= 1, "at most one goal per recorded step");
        }
    }

    #[test]
    fn simulated_goal_counts_match_the_rates() {
        let mut rng = Rng::new(3);
        let (mut home_total, mut away_total) = (0u64, 0u64);
        let n = 20_000;
        for _ in 0..n {
            let t = simulate_match(&mut rng, 1.8, 0.9);
            let last = t.last().unwrap();
            home_total += u64::from(last.home);
            away_total += u64::from(last.away);
        }
        assert!((home_total as f64 / n as f64 - 1.8).abs() < 0.05);
        assert!((away_total as f64 / n as f64 - 0.9).abs() < 0.05);
    }

    #[test]
    fn win_prob_path_starts_pre_match_and_settles_at_full_time() {
        // Home scores at 30' and holds on.
        let timeline = vec![st(0, 0, 0), st(30, 1, 0), st(90, 1, 0)];
        let path = win_prob_path(&timeline, 1.4, 1.2, Outcome::HomeWin, 5);

        assert_eq!(path.first().unwrap().minute, 0);
        assert_eq!(path.last().unwrap().minute, 90);
        // Kickoff point equals the pre-match home probability, strictly between 0 and 1.
        let pre = win_probabilities(MatchState::kickoff(), 1.4, 1.2).home_win;
        assert!((path[0].prob - pre).abs() < 1e-9 && path[0].prob > 0.0 && path[0].prob < 1.0);
        // Home held the lead, so the path settles to 1.
        assert!((path.last().unwrap().prob - 1.0).abs() < 1e-9);
        // Minutes increase, probabilities stay in range.
        for w in path.windows(2) {
            assert!(w[1].minute > w[0].minute);
        }
        for p in &path {
            assert!((0.0..=1.0).contains(&p.prob));
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    // Profit if the backed outcome wins / loses, after laying `hedge` at `lay_odds`.
    fn settle(back_stake: f64, back_odds: f64, lay_odds: f64) -> (f64, f64) {
        let hedge = hedge_stake(back_stake, back_odds, lay_odds);
        let win = back_stake * (back_odds - 1.0) - (lay_odds - 1.0) * hedge;
        let lose = -back_stake + hedge;
        (win, lose)
    }

    #[test]
    fn fair_odds_invert_probability() {
        approx(fair_odds(0.5), 2.0);
        approx(fair_odds(0.25), 4.0);
        approx(fair_odds(1.0), 1.0);
    }

    #[test]
    fn hedging_locks_an_equal_profit_both_ways() {
        // Odds shortened (3.0 -> 2.0): the hedge locks a profit, equal whichever way it settles.
        let (win, lose) = settle(10.0, 3.0, 2.0);
        approx(win, lose);
        approx(win, locked_profit(10.0, 3.0, 2.0));
        approx(locked_profit(10.0, 3.0, 2.0), 5.0);
        // Odds drifted out (2.0 -> 4.0): the locked figure is a loss, still equal both ways.
        let (dw, dl) = settle(10.0, 2.0, 4.0);
        approx(dw, dl);
        approx(dw, locked_profit(10.0, 2.0, 4.0));
        approx(locked_profit(10.0, 2.0, 4.0), -5.0);
    }

    #[test]
    fn cash_out_value_tracks_the_move_since_the_back() {
        // Backed at 2.0 (break-even prob 0.5). Unchanged -> zero.
        approx(cash_out_value(10.0, 2.0, 0.5), 0.0);
        // More likely now -> a positive cash-out; less likely -> negative.
        assert!(cash_out_value(10.0, 2.0, 0.6) > 0.0);
        assert!(cash_out_value(10.0, 2.0, 0.4) < 0.0);
        // Certain now -> the full back winnings.
        approx(cash_out_value(10.0, 2.0, 1.0), 10.0);
    }

    fn pt(minute: u16, prob: f64) -> PathPoint {
        PathPoint { minute, prob }
    }

    #[test]
    fn a_winning_position_cashes_out_at_the_profit_target() {
        let path = vec![pt(0, 0.5), pt(45, 0.8), pt(89, 0.95), pt(90, 1.0)];
        let cfg = TradeConfig {
            stake: 10.0,
            profit_target: 0.5,
            stop_loss: 0.5,
        };
        let r = trade_match(2.0, &path, true, &cfg);
        assert!(r.cashed_out && r.exit_minute == 45);
        approx(r.pnl, 6.0); // locked profit at prob 0.8, backed at 2.0
    }

    #[test]
    fn a_fading_position_stops_out() {
        let path = vec![pt(0, 0.5), pt(45, 0.2), pt(90, 0.0)];
        let r = trade_match(2.0, &path, false, &TradeConfig::default());
        assert!(r.cashed_out && r.exit_minute == 45 && r.pnl < 0.0);
    }

    #[test]
    fn without_a_trigger_the_position_is_held_to_settlement() {
        let path = vec![pt(0, 0.5), pt(45, 0.55), pt(90, 1.0)];
        let patient = TradeConfig {
            stake: 10.0,
            profit_target: 100.0,
            stop_loss: 100.0,
        };
        let won = trade_match(2.0, &path, true, &patient);
        assert!(!won.cashed_out && won.exit_minute == 90);
        approx(won.pnl, 10.0);
        let lost = trade_match(2.0, &path, false, &patient);
        approx(lost.pnl, -10.0);
    }

    #[test]
    fn inplay_backtest_is_reproducible_and_edge_free() {
        let cfg = InPlayConfig {
            iters: 20_000,
            ..InPlayConfig::default()
        };
        let a = inplay_backtest(1.7, 1.0, &cfg);
        let b = inplay_backtest(1.7, 1.0, &cfg);

        // Reproducible from the seed.
        approx(a.mean_pnl, b.mean_pnl);
        assert_eq!(a.trades, 20_000);
        // Rates are proportions; some matches genuinely cash out in play.
        assert!((0.0..=1.0).contains(&a.cash_out_rate) && a.cash_out_rate > 0.0);
        assert!((0.0..=1.0).contains(&a.profitable_rate));
        assert!(a.max_drawdown >= 0.0);
        // Fair odds mean no edge: both the traded and the held P&L average near zero.
        let stake = cfg.trade.stake;
        assert!(a.mean_pnl.abs() < 0.1 * stake, "traded mean {}", a.mean_pnl);
        assert!(
            a.hold_mean_pnl.abs() < 0.1 * stake,
            "held mean {}",
            a.hold_mean_pnl
        );
    }

    #[test]
    fn match_state_starts_goalless_at_kickoff() {
        let s = MatchState::kickoff();
        assert_eq!((s.minute, s.home, s.away), (0, 0, 0));
    }
}
