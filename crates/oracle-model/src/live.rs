//! Bayesian in-match updating.
//!
//! Pre-match, the goal model gives full-90 scoring rates `(λ, μ)`. Once a match is
//! live we *condition on what has already happened* — the current scoreline, the
//! minute, and any red cards — and re-derive the distribution over the **final**
//! result.
//!
//! The remaining goals for each side are modelled as independent Poisson processes
//! over the time left, so the expected remaining goals scale with the fraction of
//! the match still to play. Red cards perturb the live intensities (a team down to
//! ten men scores less and concedes more). Combining the "goals already in" with the
//! posterior over "goals still to come" yields live win/draw/win probabilities that
//! update event-by-event — the engine recomputes these every time a material event
//! arrives.

use crate::poisson::poisson_pmf;
use oracle_domain::{fixture::REGULATION_MINUTES, Probabilities, ScoreGrid, Scoreline};
use serde::{Deserialize, Serialize};

/// Knobs for the live model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LiveConfig {
    /// Max *additional* goals (per side) modelled when convolving the remainder.
    pub max_remaining_goals: usize,
    /// Multiplier applied to a team's scoring rate per red card it has received.
    pub red_card_self_penalty: f64,
    /// Multiplier applied to the opponent's scoring rate per red card a team receives.
    pub red_card_opponent_bonus: f64,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            max_remaining_goals: 8,
            red_card_self_penalty: 0.75,
            red_card_opponent_bonus: 1.15,
        }
    }
}

/// Snapshot of a live match's observable state.
#[derive(Debug, Clone, Copy)]
pub struct LiveState {
    pub current: Scoreline,
    pub minute: u16,
    pub home_red_cards: u8,
    pub away_red_cards: u8,
}

impl LiveState {
    pub fn new(current: Scoreline, minute: u16) -> Self {
        Self {
            current,
            minute,
            home_red_cards: 0,
            away_red_cards: 0,
        }
    }
}

/// Expected goals each side will *still* score, given the time left and red cards.
fn remaining_rates(
    base_lambda: f64,
    base_mu: f64,
    state: &LiveState,
    config: &LiveConfig,
) -> (f64, f64) {
    let fraction_left =
        f64::from(REGULATION_MINUTES.saturating_sub(state.minute)) / f64::from(REGULATION_MINUTES);
    let fraction_left = fraction_left.clamp(0.0, 1.0);

    let home_mult = config
        .red_card_self_penalty
        .powi(i32::from(state.home_red_cards))
        * config
            .red_card_opponent_bonus
            .powi(i32::from(state.away_red_cards));
    let away_mult = config
        .red_card_self_penalty
        .powi(i32::from(state.away_red_cards))
        * config
            .red_card_opponent_bonus
            .powi(i32::from(state.home_red_cards));

    (
        base_lambda * fraction_left * home_mult,
        base_mu * fraction_left * away_mult,
    )
}

/// The distribution over the **final** scoreline given the current live state.
pub fn live_score_grid(
    base_lambda: f64,
    base_mu: f64,
    state: &LiveState,
    config: &LiveConfig,
) -> ScoreGrid {
    let (rem_h, rem_a) = remaining_rates(base_lambda, base_mu, state, config);
    let m = config.max_remaining_goals;
    let cur_h = state.current.home as usize;
    let cur_a = state.current.away as usize;
    let max_goals = cur_h.max(cur_a) + m;

    ScoreGrid::from_fn(max_goals, |h, a| {
        // Final score (h,a) requires (h-cur_h, a-cur_a) further goals; impossible if
        // fewer than already scored.
        if h < cur_h || a < cur_a {
            return 0.0;
        }
        poisson_pmf((h - cur_h) as u32, rem_h) * poisson_pmf((a - cur_a) as u32, rem_a)
    })
}

/// Live win/draw/win probabilities for the final result.
pub fn live_probabilities(
    base_lambda: f64,
    base_mu: f64,
    state: &LiveState,
    config: &LiveConfig,
) -> Probabilities {
    live_score_grid(base_lambda, base_mu, state, config).outcome_probabilities()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_kickoff_live_matches_prematch_shape() {
        // 0-0 at minute 0 should resemble a full-match Poisson with rates (λ, μ).
        let cfg = LiveConfig::default();
        let state = LiveState::new(Scoreline::new(0, 0), 0);
        let p = live_probabilities(1.6, 1.1, &state, &cfg);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!(p.home_win > p.away_win, "higher λ favours home");
    }

    #[test]
    fn late_lead_is_nearly_decisive() {
        let cfg = LiveConfig::default();
        // Home leads 1-0 with one minute to play.
        let state = LiveState::new(Scoreline::new(1, 0), 89);
        let p = live_probabilities(1.5, 1.5, &state, &cfg);
        assert!(p.home_win > 0.9, "a 1-0 lead at 89' should be ~won");
    }

    #[test]
    fn full_time_is_certain() {
        let cfg = LiveConfig::default();
        let state = LiveState::new(Scoreline::new(2, 1), 90);
        let p = live_probabilities(1.5, 1.5, &state, &cfg);
        assert!(
            (p.home_win - 1.0).abs() < 1e-9,
            "no time left ⇒ result is fixed"
        );
    }

    #[test]
    fn red_card_hurts_the_carded_team() {
        let cfg = LiveConfig::default();
        let base = LiveState::new(Scoreline::new(0, 0), 45);
        let mut carded = base;
        carded.home_red_cards = 1;
        let p_base = live_probabilities(1.4, 1.4, &base, &cfg);
        let p_red = live_probabilities(1.4, 1.4, &carded, &cfg);
        assert!(
            p_red.home_win < p_base.home_win,
            "a home red card should lower home's win probability"
        );
    }
}
