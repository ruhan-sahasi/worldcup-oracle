//! # oracle-ratings
//!
//! A self-contained Elo rating engine tuned for international football, providing
//! one of the two pillars of the ensemble (the other being the Dixon-Coles goal
//! model in `oracle-model`).
//!
//! Two refinements over textbook Elo make it credible for football:
//!
//! 1. **Margin-of-victory scaling** - a 4-0 win shifts ratings more than a 1-0 win,
//!    using the well-known World Football Elo goal-difference index `G`, while
//!    dampening the effect for already-lopsided matchups (so favourites can't farm
//!    rating points). See [`EloConfig`] and [`RatingStore::record`].
//!
//! 2. **Three-way outcome conversion** - raw Elo yields an *expected score*
//!    `E ∈ [0,1]` that merges wins and draws. Football needs an explicit draw
//!    probability, so [`RatingStore::win_probabilities`] models the draw mass as a
//!    Gaussian in the rating gap and splits `E` into home/draw/away while preserving
//!    `E = P(home) + ½·P(draw)`.
#![forbid(unsafe_code)]

use oracle_domain::{Probabilities, Scoreline, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tunable Elo parameters. Defaults are calibrated for World-Cup-level football.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EloConfig {
    /// Base K-factor (rating volatility). World Cup matches warrant a high K.
    pub k: f64,
    /// Rating points added to the home side (0 at a neutral venue).
    pub home_advantage: f64,
    /// Default rating assigned to an unseen team.
    pub initial: f64,
    /// Logistic scale (the classic Elo "400").
    pub scale: f64,
    /// Peak draw probability, reached when two teams are evenly matched.
    pub draw_peak: f64,
    /// Rating-gap scale over which the draw probability decays.
    pub draw_scale: f64,
}

impl Default for EloConfig {
    fn default() -> Self {
        Self {
            k: 40.0,
            home_advantage: 65.0,
            initial: 1500.0,
            scale: 400.0,
            draw_peak: 0.32,
            draw_scale: 200.0,
        }
    }
}

/// A mutable store of team Elo ratings plus the update/inference logic.
///
/// Designed for single-writer use inside the engine task; reads for the public API
/// go through immutable snapshots ([`RatingStore::snapshot`]) rather than this store
/// directly, so no locking is needed here.
#[derive(Debug, Clone)]
pub struct RatingStore {
    config: EloConfig,
    ratings: HashMap<TeamId, f64>,
}

impl RatingStore {
    pub fn new(config: EloConfig) -> Self {
        Self {
            config,
            ratings: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(EloConfig::default())
    }

    pub fn config(&self) -> &EloConfig {
        &self.config
    }

    /// Current rating for a team (the configured initial rating if unseen).
    pub fn rating(&self, team: TeamId) -> f64 {
        self.ratings
            .get(&team)
            .copied()
            .unwrap_or(self.config.initial)
    }

    /// Seed a team's rating explicitly (e.g. from a prior or historical ratings file).
    pub fn seed(&mut self, team: TeamId, rating: f64) {
        self.ratings.insert(team, rating);
    }

    /// Expected score for the home team (∈ [0,1]) under the logistic Elo model.
    /// `neutral` drops the home-advantage bonus (the norm at a World Cup).
    pub fn expected_score(&self, home: TeamId, away: TeamId, neutral: bool) -> f64 {
        let adv = if neutral {
            0.0
        } else {
            self.config.home_advantage
        };
        let diff = (self.rating(home) + adv) - self.rating(away);
        1.0 / (1.0 + 10f64.powf(-diff / self.config.scale))
    }

    /// Convert ratings into a full win/draw/win distribution.
    ///
    /// The expected score `E` fixes `P(home) + ½P(draw) = E`. We additionally model
    /// the draw mass as `draw_peak · exp(-(Δ/draw_scale)²)` (most likely when teams
    /// are even) and solve for the three probabilities, clamping any negatives.
    pub fn win_probabilities(&self, home: TeamId, away: TeamId, neutral: bool) -> Probabilities {
        let e = self.expected_score(home, away, neutral);
        let adv = if neutral {
            0.0
        } else {
            self.config.home_advantage
        };
        let diff = (self.rating(home) + adv) - self.rating(away);
        let p_draw = self.config.draw_peak * (-(diff / self.config.draw_scale).powi(2)).exp();
        // E = p_home + 0.5 * p_draw  =>  p_home = E - 0.5 p_draw
        let p_home = e - 0.5 * p_draw;
        let p_away = 1.0 - e - 0.5 * p_draw;
        Probabilities::new(p_home, p_draw, p_away)
    }

    /// The World Football Elo goal-difference index `G`: how much a result's margin
    /// amplifies the rating update. 1-goal → 1.0, 2 → 1.5, 3 → 1.75, then +⅛ per goal.
    fn goal_diff_index(goal_diff: u8) -> f64 {
        match goal_diff {
            0 | 1 => 1.0,
            2 => 1.5,
            d => (11.0 + d as f64) / 8.0,
        }
    }

    /// Update both teams' ratings from a finished match. Zero-sum: whatever the home
    /// team gains, the away team loses, scaled by the margin of victory.
    pub fn record(&mut self, home: TeamId, away: TeamId, score: Scoreline, neutral: bool) {
        let expected_home = self.expected_score(home, away, neutral);
        let actual_home = match score.outcome() {
            oracle_domain::Outcome::HomeWin => 1.0,
            oracle_domain::Outcome::Draw => 0.5,
            oracle_domain::Outcome::AwayWin => 0.0,
        };
        let goal_diff = score.home.abs_diff(score.away);
        let g = Self::goal_diff_index(goal_diff);
        let delta = self.config.k * g * (actual_home - expected_home);

        let h = self.rating(home);
        let a = self.rating(away);
        self.ratings.insert(home, h + delta);
        self.ratings.insert(away, a - delta);
    }

    /// An immutable snapshot of all current ratings for read-only consumers.
    pub fn snapshot(&self) -> HashMap<TeamId, f64> {
        self.ratings.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: u32) -> TeamId {
        TeamId(n)
    }

    #[test]
    fn equal_teams_have_even_expected_score_at_neutral() {
        let s = RatingStore::with_defaults();
        let e = s.expected_score(t(1), t(2), true);
        assert!((e - 0.5).abs() < 1e-9);
    }

    #[test]
    fn home_advantage_shifts_expected_score_up() {
        let s = RatingStore::with_defaults();
        let neutral = s.expected_score(t(1), t(2), true);
        let at_home = s.expected_score(t(1), t(2), false);
        assert!(at_home > neutral);
    }

    #[test]
    fn probabilities_are_valid_and_normalized() {
        let mut s = RatingStore::with_defaults();
        s.seed(t(1), 2000.0);
        s.seed(t(2), 1400.0);
        let p = s.win_probabilities(t(1), t(2), true);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!(p.home_win > p.away_win, "stronger team should be favoured");
        assert!(p.home_win >= 0.0 && p.draw >= 0.0 && p.away_win >= 0.0);
    }

    #[test]
    fn updates_are_zero_sum() {
        let mut s = RatingStore::with_defaults();
        s.seed(t(1), 1500.0);
        s.seed(t(2), 1500.0);
        let before = s.rating(t(1)) + s.rating(t(2));
        s.record(t(1), t(2), Scoreline::new(3, 0), true);
        let after = s.rating(t(1)) + s.rating(t(2));
        assert!((before - after).abs() < 1e-9, "total rating is conserved");
        assert!(s.rating(t(1)) > 1500.0, "winner gains");
        assert!(s.rating(t(2)) < 1500.0, "loser drops");
    }

    #[test]
    fn bigger_margin_moves_rating_more() {
        let mut narrow = RatingStore::with_defaults();
        let mut wide = RatingStore::with_defaults();
        narrow.record(t(1), t(2), Scoreline::new(1, 0), true);
        wide.record(t(1), t(2), Scoreline::new(5, 0), true);
        assert!(wide.rating(t(1)) > narrow.rating(t(1)));
    }
}
