//! The Dixon-Coles bivariate-Poisson goal model.
//!
//! Each team carries an **attack** and **defense** coefficient. For a match between
//! home `i` and away `j`:
//!
//! ```text
//! log λ = intercept + attack_i − defense_j + home_advantage   (home goals)
//! log μ = intercept + attack_j − defense_i                     (away goals)
//! ```
//!
//! Goals are approximately independent Poisson draws with rates `λ`, `μ`, except
//! that Dixon & Coles (1997) correct the four low-score cells (0-0, 1-0, 0-1, 1-1)
//! with a dependence parameter `ρ`, since real football has more draws than
//! independence implies.
//!
//! ## Fitting
//! Parameters are fit by **maximum likelihood** on historical results, with each
//! match down-weighted by `exp(−ξ · age_days)` so recent form counts for more
//! (the Dixon-Coles time-decay trick). We ascend the time-weighted Poisson
//! log-likelihood analytically for the attack/defense/intercept/home terms - the
//! Poisson score equations reduce neatly to `(observed − expected) goals` - and fit
//! `ρ` with a one-dimensional search over the full corrected likelihood.

use crate::poisson::poisson_pmf;
use oracle_domain::{Probabilities, ScoreGrid, Scoreline, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One historical result used to fit the model.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub home: TeamId,
    pub away: TeamId,
    pub score: Scoreline,
    /// Age of the match in days; older matches are exponentially down-weighted.
    pub age_days: f64,
}

impl Observation {
    pub fn new(home: TeamId, away: TeamId, score: Scoreline, age_days: f64) -> Self {
        Self {
            home,
            away,
            score,
            age_days,
        }
    }
}

/// Hyper-parameters controlling the fit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DixonColesConfig {
    /// Time-decay rate ξ (per day). ~0.003 ≈ a half-life of ~230 days.
    pub xi: f64,
    /// Largest goal count modelled exactly in the score grid.
    pub max_goals: usize,
    /// Gradient-ascent iterations.
    pub iterations: usize,
    /// Gradient-ascent learning rate.
    pub learning_rate: f64,
}

impl Default for DixonColesConfig {
    fn default() -> Self {
        Self {
            xi: 0.003,
            max_goals: 10,
            iterations: 400,
            learning_rate: 0.06,
        }
    }
}

/// A fitted Dixon-Coles model. Cheap to clone and query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalModel {
    intercept: f64,
    home_advantage: f64,
    rho: f64,
    max_goals: usize,
    attack: HashMap<TeamId, f64>,
    defense: HashMap<TeamId, f64>,
}

impl Default for GoalModel {
    fn default() -> Self {
        // A neutral model: league-average scoring, modest home edge, no team data.
        Self {
            intercept: 0.3,       // exp(0.3) ≈ 1.35 baseline goals
            home_advantage: 0.25, // exp(0.25) ≈ 1.28× at home
            rho: -0.05,
            max_goals: 10,
            attack: HashMap::new(),
            defense: HashMap::new(),
        }
    }
}

impl GoalModel {
    /// Fit a model to historical results by time-weighted maximum likelihood.
    /// Returns a sensible default model if `observations` is empty.
    pub fn fit(observations: &[Observation], config: DixonColesConfig) -> Self {
        if observations.is_empty() {
            return GoalModel {
                max_goals: config.max_goals,
                ..GoalModel::default()
            };
        }

        // Collect the team universe and precompute time-decay weights.
        let mut teams: Vec<TeamId> = Vec::new();
        let mut seen = HashMap::new();
        for o in observations {
            for t in [o.home, o.away] {
                if seen.insert(t, ()).is_none() {
                    teams.push(t);
                }
            }
        }
        let weights: Vec<f64> = observations
            .iter()
            .map(|o| (-config.xi * o.age_days).exp())
            .collect();
        let total_weight: f64 = weights.iter().sum();

        let mut attack: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
        let mut defense: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
        // Seed intercept from the (weighted) average goals per team per match.
        let avg_goals: f64 = observations
            .iter()
            .zip(&weights)
            .map(|(o, w)| w * f64::from(o.score.home + o.score.away) / 2.0)
            .sum::<f64>()
            / total_weight;
        let mut intercept = avg_goals.max(0.2).ln();
        let mut home_advantage = 0.25_f64;

        // ---- Gradient ascent on the time-weighted Poisson log-likelihood ----
        for _ in 0..config.iterations {
            let mut g_attack: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
            let mut g_defense: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
            let mut g_intercept = 0.0;
            let mut g_home = 0.0;

            for (o, &w) in observations.iter().zip(&weights) {
                let a_h = attack[&o.home];
                let a_a = attack[&o.away];
                let d_h = defense[&o.home];
                let d_a = defense[&o.away];
                let lambda = (intercept + a_h - d_a + home_advantage).exp();
                let mu = (intercept + a_a - d_h).exp();

                let res_h = f64::from(o.score.home) - lambda; // ∂/∂(logλ)
                let res_a = f64::from(o.score.away) - mu; // ∂/∂(logμ)

                *g_attack.get_mut(&o.home).unwrap() += w * res_h;
                *g_attack.get_mut(&o.away).unwrap() += w * res_a;
                *g_defense.get_mut(&o.away).unwrap() -= w * res_h;
                *g_defense.get_mut(&o.home).unwrap() -= w * res_a;
                g_intercept += w * (res_h + res_a);
                g_home += w * res_h;
            }

            let step = config.learning_rate / total_weight;
            for t in &teams {
                *attack.get_mut(t).unwrap() += step * g_attack[t];
                *defense.get_mut(t).unwrap() += step * g_defense[t];
            }
            intercept += step * g_intercept;
            home_advantage += step * g_home;

            // Identifiability: pin mean(attack) = mean(defense) = 0 so the levels are
            // absorbed by the intercept rather than drifting freely.
            recenter(&mut attack);
            recenter(&mut defense);
        }

        // ---- Fit ρ on the fully-corrected likelihood via a refined grid search ----
        let rho = fit_rho(
            observations,
            &weights,
            &attack,
            &defense,
            intercept,
            home_advantage,
        );

        GoalModel {
            intercept,
            home_advantage,
            rho,
            max_goals: config.max_goals,
            attack,
            defense,
        }
    }

    pub fn home_advantage(&self) -> f64 {
        self.home_advantage
    }

    pub fn rho(&self) -> f64 {
        self.rho
    }

    fn attack_of(&self, t: TeamId) -> f64 {
        self.attack.get(&t).copied().unwrap_or(0.0)
    }

    fn defense_of(&self, t: TeamId) -> f64 {
        self.defense.get(&t).copied().unwrap_or(0.0)
    }

    /// Expected goals `(λ_home, μ_away)` for a matchup. `neutral` removes the home
    /// edge (the World Cup default).
    pub fn expected_goals(&self, home: TeamId, away: TeamId, neutral: bool) -> (f64, f64) {
        self.expected_goals_adjusted(home, away, neutral, (0.0, 0.0), (0.0, 0.0))
    }

    /// Expected goals with per-team lineup adjustments, each `(attack_delta, defense_delta)`
    /// in log space where positive means stronger. A confirmed lineup missing a key player
    /// supplies a negative attack delta, which lowers that team's expected goals and raises
    /// the opponent's. With zero adjustments this equals [`expected_goals`].
    pub fn expected_goals_adjusted(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
        home_adj: (f64, f64),
        away_adj: (f64, f64),
    ) -> (f64, f64) {
        let adv = if neutral { 0.0 } else { self.home_advantage };
        let (h_atk, h_def) = home_adj;
        let (a_atk, a_def) = away_adj;
        let lambda = (self.intercept + self.attack_of(home) - self.defense_of(away) + adv + h_atk
            - a_def)
            .exp();
        let mu =
            (self.intercept + self.attack_of(away) - self.defense_of(home) + a_atk - h_def).exp();
        (lambda, mu)
    }

    /// Build the Dixon-Coles score grid from explicit goal rates.
    fn grid_from(&self, lambda: f64, mu: f64) -> ScoreGrid {
        ScoreGrid::from_fn(self.max_goals, |h, a| {
            poisson_pmf(h as u32, lambda)
                * poisson_pmf(a as u32, mu)
                * tau(h, a, lambda, mu, self.rho)
        })
    }

    /// The full joint score distribution with the Dixon-Coles low-score correction.
    pub fn score_grid(&self, home: TeamId, away: TeamId, neutral: bool) -> ScoreGrid {
        let (lambda, mu) = self.expected_goals(home, away, neutral);
        self.grid_from(lambda, mu)
    }

    /// As [`score_grid`], with per-team lineup adjustments applied to the goal rates.
    pub fn score_grid_adjusted(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
        home_adj: (f64, f64),
        away_adj: (f64, f64),
    ) -> ScoreGrid {
        let (lambda, mu) = self.expected_goals_adjusted(home, away, neutral, home_adj, away_adj);
        self.grid_from(lambda, mu)
    }

    /// Pre-match win/draw/win probabilities for a matchup.
    pub fn outcome_probabilities(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
    ) -> Probabilities {
        self.score_grid(home, away, neutral).outcome_probabilities()
    }

    /// Teams ranked by overall strength (attack − defense), strongest first.
    /// A nice human-readable view of what the model learned.
    pub fn strength_ranking(&self) -> Vec<(TeamId, f64)> {
        let mut v: Vec<(TeamId, f64)> = self
            .attack
            .keys()
            .map(|&t| (t, self.attack_of(t) - self.defense_of(t)))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// The Dixon-Coles dependence correction `τ` for the four low-score cells.
fn tau(x: usize, y: usize, lambda: f64, mu: f64, rho: f64) -> f64 {
    match (x, y) {
        (0, 0) => 1.0 - lambda * mu * rho,
        (0, 1) => 1.0 + lambda * rho,
        (1, 0) => 1.0 + mu * rho,
        (1, 1) => 1.0 - rho,
        _ => 1.0,
    }
    .max(0.0001) // keep the likelihood strictly positive
}

fn recenter(map: &mut HashMap<TeamId, f64>) {
    if map.is_empty() {
        return;
    }
    let mean: f64 = map.values().sum::<f64>() / map.len() as f64;
    for v in map.values_mut() {
        *v -= mean;
    }
}

/// One-dimensional search for the ρ that maximizes the time-weighted corrected
/// log-likelihood, given the already-fit attack/defense parameters.
fn fit_rho(
    obs: &[Observation],
    weights: &[f64],
    attack: &HashMap<TeamId, f64>,
    defense: &HashMap<TeamId, f64>,
    intercept: f64,
    home_advantage: f64,
) -> f64 {
    let ll = |rho: f64| -> f64 {
        obs.iter()
            .zip(weights)
            .map(|(o, &w)| {
                let lambda =
                    (intercept + attack[&o.home] - defense[&o.away] + home_advantage).exp();
                let mu = (intercept + attack[&o.away] - defense[&o.home]).exp();
                let x = o.score.home as usize;
                let y = o.score.away as usize;
                let p = poisson_pmf(x as u32, lambda)
                    * poisson_pmf(y as u32, mu)
                    * tau(x, y, lambda, mu, rho);
                w * p.max(1e-12).ln()
            })
            .sum()
    };

    // Coarse grid then a local refinement; ρ is physically constrained near 0.
    let mut best_rho = 0.0;
    let mut best_ll = f64::NEG_INFINITY;
    let mut lo = -0.2;
    let mut hi = 0.2;
    for _ in 0..4 {
        let mut local_best = best_rho;
        let steps = 20;
        for i in 0..=steps {
            let rho = lo + (hi - lo) * i as f64 / steps as f64;
            let cur = ll(rho);
            if cur > best_ll {
                best_ll = cur;
                local_best = rho;
            }
        }
        best_rho = local_best;
        let span = (hi - lo) / steps as f64;
        lo = best_rho - span;
        hi = best_rho + span;
    }
    best_rho
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: u32) -> TeamId {
        TeamId(n)
    }

    /// Build a synthetic history where team 1 is strong and team 2 is weak, then
    /// check the fitted model recovers that ordering and produces valid grids.
    fn synthetic_history() -> Vec<Observation> {
        let mut obs = Vec::new();
        // Strong team (1) beats weak team (2) repeatedly; mid team (3) in between.
        for i in 0..30 {
            let age = i as f64;
            obs.push(Observation::new(t(1), t(2), Scoreline::new(3, 0), age));
            obs.push(Observation::new(t(2), t(1), Scoreline::new(0, 2), age));
            obs.push(Observation::new(t(1), t(3), Scoreline::new(2, 1), age));
            obs.push(Observation::new(t(3), t(2), Scoreline::new(2, 0), age));
        }
        obs
    }

    #[test]
    fn fit_recovers_strength_ordering() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let s1 = model.attack_of(t(1)) - model.defense_of(t(1));
        let s2 = model.attack_of(t(2)) - model.defense_of(t(2));
        let s3 = model.attack_of(t(3)) - model.defense_of(t(3));
        assert!(s1 > s3, "team 1 should outrank team 3");
        assert!(s3 > s2, "team 3 should outrank team 2");
    }

    #[test]
    fn strong_team_is_favoured() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let p = model.outcome_probabilities(t(1), t(2), true);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!(p.home_win > p.away_win);
        assert!(p.home_win > 0.5);
    }

    #[test]
    fn lineup_penalty_lowers_expected_goals_and_win_prob() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let (base_lambda, _) = model.expected_goals(t(1), t(2), true);
        // Home team fields a weakened attack (a missing key player).
        let (weak_lambda, _) =
            model.expected_goals_adjusted(t(1), t(2), true, (-0.4, 0.0), (0.0, 0.0));
        assert!(
            weak_lambda < base_lambda,
            "a negative attack delta lowers home xG"
        );

        let base = model.outcome_probabilities(t(1), t(2), true);
        let weak = model
            .score_grid_adjusted(t(1), t(2), true, (-0.4, 0.0), (0.0, 0.0))
            .outcome_probabilities();
        assert!(
            weak.home_win < base.home_win,
            "weakened home team is less favoured"
        );
        assert!((weak.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn score_grid_is_normalized_and_positive() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let g = model.score_grid(t(1), t(3), false);
        assert!((g.sum() - 1.0).abs() < 1e-6);
        assert!(g.grid.iter().flatten().all(|&p| p >= 0.0));
    }

    #[test]
    fn expected_goals_positive_and_home_edge_applies() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let (l_home, _) = model.expected_goals(t(1), t(2), false);
        let (l_neutral, _) = model.expected_goals(t(1), t(2), true);
        assert!(l_home > 0.0);
        assert!(l_home > l_neutral, "home advantage raises home xG");
    }

    #[test]
    fn empty_history_yields_usable_default() {
        let model = GoalModel::fit(&[], DixonColesConfig::default());
        let p = model.outcome_probabilities(t(1), t(2), false);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!(
            p.home_win > p.away_win,
            "default model still has a home edge"
        );
    }
}
