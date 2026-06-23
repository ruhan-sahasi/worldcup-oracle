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
//! log-likelihood for the attack/defense/intercept/home terms with a backtracking,
//! convergence-checked step (a non-improving step is rolled back and the learning rate
//! halved, so the fit cannot oscillate or diverge), then fit the dependence parameter by a
//! one-dimensional search over the full corrected likelihood.
//!
//! Two dependence models are available (see [`ScoreModel`]): independent margins with the
//! Dixon-Coles low-score correction `ρ`, or a **bivariate Poisson** whose shared component
//! models positive correlation directly. The `wc-oracle tune` command picks between them,
//! and the rest of the hyperparameters, by held-out log-loss.

use crate::poisson::{bivariate_poisson_pmf, poisson_pmf};
use oracle_domain::{Probabilities, ScoreGrid, Scoreline, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One historical result used to fit the model.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub home: TeamId,
    pub away: TeamId,
    pub score: Scoreline,
    /// Expected goals (xG) for each side, when available. The fit regresses on xG instead
    /// of goals when present: xG is a far less noisy signal than the realized scoreline.
    pub home_xg: Option<f64>,
    pub away_xg: Option<f64>,
    /// Age of the match in days; older matches are exponentially down-weighted.
    pub age_days: f64,
}

impl Observation {
    pub fn new(home: TeamId, away: TeamId, score: Scoreline, age_days: f64) -> Self {
        Self {
            home,
            away,
            score,
            home_xg: None,
            away_xg: None,
            age_days,
        }
    }

    /// As [`new`], but with expected goals attached, which the fit prefers over goals.
    pub fn with_xg(
        home: TeamId,
        away: TeamId,
        score: Scoreline,
        home_xg: f64,
        away_xg: f64,
        age_days: f64,
    ) -> Self {
        Self {
            home,
            away,
            score,
            home_xg: Some(home_xg),
            away_xg: Some(away_xg),
            age_days,
        }
    }

    /// Regression target for the home rate: xG if present, else the realized goals.
    fn target_home(&self) -> f64 {
        self.home_xg.unwrap_or_else(|| f64::from(self.score.home))
    }

    /// Regression target for the away rate: xG if present, else the realized goals.
    fn target_away(&self) -> f64 {
        self.away_xg.unwrap_or_else(|| f64::from(self.score.away))
    }
}

/// How the joint scoreline distribution models the dependence between the two teams' goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ScoreModel {
    /// Independent Poisson margins with the Dixon-Coles low-score correction `ρ`.
    #[default]
    Independent,
    /// Bivariate Poisson: a shared component induces positive correlation directly.
    Bivariate,
}

/// Hyper-parameters controlling the fit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DixonColesConfig {
    /// Time-decay rate ξ (per day). ~0.003 ≈ a half-life of ~230 days.
    pub xi: f64,
    /// Largest goal count modelled exactly in the score grid.
    pub max_goals: usize,
    /// Maximum gradient-ascent iterations (the fit stops early once it converges).
    pub iterations: usize,
    /// Gradient-ascent learning rate.
    pub learning_rate: f64,
    /// L2 (ridge) shrinkage on the attack/defense coefficients toward the mean. Because a
    /// data-rich team accumulates a larger gradient than a data-poor one, this shrinks
    /// sparse-data teams more, which is exactly the regularization a World Cup needs.
    pub ridge: f64,
    /// Relative-improvement tolerance for the convergence check.
    pub tol: f64,
    /// Which dependence model the score grid uses.
    pub model: ScoreModel,
}

impl Default for DixonColesConfig {
    fn default() -> Self {
        Self {
            xi: 0.003,
            max_goals: 10,
            iterations: 400,
            learning_rate: 0.06,
            ridge: 0.01,
            tol: 1e-7,
            model: ScoreModel::Independent,
        }
    }
}

/// A fitted Dixon-Coles model. Cheap to clone and query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalModel {
    intercept: f64,
    home_advantage: f64,
    /// Dixon-Coles low-score correction (used by [`ScoreModel::Independent`]).
    rho: f64,
    /// Bivariate-Poisson covariance term λ3 (used by [`ScoreModel::Bivariate`]).
    #[serde(default)]
    covariance: f64,
    #[serde(default)]
    model: ScoreModel,
    max_goals: usize,
    attack: HashMap<TeamId, f64>,
    defense: HashMap<TeamId, f64>,
    /// Total time-decay weight of the matches that informed each team, used to gauge how
    /// confident the team's rating is (more data -> lower strength uncertainty).
    #[serde(default)]
    match_weight: HashMap<TeamId, f64>,
}

impl Default for GoalModel {
    fn default() -> Self {
        // A neutral model: league-average scoring, modest home edge, no team data.
        Self {
            intercept: 0.3,       // exp(0.3) ≈ 1.35 baseline goals
            home_advantage: 0.25, // exp(0.25) ≈ 1.28× at home
            rho: -0.05,
            covariance: 0.0,
            model: ScoreModel::Independent,
            max_goals: 10,
            attack: HashMap::new(),
            defense: HashMap::new(),
            match_weight: HashMap::new(),
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

        // How much (time-decayed) data informed each team, for the strength-uncertainty.
        let mut match_weight: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
        for (o, &w) in observations.iter().zip(&weights) {
            *match_weight.get_mut(&o.home).unwrap() += w;
            *match_weight.get_mut(&o.away).unwrap() += w;
        }

        let mut attack: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
        let mut defense: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
        // Seed intercept from the (weighted) average expected goals per team per match
        // (xG when available, else realized goals).
        let avg_goals: f64 = observations
            .iter()
            .zip(&weights)
            .map(|(o, w)| w * (o.target_home() + o.target_away()) / 2.0)
            .sum::<f64>()
            / total_weight;
        let mut intercept = avg_goals.max(0.2).ln();
        let mut home_advantage = 0.25_f64;

        // The time-weighted Poisson mean-fit log-likelihood: exactly what the gradient step
        // ascends (regressing the marginal rates on the targets), so it is the right
        // objective to monitor for convergence and to guard against an overshooting step.
        let mean_ll = |attack: &HashMap<TeamId, f64>,
                       defense: &HashMap<TeamId, f64>,
                       intercept: f64,
                       home: f64|
         -> f64 {
            observations
                .iter()
                .zip(&weights)
                .map(|(o, &w)| {
                    let lambda = (intercept + attack[&o.home] - defense[&o.away] + home).exp();
                    let mu = (intercept + attack[&o.away] - defense[&o.home]).exp();
                    w * (o.target_home() * lambda.max(1e-12).ln() - lambda
                        + o.target_away() * mu.max(1e-12).ln()
                        - mu)
                })
                .sum()
        };

        // ---- Gradient ascent to convergence, with an objective-monotone (backtracking)
        // step: a step that fails to improve the objective is rolled back and the learning
        // rate halved, so the fit cannot oscillate or diverge and stops once it has settled.
        let mut lr = config.learning_rate;
        let mut prev_ll = mean_ll(&attack, &defense, intercept, home_advantage);
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

                // Regress on xG when present (sharper than goals); the Poisson score
                // equation `target - rate` is a valid estimating equation either way.
                let res_h = o.target_home() - lambda; // ∂/∂(logλ)
                let res_a = o.target_away() - mu; // ∂/∂(logμ)

                *g_attack.get_mut(&o.home).unwrap() += w * res_h;
                *g_attack.get_mut(&o.away).unwrap() += w * res_a;
                *g_defense.get_mut(&o.away).unwrap() -= w * res_h;
                *g_defense.get_mut(&o.home).unwrap() -= w * res_a;
                g_intercept += w * (res_h + res_a);
                g_home += w * res_h;
            }

            // Snapshot so a non-improving step can be rolled back.
            let (snap_a, snap_d, snap_i, snap_h) =
                (attack.clone(), defense.clone(), intercept, home_advantage);

            let step = lr / total_weight;
            let shrink = lr * config.ridge;
            for t in &teams {
                let (a, d) = (attack[t], defense[t]);
                // Gradient ascent on the log-likelihood, minus an L2 penalty (ridge).
                *attack.get_mut(t).unwrap() += step * g_attack[t] - shrink * a;
                *defense.get_mut(t).unwrap() += step * g_defense[t] - shrink * d;
            }
            intercept += step * g_intercept;
            home_advantage += step * g_home;

            // Identifiability: pin mean(attack) = mean(defense) = 0 so the levels are
            // absorbed by the intercept rather than drifting freely.
            recenter(&mut attack);
            recenter(&mut defense);

            let ll = mean_ll(&attack, &defense, intercept, home_advantage);
            if !ll.is_finite() || ll < prev_ll {
                // Overshoot: undo the step and take a smaller one next time.
                attack = snap_a;
                defense = snap_d;
                intercept = snap_i;
                home_advantage = snap_h;
                lr *= 0.5;
                if lr <= 1e-9 {
                    break;
                }
                continue;
            }
            if ll - prev_ll <= config.tol * prev_ll.abs().max(1.0) {
                break; // converged
            }
            prev_ll = ll;
        }

        // ---- Fit the dependence parameter on the fully-corrected (integer-goal) likelihood.
        let (rho, covariance) = match config.model {
            ScoreModel::Independent => (
                fit_rho(
                    observations,
                    &weights,
                    &attack,
                    &defense,
                    intercept,
                    home_advantage,
                ),
                0.0,
            ),
            ScoreModel::Bivariate => (
                0.0,
                fit_covariance(
                    observations,
                    &weights,
                    &attack,
                    &defense,
                    intercept,
                    home_advantage,
                ),
            ),
        };

        GoalModel {
            intercept,
            home_advantage,
            rho,
            covariance,
            model: config.model,
            max_goals: config.max_goals,
            attack,
            defense,
            match_weight,
        }
    }

    /// Log-space standard deviation of a team's attack/defense rating, reflecting how much
    /// data informed it: a team with little match history is more uncertain. Used by the
    /// Monte-Carlo to resample team strength per iteration, so champion odds are not
    /// over-concentrated (a point estimate treated as certain). Returns 0 for the default
    /// (data-free) model.
    pub fn strength_uncertainty(&self, team: TeamId) -> f64 {
        // SE ~ sigma0 / sqrt(effective sample size), floored so a thin record stays bounded.
        const SIGMA0: f64 = 1.0;
        const CAP: f64 = 0.6;
        let weight = self.match_weight.get(&team).copied().unwrap_or(0.0);
        if weight <= 0.0 {
            return 0.0;
        }
        (SIGMA0 / weight.max(1.0).sqrt()).min(CAP)
    }

    /// The fitted bivariate-Poisson covariance term (`0` for the independent model).
    pub fn covariance(&self) -> f64 {
        self.covariance
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

    /// Online update from a single finished match: one gradient-ascent step on the two
    /// teams' attack/defense coefficients toward the observed score, at learning rate `lr`.
    ///
    /// This is the per-observation step of [`fit`] applied incrementally, so the model
    /// learns from in-tournament results as they land instead of staying frozen at the
    /// offline fit. Only the four team-specific coefficients move, leaving the league-level
    /// intercept and home advantage fixed, so global drift stays bounded; the residual is
    /// clamped so one blowout cannot swing a rating. We intentionally do not recenter
    /// (recentering is a global batch-fit operation and would perturb every other team).
    pub fn update_with_result(
        &mut self,
        home: TeamId,
        away: TeamId,
        score: Scoreline,
        neutral: bool,
        lr: f64,
    ) {
        if lr == 0.0 {
            return;
        }
        let (lambda, mu) = self.expected_goals(home, away, neutral);
        let res_h = (f64::from(score.home) - lambda).clamp(-4.0, 4.0);
        let res_a = (f64::from(score.away) - mu).clamp(-4.0, 4.0);

        // attack rises / opponent defense falls when a side outscores its expectation.
        *self.attack.entry(home).or_insert(0.0) += lr * res_h;
        *self.defense.entry(away).or_insert(0.0) -= lr * res_h;
        *self.attack.entry(away).or_insert(0.0) += lr * res_a;
        *self.defense.entry(home).or_insert(0.0) -= lr * res_a;
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

    /// Build the joint score grid from explicit marginal goal rates, using whichever
    /// dependence model this was fit with.
    fn grid_from(&self, lambda: f64, mu: f64) -> ScoreGrid {
        match self.model {
            ScoreModel::Independent => ScoreGrid::from_fn(self.max_goals, |h, a| {
                poisson_pmf(h as u32, lambda)
                    * poisson_pmf(a as u32, mu)
                    * tau(h, a, lambda, mu, self.rho)
            }),
            ScoreModel::Bivariate => {
                // Preserve the marginal means: lambda3 is the shared component, so the home
                // own-rate is lambda - lambda3 (clamped to keep both own-rates positive).
                let l3 = self.covariance.min(0.95 * lambda.min(mu)).max(0.0);
                let (l1, l2) = (lambda - l3, mu - l3);
                ScoreGrid::from_fn(self.max_goals, |h, a| {
                    bivariate_poisson_pmf(h as u32, a as u32, l1, l2, l3)
                })
            }
        }
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

/// One-dimensional search for the bivariate-Poisson covariance `λ3` that maximizes the
/// time-weighted log-likelihood, holding the fitted marginal means fixed (`λ1 = μ_home - λ3`,
/// `λ2 = μ_away - λ3`, clamped). `λ3 = 0` is the independent model, so the search only helps.
fn fit_covariance(
    obs: &[Observation],
    weights: &[f64],
    attack: &HashMap<TeamId, f64>,
    defense: &HashMap<TeamId, f64>,
    intercept: f64,
    home_advantage: f64,
) -> f64 {
    let ll = |cov: f64| -> f64 {
        obs.iter()
            .zip(weights)
            .map(|(o, &w)| {
                let lambda =
                    (intercept + attack[&o.home] - defense[&o.away] + home_advantage).exp();
                let mu = (intercept + attack[&o.away] - defense[&o.home]).exp();
                let l3 = cov.min(0.95 * lambda.min(mu)).max(0.0);
                let p = bivariate_poisson_pmf(
                    o.score.home as u32,
                    o.score.away as u32,
                    lambda - l3,
                    mu - l3,
                    l3,
                );
                w * p.max(1e-12).ln()
            })
            .sum()
    };

    // Covariance is non-negative and small; coarse grid then local refinement.
    let mut best = 0.0;
    let mut best_ll = ll(0.0);
    let (mut lo, mut hi) = (0.0, 0.6);
    for _ in 0..4 {
        let mut local_best = best;
        let steps = 20;
        for i in 0..=steps {
            let cov = lo + (hi - lo) * i as f64 / steps as f64;
            let cur = ll(cov);
            if cur > best_ll {
                best_ll = cur;
                local_best = cov;
            }
        }
        best = local_best;
        let span = (hi - lo) / steps as f64;
        lo = (best - span).max(0.0);
        hi = best + span;
    }
    best
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
    fn strength_uncertainty_is_larger_for_data_poor_teams() {
        // Teams 1-3 appear in many matches; team 9 appears in only two.
        let mut obs = synthetic_history();
        obs.push(Observation::new(t(9), t(2), Scoreline::new(1, 0), 1.0));
        obs.push(Observation::new(t(2), t(9), Scoreline::new(0, 1), 1.0));
        let model = GoalModel::fit(&obs, DixonColesConfig::default());
        assert!(
            model.strength_uncertainty(t(9)) > model.strength_uncertainty(t(1)),
            "the data-poor team should be more uncertain ({} vs {})",
            model.strength_uncertainty(t(9)),
            model.strength_uncertainty(t(1)),
        );
        // The data-free default model claims no uncertainty.
        assert_eq!(GoalModel::default().strength_uncertainty(t(1)), 0.0);
    }

    #[test]
    fn bivariate_model_fits_and_normalizes() {
        let cfg = DixonColesConfig {
            model: ScoreModel::Bivariate,
            ..DixonColesConfig::default()
        };
        let model = GoalModel::fit(&synthetic_history(), cfg);
        assert!(model.covariance() >= 0.0, "covariance is non-negative");
        let g = model.score_grid(t(1), t(2), true);
        assert!((g.sum() - 1.0).abs() < 1e-6, "bivariate grid normalizes");
        // Strength ordering still recovered under the bivariate model.
        let p = model.outcome_probabilities(t(1), t(2), true);
        assert!(p.home_win > p.away_win);
    }

    #[test]
    fn bivariate_with_more_draws_in_data_learns_positive_covariance() {
        // A history dominated by draws should pull the covariance above zero (the shared
        // component is how the bivariate model represents extra draw mass).
        let mut obs = Vec::new();
        for i in 0..60 {
            let age = i as f64;
            obs.push(Observation::new(t(1), t(2), Scoreline::new(1, 1), age));
            obs.push(Observation::new(t(2), t(1), Scoreline::new(2, 2), age));
            obs.push(Observation::new(t(1), t(2), Scoreline::new(0, 0), age));
        }
        let cfg = DixonColesConfig {
            model: ScoreModel::Bivariate,
            ..DixonColesConfig::default()
        };
        let model = GoalModel::fit(&obs, cfg);
        assert!(
            model.covariance() > 0.0,
            "draw-heavy data should learn a positive covariance (got {})",
            model.covariance()
        );
    }

    #[test]
    fn ridge_shrinks_sparse_data_teams_more() {
        // Rich history among teams 1-3, plus a team (9) with just two lopsided wins.
        let mut obs = synthetic_history();
        obs.push(Observation::new(t(9), t(2), Scoreline::new(5, 0), 1.0));
        obs.push(Observation::new(t(9), t(3), Scoreline::new(5, 0), 1.0));

        let strength = |m: &GoalModel| (m.attack_of(t(9)) - m.defense_of(t(9))).abs();
        let no_ridge = GoalModel::fit(
            &obs,
            DixonColesConfig {
                ridge: 0.0,
                ..DixonColesConfig::default()
            },
        );
        let heavy_ridge = GoalModel::fit(
            &obs,
            DixonColesConfig {
                ridge: 0.5,
                ..DixonColesConfig::default()
            },
        );
        assert!(
            strength(&heavy_ridge) < strength(&no_ridge),
            "ridge should shrink the sparse team's strength ({:.3} -> {:.3})",
            strength(&no_ridge),
            strength(&heavy_ridge),
        );
    }

    #[test]
    fn online_update_strengthens_a_repeated_winner() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let before = model.outcome_probabilities(t(2), t(1), true).home_win;

        // Team 2 (the weak side) now thrashes team 1 several times in the tournament.
        let mut learned = model.clone();
        let s2 = learned.attack_of(t(2)) - learned.defense_of(t(2));
        for _ in 0..6 {
            learned.update_with_result(t(2), t(1), Scoreline::new(4, 0), true, 0.05);
        }
        let s2_after = learned.attack_of(t(2)) - learned.defense_of(t(2));
        assert!(s2_after > s2, "a repeated big winner should strengthen");

        let after = learned.outcome_probabilities(t(2), t(1), true).home_win;
        assert!(
            after > before,
            "its win probability vs team 1 should rise ({before:.3} -> {after:.3})"
        );
    }

    #[test]
    fn online_update_with_zero_lr_is_a_noop() {
        let mut model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        let before = model.expected_goals(t(1), t(2), true);
        model.update_with_result(t(1), t(2), Scoreline::new(5, 0), true, 0.0);
        let after = model.expected_goals(t(1), t(2), true);
        assert_eq!(before, after);
    }

    #[test]
    fn xg_drives_the_fit_when_goals_are_uninformative() {
        // Every match ends 0-0 (the scoreline carries no signal), but team 1 consistently
        // out-creates team 2 on xG. A goals-only fit would rate them equal; the xG fit must
        // make team 1 the favourite.
        let mut obs = Vec::new();
        for i in 0..40 {
            let age = i as f64;
            obs.push(Observation::with_xg(
                t(1),
                t(2),
                Scoreline::new(0, 0),
                2.2,
                0.5,
                age,
            ));
            obs.push(Observation::with_xg(
                t(2),
                t(1),
                Scoreline::new(0, 0),
                0.5,
                2.2,
                age,
            ));
        }
        let model = GoalModel::fit(&obs, DixonColesConfig::default());
        let (l, m) = model.expected_goals(t(1), t(2), true);
        assert!(
            l > m,
            "xG should make team 1 the favourite ({l:.2} vs {m:.2})"
        );
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
