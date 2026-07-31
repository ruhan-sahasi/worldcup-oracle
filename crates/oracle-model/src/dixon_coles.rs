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

use crate::poisson::{bivariate_poisson_pmf, neg_binomial_pmf, poisson_pmf};
use oracle_domain::{Confederation, Probabilities, ScoreGrid, Scoreline, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How strongly a team's coefficients are pooled toward its confederation mean (vs the global
/// mean) during the hierarchical fit. 0 = no pooling (plain global ridge), 1 = shrink fully to
/// the confederation level. The remaining `1 - POOL` keeps a sparse confederation's level from
/// drifting too far from neutral.
const CONFEDERATION_POOL: f64 = 0.7;

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
    /// Negative-binomial dispersion (the NB "size"/`r`) for the [`ScoreModel::Independent`]
    /// margins, fitted from the data. `0` = Poisson (no overdispersion); a finite value gives
    /// fatter scoreline tails. The variance of each margin is `mean + mean²/dispersion`.
    #[serde(default)]
    dispersion: f64,
    #[serde(default)]
    model: ScoreModel,
    max_goals: usize,
    attack: HashMap<TeamId, f64>,
    defense: HashMap<TeamId, f64>,
    /// Per-team **Fisher information** at the fitted optimum (`Σ wᵢ · rateᵢ` over the team's
    /// matches). With the ridge prior precision it gives a Laplace posterior standard deviation
    /// per team (see [`strength_uncertainty`](Self::strength_uncertainty)): more (and more recent)
    /// data means more information, hence a tighter posterior.
    #[serde(default)]
    fisher_info: HashMap<TeamId, f64>,
    /// Prior precision (the ridge strength) used in the fit, i.e. the precision of the Gaussian
    /// prior the L2 penalty corresponds to. Adds to the Fisher information in the Laplace posterior.
    #[serde(default)]
    ridge_prior: f64,
    /// Time-decay rate `ξ` used in the fit, kept so the posterior sampler can reconstruct the same
    /// time-weighted likelihood.
    #[serde(default)]
    xi: f64,
    /// Each team's confederation (when supplied to the hierarchical fit), so the fitted
    /// confederation strength levels can be reported. Empty for the confederation-agnostic model.
    #[serde(default)]
    confederation: HashMap<TeamId, Confederation>,
}

impl Default for GoalModel {
    fn default() -> Self {
        // A neutral model: league-average scoring, modest home edge, no team data.
        Self {
            intercept: 0.3,       // exp(0.3) ≈ 1.35 baseline goals
            home_advantage: 0.25, // exp(0.25) ≈ 1.28× at home
            rho: -0.05,
            covariance: 0.0,
            dispersion: 0.0,
            model: ScoreModel::Independent,
            max_goals: 10,
            attack: HashMap::new(),
            defense: HashMap::new(),
            fisher_info: HashMap::new(),
            ridge_prior: 0.0,
            xi: 0.0,
            confederation: HashMap::new(),
        }
    }
}

/// A named additive breakdown of a matchup's log expected-goal edge (see
/// [`GoalModel::rate_breakdown`]). The three edge terms sum to `ln λ_home - ln μ_away`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateBreakdown {
    pub expected_home: f64,
    pub expected_away: f64,
    /// Home attack minus away attack (who creates more).
    pub attack_edge: f64,
    /// Home defense minus away defense (who concedes less).
    pub defense_edge: f64,
    /// Home advantage in log-rate (0 at a neutral venue).
    pub home_advantage: f64,
}

impl GoalModel {
    /// Fit a model to historical results by time-weighted maximum likelihood.
    /// Returns a sensible default model if `observations` is empty.
    ///
    /// This is the confederation-agnostic fit (plain global ridge, no cross-confederation offset).
    /// Pass a team-to-confederation map to [`fit_with_confederations`](Self::fit_with_confederations)
    /// for the hierarchical variant.
    pub fn fit(observations: &[Observation], config: DixonColesConfig) -> Self {
        Self::fit_with_confederations(observations, config, &HashMap::new())
    }

    /// Override the negative-binomial dispersion (the NB size `r`; 0 = Poisson, no overdispersion).
    /// Lets a fitted model be re-run with overdispersion disabled, which the signal sensitivity
    /// analysis uses to isolate the effect of the fat-tailed (negative-binomial) margins.
    pub fn set_dispersion(&mut self, dispersion: f64) {
        self.dispersion = dispersion.max(0.0);
    }

    /// Like [`fit`](Self::fit), but **confederation-aware**. Two additions, both targeting the
    /// World Cup's core blind spot (confederations rarely play each other, so cross-confederation
    /// strength is poorly pinned down):
    ///
    /// - **Hierarchical partial pooling.** Each team's attack/defense is shrunk toward its
    ///   *confederation* mean rather than the global mean, so a data-poor team borrows strength
    ///   from its confederation instead of being dragged to the world average.
    /// - **Cross-confederation offset.** A per-confederation log-rate adjustment, fitted from the
    ///   inter-confederation results and applied only in cross-confederation matches, captures
    ///   systematic over/under-performance a confederation shows against outsiders.
    ///
    /// An empty `confederations` map reduces exactly to [`fit`](Self::fit).
    ///
    /// # Panics
    /// If an observation names a team missing from the fitted team universe. It cannot: that
    /// universe is collected from these same observations, so every `home` and `away` is a key.
    pub fn fit_with_confederations(
        observations: &[Observation],
        config: DixonColesConfig,
        confederations: &HashMap<TeamId, Confederation>,
    ) -> Self {
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

        // ---- Fit by L-BFGS on the penalized negative log-likelihood. The coefficients are
        // packed into one vector `[attack.., defense.., intercept, home]`; the penalty is the
        // ridge toward the (hierarchical) pooling target. An outer loop refreshes the
        // confederation-mean targets between solves (a block-coordinate / MM scheme) so each
        // inner objective is well defined and its gradient exact. L-BFGS converges in tens of
        // iterations where the old hand-rolled gradient ascent took hundreds.
        let index: HashMap<TeamId, usize> =
            teams.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        let dim = 2 * teams.len() + 2;
        let n_t = teams.len();
        let ridge = config.ridge;

        let mut x = vec![0.0; dim];
        x[2 * n_t] = intercept;
        x[2 * n_t + 1] = home_advantage;

        // Pooling targets (toward 0 = plain ridge when there are no confederations).
        let mut ta = vec![0.0; n_t];
        let mut td = vec![0.0; n_t];
        let outer_passes = if confederations.is_empty() { 1 } else { 4 };
        for _ in 0..outer_passes {
            {
                // Penalized negative log-likelihood and its gradient at `xv`.
                let eval = |xv: &[f64]| -> (f64, Vec<f64>) {
                    let mut grad = vec![0.0; dim];
                    let mut nll = 0.0;
                    let (icpt, home) = (xv[2 * n_t], xv[2 * n_t + 1]);
                    for (o, &w) in observations.iter().zip(&weights) {
                        let (hi, ai) = (index[&o.home], index[&o.away]);
                        let lambda = (icpt + xv[hi] - xv[n_t + ai] + home).exp();
                        let mu = (icpt + xv[ai] - xv[n_t + hi]).exp();
                        nll -= w
                            * (o.target_home() * lambda.max(1e-12).ln() - lambda
                                + o.target_away() * mu.max(1e-12).ln()
                                - mu);
                        let res_h = o.target_home() - lambda;
                        let res_a = o.target_away() - mu;
                        // Gradient of the negative log-likelihood (so the LL score is negated).
                        grad[hi] -= w * res_h;
                        grad[ai] -= w * res_a;
                        grad[n_t + ai] += w * res_h;
                        grad[n_t + hi] += w * res_a;
                        grad[2 * n_t] -= w * (res_h + res_a);
                        grad[2 * n_t + 1] -= w * res_h;
                    }
                    // Ridge penalty toward the (fixed-this-pass) pooling targets.
                    for i in 0..n_t {
                        let (ea, ed) = (xv[i] - ta[i], xv[n_t + i] - td[i]);
                        nll += 0.5 * ridge * (ea * ea + ed * ed);
                        grad[i] += ridge * ea;
                        grad[n_t + i] += ridge * ed;
                    }
                    (nll, grad)
                };
                x = lbfgs(x, &eval, config.iterations, config.tol);
            }
            if confederations.is_empty() {
                break;
            }
            // Refresh confederation-mean targets from the current solution.
            let attack_now: HashMap<TeamId, f64> =
                teams.iter().enumerate().map(|(i, &t)| (t, x[i])).collect();
            let defense_now: HashMap<TeamId, f64> = teams
                .iter()
                .enumerate()
                .map(|(i, &t)| (t, x[n_t + i]))
                .collect();
            let (cm_a, cm_d) =
                confederation_means(&teams, &attack_now, &defense_now, confederations);
            for (i, &t) in teams.iter().enumerate() {
                if let Some(c) = confederations.get(&t) {
                    ta[i] = CONFEDERATION_POOL * cm_a.get(c).copied().unwrap_or(0.0);
                    td[i] = CONFEDERATION_POOL * cm_d.get(c).copied().unwrap_or(0.0);
                }
            }
        }

        // Unpack and pin the gauge: fold the coefficient means into the intercept (which leaves
        // every prediction unchanged) so mean(attack) = mean(defense) = 0.
        for (i, &t) in teams.iter().enumerate() {
            attack.insert(t, x[i]);
            defense.insert(t, x[n_t + i]);
        }
        intercept = x[2 * n_t];
        home_advantage = x[2 * n_t + 1];
        // Summed over `teams` rather than over the maps' own iteration order. A `HashMap` in Rust
        // gets a fresh hash seed per instance, so `attack.values().sum()` adds the same numbers in a
        // different order on every fit - and floating-point addition is not associative, so the mean
        // differs in its last bits, shifting every fitted coefficient with it. The effect is around
        // 1e-16 per coefficient and harmless numerically, but it makes the fit irreproducible: two
        // fits on identical data disagree. `teams` is a slice in a fixed order, so this is stable.
        let mean_a: f64 = teams.iter().map(|t| attack[t]).sum::<f64>() / n_t as f64;
        let mean_d: f64 = teams.iter().map(|t| defense[t]).sum::<f64>() / n_t as f64;
        for v in attack.values_mut() {
            *v -= mean_a;
        }
        for v in defense.values_mut() {
            *v -= mean_d;
        }
        intercept += mean_a - mean_d;

        // Per-team Fisher information at the fitted optimum: `Σ wᵢ · rateᵢ` over the team's
        // matches (the Poisson information for its attacking log-rate). Feeds the Laplace
        // posterior standard deviation the Monte-Carlo resamples from.
        let mut fisher_info: HashMap<TeamId, f64> = teams.iter().map(|&t| (t, 0.0)).collect();
        for (o, &w) in observations.iter().zip(&weights) {
            let lambda = (intercept + attack[&o.home] - defense[&o.away] + home_advantage).exp();
            let mu = (intercept + attack[&o.away] - defense[&o.home]).exp();
            *fisher_info.get_mut(&o.home).unwrap() += w * lambda;
            *fisher_info.get_mut(&o.away).unwrap() += w * mu;
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

        // Fit the goal-count overdispersion (a finite NB size if the data has fatter tails than
        // Poisson, else 0). Applies to the Independent margins; the Bivariate model has its own
        // dependence mechanism, so leave it Poisson-marginal.
        let dispersion = match config.model {
            ScoreModel::Independent => fit_dispersion(
                observations,
                &weights,
                &attack,
                &defense,
                intercept,
                home_advantage,
            ),
            ScoreModel::Bivariate => 0.0,
        };

        GoalModel {
            intercept,
            home_advantage,
            rho,
            covariance,
            dispersion,
            model: config.model,
            max_goals: config.max_goals,
            attack,
            defense,
            fisher_info,
            ridge_prior: config.ridge,
            xi: config.xi,
            confederation: confederations.clone(),
        }
    }

    /// The fitted strength level of each confederation: the mean overall rating (`attack +
    /// defense`, i.e. scores more *and* concedes less) across its members. A regularized, shared
    /// quantity thanks to the hierarchical pooling, so it is far better behaved than averaging
    /// freely-floating per-team ratings - the antidote to the World Cup's cross-confederation
    /// calibration blind spot. Empty unless the model was fit with
    /// [`fit_with_confederations`](Self::fit_with_confederations).
    pub fn confederation_levels(&self) -> HashMap<Confederation, f64> {
        let mut acc: HashMap<Confederation, (f64, usize)> = HashMap::new();
        for (&team, &conf) in &self.confederation {
            let e = acc.entry(conf).or_insert((0.0, 0));
            e.0 += self.attack_of(team) + self.defense_of(team);
            e.1 += 1;
        }
        acc.into_iter()
            .filter(|(_, (_, n))| *n > 0)
            .map(|(c, (s, n))| (c, s / n as f64))
            .collect()
    }

    /// Log-space standard deviation of a team's strength: the **Laplace (Fisher-information)
    /// posterior** standard deviation at the fitted optimum. Treating the ridge penalty as a
    /// Gaussian prior, the posterior precision is `prior + Fisher information`, so the SD is
    /// `1 / sqrt(ridge + Σ wᵢ·rateᵢ)` - it shrinks as a team accumulates (recent) data and is
    /// wide for a thinly-observed team. The Monte-Carlo resamples each team's strength from this
    /// once per iteration, so champion odds carry parameter uncertainty rather than treating a
    /// point estimate as certain. Returns 0 for the default (data-free) model.
    pub fn strength_uncertainty(&self, team: TeamId) -> f64 {
        const CAP: f64 = 0.6; // a thinly-observed team stays bounded
        let info = self.fisher_info.get(&team).copied().unwrap_or(0.0);
        if info <= 0.0 {
            return 0.0;
        }
        (1.0 / (self.ridge_prior + info).sqrt()).min(CAP)
    }

    /// The fitted bivariate-Poisson covariance term (`0` for the independent model).
    pub fn covariance(&self) -> f64 {
        self.covariance
    }

    /// The fitted negative-binomial dispersion (the NB "size"/`r`); `0` means Poisson margins.
    /// Smaller positive values mean more overdispersion (fatter scoreline tails).
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    pub fn home_advantage(&self) -> f64 {
        self.home_advantage
    }

    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Draw the **posterior distribution of a matchup's win/draw/win probabilities** via
    /// Hamiltonian Monte Carlo, returning one [`Probabilities`] per HMC sample. Where
    /// [`outcome_probabilities`](Self::outcome_probabilities) gives a single point forecast, this
    /// reflects the model's *uncertainty about its own forecast* (a credible interval), sampling
    /// the full posterior over the attack/defense/intercept/home parameters rather than the
    /// Gaussian Laplace approximation. The dependence parameter (`ρ`/`λ3`) and dispersion are held
    /// at their fitted values. `observations` must be the data the model was fit on; `neutral`
    /// drops the home edge. Returns empty for a data-free model or an unknown team.
    pub fn posterior_outcome_samples(
        &self,
        observations: &[Observation],
        confederations: &HashMap<TeamId, Confederation>,
        home: TeamId,
        away: TeamId,
        neutral: bool,
        hmc: crate::hmc::HmcConfig,
    ) -> Vec<Probabilities> {
        let mut teams: Vec<TeamId> = self.attack.keys().copied().collect();
        teams.sort_by_key(|t| t.0);
        let index: HashMap<TeamId, usize> =
            teams.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        let (Some(&hx), Some(&ax)) = (index.get(&home), index.get(&away)) else {
            return Vec::new();
        };
        let n_t = teams.len();
        let dim = 2 * n_t + 2;

        // Same time-decay weighting the fit used.
        let weights: Vec<f64> = observations
            .iter()
            .map(|o| (-self.xi * o.age_days).exp())
            .collect();

        // Fixed (empirical-Bayes) pooling targets from the MAP confederation means.
        let (mut ta, mut td) = (vec![0.0; n_t], vec![0.0; n_t]);
        if !confederations.is_empty() {
            let (cm_a, cm_d) =
                confederation_means(&teams, &self.attack, &self.defense, confederations);
            for (i, &t) in teams.iter().enumerate() {
                if let Some(c) = confederations.get(&t) {
                    ta[i] = CONFEDERATION_POOL * cm_a.get(c).copied().unwrap_or(0.0);
                    td[i] = CONFEDERATION_POOL * cm_d.get(c).copied().unwrap_or(0.0);
                }
            }
        }
        let ridge = self.ridge_prior;

        // Initialize the chain at the MAP (the fitted coefficients).
        let mut init = vec![0.0; dim];
        for (i, &t) in teams.iter().enumerate() {
            init[i] = self.attack_of(t);
            init[n_t + i] = self.defense_of(t);
        }
        init[2 * n_t] = self.intercept;
        init[2 * n_t + 1] = self.home_advantage;

        // Log-posterior and its gradient: the negative of the penalized NLL the fit minimizes.
        let target = |x: &[f64]| -> (f64, Vec<f64>) {
            let mut grad = vec![0.0; dim];
            let mut nll = 0.0;
            let (icpt, hadv) = (x[2 * n_t], x[2 * n_t + 1]);
            for (o, &w) in observations.iter().zip(&weights) {
                let (Some(&hi), Some(&ai)) = (index.get(&o.home), index.get(&o.away)) else {
                    continue;
                };
                let lambda = (icpt + x[hi] - x[n_t + ai] + hadv).exp();
                let mu = (icpt + x[ai] - x[n_t + hi]).exp();
                nll -= w
                    * (o.target_home() * lambda.max(1e-12).ln() - lambda
                        + o.target_away() * mu.max(1e-12).ln()
                        - mu);
                let (res_h, res_a) = (o.target_home() - lambda, o.target_away() - mu);
                grad[hi] -= w * res_h;
                grad[ai] -= w * res_a;
                grad[n_t + ai] += w * res_h;
                grad[n_t + hi] += w * res_a;
                grad[2 * n_t] -= w * (res_h + res_a);
                grad[2 * n_t + 1] -= w * res_h;
            }
            for i in 0..n_t {
                let (ea, ed) = (x[i] - ta[i], x[n_t + i] - td[i]);
                nll += 0.5 * ridge * (ea * ea + ed * ed);
                grad[i] += ridge * ea;
                grad[n_t + i] += ridge * ed;
            }
            (-nll, grad.iter().map(|g| -g).collect())
        };

        // Diagonal preconditioner = Laplace posterior variances (1 / precision) at the MAP.
        let mut precision = vec![0.0; dim];
        for i in 0..n_t {
            precision[i] = ridge;
            precision[n_t + i] = ridge;
        }
        for (o, &w) in observations.iter().zip(&weights) {
            let (Some(&hi), Some(&ai)) = (index.get(&o.home), index.get(&o.away)) else {
                continue;
            };
            let lambda = (self.intercept + self.attack_of(o.home) - self.defense_of(o.away)
                + self.home_advantage)
                .exp();
            let mu = (self.intercept + self.attack_of(o.away) - self.defense_of(o.home)).exp();
            precision[hi] += w * lambda;
            precision[ai] += w * mu;
            precision[n_t + ai] += w * lambda;
            precision[n_t + hi] += w * mu;
            precision[2 * n_t] += w * (lambda + mu);
            precision[2 * n_t + 1] += w * lambda;
        }
        let inv_mass: Vec<f64> = precision.iter().map(|p| 1.0 / p.max(1e-9)).collect();

        let res = crate::hmc::sample(init, &inv_mass, hmc, target);
        res.samples
            .iter()
            .map(|x| {
                let hadv = if neutral { 0.0 } else { x[2 * n_t + 1] };
                let lambda = (x[2 * n_t] + x[hx] - x[n_t + ax] + hadv).exp();
                let mu = (x[2 * n_t] + x[ax] - x[n_t + hx]).exp();
                self.grid_from(lambda, mu).outcome_probabilities()
            })
            .collect()
    }

    fn attack_of(&self, t: TeamId) -> f64 {
        self.attack.get(&t).copied().unwrap_or(0.0)
    }

    fn defense_of(&self, t: TeamId) -> f64 {
        self.defense.get(&t).copied().unwrap_or(0.0)
    }

    /// Whether the fit produced coefficients for `team`. A team the fit never saw is treated as
    /// league-average by [`Self::expected_goals`] (both coefficients default to 0), which
    /// over-rates a genuine minnow - callers can detect that here and seed a rating instead.
    pub fn contains_team(&self, team: TeamId) -> bool {
        self.attack.contains_key(&team)
    }

    /// The `(attack, defense)` coefficients of the weakest fitted team: the lowest attack and the
    /// lowest defense across the table (both reduce a side's edge). Used to give a plausible floor
    /// rating to a real team the offline fit never saw, rather than the misleading average default.
    pub fn weakest_coefficients(&self) -> (f64, f64) {
        let min_atk = self.attack.values().copied().fold(f64::INFINITY, f64::min);
        let min_def = self.defense.values().copied().fold(f64::INFINITY, f64::min);
        (
            if min_atk.is_finite() { min_atk } else { 0.0 },
            if min_def.is_finite() { min_def } else { 0.0 },
        )
    }

    /// Seed a team's attack/defense coefficients directly, overwriting any existing values. Used to
    /// give a rating to a real team absent from the offline fit before a stage-conditioned forecast.
    pub fn set_team_coefficients(&mut self, team: TeamId, attack: f64, defense: f64) {
        self.attack.insert(team, attack);
        self.defense.insert(team, defense);
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
                // Negative-binomial margins (Poisson when dispersion is 0) with the Dixon-Coles
                // low-score correction; the grid is renormalized by `from_fn`.
                neg_binomial_pmf(h as u32, lambda, self.dispersion)
                    * neg_binomial_pmf(a as u32, mu, self.dispersion)
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

    /// A named additive breakdown of a matchup's log expected-goal **edge** (`ln λ_home - ln μ_away`),
    /// exactly as [`expected_goals`](Self::expected_goals) computes it (the shared intercept
    /// cancels): the attack edge, the defense edge, and the home advantage. Used to explain *why*
    /// the goal model favours one side.
    pub fn rate_breakdown(&self, home: TeamId, away: TeamId, neutral: bool) -> RateBreakdown {
        let (expected_home, expected_away) = self.expected_goals(home, away, neutral);
        RateBreakdown {
            expected_home,
            expected_away,
            attack_edge: self.attack_of(home) - self.attack_of(away),
            defense_edge: self.defense_of(home) - self.defense_of(away),
            home_advantage: if neutral { 0.0 } else { self.home_advantage },
        }
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimize a smooth function by **limited-memory BFGS**. `eval(x) -> (value, gradient)` returns
/// the objective and its gradient; the last `M` curvature pairs approximate the inverse Hessian
/// via the two-loop recursion, and a backtracking Armijo line search guarantees descent. Curvature
/// pairs with non-positive `sᵀy` are skipped (the standard safeguard). Converges in tens of
/// iterations on the goal-model likelihood where plain gradient ascent needed hundreds.
fn lbfgs<F: Fn(&[f64]) -> (f64, Vec<f64>)>(
    mut x: Vec<f64>,
    eval: &F,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    const M: usize = 8;
    let n = x.len();
    let (mut s_hist, mut y_hist, mut rho_hist): (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut f, mut g) = eval(&x);

    for _ in 0..max_iter {
        let gnorm = dot(&g, &g).sqrt();
        if gnorm < tol {
            break;
        }
        // Two-loop recursion: q := H·g, the (approximate) Newton direction's negative.
        let mut q = g.clone();
        let k = s_hist.len();
        let mut alpha = vec![0.0; k];
        for i in (0..k).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            for j in 0..n {
                q[j] -= a * y_hist[i][j];
            }
        }
        let gamma = if k > 0 {
            (dot(&s_hist[k - 1], &y_hist[k - 1]) / dot(&y_hist[k - 1], &y_hist[k - 1]).max(1e-12))
                .max(1e-8)
        } else {
            1.0
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..k {
            let b = rho_hist[i] * dot(&y_hist[i], &q);
            for j in 0..n {
                q[j] += (alpha[i] - b) * s_hist[i][j];
            }
        }
        let mut dir: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut dg = dot(&dir, &g);
        if !dg.is_finite() || dg > 0.0 {
            // Not a descent direction: fall back to steepest descent.
            dir = g.iter().map(|v| -v).collect();
            dg = dot(&dir, &g);
        }

        // Backtracking Armijo line search.
        let mut step = if k == 0 { (1.0 / gnorm).min(1.0) } else { 1.0 };
        let mut x_new = x.clone();
        let mut accepted: Option<(f64, Vec<f64>)> = None;
        for _ in 0..40 {
            for j in 0..n {
                x_new[j] = x[j] + step * dir[j];
            }
            let (fn_, gn_) = eval(&x_new);
            if fn_.is_finite() && fn_ <= f + 1e-4 * step * dg {
                accepted = Some((fn_, gn_));
                break;
            }
            step *= 0.5;
        }
        let Some((f_new, g_new)) = accepted else {
            break; // line search stalled: treat as converged
        };

        let s: Vec<f64> = (0..n).map(|j| step * dir[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            if s_hist.len() == M {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

        let improved = (f - f_new).abs();
        x = x_new.clone();
        let prev_f = f;
        f = f_new;
        g = g_new;
        if improved <= tol * prev_f.abs().max(1.0) {
            break; // converged
        }
    }
    x
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

/// One-dimensional search for the negative-binomial dispersion (the "size"/`r`) that maximizes
/// the time-weighted log-likelihood of the goal counts, given the fitted means. Returns `0`
/// (Poisson) when the data shows no overdispersion, else the finite `r` that best fits the
/// goal-count spread (smaller = more overdispersed). Both teams' goals contribute.
fn fit_dispersion(
    obs: &[Observation],
    weights: &[f64],
    attack: &HashMap<TeamId, f64>,
    defense: &HashMap<TeamId, f64>,
    intercept: f64,
    home_advantage: f64,
) -> f64 {
    let ll = |size: f64| -> f64 {
        obs.iter()
            .zip(weights)
            .map(|(o, &w)| {
                let lambda =
                    (intercept + attack[&o.home] - defense[&o.away] + home_advantage).exp();
                let mu = (intercept + attack[&o.away] - defense[&o.home]).exp();
                let ph = neg_binomial_pmf(o.score.home as u32, lambda, size).max(1e-12);
                let pa = neg_binomial_pmf(o.score.away as u32, mu, size).max(1e-12);
                w * (ph.ln() + pa.ln())
            })
            .sum()
    };
    // Candidate sizes: Poisson (0) plus finite sizes from heavy to light overdispersion. A large
    // size is already indistinguishable from Poisson, so the grid need not extend further.
    const SIZES: &[f64] = &[0.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0];
    let mut best = 0.0;
    let mut best_ll = ll(0.0);
    for &s in &SIZES[1..] {
        let cur = ll(s);
        if cur > best_ll {
            best_ll = cur;
            best = s;
        }
    }
    best
}

/// Mean attack and mean defense coefficient within each confederation, for hierarchical pooling.
fn confederation_means(
    teams: &[TeamId],
    attack: &HashMap<TeamId, f64>,
    defense: &HashMap<TeamId, f64>,
    confederations: &HashMap<TeamId, Confederation>,
) -> (HashMap<Confederation, f64>, HashMap<Confederation, f64>) {
    let mut acc: HashMap<Confederation, (f64, f64, usize)> = HashMap::new();
    for &t in teams {
        if let Some(&c) = confederations.get(&t) {
            let e = acc.entry(c).or_insert((0.0, 0.0, 0));
            e.0 += attack.get(&t).copied().unwrap_or(0.0);
            e.1 += defense.get(&t).copied().unwrap_or(0.0);
            e.2 += 1;
        }
    }
    let mut mean_attack = HashMap::new();
    let mut mean_defense = HashMap::new();
    for (c, (sa, sd, n)) in acc {
        if n > 0 {
            mean_attack.insert(c, sa / n as f64);
            mean_defense.insert(c, sd / n as f64);
        }
    }
    (mean_attack, mean_defense)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: u32) -> TeamId {
        TeamId(n)
    }

    #[test]
    fn fit_learns_overdispersion_from_fat_tailed_scores() {
        // The same matchup repeated with a high-variance goal pattern (blowouts and blanks),
        // which a fixed-rate Poisson cannot fit: the fit should pick a finite NB dispersion.
        let scores = [
            (5, 0),
            (0, 0),
            (4, 1),
            (0, 2),
            (6, 0),
            (0, 0),
            (3, 0),
            (1, 4),
        ];
        let obs: Vec<Observation> = (0..40)
            .map(|i| {
                let (h, a) = scores[i % scores.len()];
                Observation::new(t(1), t(2), Scoreline::new(h, a), i as f64)
            })
            .collect();
        let model = GoalModel::fit(&obs, DixonColesConfig::default());
        assert!(
            model.dispersion() > 0.0,
            "fat-tailed scores should yield a finite NB dispersion, got {}",
            model.dispersion()
        );
    }

    #[test]
    fn hierarchical_pooling_lifts_a_data_poor_team_toward_its_confederation() {
        use oracle_domain::Confederation::{Afc, Uefa};
        let mut confs = HashMap::new();
        for id in [1, 2, 3, 7] {
            confs.insert(t(id), Uefa); // strong confederation
        }
        for id in [4, 5, 6] {
            confs.insert(t(id), Afc); // weak confederation
        }
        let mut obs = Vec::new();
        // Strong UEFA teams 1-3 thrash weak AFC teams 4-6 repeatedly, alternating venue so the
        // gap lands in attack/defense rather than being absorbed by home advantage.
        for i in 0..30u32 {
            let (uefa, afc) = (t(1 + i % 3), t(4 + i % 3));
            if i % 2 == 0 {
                obs.push(Observation::new(uefa, afc, Scoreline::new(3, 0), i as f64));
            } else {
                obs.push(Observation::new(afc, uefa, Scoreline::new(0, 3), i as f64));
            }
        }
        // Team 7 (UEFA) is data-poor: two modest draws, so its own record looks average.
        obs.push(Observation::new(t(7), t(4), Scoreline::new(1, 1), 1.0));
        obs.push(Observation::new(t(7), t(5), Scoreline::new(1, 1), 0.0));

        let cfg = DixonColesConfig {
            ridge: 0.1,
            ..Default::default()
        };
        let flat = GoalModel::fit(&obs, cfg);
        let hier = GoalModel::fit_with_confederations(&obs, cfg, &confs);
        assert!(
            hier.attack_of(t(7)) > flat.attack_of(t(7)),
            "pooling should pull the sparse team toward its strong confederation: {} vs {}",
            hier.attack_of(t(7)),
            flat.attack_of(t(7))
        );
        // The fitted confederation levels rank the strong confederation above the weak one.
        let levels = hier.confederation_levels();
        assert!(levels[&Uefa] > levels[&Afc], "UEFA level should exceed AFC");
    }

    #[test]
    fn confederation_agnostic_fit_has_no_confederation_levels() {
        let model = GoalModel::fit(&synthetic_history(), DixonColesConfig::default());
        assert!(
            model.confederation_levels().is_empty(),
            "plain fit carries no confederation structure"
        );
    }

    #[test]
    fn overdispersion_fattens_both_grid_tails() {
        // Same means (the data-free default), Poisson vs negative-binomial margins.
        let poisson = GoalModel::default();
        let nb = GoalModel {
            dispersion: 5.0,
            ..GoalModel::default()
        };
        let gp = poisson.score_grid(t(1), t(2), false);
        let gnb = nb.score_grid(t(1), t(2), false);
        assert!(gnb.grid[0][0] > gp.grid[0][0], "NB lifts the 0-0 mass");
        // Use a line well into the tail (mean total is ~3): overdispersion adds mass to the
        // extremes, so the high-scoring tail is clearly fatter.
        assert!(
            gnb.prob_over(5.5) > gp.prob_over(5.5),
            "NB lifts the blowout tail"
        );
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
    fn posterior_samples_center_on_the_point_estimate_with_spread() {
        let obs = synthetic_history();
        let model = GoalModel::fit(&obs, DixonColesConfig::default());
        let point = model.outcome_probabilities(t(1), t(2), true);
        let samples = model.posterior_outcome_samples(
            &obs,
            &HashMap::new(),
            t(1),
            t(2),
            true,
            crate::hmc::HmcConfig {
                n_samples: 500,
                n_warmup: 150,
                step_size: 0.25,
                n_leapfrog: 12,
                seed: 3,
            },
        );
        assert_eq!(samples.len(), 500);
        for p in &samples {
            assert!((p.sum() - 1.0).abs() < 1e-9);
        }
        let mean_home = samples.iter().map(|p| p.home_win).sum::<f64>() / samples.len() as f64;
        // The posterior mean of the win probability sits near the MAP point estimate ...
        assert!(
            (mean_home - point.home_win).abs() < 0.06,
            "posterior mean {mean_home} vs point {}",
            point.home_win
        );
        // ... and there is genuine spread: the model is uncertain about its own forecast.
        let var = samples
            .iter()
            .map(|p| (p.home_win - mean_home).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        assert!(var > 1e-5, "posterior should have real spread, var = {var}");
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
