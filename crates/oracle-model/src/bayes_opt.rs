//! Bayesian optimization of an expensive black-box objective over a box.
//!
//! Where a grid search blindly evaluates a fixed lattice, Bayesian optimization fits a cheap
//! probabilistic **surrogate** (a Gaussian process) to the points seen so far and uses it to
//! decide where to look next: the **Expected-Improvement** acquisition balances exploiting
//! regions that scored well against exploring regions the surrogate is unsure about. It finds a
//! better optimum in far fewer (expensive) evaluations, and over a *continuous* space the grid
//! cannot reach.
//!
//! This is a compact, dependency-free implementation: an RBF-kernel GP solved by Cholesky, EI
//! over randomly sampled candidates, and `oracle-numeric`'s seeded generator so a run is fully
//! reproducible. It powers `wc-oracle tune`, replacing the hand-specified grid.
// Explicit index loops read more naturally than iterators for the small matrix routines here.
#![allow(clippy::needless_range_loop)]

use oracle_numeric::normal::{normal_cdf, normal_pdf};
use oracle_numeric::Rng;

/// Tuning for a Bayesian-optimization run.
#[derive(Debug, Clone, Copy)]
pub struct BoConfig {
    /// Random initial design points evaluated before the surrogate is used.
    pub n_init: usize,
    /// Acquisition-guided evaluations after the initial design.
    pub n_iter: usize,
    /// Seed for the (reproducible) candidate sampling.
    pub seed: u64,
}

/// The outcome of a run: the best point found, its objective value, and how many evaluations it took.
#[derive(Debug, Clone)]
pub struct BoResult {
    pub best_x: Vec<f64>,
    pub best_value: f64,
    pub evaluations: usize,
}

/// Minimize `objective` over the box `bounds` (`[(lo, hi); d]`) by Bayesian optimization with a
/// Gaussian-process surrogate and Expected-Improvement acquisition. The objective is treated as a
/// black box (only its values are used), so it can be a noisy, expensive model-fit-and-score.
pub fn minimize<F: FnMut(&[f64]) -> f64>(
    bounds: &[(f64, f64)],
    cfg: BoConfig,
    mut objective: F,
) -> BoResult {
    let d = bounds.len();
    let mut rng = Rng::new(cfg.seed);
    // The GP works in the unit cube; map a normalized point back to the real box for evaluation.
    let to_box = |u: &[f64]| -> Vec<f64> {
        (0..d)
            .map(|i| bounds[i].0 + u[i] * (bounds[i].1 - bounds[i].0))
            .collect()
    };

    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut evals = 0usize;

    // Initial design: the box centre (a stable anchor) plus random points.
    let n_init = cfg.n_init.max(2);
    for k in 0..n_init {
        let u: Vec<f64> = if k == 0 {
            vec![0.5; d]
        } else {
            (0..d).map(|_| rng.unit()).collect()
        };
        let y = objective(&to_box(&u));
        ys.push(y);
        xs.push(u);
        evals += 1;
    }

    for _ in 0..cfg.n_iter {
        let gp = Gp::fit(&xs, &ys);
        let y_best = ys.iter().copied().fold(f64::INFINITY, f64::min);
        // Maximize Expected Improvement over a cloud of random candidates.
        let mut best_u = vec![0.5; d];
        let mut best_ei = -1.0;
        for _ in 0..512 {
            let u: Vec<f64> = (0..d).map(|_| rng.unit()).collect();
            let (mean, var) = gp.predict(&u);
            let ei = expected_improvement(y_best, mean, var.max(0.0).sqrt());
            if ei > best_ei {
                best_ei = ei;
                best_u = u;
            }
        }
        let y = objective(&to_box(&best_u));
        ys.push(y);
        xs.push(best_u);
        evals += 1;
    }

    let (bi, &by) = ys
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    BoResult {
        best_x: to_box(&xs[bi]),
        best_value: by,
        evaluations: evals,
    }
}

/// A Gaussian-process regressor with an RBF kernel and fixed hyperparameters, fit by Cholesky.
struct Gp {
    xs: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    chol: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
}

const GP_LENGTH: f64 = 0.2; // RBF length scale in the unit cube
const GP_NOISE: f64 = 1e-4; // observation noise / jitter for numerical stability

impl Gp {
    fn fit(xs: &[Vec<f64>], ys: &[f64]) -> Self {
        let n = xs.len();
        let y_mean = ys.iter().sum::<f64>() / n as f64;
        let var = ys.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n as f64;
        let y_std = var.sqrt().max(1e-6);
        let yn: Vec<f64> = ys.iter().map(|y| (y - y_mean) / y_std).collect();

        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = rbf(&xs[i], &xs[j]) + if i == j { GP_NOISE } else { 0.0 };
            }
        }
        let chol = cholesky(&k);
        let alpha = chol_solve(&chol, &yn);
        Gp {
            xs: xs.to_vec(),
            alpha,
            chol,
            y_mean,
            y_std,
        }
    }

    /// Posterior `(mean, variance)` at a normalized point.
    fn predict(&self, x: &[f64]) -> (f64, f64) {
        let kstar: Vec<f64> = self.xs.iter().map(|xi| rbf(xi, x)).collect();
        let mean_n = dot(&kstar, &self.alpha);
        let v = solve_lower(&self.chol, &kstar);
        let var_n = (1.0 + GP_NOISE - dot(&v, &v)).max(0.0);
        (
            mean_n * self.y_std + self.y_mean,
            var_n * self.y_std * self.y_std,
        )
    }
}

fn rbf(a: &[f64], b: &[f64]) -> f64 {
    let d2: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    (-d2 / (2.0 * GP_LENGTH * GP_LENGTH)).exp()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lower-triangular Cholesky factor `L` of a symmetric positive-definite matrix (`A = L Lᵀ`).
fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = s.max(1e-12).sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    l
}

/// Forward substitution: solve `L y = b` for a lower-triangular `L`.
fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    y
}

/// Solve `A x = b` given `A = L Lᵀ`: forward then back substitution.
fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let z = solve_lower(l, b);
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Expected improvement (for minimization) of a candidate with predictive `mean`/`sd`, relative
/// to the best objective seen so far.
fn expected_improvement(y_best: f64, mean: f64, sd: f64) -> f64 {
    let improvement = y_best - mean;
    if sd < 1e-12 {
        return improvement.max(0.0);
    }
    let z = improvement / sd;
    improvement * normal_cdf(z) + sd * normal_pdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_minimum_of_a_bowl() {
        // f(x, y) = (x - 0.3)^2 + (y - 0.7)^2 on the unit square; optimum at (0.3, 0.7).
        let bounds = [(0.0, 1.0), (0.0, 1.0)];
        let res = minimize(
            &bounds,
            BoConfig {
                n_init: 6,
                n_iter: 30,
                seed: 1,
            },
            |x| (x[0] - 0.3).powi(2) + (x[1] - 0.7).powi(2),
        );
        assert!(
            res.best_value < 0.01,
            "should get close to the optimum, got {} at {:?}",
            res.best_value,
            res.best_x
        );
        assert!((res.best_x[0] - 0.3).abs() < 0.12 && (res.best_x[1] - 0.7).abs() < 0.12);
        assert_eq!(res.evaluations, 36);
    }

    #[test]
    fn beats_a_random_search_of_the_same_budget() {
        // On a slightly trickier objective BO should beat blind random search at equal budget.
        let f = |x: &[f64]| (x[0] - 0.8).powi(2) + 0.5 * (x[1] - 0.2).powi(2) + 0.1 * x[0] * x[1];
        let bounds = [(0.0, 1.0), (0.0, 1.0)];
        let bo = minimize(
            &bounds,
            BoConfig {
                n_init: 5,
                n_iter: 20,
                seed: 7,
            },
            f,
        );
        let mut rng = Rng::new(7);
        let mut rand_best = f64::INFINITY;
        for _ in 0..25 {
            let x = [rng.unit(), rng.unit()];
            rand_best = rand_best.min(f(&x));
        }
        assert!(
            bo.best_value <= rand_best + 1e-9,
            "BO {} should not be worse than random {}",
            bo.best_value,
            rand_best
        );
    }

    #[test]
    fn is_reproducible_for_a_fixed_seed() {
        let f = |x: &[f64]| (x[0] - 0.4).powi(2);
        let bounds = [(0.0, 1.0)];
        let cfg = BoConfig {
            n_init: 4,
            n_iter: 10,
            seed: 42,
        };
        let a = minimize(&bounds, cfg, f);
        let b = minimize(&bounds, cfg, f);
        assert_eq!(a.best_x, b.best_x);
        assert_eq!(a.best_value, b.best_value);
    }
}
