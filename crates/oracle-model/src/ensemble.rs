//! Ensemble blending of independent predictors - with **learned** weights.
//!
//! The engine produces two pre-match signals for every match - the Dixon-Coles goal
//! model and the Elo rating model. We combine them with a **logarithmic opinion
//! pool** (a temperature-scaled weighted geometric mean):
//!
//! ```text
//! q(o) ∝ exp( τ · Σ_k a_k · ln p_k(o) ),    a_k ≥ 0, Σ a_k = 1, τ > 0
//! ```
//!
//! - the **mixture weights** `a_k` decide which member to trust, and
//! - the **temperature** `τ` fixes systematic under/over-confidence (`τ > 1` sharpens).
//!
//! Both are fit by **stacking** - [`Ensemble::fit`] minimizes out-of-sample log-loss
//! on a held-out validation set. Because `a = [1, 0], τ = 1` recovers a single member
//! exactly, a fitted ensemble can never be worse than its best member on the fit set:
//! this is the guard that stops the ensemble from degrading the Dixon-Coles forecast
//! (which is exactly what hardcoded weights were doing before).

use oracle_domain::{Outcome, Probabilities};
use serde::{Deserialize, Serialize};

const OUTCOMES: [Outcome; 3] = [Outcome::HomeWin, Outcome::Draw, Outcome::AwayWin];

fn one() -> f64 {
    1.0
}

/// A temperature-scaled, weighted log-opinion-pool ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ensemble {
    /// Weight per member, in the order members are supplied to [`Ensemble::blend`].
    /// Internally normalized to a simplex, so only the ratios matter.
    pub weights: Vec<f64>,
    /// Global sharpness. 1.0 = plain geometric pool; >1 sharpens, <1 flattens.
    #[serde(default = "one")]
    pub temperature: f64,
}

/// Hyper-parameters for [`Ensemble::fit`].
#[derive(Debug, Clone, Copy)]
pub struct EnsembleFitConfig {
    pub iterations: usize,
    pub learning_rate: f64,
}

impl Default for EnsembleFitConfig {
    fn default() -> Self {
        Self {
            iterations: 300,
            learning_rate: 0.3,
        }
    }
}

impl Ensemble {
    pub fn new(weights: Vec<f64>) -> Self {
        Self {
            weights,
            temperature: 1.0,
        }
    }

    /// The default two-model prior: Dixon-Coles weighted a little above Elo. Used until
    /// [`fit`](Ensemble::fit) replaces it with learned values.
    pub fn dixon_coles_and_elo() -> Self {
        Self {
            weights: vec![0.65, 0.35],
            temperature: 1.0,
        }
    }

    /// Blend member forecasts via the temperature-scaled log-opinion pool. An empty
    /// input returns a uniform distribution; missing weights default to 1.0.
    pub fn blend(&self, members: &[Probabilities]) -> Probabilities {
        if members.is_empty() {
            return Probabilities::uniform();
        }
        let mut acc = [0.0f64; 3];
        let mut total_w = 0.0;
        for (idx, m) in members.iter().enumerate() {
            let w = self.weights.get(idx).copied().unwrap_or(1.0);
            total_w += w;
            for (slot, outcome) in OUTCOMES.into_iter().enumerate() {
                // ln of each member's probability, floored to avoid -inf on p = 0.
                acc[slot] += w * m.of(outcome).max(1e-12).ln();
            }
        }
        // Normalize by total weight (→ weighted geometric mean) then apply temperature.
        let scale = if total_w > f64::EPSILON {
            self.temperature / total_w
        } else {
            self.temperature
        };
        for a in &mut acc {
            *a *= scale;
        }
        // Softmax with the usual max-subtraction for numerical stability.
        let max = acc[0].max(acc[1]).max(acc[2]);
        Probabilities::new(
            (acc[0] - max).exp(),
            (acc[1] - max).exp(),
            (acc[2] - max).exp(),
        )
    }

    /// Fit weights + temperature by minimizing mean log-loss on validation data, using
    /// the default [`EnsembleFitConfig`].
    ///
    /// `member_preds[i]` holds each member's forecast for match `i` (same order every
    /// time); `actuals[i]` is what happened. `n_members` is the number of members.
    pub fn fit(member_preds: &[Vec<Probabilities>], actuals: &[Outcome], n_members: usize) -> Self {
        Self::fit_with_config(
            member_preds,
            actuals,
            n_members,
            EnsembleFitConfig::default(),
        )
    }

    /// As [`fit`](Ensemble::fit), with explicit hyper-parameters.
    pub fn fit_with_config(
        member_preds: &[Vec<Probabilities>],
        actuals: &[Outcome],
        n_members: usize,
        config: EnsembleFitConfig,
    ) -> Self {
        if member_preds.is_empty() || n_members == 0 {
            return Ensemble::dixon_coles_and_elo();
        }

        // Parameters: unconstrained logits `r` (softmax → simplex weights) and
        // `b = ln τ`. Start from uniform weights and τ = 1.
        let mut r = vec![0.0f64; n_members];
        let mut b = 0.0f64;
        const EPS: f64 = 1e-4;
        let (lo_b, hi_b) = (0.1f64.ln(), 5.0f64.ln());

        let loss = |r: &[f64], b: f64| -> f64 {
            mean_log_loss(member_preds, actuals, &softmax(r), b.exp())
        };

        for _ in 0..config.iterations {
            // Central-difference numerical gradient over the n_members + 1 parameters.
            let mut grad_r = vec![0.0f64; n_members];
            for k in 0..n_members {
                let mut up = r.clone();
                let mut dn = r.clone();
                up[k] += EPS;
                dn[k] -= EPS;
                grad_r[k] = (loss(&up, b) - loss(&dn, b)) / (2.0 * EPS);
            }
            let grad_b = (loss(&r, b + EPS) - loss(&r, b - EPS)) / (2.0 * EPS);

            for k in 0..n_members {
                r[k] -= config.learning_rate * grad_r[k];
            }
            b = (b - config.learning_rate * grad_b).clamp(lo_b, hi_b);
        }

        Ensemble {
            weights: softmax(&r),
            temperature: b.exp(),
        }
    }
}

impl Default for Ensemble {
    fn default() -> Self {
        Self::dixon_coles_and_elo()
    }
}

fn softmax(xs: &[f64]) -> Vec<f64> {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / xs.len() as f64; xs.len()]
    }
}

fn mean_log_loss(
    member_preds: &[Vec<Probabilities>],
    actuals: &[Outcome],
    weights: &[f64],
    temperature: f64,
) -> f64 {
    let ensemble = Ensemble {
        weights: weights.to_vec(),
        temperature,
    };
    let n = member_preds.len().max(1) as f64;
    member_preds
        .iter()
        .zip(actuals)
        .map(|(preds, &actual)| -ensemble.blend(preds).of(actual).max(1e-12).ln())
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_of_identical_members_is_unchanged() {
        let p = Probabilities::new(0.5, 0.3, 0.2);
        let e = Ensemble::new(vec![1.0, 1.0]);
        let blended = e.blend(&[p, p]);
        assert!((blended.home_win - 0.5).abs() < 1e-9);
        assert!((blended.draw - 0.3).abs() < 1e-9);
        assert!((blended.away_win - 0.2).abs() < 1e-9);
    }

    #[test]
    fn blend_is_normalized() {
        let a = Probabilities::new(0.7, 0.2, 0.1);
        let b = Probabilities::new(0.3, 0.3, 0.4);
        let blended = Ensemble::default().blend(&[a, b]);
        assert!((blended.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn blend_lands_between_members() {
        let a = Probabilities::new(0.8, 0.1, 0.1);
        let b = Probabilities::new(0.2, 0.1, 0.7);
        let blended = Ensemble::new(vec![1.0, 1.0]).blend(&[a, b]);
        assert!(blended.home_win < 0.8 && blended.home_win > 0.2);
    }

    #[test]
    fn empty_blend_is_uniform() {
        let blended = Ensemble::default().blend(&[]);
        assert!((blended.home_win - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn higher_temperature_sharpens() {
        let members = [Probabilities::new(0.5, 0.3, 0.2)];
        let cool = Ensemble {
            weights: vec![1.0],
            temperature: 1.0,
        }
        .blend(&members);
        let hot = Ensemble {
            weights: vec![1.0],
            temperature: 3.0,
        }
        .blend(&members);
        assert!(
            hot.home_win > cool.home_win,
            "temperature should sharpen the mode"
        );
    }

    /// Fitting must down-weight a useless member and never do worse than the good one.
    #[test]
    fn fit_downweights_a_useless_member_and_beats_it() {
        // Member 0 is informative (leans toward the truth); member 1 is pure noise.
        let mut member_preds = Vec::new();
        let mut actuals = Vec::new();
        for i in 0..300u32 {
            let actual = match i % 3 {
                0 => Outcome::HomeWin,
                1 => Outcome::Draw,
                _ => Outcome::AwayWin,
            };
            let good = match actual {
                Outcome::HomeWin => Probabilities::new(0.6, 0.25, 0.15),
                Outcome::Draw => Probabilities::new(0.25, 0.5, 0.25),
                Outcome::AwayWin => Probabilities::new(0.15, 0.25, 0.6),
            };
            let noise = Probabilities::uniform();
            member_preds.push(vec![good, noise]);
            actuals.push(actual);
        }

        let fitted = Ensemble::fit(&member_preds, &actuals, 2);
        let total: f64 = fitted.weights.iter().sum();
        let (w_good, w_noise) = (fitted.weights[0] / total, fitted.weights[1] / total);
        assert!(
            w_good > w_noise,
            "informative member should outweigh noise ({w_good:.3} vs {w_noise:.3})"
        );

        // The fitted ensemble must beat the good member used alone.
        let ll_fit = mean_log_loss(&member_preds, &actuals, &fitted.weights, fitted.temperature);
        let ll_good = mean_log_loss(&member_preds, &actuals, &[1.0, 0.0], 1.0);
        assert!(
            ll_fit <= ll_good + 1e-6,
            "fitted log-loss {ll_fit:.4} should be ≤ good-member-only {ll_good:.4}"
        );
    }
}
