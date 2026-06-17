//! Ensemble blending of independent predictors.
//!
//! The engine produces two pre-match signals for every match — the Dixon-Coles goal
//! model and the Elo rating model — and they disagree in useful, complementary ways.
//! We combine them with a **logarithmic opinion pool**: the blended probability of
//! an outcome is proportional to the *weighted geometric mean* of the members'
//! probabilities,
//!
//! ```text
//! p(o) ∝ Π_k  p_k(o)^{w_k}
//! ```
//!
//! Geometric (log-space) pooling is the natural choice for combining probabilistic
//! forecasts: it is externally Bayesian and stays sharp, where a plain arithmetic
//! average tends to wash out toward the uniform. Weights are tunable and can be
//! calibrated against history (see [`crate::calibration`]).

use oracle_domain::{Outcome, Probabilities};
use serde::{Deserialize, Serialize};

/// A weighted ensemble of probability forecasts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ensemble {
    /// Weight per member, in the same order members are supplied to [`Ensemble::blend`].
    pub weights: Vec<f64>,
}

impl Ensemble {
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }

    /// The default two-model ensemble: Dixon-Coles weighted a little above Elo,
    /// reflecting that the goal model carries more match-specific signal.
    pub fn dixon_coles_and_elo() -> Self {
        Self {
            weights: vec![0.65, 0.35],
        }
    }

    /// Blend member forecasts via the logarithmic opinion pool. Missing weights
    /// default to 1.0; an empty input returns a uniform distribution.
    pub fn blend(&self, members: &[Probabilities]) -> Probabilities {
        if members.is_empty() {
            return Probabilities::uniform();
        }
        let mut acc = [0.0f64; 3];
        let mut total_w = 0.0;
        for (idx, m) in members.iter().enumerate() {
            let w = self.weights.get(idx).copied().unwrap_or(1.0);
            total_w += w;
            for (slot, outcome) in [Outcome::HomeWin, Outcome::Draw, Outcome::AwayWin]
                .into_iter()
                .enumerate()
            {
                // ln of each member's probability, floored to avoid -inf on p = 0.
                acc[slot] += w * m.of(outcome).max(1e-12).ln();
            }
        }
        // Normalize by total weight so the pool is a weighted *geometric mean*: this
        // makes the blend of identical members the identity, regardless of scale.
        if total_w > f64::EPSILON {
            for a in &mut acc {
                *a /= total_w;
            }
        }
        // exp and let `Probabilities::new` normalize. Subtract the max first for
        // numerical stability (softmax trick).
        let max = acc[0].max(acc[1]).max(acc[2]);
        Probabilities::new(
            (acc[0] - max).exp(),
            (acc[1] - max).exp(),
            (acc[2] - max).exp(),
        )
    }
}

impl Default for Ensemble {
    fn default() -> Self {
        Self::dixon_coles_and_elo()
    }
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
}
