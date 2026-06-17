//! Scoring rules for evaluating forecast quality.
//!
//! A prediction engine is only as good as it is *calibrated*. We report the two
//! standard proper scoring rules for probabilistic classification, plus plain
//! accuracy as an intuition check:
//!
//! - **Brier score** — mean squared error between the predicted distribution and the
//!   one-hot actual outcome (lower is better; 0 is perfect, ~0.667 is the worst).
//! - **Log loss** — mean negative log-probability assigned to the actual outcome
//!   (lower is better; punishes confident wrong calls hard).
//!
//! These power the `backtest` CLI command and the integration test that guards
//! against model regressions.

use oracle_domain::{Outcome, Probabilities};
use serde::{Deserialize, Serialize};

/// Aggregate quality of a set of forecasts against realized outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub n: usize,
    pub brier: f64,
    pub log_loss: f64,
    /// Fraction of matches whose most-likely outcome was correct.
    pub accuracy: f64,
}

impl CalibrationReport {
    /// Brier/log-loss of a naive uniform (⅓, ⅓, ⅓) baseline — the bar any real
    /// model must clear. Handy for the backtest report.
    pub fn uniform_baseline(n: usize) -> Self {
        let third: f64 = 1.0 / 3.0;
        Self {
            n,
            // Σ over 3 classes of (1/3)^2 with one (1-1/3)^2 term:
            brier: 2.0 * third * third + (1.0 - third).powi(2),
            log_loss: -third.ln(),
            accuracy: third,
        }
    }
}

/// Score a batch of `(prediction, actual outcome)` pairs.
pub fn score(predictions: &[(Probabilities, Outcome)]) -> CalibrationReport {
    let n = predictions.len();
    if n == 0 {
        return CalibrationReport {
            n: 0,
            brier: 0.0,
            log_loss: 0.0,
            accuracy: 0.0,
        };
    }

    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut correct = 0usize;

    for (p, actual) in predictions {
        for outcome in [Outcome::HomeWin, Outcome::Draw, Outcome::AwayWin] {
            let target = if outcome == *actual { 1.0 } else { 0.0 };
            brier_sum += (p.of(outcome) - target).powi(2);
        }
        log_loss_sum += -p.of(*actual).max(1e-12).ln();
        if p.most_likely() == *actual {
            correct += 1;
        }
    }

    CalibrationReport {
        n,
        brier: brier_sum / n as f64,
        log_loss: log_loss_sum / n as f64,
        accuracy: correct as f64 / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_forecasts_score_zero() {
        let preds = vec![
            (Probabilities::new(1.0, 0.0, 0.0), Outcome::HomeWin),
            (Probabilities::new(0.0, 0.0, 1.0), Outcome::AwayWin),
        ];
        let r = score(&preds);
        assert!(r.brier < 1e-9);
        assert!(r.log_loss < 1e-6);
        assert!((r.accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn confident_model_beats_uniform_baseline() {
        // A decent model that leans correctly should beat (⅓,⅓,⅓).
        let preds = vec![
            (Probabilities::new(0.6, 0.25, 0.15), Outcome::HomeWin),
            (Probabilities::new(0.15, 0.25, 0.6), Outcome::AwayWin),
            (Probabilities::new(0.25, 0.5, 0.25), Outcome::Draw),
        ];
        let model = score(&preds);
        let baseline = CalibrationReport::uniform_baseline(preds.len());
        assert!(model.brier < baseline.brier);
        assert!(model.log_loss < baseline.log_loss);
    }
}
