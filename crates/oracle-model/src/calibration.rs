//! Scoring rules for evaluating forecast quality.
//!
//! A prediction engine is only as good as it is *calibrated*. We report the two
//! standard proper scoring rules for probabilistic classification, plus plain
//! accuracy as an intuition check:
//!
//! - **Brier score** - mean squared error between the predicted distribution and the
//!   one-hot actual outcome (lower is better; 0 is perfect, ~0.667 is the worst).
//! - **Log loss** - mean negative log-probability assigned to the actual outcome
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
    /// Brier/log-loss of a naive uniform (⅓, ⅓, ⅓) baseline - the bar any real
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

/// Convert decimal odds into implied win/draw/win probabilities, normalizing out the
/// bookmaker's overround (the "vig"). Each argument is the decimal payout, e.g. `2.50`.
///
/// The raw inverse odds `1/o` sum to more than 1 because the book bakes in a margin;
/// dividing by their total recovers a proper probability distribution. This is the
/// market baseline the engine is measured against in the backtest, and bookmaker
/// implied probabilities are a notoriously hard bar to beat.
pub fn implied_probabilities(home_odds: f64, draw_odds: f64, away_odds: f64) -> Probabilities {
    let inv = |o: f64| if o > 1.0 { 1.0 / o } else { 0.0 };
    Probabilities::new(inv(home_odds), inv(draw_odds), inv(away_odds))
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

/// One probability bin of a reliability (calibration) curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReliabilityBin {
    pub lo: f64,
    pub hi: f64,
    /// Mean predicted probability of the predictions that fell in this bin.
    pub mean_pred: f64,
    /// Empirical frequency with which those predictions came true.
    pub empirical: f64,
    pub count: usize,
}

/// A reliability curve plus its expected calibration error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityReport {
    pub bins: Vec<ReliabilityBin>,
    /// Expected calibration error: count-weighted mean gap between predicted and empirical.
    pub ece: f64,
}

/// Build a reliability curve over `bins` equal-width probability bins. Every class
/// probability of every prediction contributes a `(predicted, did it happen)` pair, so a
/// well-calibrated model has empirical ≈ predicted in each bin (and a low ECE).
pub fn reliability(predictions: &[(Probabilities, Outcome)], bins: usize) -> ReliabilityReport {
    let bins = bins.max(1);
    let mut sum_pred = vec![0.0f64; bins];
    let mut sum_hit = vec![0.0f64; bins];
    let mut count = vec![0usize; bins];

    for (p, actual) in predictions {
        for outcome in [Outcome::HomeWin, Outcome::Draw, Outcome::AwayWin] {
            let pred = p.of(outcome);
            let hit = if outcome == *actual { 1.0 } else { 0.0 };
            let b = ((pred * bins as f64) as usize).min(bins - 1);
            sum_pred[b] += pred;
            sum_hit[b] += hit;
            count[b] += 1;
        }
    }

    let total: usize = count.iter().sum();
    let mut ece = 0.0;
    let bin_report = (0..bins)
        .map(|b| {
            let c = count[b];
            let (mean_pred, empirical) = if c > 0 {
                (sum_pred[b] / c as f64, sum_hit[b] / c as f64)
            } else {
                (0.0, 0.0)
            };
            if c > 0 && total > 0 {
                ece += (c as f64 / total as f64) * (mean_pred - empirical).abs();
            }
            ReliabilityBin {
                lo: b as f64 / bins as f64,
                hi: (b + 1) as f64 / bins as f64,
                mean_pred,
                empirical,
                count: c,
            }
        })
        .collect();

    ReliabilityReport {
        bins: bin_report,
        ece,
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

    #[test]
    fn implied_probabilities_remove_vig_and_rank_favourite() {
        // Odds with a built-in margin (inverse sum > 1).
        let p = implied_probabilities(2.0, 3.5, 4.0);
        assert!((p.sum() - 1.0).abs() < 1e-9, "overround normalized away");
        assert_eq!(
            p.most_likely(),
            Outcome::HomeWin,
            "shortest odds = favourite"
        );
        assert!(p.home_win > p.draw && p.home_win > p.away_win);
    }

    #[test]
    fn calibrated_set_has_lower_ece_than_overconfident() {
        // Build n predictions all equal to `p`, with `frac_home` home wins and the rest
        // split evenly between draw and away.
        let mk = |n: usize, p: Probabilities, frac_home: f64| {
            (0..n)
                .map(|i| {
                    let r = i as f64 / n as f64;
                    let actual = if r < frac_home {
                        Outcome::HomeWin
                    } else if r < frac_home + (1.0 - frac_home) / 2.0 {
                        Outcome::Draw
                    } else {
                        Outcome::AwayWin
                    };
                    (p, actual)
                })
                .collect::<Vec<_>>()
        };
        // Predictions match reality (0.5/0.25/0.25 with 50% home).
        let calibrated = mk(400, Probabilities::new(0.5, 0.25, 0.25), 0.5);
        // Wildly overconfident (0.9 home) but still only 50% home wins.
        let overconfident = mk(400, Probabilities::new(0.9, 0.05, 0.05), 0.5);

        let r_cal = reliability(&calibrated, 10);
        let r_over = reliability(&overconfident, 10);
        assert!(
            r_cal.ece < 0.05,
            "well-calibrated ECE should be small: {}",
            r_cal.ece
        );
        assert!(
            r_cal.ece < r_over.ece,
            "calibrated ECE {} should beat overconfident {}",
            r_cal.ece,
            r_over.ece
        );
    }
}
