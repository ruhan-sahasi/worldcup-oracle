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
use oracle_numeric::Rng;
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

/// Rescale a probability triple by a temperature: multiply the log-probabilities by `temperature`
/// and renormalize (a softmax over `temperature * ln p`). This matches [`Ensemble`](crate::Ensemble)'s
/// convention: `temperature > 1` sharpens the distribution (more confident), `< 1` flattens it
/// (less confident), and `1` is the identity. Used to apply a fitted calibration correction.
pub fn apply_temperature(p: Probabilities, temperature: f64) -> Probabilities {
    let l = [
        p.home_win.max(1e-12).ln() * temperature,
        p.draw.max(1e-12).ln() * temperature,
        p.away_win.max(1e-12).ln() * temperature,
    ];
    let max = l[0].max(l[1]).max(l[2]);
    Probabilities::new((l[0] - max).exp(), (l[1] - max).exp(), (l[2] - max).exp())
}

/// Fit the single temperature that minimizes log-loss over `(prediction, outcome)` pairs, the
/// standard **temperature-scaling** post-hoc calibration. `> 1` means the raw predictions were
/// under-confident (sharpen them), `< 1` over-confident (soften them). Returns `1.0` (the identity)
/// for fewer than two samples. Found by golden-section search over `[0.1, 5.0]`, since the log-loss
/// is unimodal in the temperature.
pub fn fit_temperature(predictions: &[(Probabilities, Outcome)]) -> f64 {
    if predictions.len() < 2 {
        return 1.0;
    }
    let loss = |t: f64| -> f64 {
        predictions
            .iter()
            .map(|(p, o)| -apply_temperature(*p, t).of(*o).max(1e-12).ln())
            .sum::<f64>()
    };
    // Golden-section search: shrink [a, b] keeping the golden-ratio interior points.
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618
    let (mut a, mut b) = (0.1_f64, 5.0_f64);
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let (mut fc, mut fd) = (loss(c), loss(d));
    for _ in 0..80 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = loss(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = loss(d);
        }
        if (b - a).abs() < 1e-4 {
            break;
        }
    }
    (a + b) / 2.0
}

/// Fit a multiplicative **gain** on a predictor, shrunk toward 1: a one-dimensional Bayesian
/// ridge. Given `(x, y)` pairs it returns the `g` minimizing
/// `Σ (yᵢ - g·xᵢ)² + prior_precision·(g - 1)²`, in closed form
/// `(Σ xᵢyᵢ + prior_precision) / (Σ xᵢ² + prior_precision)`. With no data (or a huge prior) it
/// returns `1.0`, the prior. Used to recalibrate a reasoned-synthetic effect's *strength* against
/// observed results without overreacting to a small sample: `x` is the effect's predicted
/// contribution, `y` the residual it is meant to explain, so `g > 1` means the effect is playing
/// out stronger than the prior and `g < 1` weaker.
pub fn fit_gain_toward_one(pairs: &[(f64, f64)], prior_precision: f64) -> f64 {
    let lambda = prior_precision.max(0.0);
    // Prior N(1, 1/lambda) contributes `lambda` to the denominator and `lambda * 1` to the numerator.
    let mut sxx = lambda;
    let mut sxy = lambda;
    for &(x, y) in pairs {
        sxx += x * x;
        sxy += x * y;
    }
    if sxx <= 0.0 {
        return 1.0;
    }
    sxy / sxx
}

/// A skill metric with a bootstrap confidence interval: the point estimate on the full sample
/// plus the 2.5th and 97.5th percentiles over the resamples.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricCi {
    pub point: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Bootstrap 95% confidence intervals for Brier, log-loss, and accuracy over a set of
/// `(prediction, actual)` pairs (e.g. the pooled out-of-fold predictions of a rolling-origin
/// cross-validation). Resampling with replacement is driven by `oracle-numeric`'s seeded generator,
/// so the intervals are fully reproducible for a given `(seed, n_boot)` - no nondeterminism.
/// Non-overlapping intervals between two models are evidence the skill gap is real rather than a
/// single-split fluke.
pub fn bootstrap_score_ci(
    predictions: &[(Probabilities, Outcome)],
    n_boot: usize,
    seed: u64,
) -> (MetricCi, MetricCi, MetricCi) {
    let point = score(predictions);
    let n = predictions.len();
    if n == 0 || n_boot == 0 {
        let z = MetricCi {
            point: 0.0,
            lo: 0.0,
            hi: 0.0,
        };
        return (z, z, z);
    }

    let mut rng = Rng::new(seed);
    let (mut briers, mut losses, mut accs) = (
        Vec::with_capacity(n_boot),
        Vec::with_capacity(n_boot),
        Vec::with_capacity(n_boot),
    );
    let mut sample: Vec<(Probabilities, Outcome)> = Vec::with_capacity(n);
    for _ in 0..n_boot {
        sample.clear();
        for _ in 0..n {
            sample.push(predictions[rng.index_below(n)]);
        }
        let r = score(&sample);
        briers.push(r.brier);
        losses.push(r.log_loss);
        accs.push(r.accuracy);
    }

    let ci = |point: f64, mut v: Vec<f64>| -> MetricCi {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f64| {
            let idx = ((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
            v[idx]
        };
        MetricCi {
            point,
            lo: pct(0.025),
            hi: pct(0.975),
        }
    };
    (
        ci(point.brier, briers),
        ci(point.log_loss, losses),
        ci(point.accuracy, accs),
    )
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
    fn temperature_one_is_the_identity() {
        let p = Probabilities::new(0.6, 0.25, 0.15);
        let q = apply_temperature(p, 1.0);
        assert!((p.home_win - q.home_win).abs() < 1e-9);
        assert!((p.draw - q.draw).abs() < 1e-9);
        assert!((p.away_win - q.away_win).abs() < 1e-9);
    }

    #[test]
    fn temperature_above_one_sharpens_below_one_flattens() {
        let p = Probabilities::new(0.6, 0.25, 0.15);
        let sharp = apply_temperature(p, 2.0);
        let flat = apply_temperature(p, 0.5);
        assert!(sharp.home_win > p.home_win, "sharpen raises the peak");
        assert!(flat.home_win < p.home_win, "flatten lowers the peak");
        assert!((sharp.sum() - 1.0).abs() < 1e-9 && (flat.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fit_temperature_sharpens_an_under_confident_model() {
        // A model that only ever says 0.5 for the favourite, but the favourite always wins, is
        // under-confident: the best temperature sharpens it (> 1) and cuts the log-loss.
        let preds: Vec<(Probabilities, Outcome)> = (0..60)
            .map(|_| (Probabilities::new(0.5, 0.3, 0.2), Outcome::HomeWin))
            .collect();
        let t = fit_temperature(&preds);
        assert!(
            t > 1.0,
            "under-confident model should be sharpened, got {t}"
        );
        let before = score(&preds).log_loss;
        let after = score(
            &preds
                .iter()
                .map(|(p, o)| (apply_temperature(*p, t), *o))
                .collect::<Vec<_>>(),
        )
        .log_loss;
        assert!(after < before, "temperature scaling should reduce log-loss");
    }

    #[test]
    fn gain_shrinks_toward_one_and_recovers_the_slope() {
        // y = 1.5 x exactly. With no prior, recover the slope; with a strong prior, pulled toward 1.
        let pairs: Vec<(f64, f64)> = (1..=20)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, 1.5 * x)
            })
            .collect();
        assert!((fit_gain_toward_one(&pairs, 0.0) - 1.5).abs() < 1e-9);
        let shrunk = fit_gain_toward_one(&pairs, 100.0);
        assert!(
            shrunk > 1.0 && shrunk < 1.5,
            "strong prior pulls toward 1: {shrunk}"
        );
        // No data collapses to the prior mean of 1.
        assert!((fit_gain_toward_one(&[], 5.0) - 1.0).abs() < 1e-12);
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
    fn bootstrap_ci_brackets_the_point_estimate_and_is_reproducible() {
        let preds: Vec<(Probabilities, Outcome)> = (0..300)
            .map(|i| {
                let actual = match i % 3 {
                    0 => Outcome::HomeWin,
                    1 => Outcome::Draw,
                    _ => Outcome::AwayWin,
                };
                (Probabilities::new(0.5, 0.3, 0.2), actual)
            })
            .collect();
        let point = score(&preds);
        let (brier, log_loss, _acc) = bootstrap_score_ci(&preds, 400, 7);
        // The point estimate matches `score`, and the interval brackets it.
        assert!((brier.point - point.brier).abs() < 1e-12);
        assert!(brier.lo <= brier.point && brier.point <= brier.hi);
        assert!(log_loss.lo <= log_loss.point && log_loss.point <= log_loss.hi);
        assert!(
            brier.lo < brier.hi,
            "a non-degenerate sample has a real interval"
        );
        // Same seed reproduces exactly.
        let again = bootstrap_score_ci(&preds, 400, 7);
        assert_eq!(brier, again.0);
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
