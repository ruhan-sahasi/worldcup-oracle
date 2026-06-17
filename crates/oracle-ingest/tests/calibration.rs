//! Integration test: the fitted goal model must beat a naive baseline out-of-sample.
//!
//! This is the regression guard for model quality. We fit Dixon-Coles on the older
//! 80% of the synthetic history and score it on the most recent 20% - a proper
//! temporal split - then assert it beats the uniform (⅓,⅓,⅓) baseline on both
//! proper scoring rules. If a change to the model or fit silently degrades accuracy,
//! this test fails.

use oracle_ingest::data;
use oracle_model::{score, CalibrationReport, DixonColesConfig, GoalModel};

#[test]
fn fitted_model_beats_uniform_baseline_out_of_sample() {
    let mut history = data::synthetic_history(5000, 7);
    // Oldest first → train on the past, test on the most recent matches.
    history.sort_by(|a, b| {
        b.age_days
            .partial_cmp(&a.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let split = history.len() * 4 / 5;
    let (train, test) = history.split_at(split);

    let model = GoalModel::fit(train, DixonColesConfig::default());
    let predictions: Vec<_> = test
        .iter()
        .map(|o| {
            (
                model.outcome_probabilities(o.home, o.away, true),
                o.score.outcome(),
            )
        })
        .collect();

    let report = score(&predictions);
    let baseline = CalibrationReport::uniform_baseline(test.len());

    assert!(
        report.brier < baseline.brier,
        "Brier {:.4} should beat baseline {:.4}",
        report.brier,
        baseline.brier
    );
    assert!(
        report.log_loss < baseline.log_loss,
        "log-loss {:.4} should beat baseline {:.4}",
        report.log_loss,
        baseline.log_loss
    );
    assert!(
        report.accuracy > 0.40,
        "accuracy {:.3} unexpectedly low",
        report.accuracy
    );
}
