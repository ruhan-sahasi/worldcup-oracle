//! The skill gate, enforced from `cargo test`.
//!
//! CI runs the gate as its own step so a failure is labelled as a skill regression rather than a
//! generic test failure. This runs it again from the test suite, which is deliberate duplication: a
//! contributor gets the answer from `cargo test` before pushing, and the gate keeps working for
//! anyone who runs the tests without the CI workflow.
//!
//! An integration test rather than a unit test because it reads committed files - the frozen fixture
//! and the recorded baseline - and their relationship to each other is the thing being checked.

use oracle_eval::gate::{compare, Discrepancy};
use oracle_eval::{EvalConfig, SkillBaseline};

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/skill_v1.csv")
}

fn baseline_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines/skill_v1.json")
}

#[test]
fn the_model_has_not_regressed_against_the_recorded_baseline() {
    let (records, hash) = oracle_eval::fixture::load(fixture_path()).expect("the frozen fixture");
    let baseline = SkillBaseline::load(baseline_path()).expect("the recorded baseline");
    let report = oracle_eval::evaluate(&records, EvalConfig::default()).expect("evaluation");
    let verdict = compare(&baseline, &report, &hash);

    // Report everything before asserting, so a failure shows the whole table rather than one line.
    for c in &verdict.comparisons {
        for d in &c.deltas {
            println!(
                "{:<30} {:.6} -> {:.6} ({:+.6}, tolerance {:.6}){}",
                format!("{} {}", c.model.label(), d.metric),
                d.baseline,
                d.current,
                d.current - d.baseline,
                d.tolerance,
                if d.breached() { "  REGRESSION" } else { "" }
            );
        }
    }

    // A changed fixture or split is not a regression, and saying so plainly saves the next person
    // from debugging the model when the data moved.
    for discrepancy in &verdict.discrepancies {
        match discrepancy {
            Discrepancy::FixtureChanged { expected, found } => panic!(
                "the fixture has changed (baseline hash {expected}, current {found}).\n\
                 This is not a model regression - the comparison is invalid. Restore \
                 fixtures/skill_v1.csv, or freeze a new fixture and record a fresh baseline."
            ),
            Discrepancy::SplitChanged { expected, found } => panic!(
                "the evaluation split changed: baseline {expected:?} vs current {found:?}. The two \
                 measurements are over different test sets, so they are not comparable."
            ),
            Discrepancy::ModelMissing { model } => panic!(
                "{} is in the baseline but was not scored; a dropped forecaster hides its skill.",
                model.label()
            ),
            // A new forecaster is a legitimate addition, not a failure.
            Discrepancy::ModelAdded { .. } => {}
        }
    }

    if let Some((model, d)) = verdict.worst_regression() {
        panic!(
            "skill regression: {} {} moved {:+.6}, tolerance {:.6}.\n\
             If this change is deliberate and understood, record a new baseline:\n  \
             cargo run -p oracle-cli -- skill-gate-update --note \"why\"",
            model.label(),
            d.metric,
            d.current - d.baseline,
            d.tolerance
        );
    }
    assert!(verdict.passed());
}

#[test]
fn the_baseline_describes_the_committed_fixture() {
    // Guards the pairing itself. If the baseline and fixture were committed out of step, every other
    // assertion here would be about the wrong comparison - and the failure would look like a model
    // regression rather than a bookkeeping mistake.
    let (records, hash) = oracle_eval::fixture::load(fixture_path()).expect("the frozen fixture");
    let baseline = SkillBaseline::load(baseline_path()).expect("the recorded baseline");

    assert_eq!(
        baseline.fixture_hash, hash,
        "the committed baseline was measured on different fixture bytes"
    );
    assert!(
        baseline.fixture.ends_with("skill_v1.csv"),
        "baseline points at {}",
        baseline.fixture
    );
    assert_eq!(records.len(), 4000);
    assert_eq!(
        (baseline.train, baseline.validation, baseline.test),
        (2400, 800, 800),
        "the recorded split does not match a 60/20/20 division of 4000"
    );
    assert!(
        !baseline.note.trim().is_empty(),
        "a baseline must record why it was written"
    );
}

#[test]
fn the_baseline_records_a_model_that_beats_the_uniform_bar() {
    // A sanity floor independent of the delta comparison. If a future baseline were recorded from a
    // thoroughly broken model, every delta would be zero against it and the gate would happily
    // protect the broken state. This catches that: whatever the baseline says, the ensemble in it has
    // to be better than guessing.
    let baseline = SkillBaseline::load(baseline_path()).expect("the recorded baseline");
    let uniform = baseline
        .get(oracle_eval::Model::UniformBaseline)
        .expect("a uniform row");
    let ensemble = baseline
        .get(oracle_eval::Model::Ensemble)
        .expect("an ensemble row");

    assert!(
        ensemble.brier < uniform.brier,
        "the recorded ensemble Brier {} does not beat the uniform baseline {}",
        ensemble.brier,
        uniform.brier
    );
    assert!(
        ensemble.log_loss < uniform.log_loss,
        "the recorded ensemble log loss {} does not beat uniform {}",
        ensemble.log_loss,
        uniform.log_loss
    );
    // And it must be within reach of the market, which is the bar the project claims to match.
    let market = baseline
        .get(oracle_eval::Model::Market)
        .expect("a market row");
    assert!(
        ensemble.brier < market.brier + 0.01,
        "the recorded ensemble Brier {} has fallen well behind the market's {}",
        ensemble.brier,
        market.brier
    );
}
