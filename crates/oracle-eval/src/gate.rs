//! The skill regression gate: a recorded baseline, and a verdict on a fresh evaluation.
//!
//! # What this protects
//!
//! The project claims forecasting skill and nothing enforced it. Tests covered the machinery -
//! probabilities normalize, grids sum to one, the simulator reproduces its seed - and said nothing
//! about whether the model was any *good*. A change that quietly cost ten percent of skill passed CI
//! green.
//!
//! # What it deliberately does not do
//!
//! It does not measure absolute skill. The fixture is synthetic, so the numbers here are not evidence
//! the model works on real football; `docs/VALIDATION.md` is where that lives. The gate answers a
//! narrower and more useful question: *did this change make the model worse?* For that, the dataset
//! only has to be fixed, not realistic.
//!
//! It also does not stop the bar being moved. A baseline is a committed file and can be rewritten -
//! that is necessary, since a genuine model improvement should be recorded. What it makes impossible
//! is moving the bar *silently*: updating a baseline is an explicit command that rewrites a tracked
//! file and shows up in review as a diff of the numbers being claimed.

use crate::skill::{Model, SkillReport};
use serde::{Deserialize, Serialize};

/// The schema version new baselines are written with.
pub const BASELINE_SCHEMA: u32 = 1;

/// One forecaster's recorded skill, and the interval it was measured with.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub model: Model,
    pub scored: usize,
    pub brier: f64,
    pub log_loss: f64,
    pub accuracy: f64,
}

/// A recorded skill measurement to compare future evaluations against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillBaseline {
    #[serde(default = "default_baseline_schema")]
    pub schema: u32,
    /// The fixture this was measured on, as a repo-relative path, for the error message when a
    /// mismatch is found.
    pub fixture: String,
    /// [`crate::fixture::content_hash`] of that file's bytes. A mismatch means the comparison is invalid,
    /// not that the model regressed.
    pub fixture_hash: String,
    /// Matches in each split, recorded so a changed split is caught as a distinct problem from a
    /// changed model.
    pub train: usize,
    pub validation: usize,
    pub test: usize,
    pub models: Vec<BaselineEntry>,
    /// How much each metric may worsen before the gate fails. Absolute, in metric units.
    pub tolerance: Tolerance,
    /// Free-text note recorded when the baseline was written, e.g. why it was updated.
    #[serde(default)]
    pub note: String,
}

fn default_baseline_schema() -> u32 {
    1
}

/// How much worse than baseline a fresh measurement may be.
///
/// Absolute rather than relative, because these metrics do not span orders of magnitude - a Brier
/// score lives between 0 and 2 - and a relative tolerance would be quietly stricter on the models
/// that already score well.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerance {
    pub brier: f64,
    pub log_loss: f64,
    /// Accuracy is reported but not gated by default, because it is not a proper scoring rule: a
    /// model can improve its Brier and log loss while calling one more match wrong. Set this to gate
    /// on it anyway.
    pub accuracy: Option<f64>,
}

impl Default for Tolerance {
    fn default() -> Self {
        // Calibrated by measuring what a real degradation actually costs, not picked round.
        //
        // The evaluation is bit-reproducible over a frozen fixture, so there is no run-to-run noise
        // to absorb - only the last-bits drift an unrelated refactor might introduce, on the order of
        // 1e-14. That leaves a wide range to choose from, and the choice matters: a first attempt at
        // 0.002 Brier turned out to pass a change that disabled the learned ensemble stacking
        // entirely (equal weights, no temperature scaling), which costs only 0.0006 Brier on this
        // fixture. A gate that lets a whole model component be deleted is decoration.
        //
        // 0.0002 catches that with room to spare while sitting ten orders of magnitude above the
        // floating-point floor. Log loss is set proportionally: it moved about twice as far as Brier
        // in the same experiment.
        Self {
            brier: 0.0002,
            log_loss: 0.0005,
            accuracy: None,
        }
    }
}

impl SkillBaseline {
    /// Record a fresh evaluation as the baseline.
    pub fn record(
        report: &SkillReport,
        fixture_path: impl Into<String>,
        fixture_hash: impl Into<String>,
        tolerance: Tolerance,
        note: impl Into<String>,
    ) -> Self {
        Self {
            schema: BASELINE_SCHEMA,
            fixture: fixture_path.into(),
            fixture_hash: fixture_hash.into(),
            train: report.train,
            validation: report.validation,
            test: report.test,
            models: report
                .models
                .iter()
                .map(|m| BaselineEntry {
                    model: m.model,
                    scored: m.scored,
                    brier: m.brier,
                    log_loss: m.log_loss,
                    accuracy: m.accuracy,
                })
                .collect(),
            tolerance,
            note: note.into(),
        }
    }

    /// One forecaster's recorded entry.
    pub fn get(&self, model: Model) -> Option<&BaselineEntry> {
        self.models.iter().find(|m| m.model == model)
    }

    /// Serialize as pretty JSON with a trailing newline, so a committed baseline diffs one metric per
    /// line and a reviewer can see exactly which number moved.
    ///
    /// # Panics
    /// If serialization fails. It cannot: every field is a plain number, string or enum, and the only
    /// documented failure of `to_string_pretty` is a type whose `Serialize` impl errors or a map with
    /// non-string keys, neither of which is reachable here.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a baseline always serializes");
        s.push('\n');
        s
    }

    /// Parse a baseline, rejecting one written by a newer build.
    ///
    /// Same reasoning as the forecast journal's schema check: a future baseline's known fields still
    /// deserialize, so it would load looking valid while dropping whatever the newer version added -
    /// a stricter rule, an extra gated metric - and the gate would pass on a comparison it did not
    /// fully understand.
    pub fn from_json(text: &str) -> Result<Self, GateError> {
        let parsed: Self =
            serde_json::from_str(text).map_err(|e| GateError::Malformed(e.to_string()))?;
        if parsed.schema > BASELINE_SCHEMA {
            return Err(GateError::FutureSchema {
                found: parsed.schema,
                understood: BASELINE_SCHEMA,
            });
        }
        Ok(parsed)
    }

    /// Load a baseline from disk.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, GateError> {
        let text = std::fs::read_to_string(path).map_err(|e| GateError::Io(e.to_string()))?;
        Self::from_json(&text)
    }
}

/// Why a gate run could not reach a verdict.
///
/// Separate from a failing verdict on purpose. "The model got worse" and "I could not tell whether
/// the model got worse" call for different reactions, and collapsing them would let a broken setup
/// read as a pass.
#[derive(Debug, Clone, PartialEq)]
pub enum GateError {
    Io(String),
    Malformed(String),
    FutureSchema { found: u32, understood: u32 },
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateError::Io(e) => write!(f, "reading the baseline: {e}"),
            GateError::Malformed(e) => write!(f, "the baseline is not valid JSON: {e}"),
            GateError::FutureSchema { found, understood } => write!(
                f,
                "the baseline was written by a newer build (schema {found}, this build understands \
                 {understood}); upgrade rather than comparing against it partially"
            ),
        }
    }
}

impl std::error::Error for GateError {}

/// How one metric moved against its recorded value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MetricDelta {
    pub metric: &'static str,
    pub baseline: f64,
    pub current: f64,
    /// Positive means worse than baseline, for every metric here. Brier and log loss are losses, so
    /// worse is larger; accuracy is a score, so worse is smaller and the sign is flipped. Normalizing
    /// the direction means the comparison logic never has to remember which is which.
    pub regression: f64,
    pub tolerance: f64,
}

impl MetricDelta {
    /// Whether this metric moved further than its tolerance allows.
    pub fn breached(&self) -> bool {
        self.regression > self.tolerance
    }
}

/// One forecaster's comparison against its recorded entry.
#[derive(Debug, Clone, Serialize)]
pub struct ModelComparison {
    pub model: Model,
    pub deltas: Vec<MetricDelta>,
}

impl ModelComparison {
    /// The metrics that breached tolerance.
    pub fn breaches(&self) -> impl Iterator<Item = &MetricDelta> {
        self.deltas.iter().filter(|d| d.breached())
    }
}

/// Everything that can make a gate run fail, other than a metric moving.
///
/// Each of these means the comparison is not valid, which is different from the model being worse -
/// so they are reported separately and never silently treated as a pass.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Discrepancy {
    /// The fixture's bytes are not the ones the baseline was measured on.
    FixtureChanged { expected: String, found: String },
    /// The split sizes differ, so the two measurements are over different test sets.
    SplitChanged {
        expected: (usize, usize, usize),
        found: (usize, usize, usize),
    },
    /// A model in the baseline is missing from the fresh evaluation.
    ModelMissing { model: Model },
    /// A model appeared that the baseline has no entry for. Not a failure on its own - a new
    /// forecaster is a legitimate addition - but reported so it is noticed rather than ignored.
    ModelAdded { model: Model },
}

/// The outcome of comparing an evaluation against a baseline.
#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub comparisons: Vec<ModelComparison>,
    pub discrepancies: Vec<Discrepancy>,
}

impl Verdict {
    /// Whether the gate passes.
    ///
    /// Fails on any breached metric, and on any discrepancy other than an added model. A changed
    /// fixture or split is not evidence the model regressed, but it *is* evidence the comparison
    /// cannot be trusted, and a gate that passes when it does not know the answer is worse than no
    /// gate.
    pub fn passed(&self) -> bool {
        self.regressions().next().is_none() && !self.has_blocking_discrepancy()
    }

    /// Every metric that moved beyond tolerance, with the model it belongs to.
    pub fn regressions(&self) -> impl Iterator<Item = (Model, &MetricDelta)> {
        self.comparisons
            .iter()
            .flat_map(|c| c.breaches().map(move |d| (c.model, d)))
    }

    /// Whether any discrepancy invalidates the comparison.
    pub fn has_blocking_discrepancy(&self) -> bool {
        self.discrepancies
            .iter()
            .any(|d| !matches!(d, Discrepancy::ModelAdded { .. }))
    }

    /// The largest regression seen, for a one-line summary.
    pub fn worst_regression(&self) -> Option<(Model, MetricDelta)> {
        self.regressions()
            .max_by(|(_, a), (_, b)| {
                (a.regression - a.tolerance)
                    .partial_cmp(&(b.regression - b.tolerance))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(m, d)| (m, *d))
    }
}

/// Compare a fresh evaluation against a recorded baseline.
///
/// `current_fixture_hash` is the hash of the bytes actually evaluated, so a mismatch against the
/// baseline's recorded hash is detected here rather than assumed away.
pub fn compare(
    baseline: &SkillBaseline,
    report: &SkillReport,
    current_fixture_hash: &str,
) -> Verdict {
    let mut discrepancies = Vec::new();
    if baseline.fixture_hash != current_fixture_hash {
        discrepancies.push(Discrepancy::FixtureChanged {
            expected: baseline.fixture_hash.clone(),
            found: current_fixture_hash.to_string(),
        });
    }
    let recorded_split = (baseline.train, baseline.validation, baseline.test);
    let current_split = (report.train, report.validation, report.test);
    if recorded_split != current_split {
        discrepancies.push(Discrepancy::SplitChanged {
            expected: recorded_split,
            found: current_split,
        });
    }
    for entry in &baseline.models {
        if report.get(entry.model).is_none() {
            discrepancies.push(Discrepancy::ModelMissing { model: entry.model });
        }
    }
    for m in &report.models {
        if baseline.get(m.model).is_none() {
            discrepancies.push(Discrepancy::ModelAdded { model: m.model });
        }
    }

    let t = baseline.tolerance;
    let comparisons = baseline
        .models
        .iter()
        .filter_map(|entry| {
            let current = report.get(entry.model)?;
            let mut deltas = vec![
                // Losses: worse is larger.
                MetricDelta {
                    metric: "brier",
                    baseline: entry.brier,
                    current: current.brier,
                    regression: current.brier - entry.brier,
                    tolerance: t.brier,
                },
                MetricDelta {
                    metric: "log_loss",
                    baseline: entry.log_loss,
                    current: current.log_loss,
                    regression: current.log_loss - entry.log_loss,
                    tolerance: t.log_loss,
                },
            ];
            if let Some(tol) = t.accuracy {
                // A score: worse is smaller, so the sign flips.
                deltas.push(MetricDelta {
                    metric: "accuracy",
                    baseline: entry.accuracy,
                    current: current.accuracy,
                    regression: entry.accuracy - current.accuracy,
                    tolerance: tol,
                });
            }
            Some(ModelComparison {
                model: entry.model,
                deltas,
            })
        })
        .collect();

    Verdict {
        comparisons,
        discrepancies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evaluate, EvalConfig};
    use oracle_ingest::data;

    fn report() -> SkillReport {
        evaluate(
            &data::synthetic_history_with_market(1000, 7),
            EvalConfig::default(),
        )
        .unwrap()
    }

    fn baseline() -> SkillBaseline {
        SkillBaseline::record(
            &report(),
            "fixtures/skill_v1.csv",
            "abc123",
            Tolerance::default(),
            "",
        )
    }

    /// A baseline over `report`, then a report with `model`'s metrics shifted by `delta` (positive =
    /// worse), so a regression can be simulated without changing the model.
    fn worsened(model: Model, delta: f64) -> (SkillBaseline, SkillReport) {
        let r = report();
        let b = SkillBaseline::record(&r, "f.csv", "H", Tolerance::default(), "");
        let mut worse = r.clone();
        for m in worse.models.iter_mut().filter(|m| m.model == model) {
            m.brier += delta;
            m.log_loss += delta;
        }
        (b, worse)
    }

    #[test]
    fn an_unchanged_evaluation_passes() {
        let r = report();
        let b = SkillBaseline::record(&r, "f.csv", "H", Tolerance::default(), "");
        let v = compare(&b, &r, "H");
        assert!(v.passed(), "identical metrics must pass");
        assert!(v.discrepancies.is_empty());
        assert_eq!(v.regressions().count(), 0);
        assert!(v.worst_regression().is_none());
    }

    #[test]
    fn a_regression_beyond_tolerance_fails_and_names_the_model() {
        let (b, worse) = worsened(Model::Ensemble, 0.01);
        let v = compare(&b, &worse, "H");
        assert!(!v.passed());
        let (model, delta) = v.worst_regression().expect("a regression");
        assert_eq!(model, Model::Ensemble, "the gate names the culprit");
        assert!(delta.breached());
        assert!(delta.regression > delta.tolerance);
    }

    #[test]
    fn a_movement_inside_tolerance_passes() {
        // Derived from the tolerance rather than hardcoded. An earlier version of this test used a
        // literal 0.0009, which was half the tolerance at the time; tightening the tolerance to
        // 0.0002 then made the test fail for a reason that had nothing to do with what it checks.
        let tol = Tolerance::default();
        let inside = tol.brier.min(tol.log_loss) / 2.0;
        let (b, slightly_worse) = worsened(Model::Ensemble, inside);
        let v = compare(&b, &slightly_worse, "H");
        assert!(
            v.passed(),
            "a move of {inside} is inside tolerance and must not fail: {:?}",
            v.regressions().collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_movement_just_outside_tolerance_fails() {
        // The other side of the same boundary, also derived, so the two together pin where the line
        // actually is rather than asserting only that some large move fails.
        let tol = Tolerance::default();
        let outside = tol.brier.max(tol.log_loss) * 1.5;
        let (b, worse) = worsened(Model::Ensemble, outside);
        let v = compare(&b, &worse, "H");
        assert!(!v.passed(), "a move of {outside} must fail");
    }

    #[test]
    fn an_improvement_passes_and_is_not_reported_as_a_regression() {
        let (b, better) = worsened(Model::Ensemble, -0.05);
        let v = compare(&b, &better, "H");
        assert!(v.passed());
        assert_eq!(v.regressions().count(), 0, "getting better is not a breach");
        // The delta is still recorded, with a negative regression, so an improvement is visible.
        let ens = v
            .comparisons
            .iter()
            .find(|c| c.model == Model::Ensemble)
            .unwrap();
        assert!(ens.deltas.iter().all(|d| d.regression < 0.0));
    }

    #[test]
    fn a_changed_fixture_blocks_rather_than_passing_or_blaming_the_model() {
        let r = report();
        let b = SkillBaseline::record(&r, "f.csv", "expected-hash", Tolerance::default(), "");
        let v = compare(&b, &r, "different-hash");
        assert!(!v.passed(), "an invalid comparison must not pass");
        assert_eq!(v.regressions().count(), 0, "the model did not regress");
        assert!(matches!(
            v.discrepancies.first(),
            Some(Discrepancy::FixtureChanged { .. })
        ));
    }

    #[test]
    fn a_changed_split_blocks() {
        let r = report();
        let mut b = SkillBaseline::record(&r, "f.csv", "H", Tolerance::default(), "");
        b.test += 1;
        let v = compare(&b, &r, "H");
        assert!(!v.passed());
        assert!(v
            .discrepancies
            .iter()
            .any(|d| matches!(d, Discrepancy::SplitChanged { .. })));
    }

    #[test]
    fn a_missing_model_blocks_but_a_new_one_does_not() {
        let r = report();
        let b = SkillBaseline::record(&r, "f.csv", "H", Tolerance::default(), "");

        // Dropping a forecaster hides its skill, so it blocks.
        let mut without_elo = r.clone();
        without_elo.models.retain(|m| m.model != Model::Elo);
        let v = compare(&b, &without_elo, "H");
        assert!(!v.passed());
        assert!(v
            .discrepancies
            .iter()
            .any(|d| matches!(d, Discrepancy::ModelMissing { model } if *model == Model::Elo)));

        // Adding one is a legitimate change, reported but not blocking.
        let mut trimmed_baseline = b.clone();
        trimmed_baseline.models.retain(|m| m.model != Model::Elo);
        let v = compare(&trimmed_baseline, &r, "H");
        assert!(v.passed(), "a new forecaster must not fail the gate");
        assert!(v
            .discrepancies
            .iter()
            .any(|d| matches!(d, Discrepancy::ModelAdded { model } if *model == Model::Elo)));
    }

    #[test]
    fn accuracy_is_gated_only_when_a_tolerance_is_set() {
        let r = report();
        let mut worse = r.clone();
        for m in worse.models.iter_mut() {
            m.accuracy -= 0.10; // a big accuracy drop, metrics otherwise untouched
        }

        let ungated = SkillBaseline::record(&r, "f.csv", "H", Tolerance::default(), "");
        assert!(
            compare(&ungated, &worse, "H").passed(),
            "accuracy is not gated by default"
        );

        let gated = SkillBaseline::record(
            &r,
            "f.csv",
            "H",
            Tolerance {
                accuracy: Some(0.02),
                ..Tolerance::default()
            },
            "",
        );
        let v = compare(&gated, &worse, "H");
        assert!(!v.passed(), "an explicit accuracy tolerance is enforced");
        assert!(v.regressions().any(|(_, d)| d.metric == "accuracy"));
    }

    #[test]
    fn the_regression_sign_is_normalized_across_metrics() {
        // Brier and log loss are losses; accuracy is a score. `regression` must be positive-is-worse
        // for all three, or the comparison logic would have to remember which is which - the sort of
        // asymmetry that makes a gate silently one-sided.
        let r = report();
        let mut changed = r.clone();
        for m in changed.models.iter_mut() {
            m.brier += 0.01; // worse
            m.log_loss += 0.01; // worse
            m.accuracy += 0.01; // better
        }
        let b = SkillBaseline::record(
            &r,
            "f.csv",
            "H",
            Tolerance {
                accuracy: Some(0.02),
                ..Tolerance::default()
            },
            "",
        );
        let v = compare(&b, &changed, "H");
        let first = &v.comparisons[0];
        let d = |name: &str| first.deltas.iter().find(|d| d.metric == name).unwrap();
        assert!(d("brier").regression > 0.0, "a higher Brier is worse");
        assert!(d("log_loss").regression > 0.0, "a higher log loss is worse");
        assert!(
            d("accuracy").regression < 0.0,
            "a higher accuracy is better"
        );
    }

    #[test]
    fn a_baseline_round_trips_through_json() {
        let b = baseline();
        let back = SkillBaseline::from_json(&b.to_json()).unwrap();

        // Everything discrete must survive exactly.
        assert_eq!(b.schema, back.schema);
        assert_eq!(b.fixture, back.fixture);
        assert_eq!(b.fixture_hash, back.fixture_hash);
        assert_eq!(
            (b.train, b.validation, b.test),
            (back.train, back.validation, back.test)
        );
        assert_eq!(b.tolerance, back.tolerance);
        assert_eq!(b.note, back.note);
        assert_eq!(b.models.len(), back.models.len());

        // The metrics survive to within a floating-point unit in the last place. Not exactly: a JSON
        // round trip of an f64 can land one ULP away, and one of these does. That is eleven orders of
        // magnitude below the gate's tolerance, so it cannot mask or manufacture a regression - but it
        // does mean a baseline is a record of a measurement to ~1e-16, not a bit-preserving snapshot,
        // which is worth asserting explicitly rather than discovering later.
        for (x, y) in b.models.iter().zip(&back.models) {
            assert_eq!(x.model, y.model);
            assert_eq!(x.scored, y.scored);
            for (label, p, q) in [
                ("brier", x.brier, y.brier),
                ("log_loss", x.log_loss, y.log_loss),
                ("accuracy", x.accuracy, y.accuracy),
            ] {
                assert!(
                    (p - q).abs() <= f64::EPSILON * p.abs().max(1.0),
                    "{} {label} moved by {:.3e} through JSON",
                    x.model.label(),
                    (p - q).abs()
                );
            }
        }
    }

    #[test]
    fn a_baseline_records_every_scored_model() {
        let r = report();
        let b = SkillBaseline::record(&r, "f.csv", "h", Tolerance::default(), "");
        assert_eq!(b.models.len(), r.models.len());
        for m in &r.models {
            let entry = b.get(m.model).expect("every model recorded");
            assert_eq!(entry.brier, m.brier, "metrics recorded exactly");
            assert_eq!(entry.log_loss, m.log_loss);
            assert_eq!(entry.scored, m.scored);
        }
    }

    #[test]
    fn a_baseline_records_the_split_sizes() {
        // A changed split is a different problem from a changed model, so it has to be detectable
        // separately rather than showing up as a mysterious metric shift.
        let r = report();
        let b = SkillBaseline::record(&r, "f.csv", "h", Tolerance::default(), "");
        assert_eq!((b.train, b.validation, b.test), (600, 200, 200));
    }

    #[test]
    fn json_is_pretty_printed_with_a_trailing_newline() {
        // A committed baseline should diff one metric per line, and end with a newline like every
        // other text file in the repo.
        let json = baseline().to_json();
        assert!(json.contains('\n'), "not pretty-printed");
        assert!(json.ends_with('\n'), "no trailing newline");
        assert!(json.contains("\"brier\""), "field names are readable");
    }

    #[test]
    fn a_baseline_from_a_future_schema_is_refused() {
        let mut b = baseline();
        b.schema = BASELINE_SCHEMA + 1;
        let err = SkillBaseline::from_json(&b.to_json()).unwrap_err();
        assert_eq!(
            err,
            GateError::FutureSchema {
                found: BASELINE_SCHEMA + 1,
                understood: BASELINE_SCHEMA
            }
        );
        // And the message says what to do about it rather than only what happened.
        assert!(err.to_string().contains("upgrade"));
    }

    #[test]
    fn a_baseline_without_a_schema_field_reads_as_version_one() {
        let json = r#"{"fixture":"f.csv","fixture_hash":"h","train":600,"validation":200,
            "test":200,"models":[],"tolerance":{"brier":0.002,"log_loss":0.004,"accuracy":null}}"#;
        let b = SkillBaseline::from_json(json).unwrap();
        assert_eq!(b.schema, 1);
        assert_eq!(b.note, "", "an absent note defaults to empty");
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        let err = SkillBaseline::from_json("{ not json").unwrap_err();
        assert!(matches!(err, GateError::Malformed(_)));
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn a_missing_baseline_file_is_an_io_error() {
        let err = SkillBaseline::load("/nonexistent/baseline.json").unwrap_err();
        assert!(matches!(err, GateError::Io(_)));
    }

    #[test]
    fn the_default_tolerance_does_not_gate_accuracy() {
        // Accuracy is not a proper scoring rule: a model can improve its Brier and log loss while
        // calling one more match wrong. Gating it by default would fail good changes.
        assert!(Tolerance::default().accuracy.is_none());
        assert!(Tolerance::default().brier > 0.0);
        assert!(Tolerance::default().log_loss > 0.0);
    }
}
