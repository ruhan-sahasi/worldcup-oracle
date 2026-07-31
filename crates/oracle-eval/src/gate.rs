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

use crate::fixture;
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
    /// [`fixture::content_hash`] of that file's bytes. A mismatch means the comparison is invalid,
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
        // Chosen against the noise floor rather than picked round. The evaluation is exactly
        // reproducible, so run-to-run noise is zero; what these have to absorb is the last-bits drift
        // an unrelated refactor can cause (measured at ~1e-14 before the fit was made
        // bit-reproducible, and zero after). 0.002 Brier on an 800-match test split is roughly a
        // fifth of one standard error, so it catches any regression worth the name while leaving
        // room for genuinely inconsequential change.
        Self {
            brier: 0.002,
            log_loss: 0.004,
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

/// Hash a fixture's bytes the way a baseline records it.
pub fn fixture_hash(bytes: &[u8]) -> String {
    fixture::content_hash(bytes)
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
