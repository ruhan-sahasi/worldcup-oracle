//! # oracle-eval
//!
//! Offline evaluation of the oracle's forecasting skill, and the regression gate that protects it.
//!
//! ## Why this is its own crate
//!
//! The project's central claim is that the model has real skill - the README quotes Brier and log
//! loss against the bookmaker, and `docs/VALIDATION.md` is entirely about that. Nothing enforced it.
//! CI ran the tests and the tests said nothing about skill, so a change that degraded the model by
//! ten percent would have gone green.
//!
//! The evaluation needed to enforce it already existed, inside the `backtest` CLI command,
//! interleaved with its printing. Nothing else could call it. So this crate owns the evaluation
//! instead: the CLI prints from it, and the gate compares its output against a recorded baseline.
//! Both then report the same numbers by construction rather than by coincidence.
//!
//! It is a separate crate because offline evaluation belongs to neither of its plausible homes. The
//! engine is the live runtime and has no business fitting historical splits; the CLI is a front end
//! and anything buried there is unreachable, which is exactly the problem being fixed.
//!
//! ## Module map
//! - [`skill`] - fit on a dataset, score every forecaster out of sample
//! - [`fixture`] - freeze a dataset to CSV so the gate's data stops being a function of the code
//! - [`gate`] - the recorded baseline and the regression verdict
#![forbid(unsafe_code)]

pub mod fixture;
pub mod gate;
pub mod skill;

pub use gate::{compare, Discrepancy, GateError, SkillBaseline, Tolerance, Verdict};
pub use skill::{evaluate, EvalConfig, EvalError, Model, ModelSkill, SkillReport, MIN_MATCHES};
