//! # oracle-model
//!
//! The statistical "brains" of worldcup-oracle. Three cooperating pieces:
//!
//! - [`dixon_coles`] - a time-weighted Dixon-Coles bivariate-Poisson **goal model**
//!   fit by maximum likelihood; turns team histories into exact-score distributions.
//! - [`live`] - **Bayesian in-match updating**: conditions on the live scoreline,
//!   minute, and red cards to re-derive final-result probabilities event by event.
//! - [`ensemble`] - a **logarithmic opinion pool** that blends the goal model with
//!   the Elo ratings from `oracle-ratings` into a single sharper forecast.
//!
//! [`calibration`] provides the proper scoring rules (Brier, log loss) used to tune
//! and regression-test all of the above.
#![forbid(unsafe_code)]

pub mod calibration;
pub mod context;
pub mod dixon_coles;
pub mod ensemble;
pub mod live;
pub mod poisson;
pub mod style;

pub use calibration::{
    bootstrap_score_ci, implied_probabilities, reliability, score, CalibrationReport, MetricCi,
    ReliabilityBin, ReliabilityReport,
};
pub use context::{context_adjustment, Host, MatchContext};
pub use dixon_coles::{DixonColesConfig, GoalModel, Observation, ScoreModel};
pub use ensemble::{Ensemble, EnsembleFitConfig};
pub use live::{live_probabilities, live_score_grid, remaining_rates, LiveConfig, LiveState};
pub use style::{style_adjustment, style_tilt, StyleProfile};
