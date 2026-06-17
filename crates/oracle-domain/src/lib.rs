//! # oracle-domain
//!
//! The pure, dependency-light core of **worldcup-oracle**. Every other crate in the
//! workspace depends on these types, but this crate depends on *nothing* except
//! `serde` (serialization) and `chrono` (timestamps). It performs no I/O.
//!
//! This deliberate "dependency inversion at the workspace level" keeps the domain
//! model stable and trivially testable while the prediction models, data adapters,
//! and transport layers evolve around it.
//!
//! ## Module map
//! - [`team`] — teams, confederations, identifiers
//! - [`fixture`] — matches, scorelines, stages, live status
//! - [`event`] — the live event stream ([`MatchEvent`])
//! - [`probability`] — outcomes, win/draw/win probabilities, exact-score grids
//! - [`tournament`] — groups, the tournament container, and forecast outputs
#![forbid(unsafe_code)]

pub mod event;
pub mod fixture;
pub mod probability;
pub mod team;
pub mod tournament;

pub use event::{EventKind, MatchEvent};
pub use fixture::{Match, MatchId, MatchStatus, Scoreline, Stage};
pub use probability::{Outcome, Probabilities, ScoreGrid};
pub use team::{Confederation, Team, TeamId};
pub use tournament::{Group, TeamForecast, Tournament, TournamentForecast};
