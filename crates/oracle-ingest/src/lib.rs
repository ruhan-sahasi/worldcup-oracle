//! # oracle-ingest
//!
//! The data layer. It defines the [`DataProvider`] seam - the single abstraction the
//! rest of the system talks to - and ships three implementations spanning the full
//! spectrum from "works on a plane with no key" to "live off the real World Cup":
//!
//! | Provider | Source | Network | Key |
//! |----------|--------|---------|-----|
//! | [`SimProvider`] | deterministic synthetic feed | no | no |
//! | [`ReplayProvider`] | a completed tournament, re-streamed | no | no |
//! | [`FootballDataProvider`] | live football-data.org API | yes | yes |
//!
//! Supporting infrastructure: a token-bucket [`rate_limit::RateLimiter`] and a
//! [`cache::TtlCache`] (the back-pressure + caching around the live API), the
//! embedded [`data`] module (the offline 2026 tournament + synthetic training data),
//! and the [`actual_2026`] module (the *real* 2026 knockout results, R16 onward,
//! for the stage-conditioned forecast).

pub mod actual_2026;
pub mod cache;
pub mod data;
pub mod error;
pub mod football_data;
pub mod provider;
pub mod rate_limit;
pub mod replay_provider;
pub mod sim_provider;

pub use error::{IngestError, Result};
pub use football_data::FootballDataProvider;
pub use provider::DataProvider;
pub use rate_limit::RateLimiter;
pub use replay_provider::ReplayProvider;
pub use sim_provider::SimProvider;
