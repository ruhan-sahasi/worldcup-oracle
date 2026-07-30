//! A durable, append-only journal of the forecasts the engine actually published.
//!
//! # Why this exists
//!
//! The engine already scores its own pre-match calls: [`ReportCard`](crate::ReportCard) pairs each
//! finished match with the forecast made before it and reports Brier, log loss and accuracy. Those
//! forecasts were held only in memory, which made the report card something subtly weaker than a
//! track record.
//!
//! On restart the engine replays its [`EventLog`](crate::EventLog) and *recomputes* each pre-match
//! forecast as the replayed results go by. The recomputation is leak-free - each call is made before
//! the model learns that result - but it is made by **today's** model and today's fitted parameters.
//! Change the goal model, retune a hyperparameter, or alter the synthetic history the baseline is fit
//! from, and every past call silently changes with it. The scorecard then answers "how would the
//! current model have done?", which is a legitimate question but not the one a track record asks. A
//! model that has quietly improved its own history is not accountable, and the failure is invisible:
//! the numbers simply look a little better after each change.
//!
//! So a forecast, once published, is written here and never recomputed. The journal is the record of
//! what the engine *said*, in the same way the event log is the record of what it *saw*.
//!
//! # Format
//!
//! Newline-delimited JSON, one [`ForecastRecord`] per line, mirroring [`EventLog`](crate::EventLog)
//! deliberately: the same on-disk shape, the same tolerance for a torn final line, and the same lack
//! of a database dependency. Two files of append-only JSONL are easier to reason about, back up and
//! inspect by eye than a schema migration, and at the scale of one record per match per model the
//! cost of reading the whole file is nothing.

use chrono::{DateTime, Utc};
use oracle_domain::{MatchId, Probabilities};
use serde::{Deserialize, Serialize};

/// The schema version new records are written with.
///
/// A record carries its version so a journal written by an older build stays readable, and one
/// written by a *newer* build can be recognised as such rather than silently mis-parsed into
/// plausible nonsense.
pub const SCHEMA_VERSION: u32 = 1;

/// One published pre-match forecast, exactly as it was made.
///
/// The team names are stored rather than looked up from the tournament, because the point of the
/// journal is to remain interpretable on its own: a call is still readable after the tournament
/// state that produced it is gone, and a renamed or re-identified team cannot retroactively change
/// who a past call was about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastRecord {
    /// Schema version this record was written with. See [`SCHEMA_VERSION`].
    #[serde(default = "default_schema_version")]
    pub schema: u32,
    /// Which match the call is about.
    pub match_id: MatchId,
    pub home_name: String,
    pub away_name: String,
    /// Which forecaster made the call, e.g. `"Dixon-Coles ensemble"` or `"Bradley-Terry"`. Scoring
    /// groups by this, so two models journaled against the same matches can be compared directly.
    pub model: String,
    /// The published win/draw/win probabilities.
    pub forecast: Probabilities,
    /// When the call was journaled.
    pub made_at: DateTime<Utc>,
}

/// Records written before the schema field existed would be version 1 by definition.
fn default_schema_version() -> u32 {
    1
}

impl ForecastRecord {
    /// Journal a call made now.
    pub fn new(
        match_id: MatchId,
        home_name: impl Into<String>,
        away_name: impl Into<String>,
        model: impl Into<String>,
        forecast: Probabilities,
    ) -> Self {
        Self {
            schema: SCHEMA_VERSION,
            match_id,
            home_name: home_name.into(),
            away_name: away_name.into(),
            model: model.into(),
            forecast,
            made_at: Utc::now(),
        }
    }

    /// The identity of a call: one forecast per match per model.
    ///
    /// This is what makes the journal immutable in practice. A match is journaled when its result
    /// lands, and a replay of the event log walks that same result again - so without a key to
    /// deduplicate on, every restart would append another copy of every call and quietly multiply
    /// the sample the scores are computed over.
    pub fn key(&self) -> (MatchId, &str) {
        (self.match_id, self.model.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ForecastRecord {
        ForecastRecord::new(
            MatchId(42),
            "Brazil",
            "Japan",
            "Dixon-Coles ensemble",
            Probabilities::new(0.6, 0.25, 0.15),
        )
    }

    #[test]
    fn a_record_round_trips_through_json() {
        let r = sample();
        let line = serde_json::to_string(&r).unwrap();
        let back: ForecastRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn a_record_serializes_to_one_line() {
        // The journal is newline-delimited, so an embedded newline would corrupt the file.
        let line = serde_json::to_string(&sample()).unwrap();
        assert!(!line.contains('\n'), "record must fit on one line");
    }

    #[test]
    fn a_new_record_carries_the_current_schema() {
        assert_eq!(sample().schema, SCHEMA_VERSION);
    }

    #[test]
    fn a_record_without_a_schema_field_reads_as_version_one() {
        // Forward compatibility with any journal written before the field existed.
        let line = r#"{"match_id":7,"home_name":"Spain","away_name":"Malta",
            "model":"Bradley-Terry","forecast":{"home_win":0.8,"draw":0.15,"away_win":0.05},
            "made_at":"2026-06-11T18:00:00Z"}"#;
        let r: ForecastRecord = serde_json::from_str(line).unwrap();
        assert_eq!(r.schema, 1);
        assert_eq!(r.match_id, MatchId(7));
    }

    #[test]
    fn the_key_is_the_match_and_the_model() {
        let a = sample();
        let mut b = sample();
        b.model = "Bradley-Terry".to_string();
        assert_ne!(a.key(), b.key(), "two models are two distinct calls");

        let mut c = sample();
        c.match_id = MatchId(43);
        assert_ne!(a.key(), c.key(), "two matches are two distinct calls");

        // The forecast and the timestamp are not part of the identity: a second call for the same
        // match and model is the *same* call, however it was computed or whenever it arrived.
        let mut d = sample();
        d.forecast = Probabilities::uniform();
        d.made_at = Utc::now();
        assert_eq!(a.key(), d.key());
    }

    #[test]
    fn the_forecast_survives_the_round_trip_intact() {
        // Scores are computed off these numbers, so a lossy encoding would corrupt the record
        // silently rather than loudly.
        let r = sample();
        let back: ForecastRecord =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.forecast.home_win, r.forecast.home_win);
        assert_eq!(back.forecast.draw, r.forecast.draw);
        assert_eq!(back.forecast.away_win, r.forecast.away_win);
    }
}
