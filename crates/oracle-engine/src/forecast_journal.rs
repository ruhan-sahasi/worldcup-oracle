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
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

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

/// An append-only, newline-delimited-JSON journal of published forecasts.
pub struct ForecastJournal {
    writer: Mutex<BufWriter<File>>,
}

impl ForecastJournal {
    /// Open the journal at `path` for appending, creating it if absent.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    /// Append one record as a JSON line and flush.
    ///
    /// # Panics
    /// If the writer mutex is poisoned, i.e. a previous caller panicked mid-append. Failing loudly
    /// matches [`EventLog::append`](crate::EventLog::append) and for the same reason: continuing to
    /// write after a torn line would produce a journal whose scores silently rest on a record nobody
    /// can read back. I/O failures are returned as `Err` for the caller to handle.
    pub fn append(&self, record: &ForecastRecord) -> io::Result<()> {
        let line = serde_json::to_string(record)?;
        let mut w = self.writer.lock().expect("forecast-journal mutex poisoned");
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        w.flush()
    }

    /// Read every record in the journal at `path`, oldest first. A missing journal reads as empty.
    ///
    /// Unparseable lines are skipped rather than failing the read, so a process killed mid-append
    /// costs at most its final record instead of the whole history. That tolerance is the right
    /// trade here - a journal that refuses to load is a track record that has been destroyed - but
    /// it does mean a corrupt line is silently dropped, so [`read_reporting`](Self::read_reporting)
    /// exists for callers that want to know.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<ForecastRecord>> {
        Ok(Self::read_reporting(path)?.0)
    }

    /// [`read`](Self::read), plus how many lines had to be skipped.
    ///
    /// Splitting this out keeps the common path simple while letting the engine log the fact that a
    /// journal was partially unreadable, rather than that fact vanishing.
    pub fn read_reporting(path: impl AsRef<Path>) -> io::Result<(Vec<ForecastRecord>, usize)> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
            Err(e) => return Err(e),
        };
        let mut records = Vec::new();
        let mut skipped = 0usize;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<ForecastRecord>(&line) {
                Ok(r) => records.push(r),
                Err(_) => skipped += 1,
            }
        }
        Ok((records, skipped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch path unique to one test, so tests can run in parallel without colliding.
    fn scratch(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("oracle_journal_{name}.jsonl"));
        let _ = std::fs::remove_file(&p);
        p
    }

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
    fn records_round_trip_through_the_journal_in_order() {
        let path = scratch("roundtrip");
        let records: Vec<ForecastRecord> = (1..=3u32)
            .map(|i| {
                ForecastRecord::new(
                    MatchId(i),
                    format!("H{i}"),
                    format!("A{i}"),
                    "Dixon-Coles ensemble",
                    Probabilities::new(0.5, 0.3, 0.2),
                )
            })
            .collect();

        let journal = ForecastJournal::create(&path).unwrap();
        for r in &records {
            journal.append(r).unwrap();
        }
        drop(journal);

        let read = ForecastJournal::read(&path).unwrap();
        assert_eq!(read, records, "records read back in the order written");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reopening_the_journal_appends_rather_than_truncates() {
        // The whole point is durability across restarts, so a second open must not clobber the
        // first session's calls.
        let path = scratch("append");
        let first = ForecastJournal::create(&path).unwrap();
        first.append(&sample()).unwrap();
        drop(first);

        let mut second_record = sample();
        second_record.match_id = MatchId(99);
        let second = ForecastJournal::create(&path).unwrap();
        second.append(&second_record).unwrap();
        drop(second);

        let read = ForecastJournal::read(&path).unwrap();
        assert_eq!(read.len(), 2, "both sessions' calls survive");
        assert_eq!(read[0].match_id, MatchId(42));
        assert_eq!(read[1].match_id, MatchId(99));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reading_a_missing_journal_is_empty() {
        let path = scratch("absent");
        assert!(ForecastJournal::read(&path).unwrap().is_empty());
        assert_eq!(ForecastJournal::read_reporting(&path).unwrap().1, 0);
    }

    #[test]
    fn a_torn_final_line_costs_only_that_record() {
        // Simulates a process killed mid-append: the last line is truncated JSON.
        let path = scratch("torn");
        let journal = ForecastJournal::create(&path).unwrap();
        journal.append(&sample()).unwrap();
        drop(journal);
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(br#"{"match_id":43,"home_name":"Spa"#).unwrap();
        }

        let (records, skipped) = ForecastJournal::read_reporting(&path).unwrap();
        assert_eq!(records.len(), 1, "the intact record still loads");
        assert_eq!(skipped, 1, "and the torn one is reported, not hidden");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn blank_lines_are_not_counted_as_corruption() {
        let path = scratch("blank");
        let journal = ForecastJournal::create(&path).unwrap();
        journal.append(&sample()).unwrap();
        drop(journal);
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(b"\n   \n").unwrap();
        }
        let (records, skipped) = ForecastJournal::read_reporting(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(skipped, 0, "whitespace is not a damaged record");
        let _ = std::fs::remove_file(&path);
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
