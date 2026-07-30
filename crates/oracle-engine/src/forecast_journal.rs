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
use oracle_domain::{MatchId, Outcome, Probabilities, Scoreline, Tournament};
use oracle_model::{reliability, score, CalibrationReport, ReliabilityReport};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
///
/// Writes are **idempotent on `(match, model)`**: the first call journaled for a pair is the one
/// that stands, and any later call for the same pair is refused. See
/// [`append_if_new`](Self::append_if_new).
pub struct ForecastJournal {
    writer: Mutex<BufWriter<File>>,
    /// Keys already on disk or written this session, so a repeat is recognised without re-reading.
    seen: Mutex<HashSet<(MatchId, String)>>,
}

impl ForecastJournal {
    /// Open the journal at `path` for appending, creating it if absent.
    ///
    /// Existing records are read first so their keys are known: a restart must recognise the calls
    /// it already published, or replaying the event log would duplicate every one of them.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let seen = Self::read(&path)?
            .into_iter()
            .map(|r| (r.match_id, r.model))
            .collect();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
            seen: Mutex::new(seen),
        })
    }

    /// Journal a call only if this `(match, model)` has never been journaled. Returns whether it was
    /// written.
    ///
    /// This is the operation that makes the record immutable, and it is the one the engine uses. A
    /// forecast is journaled when a result lands; replaying the event log walks that same result
    /// again, and so would a re-delivered event from a flaky feed. Without a first-write-wins rule,
    /// each of those would append another copy - inflating the sample the scores are averaged over
    /// and, worse, letting a *newly recomputed* forecast for an old match enter the record, which is
    /// exactly the retroactive rewriting the journal exists to prevent.
    ///
    /// First write wins rather than last, because the first call is the one that was made without
    /// knowing the result. A later one is at best a recomputation and at worst contaminated.
    ///
    /// # Panics
    /// If either internal mutex is poisoned; see [`append`](Self::append).
    pub fn append_if_new(&self, record: &ForecastRecord) -> io::Result<bool> {
        {
            let seen = self.seen.lock().expect("forecast-journal mutex poisoned");
            if seen.contains(&(record.match_id, record.model.clone())) {
                return Ok(false);
            }
        }
        self.append(record)?;
        Ok(true)
    }

    /// How many distinct calls the journal holds.
    ///
    /// # Panics
    /// If the internal mutex is poisoned; see [`append`](Self::append).
    pub fn len(&self) -> usize {
        self.seen
            .lock()
            .expect("forecast-journal mutex poisoned")
            .len()
    }

    /// Whether the journal holds no calls yet.
    ///
    /// # Panics
    /// If the internal mutex is poisoned; see [`append`](Self::append).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append one record as a JSON line and flush, unconditionally.
    ///
    /// Prefer [`append_if_new`](Self::append_if_new); this is the lower-level write and does not
    /// enforce first-write-wins.
    ///
    /// # Panics
    /// If the writer mutex is poisoned, i.e. a previous caller panicked mid-append. Failing loudly
    /// matches [`EventLog::append`](crate::EventLog::append) and for the same reason: continuing to
    /// write after a torn line would produce a journal whose scores silently rest on a record nobody
    /// can read back. I/O failures are returned as `Err` for the caller to handle.
    pub fn append(&self, record: &ForecastRecord) -> io::Result<()> {
        let line = serde_json::to_string(record)?;
        {
            let mut w = self.writer.lock().expect("forecast-journal mutex poisoned");
            w.write_all(line.as_bytes())?;
            w.write_all(b"\n")?;
            w.flush()?;
        }
        self.seen
            .lock()
            .expect("forecast-journal mutex poisoned")
            .insert((record.match_id, record.model.clone()));
        Ok(())
    }

    /// Read every record in the journal at `path`, oldest first. A missing journal reads as empty.
    ///
    /// Two kinds of line are skipped rather than failing the read: unparseable ones, and records
    /// whose [`schema`](ForecastRecord::schema) is newer than this build understands. A process
    /// killed mid-append then costs at most its final record instead of the whole history, which is
    /// the right trade - a journal that refuses to load is a track record that has been destroyed.
    ///
    /// Skipping is silent by necessity here, which is why [`read_reporting`](Self::read_reporting)
    /// exists: the count is the only evidence that part of the record did not load.
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
                // A record from a future schema parsed into today's shape is the dangerous case: the
                // fields present still deserialize, so it looks valid while whatever the new version
                // added - a different probability basis, a market adjustment, a retraction flag - is
                // silently dropped. Scoring it would produce plausible, wrong numbers. Refusing it
                // costs a call and keeps the rest of the record trustworthy.
                Ok(r) if r.schema > SCHEMA_VERSION => skipped += 1,
                Ok(r) => records.push(r),
                Err(_) => skipped += 1,
            }
        }
        Ok((records, skipped))
    }
}

/// A journaled call paired with the result that settled it.
///
/// Settlement is deliberately a separate step from journaling. A call is written when made and the
/// result arrives later, so the join happens at scoring time against whatever results are known -
/// which also means an unfinished match simply has no settled call yet, rather than needing a
/// placeholder in the journal.
#[derive(Debug, Clone, PartialEq)]
pub struct SettledForecast {
    pub record: ForecastRecord,
    /// What actually happened.
    pub actual: Outcome,
    /// The final scoreline, kept for display alongside the call.
    pub score: Scoreline,
}

impl SettledForecast {
    /// The `(forecast, outcome)` pair the scoring functions take.
    pub fn pair(&self) -> (Probabilities, Outcome) {
        (self.record.forecast, self.actual)
    }

    /// Whether the call's most likely outcome was the one that happened.
    pub fn called_correctly(&self) -> bool {
        self.record.forecast.most_likely() == self.actual
    }

    /// The probability the call assigned to its own most likely outcome.
    pub fn confidence(&self) -> f64 {
        let f = self.record.forecast;
        f.of(f.most_likely())
    }
}

/// Pair each journaled call with its match's result, dropping calls whose match is unknown or not
/// yet finished.
///
/// Silently dropping is the correct behaviour rather than a compromise. A journal legitimately
/// contains calls on matches that have not been played, and it may outlive a tournament definition
/// entirely - the point of storing team names on the record is that such a call stays *readable*
/// even when it can no longer be *scored*. What must never happen is a call being scored against
/// the wrong match, which is why the join is on match id and nothing else.
pub fn settle(records: &[ForecastRecord], tournament: &Tournament) -> Vec<SettledForecast> {
    let finished: HashMap<MatchId, Scoreline> = tournament
        .matches
        .iter()
        .filter(|m| m.is_finished())
        .map(|m| (m.id, m.score))
        .collect();
    records
        .iter()
        .filter_map(|r| {
            let score = *finished.get(&r.match_id)?;
            Some(SettledForecast {
                record: r.clone(),
                actual: score.outcome(),
                score,
            })
        })
        .collect()
}

/// One forecaster's record over its journaled, settled calls.
#[derive(Debug, Clone, Serialize)]
pub struct JournalScore {
    pub model: String,
    /// Settled calls behind these numbers.
    pub scored: usize,
    pub winners_called: usize,
    pub accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
}

/// Score the settled calls of every model in a journal, best Brier first.
///
/// The scoring itself is [`oracle_model::score`], the same proper scoring rules the backtest and the
/// existing report card use. That reuse is the point: a track record computed by its own private
/// arithmetic could drift away from the numbers the rest of the project quotes, and the two would
/// disagree with no way to tell which was wrong.
///
/// Grouping is by model name, so two forecasters journaled against the same matches are directly
/// comparable. Ordering is by Brier ascending because lower is better - and Brier rather than
/// accuracy, since accuracy throws away everything the probabilities said and rewards a model for
/// being confidently right on easy matches.
pub fn score_by_model(settled: &[SettledForecast]) -> Vec<JournalScore> {
    let mut by_model: HashMap<&str, Vec<&SettledForecast>> = HashMap::new();
    for s in settled {
        by_model.entry(s.record.model.as_str()).or_default().push(s);
    }
    let mut scores: Vec<JournalScore> = by_model
        .into_iter()
        .map(|(model, calls)| {
            let pairs: Vec<(Probabilities, Outcome)> = calls.iter().map(|c| c.pair()).collect();
            let report = score(&pairs);
            JournalScore {
                model: model.to_string(),
                scored: pairs.len(),
                winners_called: calls.iter().filter(|c| c.called_correctly()).count(),
                accuracy: report.accuracy,
                brier: report.brier,
                log_loss: report.log_loss,
            }
        })
        .collect();
    scores.sort_by(|a, b| {
        a.brier
            .partial_cmp(&b.brier)
            .unwrap_or(std::cmp::Ordering::Equal)
            // A stable tie-break keeps the order deterministic when two models score identically,
            // which they do when a journal is small or the models agree.
            .then_with(|| a.model.cmp(&b.model))
    });
    scores
}

/// The engine's published track record: how the forecasts it actually made have held up.
///
/// Distinct from [`ReportCard`](crate::ReportCard), which re-derives its forecasts from current
/// model state. Every number here comes from journaled calls, so it cannot move except by playing
/// more matches.
#[derive(Debug, Clone, Serialize)]
pub struct TrackRecord {
    /// Journaled calls, including those whose match has not been played.
    pub calls: usize,
    /// Calls settled against a result - the sample the scores below rest on.
    pub settled: usize,
    /// Distinct matches covered by at least one settled call.
    pub matches: usize,
    /// One row per forecaster, best Brier first.
    pub models: Vec<JournalScore>,
    /// The naive uniform baseline over the same number of calls, for context.
    pub baseline_brier: f64,
    /// The leading model's reliability curve over its own journaled calls.
    pub reliability: ReliabilityReport,
    /// When the earliest and latest surviving calls were made. `None` for an empty journal.
    pub first_call: Option<DateTime<Utc>>,
    pub last_call: Option<DateTime<Utc>>,
    /// Journal lines that could not be parsed. Non-zero means part of the record is unreadable, and
    /// the scores above are over what remains.
    pub unreadable_lines: usize,
}

/// Build the track record from journaled calls and the results known so far.
///
/// `unreadable_lines` is threaded through from the read rather than recomputed, because a track
/// record that quietly omits the fact that some of it failed to load would be exactly the kind of
/// flattering silence this whole module exists to remove.
pub fn track_record(
    records: &[ForecastRecord],
    tournament: &Tournament,
    unreadable_lines: usize,
) -> TrackRecord {
    let settled = settle(records, tournament);
    let models = score_by_model(&settled);

    // The reliability curve is the leading model's, since a curve pooled across forecasters would
    // describe no single one of them.
    let leader_pairs: Vec<(Probabilities, Outcome)> = match models.first() {
        Some(best) => settled
            .iter()
            .filter(|s| s.record.model == best.model)
            .map(|s| s.pair())
            .collect(),
        None => Vec::new(),
    };

    let distinct_matches: HashSet<MatchId> = settled.iter().map(|s| s.record.match_id).collect();
    let made_at = |pick: fn(&mut dyn Iterator<Item = DateTime<Utc>>) -> Option<DateTime<Utc>>| {
        pick(&mut records.iter().map(|r| r.made_at))
    };

    TrackRecord {
        calls: records.len(),
        settled: settled.len(),
        matches: distinct_matches.len(),
        baseline_brier: CalibrationReport::uniform_baseline(settled.len().max(1)).brier,
        models,
        reliability: reliability(&leader_pairs, 10),
        first_call: made_at(|it| it.min()),
        last_call: made_at(|it| it.max()),
        unreadable_lines,
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

    /// A two-match tournament: match 1 finished 2-0 to the home side, match 2 still scheduled.
    fn tournament_with_one_result() -> Tournament {
        use oracle_domain::{Match, MatchStatus, Stage, Team};
        let mut t = Tournament::new("Test Cup");
        for i in 0..4u32 {
            t.teams.push(Team::new(
                i,
                format!("T{i}"),
                format!("{i:03}"),
                oracle_domain::Confederation::Uefa,
            ));
        }
        let kickoff = chrono::DateTime::from_timestamp(0, 0).unwrap();
        t.matches.push(Match {
            id: MatchId(1),
            home: oracle_domain::TeamId(0),
            away: oracle_domain::TeamId(1),
            stage: Stage::Group('A'),
            kickoff,
            status: MatchStatus::Finished,
            score: Scoreline::new(2, 0),
        });
        t.matches.push(Match {
            id: MatchId(2),
            home: oracle_domain::TeamId(2),
            away: oracle_domain::TeamId(3),
            stage: Stage::Group('A'),
            kickoff,
            status: MatchStatus::Scheduled,
            score: Scoreline::new(0, 0),
        });
        t
    }

    fn call_on(match_id: u32, forecast: Probabilities) -> ForecastRecord {
        ForecastRecord::new(MatchId(match_id), "H", "A", "m", forecast)
    }

    #[test]
    fn settlement_pairs_a_call_with_its_result() {
        let t = tournament_with_one_result();
        let calls = vec![call_on(1, Probabilities::new(0.7, 0.2, 0.1))];
        let settled = settle(&calls, &t);
        assert_eq!(settled.len(), 1);
        assert_eq!(settled[0].actual, Outcome::HomeWin, "2-0 is a home win");
        assert_eq!(settled[0].score, Scoreline::new(2, 0));
        assert_eq!(settled[0].pair(), (calls[0].forecast, Outcome::HomeWin));
    }

    #[test]
    fn settlement_skips_unplayed_and_unknown_matches() {
        let t = tournament_with_one_result();
        let calls = vec![
            call_on(1, Probabilities::new(0.7, 0.2, 0.1)), // finished
            call_on(2, Probabilities::new(0.4, 0.3, 0.3)), // scheduled
            call_on(999, Probabilities::uniform()),        // not in this tournament at all
        ];
        let settled = settle(&calls, &t);
        assert_eq!(settled.len(), 1, "only the finished match settles");
        assert_eq!(settled[0].record.match_id, MatchId(1));
    }

    #[test]
    fn settlement_reads_a_call_as_correct_or_not() {
        let t = tournament_with_one_result();
        // Match 1 was a home win.
        let right = settle(&[call_on(1, Probabilities::new(0.7, 0.2, 0.1))], &t);
        assert!(right[0].called_correctly());
        assert!((right[0].confidence() - 0.7).abs() < 1e-12);

        let wrong = settle(&[call_on(1, Probabilities::new(0.1, 0.2, 0.7))], &t);
        assert!(!wrong[0].called_correctly());
        assert!(
            (wrong[0].confidence() - 0.7).abs() < 1e-12,
            "confidence is in its own pick, right or wrong"
        );
    }

    #[test]
    fn settlement_of_an_empty_journal_is_empty() {
        assert!(settle(&[], &tournament_with_one_result()).is_empty());
    }

    /// A journal where `model` calls every match with `p_home`, over a tournament of `n` home wins.
    fn calls_by(model: &str, p_home: f64, ids: &[u32]) -> Vec<ForecastRecord> {
        ids.iter()
            .map(|&i| {
                ForecastRecord::new(
                    MatchId(i),
                    "H",
                    "A",
                    model,
                    Probabilities::new(p_home, (1.0 - p_home) / 2.0, (1.0 - p_home) / 2.0),
                )
            })
            .collect()
    }

    #[test]
    fn scoring_groups_by_model_and_ranks_the_sharper_one_first() {
        let t = tournament_with_one_result(); // match 1 was a home win
        let mut records = calls_by("confident", 0.9, &[1]);
        records.extend(calls_by("timid", 0.4, &[1]));
        let scores = score_by_model(&settle(&records, &t));

        assert_eq!(scores.len(), 2, "one row per model");
        // Both called the home win; the confident one is better calibrated to what happened, so it
        // takes a lower Brier and sorts first.
        assert_eq!(scores[0].model, "confident");
        assert!(scores[0].brier < scores[1].brier);
        for s in &scores {
            assert_eq!(s.scored, 1);
            assert_eq!(s.winners_called, 1);
            assert!((s.accuracy - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn scoring_counts_only_settled_calls() {
        let t = tournament_with_one_result(); // match 2 is unplayed
        let records = calls_by("m", 0.6, &[1, 2, 999]);
        let scores = score_by_model(&settle(&records, &t));
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].scored, 1, "only the finished match counts");
    }

    #[test]
    fn scoring_matches_the_shared_calibration_function() {
        // The track record must not develop its own arithmetic; it has to agree with the scoring the
        // backtest and report card already use, exactly.
        let t = tournament_with_one_result();
        let records = calls_by("m", 0.75, &[1]);
        let settled = settle(&records, &t);
        let scores = score_by_model(&settled);
        let direct = score(&[(records[0].forecast, Outcome::HomeWin)]);
        assert_eq!(scores[0].brier, direct.brier);
        assert_eq!(scores[0].log_loss, direct.log_loss);
        assert_eq!(scores[0].accuracy, direct.accuracy);
    }

    #[test]
    fn a_wrong_call_is_scored_as_wrong() {
        let t = tournament_with_one_result(); // home win
        let records = vec![call_on(1, Probabilities::new(0.1, 0.2, 0.7))];
        let scores = score_by_model(&settle(&records, &t));
        assert_eq!(scores[0].winners_called, 0);
        assert!(scores[0].accuracy.abs() < 1e-12);
    }

    #[test]
    fn scoring_an_empty_journal_yields_no_rows() {
        assert!(score_by_model(&[]).is_empty());
    }

    #[test]
    fn tied_models_keep_a_deterministic_order() {
        // Identical forecasts score identically; the order must not depend on hash iteration.
        let t = tournament_with_one_result();
        let mut records = calls_by("zeta", 0.6, &[1]);
        records.extend(calls_by("alpha", 0.6, &[1]));
        for _ in 0..5 {
            let scores = score_by_model(&settle(&records, &t));
            assert_eq!(scores[0].model, "alpha", "ties break by name");
            assert_eq!(scores[1].model, "zeta");
        }
    }

    #[test]
    fn a_track_record_summarises_the_journal() {
        let t = tournament_with_one_result();
        let mut records = calls_by("sharp", 0.9, &[1]);
        records.extend(calls_by("blunt", 0.4, &[1]));
        // A call on an unplayed match: counted as a call, not as settled.
        records.extend(calls_by("sharp", 0.5, &[2]));

        let tr = track_record(&records, &t, 0);
        assert_eq!(tr.calls, 3, "every journaled call");
        assert_eq!(tr.settled, 2, "only those with a result");
        assert_eq!(tr.matches, 1, "both settled calls are on match 1");
        assert_eq!(tr.models.len(), 2);
        assert_eq!(tr.models[0].model, "sharp", "ranked by Brier");
        assert!(tr.first_call.is_some() && tr.last_call.is_some());
        assert!(tr.last_call >= tr.first_call);
        assert_eq!(tr.unreadable_lines, 0);
    }

    #[test]
    fn a_track_record_beats_the_baseline_when_the_model_is_right() {
        let t = tournament_with_one_result();
        let tr = track_record(&calls_by("sharp", 0.9, &[1]), &t, 0);
        assert!(
            tr.models[0].brier < tr.baseline_brier,
            "a confident correct call must beat uniform: {} vs {}",
            tr.models[0].brier,
            tr.baseline_brier
        );
    }

    #[test]
    fn an_empty_journal_yields_an_honest_empty_record() {
        let tr = track_record(&[], &tournament_with_one_result(), 0);
        assert_eq!(tr.calls, 0);
        assert_eq!(tr.settled, 0);
        assert_eq!(tr.matches, 0);
        assert!(tr.models.is_empty());
        assert!(tr.first_call.is_none() && tr.last_call.is_none());
        // The baseline is still reported rather than dividing by zero.
        assert!(tr.baseline_brier.is_finite());
    }

    #[test]
    fn unreadable_lines_are_surfaced_not_swallowed() {
        // The count comes from the read, and a track record must admit part of it failed to load.
        let tr = track_record(&calls_by("m", 0.6, &[1]), &tournament_with_one_result(), 3);
        assert_eq!(tr.unreadable_lines, 3);
    }

    #[test]
    fn the_reliability_curve_describes_the_leading_model_alone() {
        // A curve pooled across forecasters would describe none of them. With only the blunt model's
        // calls settled, the curve must reflect the leader's, not an average.
        let t = tournament_with_one_result();
        let mut records = calls_by("sharp", 0.9, &[1]);
        records.extend(calls_by("blunt", 0.4, &[1]));
        let tr = track_record(&records, &t, 0);
        let leader_only = reliability(&[(records[0].forecast, Outcome::HomeWin)], 10);
        assert_eq!(tr.reliability.ece, leader_only.ece);
    }

    #[test]
    fn a_record_from_a_future_schema_is_refused_rather_than_mis_scored() {
        // The dangerous case. A future record's known fields still deserialize, so it looks valid
        // while whatever the new version added is dropped - and scoring it would give plausible,
        // wrong numbers. Refusing costs one call and keeps the rest of the record trustworthy.
        let path = scratch("future_schema");
        let journal = ForecastJournal::create(&path).unwrap();
        journal.append(&sample()).unwrap();
        drop(journal);

        let mut future = sample();
        future.match_id = MatchId(43);
        future.schema = SCHEMA_VERSION + 1;
        let mut line = serde_json::to_string(&future).unwrap();
        line.push('\n');
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(line.as_bytes()).unwrap();
        }

        let (records, skipped) = ForecastJournal::read_reporting(&path).unwrap();
        assert_eq!(records.len(), 1, "only the readable record loads");
        assert_eq!(records[0].match_id, MatchId(42));
        assert_eq!(skipped, 1, "and the future record is reported");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_record_from_the_current_or_an_older_schema_loads() {
        let path = scratch("old_schema");
        let journal = ForecastJournal::create(&path).unwrap();
        journal.append(&sample()).unwrap();
        drop(journal);
        {
            // A version-0 record, as though written before the field was introduced.
            let mut older = sample();
            older.match_id = MatchId(44);
            older.schema = 0;
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(serde_json::to_string(&older).unwrap().as_bytes())
                .unwrap();
            f.write_all(b"\n").unwrap();
        }
        let (records, skipped) = ForecastJournal::read_reporting(&path).unwrap();
        assert_eq!(records.len(), 2, "old records stay readable");
        assert_eq!(skipped, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_future_record_does_not_reserve_its_key() {
        // A refused record must not block the key, or a downgrade would leave that match
        // permanently unjournalable while looking fine.
        let path = scratch("future_key");
        let mut future = sample();
        future.schema = SCHEMA_VERSION + 1;
        {
            let journal = ForecastJournal::create(&path).unwrap();
            journal.append(&future).unwrap();
        }
        let reopened = ForecastJournal::create(&path).unwrap();
        assert_eq!(reopened.len(), 0, "the future record is not a known key");
        assert!(
            reopened.append_if_new(&sample()).unwrap(),
            "so the call can still be journaled by this build"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_repeated_call_is_refused_and_the_first_one_stands() {
        let path = scratch("idempotent");
        let journal = ForecastJournal::create(&path).unwrap();

        let first = sample();
        assert!(journal.append_if_new(&first).unwrap(), "first write lands");

        // The same match and model, but a different forecast - what a recomputation by a changed
        // model would produce. It must not enter the record.
        let mut recomputed = sample();
        recomputed.forecast = Probabilities::new(0.1, 0.2, 0.7);
        assert!(
            !journal.append_if_new(&recomputed).unwrap(),
            "a second call for the same match and model is refused"
        );
        assert_eq!(journal.len(), 1);
        drop(journal);

        let read = ForecastJournal::read(&path).unwrap();
        assert_eq!(read.len(), 1, "only one line on disk");
        assert_eq!(
            read[0].forecast, first.forecast,
            "the original call is what survived, not the recomputation"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_restart_recognises_the_calls_it_already_published() {
        // The replay case. A restart reopens the journal and walks the same results again; without
        // loading existing keys it would duplicate every call it had ever made.
        let path = scratch("restart");
        let first = ForecastJournal::create(&path).unwrap();
        assert!(first.append_if_new(&sample()).unwrap());
        drop(first);

        let reopened = ForecastJournal::create(&path).unwrap();
        assert_eq!(reopened.len(), 1, "prior calls are known on open");
        assert!(
            !reopened.append_if_new(&sample()).unwrap(),
            "a replayed result does not re-journal its call"
        );
        drop(reopened);

        assert_eq!(ForecastJournal::read(&path).unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_models_journal_the_same_match_independently() {
        // The key is (match, model), so the second forecaster's call on a match is a distinct call,
        // not a duplicate of the first's.
        let path = scratch("two_models");
        let journal = ForecastJournal::create(&path).unwrap();
        let ensemble = sample();
        let mut bt = sample();
        bt.model = "Bradley-Terry".to_string();

        assert!(journal.append_if_new(&ensemble).unwrap());
        assert!(journal.append_if_new(&bt).unwrap(), "a different model");
        assert_eq!(journal.len(), 2);
        drop(journal);

        assert_eq!(ForecastJournal::read(&path).unwrap().len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_empty_journal_reports_itself_empty() {
        let path = scratch("empty");
        let journal = ForecastJournal::create(&path).unwrap();
        assert!(journal.is_empty());
        assert_eq!(journal.len(), 0);
        journal.append_if_new(&sample()).unwrap();
        assert!(!journal.is_empty());
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
