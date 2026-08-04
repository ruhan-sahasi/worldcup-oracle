//! A durable, append-only event log for crash recovery.
//!
//! The engine is otherwise amnesiac: a restart loses all accumulated state and would
//! replay the tournament from scratch. With an event log configured, every consumed
//! [`MatchEvent`] is appended as one JSON line, and on boot the engine replays the log to
//! rebuild its state before the live feed resumes. The log doubles as an auditable
//! recording of exactly what the engine saw.
//!
//! Format: newline-delimited JSON (one [`MatchEvent`] per line). Each append is flushed so
//! an abrupt crash loses at most the in-flight event. (A production build would offload the
//! write to a dedicated task; per-event sync flushing is fine at football-match event rates.)

use oracle_domain::{EventKind, MatchEvent, MatchStatus, Tournament};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// An append-only, newline-delimited-JSON log of match events.
pub struct EventLog {
    writer: Mutex<BufWriter<File>>,
}

impl EventLog {
    /// Open the log at `path` for appending, creating it if absent.
    ///
    /// # Errors
    /// If the path cannot be opened for appending - a missing parent directory, or no write
    /// permission. Worth propagating rather than defaulting to no log: an engine that silently ran
    /// without its recovery record would look healthy right up to the restart that lost everything.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    /// Append one event as a JSON line and flush.
    ///
    /// # Errors
    /// If serialization or the write fails. A failed append means this event is not in the recovery
    /// record, so the caller has to decide whether to continue - which is why it is an `Err` and not
    /// a panic.
    ///
    /// # Panics
    /// If the writer mutex is poisoned, i.e. a previous caller panicked mid-append. Failing loudly
    /// is deliberate here: the log is the engine's recovery record, and continuing to append after
    /// a torn write would silently produce a log that cannot be replayed. I/O failures, by
    /// contrast, are returned as `Err` for the caller to handle.
    pub fn append(&self, event: &MatchEvent) -> io::Result<()> {
        let line = serde_json::to_string(event)?;
        let mut w = self.writer.lock().expect("event-log mutex poisoned");
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        w.flush()
    }

    /// Read all events previously written to the log at `path` (oldest first). Malformed
    /// lines are skipped so a partially-written final line never blocks recovery.
    ///
    /// # Errors
    /// If the file exists but cannot be read. A *missing* file is not an error - it is a first run,
    /// and returns an empty log.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<MatchEvent>> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut events = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(ev) = serde_json::from_str::<MatchEvent>(&line) {
                events.push(ev);
            }
        }
        Ok(events)
    }
}

/// Mark every match the events report a full-time score for as finished, and return how many were
/// applied.
///
/// Deliberately the smallest possible replay: only [`EventKind::FullTime`] is read, and no model is
/// touched. Settling a published forecast against its result needs the result and nothing else, and
/// rebuilding engine state here would reintroduce exactly the recomputation the forecast journal
/// exists to avoid.
///
/// Events naming a match the tournament does not contain are skipped, so a log that outlived a
/// tournament definition still applies what it can.
pub fn apply_results(tournament: &mut Tournament, events: &[MatchEvent]) -> usize {
    let mut applied = 0;
    for event in events {
        if let EventKind::FullTime { score } = event.kind {
            if let Some(m) = tournament
                .matches
                .iter_mut()
                .find(|m| m.id == event.match_id)
            {
                m.status = MatchStatus::Finished;
                m.score = score;
                applied += 1;
            }
        }
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_domain::{EventKind, MatchId, Scoreline, TeamId};

    #[test]
    fn round_trips_events_through_the_log() {
        let path = std::env::temp_dir().join("oracle_event_log_roundtrip.jsonl");
        let _ = std::fs::remove_file(&path);

        let events = vec![
            MatchEvent::new(MatchId(1), 0, EventKind::KickOff),
            MatchEvent::new(
                MatchId(1),
                23,
                EventKind::Goal {
                    team: TeamId(7),
                    scorer: None,
                },
            ),
            MatchEvent::new(
                MatchId(1),
                90,
                EventKind::FullTime {
                    score: Scoreline::new(0, 1),
                },
            ),
        ];

        let log = EventLog::create(&path).unwrap();
        for e in &events {
            log.append(e).unwrap();
        }
        drop(log);

        let read = EventLog::read(&path).unwrap();
        assert_eq!(read.len(), events.len());
        // MatchEvent is not Eq (odds carry f64), so compare structurally via JSON.
        for (a, b) in events.iter().zip(&read) {
            assert_eq!(
                serde_json::to_string(a).unwrap(),
                serde_json::to_string(b).unwrap()
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn applying_results_finishes_the_matches_it_recognises() {
        use oracle_domain::{Confederation, Match, MatchId, Stage, Team, TeamId};
        let mut t = Tournament::new("Test Cup");
        for i in 0..2u32 {
            t.teams.push(Team::new(
                i,
                format!("T{i}"),
                format!("{i:03}"),
                Confederation::Uefa,
            ));
        }
        t.matches.push(Match {
            id: MatchId(1),
            home: TeamId(0),
            away: TeamId(1),
            stage: Stage::Group('A'),
            kickoff: chrono::DateTime::from_timestamp(0, 0).unwrap(),
            status: MatchStatus::Scheduled,
            score: Scoreline::new(0, 0),
        });

        let events = vec![
            // Not a result, so it must not finish anything.
            MatchEvent::new(MatchId(1), 0, EventKind::KickOff),
            MatchEvent::new(
                MatchId(1),
                90,
                EventKind::FullTime {
                    score: Scoreline::new(2, 1),
                },
            ),
            // A match this tournament does not contain: skipped, not counted, no panic.
            MatchEvent::new(
                MatchId(999),
                90,
                EventKind::FullTime {
                    score: Scoreline::new(0, 0),
                },
            ),
        ];
        assert_eq!(apply_results(&mut t, &events), 1);
        assert!(t.matches[0].is_finished());
        assert_eq!(t.matches[0].score, Scoreline::new(2, 1));
    }

    #[test]
    fn reading_a_missing_log_is_empty() {
        let path = std::env::temp_dir().join("oracle_event_log_does_not_exist.jsonl");
        let _ = std::fs::remove_file(&path);
        assert!(EventLog::read(&path).unwrap().is_empty());
    }
}
