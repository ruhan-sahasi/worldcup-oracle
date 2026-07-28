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

use oracle_domain::MatchEvent;
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
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    /// Append one event as a JSON line and flush.
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
    fn reading_a_missing_log_is_empty() {
        let path = std::env::temp_dir().join("oracle_event_log_does_not_exist.jsonl");
        let _ = std::fs::remove_file(&path);
        assert!(EventLog::read(&path).unwrap().is_empty());
    }
}
