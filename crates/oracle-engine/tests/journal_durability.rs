//! The property the forecast journal exists for: a published call survives a model change.
//!
//! This is an integration test rather than a unit test because the claim is about two *separate
//! processes* - a run that publishes, and a later run whose model has changed - and the failure it
//! guards is that the second one silently rewrites the first one's history. A unit test with a
//! hand-planted journal can check the preference is wired up; only a test that actually replays a
//! recorded event log through a differently-fit engine can check the outcome.
//!
//! The scenario is not hypothetical. Earlier work on this repo changed every synthetic value the
//! baseline model is fit from, which moves every forecast a replay recomputes. Before the journal,
//! that would have shifted the published scorecard with no commit saying so.

use oracle_domain::{EventKind, MatchEvent, MatchId, Scoreline};
use oracle_engine::{
    settle, track_record, EventLog, ForecastJournal, ForecastRecord, ENSEMBLE_MODEL,
};
use oracle_ingest::data;

/// A scratch path unique to one test.
fn scratch(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("oracle_durability_{name}.jsonl"));
    let _ = std::fs::remove_file(&p);
    p
}

/// Full-time events for the first `n` fixtures, alternating home wins and away wins so the results
/// are not all one outcome (which would make every scoring rule agree trivially).
fn results_for(n: usize) -> Vec<(MatchId, Scoreline)> {
    data::world_cup_2026()
        .matches
        .iter()
        .take(n)
        .enumerate()
        .map(|(i, m)| {
            let score = if i % 2 == 0 {
                Scoreline::new(2, 0)
            } else {
                Scoreline::new(0, 1)
            };
            (m.id, score)
        })
        .collect()
}

#[test]
fn a_published_call_is_scored_as_published_after_the_model_changes() {
    let journal_path = scratch("journal");
    let log_path = scratch("log");
    let results = results_for(8);

    // ---- Run one: publish calls and record the events. ----
    // The forecasts are stand-ins for "whatever the model said at the time"; what matters is that
    // they are on disk and that a later, differently-fit model would not reproduce them.
    let published: Vec<ForecastRecord> = {
        let journal = ForecastJournal::create(&journal_path).unwrap();
        let log = EventLog::create(&log_path).unwrap();
        let tournament = data::world_cup_2026();

        for (i, (id, score)) in results.iter().enumerate() {
            let m = tournament.matches.iter().find(|m| m.id == *id).unwrap();
            // A deliberately mediocre forecaster: mildly home-leaning on every match.
            let forecast = oracle_domain::Probabilities::new(0.45, 0.30, 0.25);
            journal
                .append_if_new(&ForecastRecord::new(
                    *id,
                    format!("team{}", m.home.0),
                    format!("team{}", m.away.0),
                    ENSEMBLE_MODEL,
                    forecast,
                ))
                .unwrap();
            log.append(&MatchEvent::new(*id, 0, EventKind::KickOff))
                .unwrap();
            log.append(&MatchEvent::new(
                *id,
                90,
                EventKind::FullTime { score: *score },
            ))
            .unwrap();
            assert_eq!(journal.len(), i + 1);
        }
        ForecastJournal::read(&journal_path).unwrap()
    };
    assert_eq!(published.len(), 8, "eight calls published");

    // Score run one's record.
    let settled_tournament = {
        let mut t = data::world_cup_2026();
        for (id, score) in &results {
            let m = t.matches.iter_mut().find(|m| m.id == *id).unwrap();
            m.status = oracle_domain::MatchStatus::Finished;
            m.score = *score;
        }
        t
    };
    let before = track_record(&published, &settled_tournament, 0);
    assert_eq!(before.settled, 8);
    let brier_before = before.models[0].brier;

    // ---- Run two: the model has changed. ----
    // Simulated by a forecaster that would have called every match very differently. It reopens the
    // same journal and walks the same results.
    {
        let journal = ForecastJournal::create(&journal_path).unwrap();
        assert_eq!(
            journal.len(),
            8,
            "the reopened journal knows the published calls"
        );
        for (id, _) in &results {
            // What the new model would say: confident away wins, the opposite lean.
            let rewritten = ForecastRecord::new(
                *id,
                "x",
                "y",
                ENSEMBLE_MODEL,
                oracle_domain::Probabilities::new(0.05, 0.10, 0.85),
            );
            assert!(
                !journal.append_if_new(&rewritten).unwrap(),
                "a changed model must not be able to republish an existing call"
            );
        }
    }

    // ---- The record is unchanged. ----
    let after_records = ForecastJournal::read(&journal_path).unwrap();
    assert_eq!(
        after_records, published,
        "the journal is byte-identical after the second run"
    );
    let after = track_record(&after_records, &settled_tournament, 0);
    assert_eq!(after.settled, 8, "no calls were added or lost");
    assert_eq!(
        after.models[0].brier, brier_before,
        "the published Brier did not move"
    );

    // And the guard is meaningful: the new model's calls really would have scored differently.
    let rewritten_only: Vec<ForecastRecord> = results
        .iter()
        .map(|(id, _)| {
            ForecastRecord::new(
                *id,
                "x",
                "y",
                ENSEMBLE_MODEL,
                oracle_domain::Probabilities::new(0.05, 0.10, 0.85),
            )
        })
        .collect();
    let counterfactual = track_record(&rewritten_only, &settled_tournament, 0);
    assert!(
        (counterfactual.models[0].brier - brier_before).abs() > 0.05,
        "the test proves nothing unless the two models genuinely disagree: {} vs {}",
        counterfactual.models[0].brier,
        brier_before
    );

    let _ = std::fs::remove_file(&journal_path);
    let _ = std::fs::remove_file(&log_path);
}

#[test]
fn the_record_is_reconstructible_from_the_two_files_alone() {
    // The offline property the CLI relies on: calls from the journal, results from the event log, no
    // engine and no model fitting anywhere.
    let journal_path = scratch("offline_journal");
    let log_path = scratch("offline_log");
    let results = results_for(6);

    {
        let journal = ForecastJournal::create(&journal_path).unwrap();
        let log = EventLog::create(&log_path).unwrap();
        for (id, score) in &results {
            journal
                .append_if_new(&ForecastRecord::new(
                    *id,
                    "H",
                    "A",
                    ENSEMBLE_MODEL,
                    oracle_domain::Probabilities::new(0.6, 0.25, 0.15),
                ))
                .unwrap();
            log.append(&MatchEvent::new(
                *id,
                90,
                EventKind::FullTime { score: *score },
            ))
            .unwrap();
        }
    }

    // Rebuild using only the files.
    let records = ForecastJournal::read(&journal_path).unwrap();
    let mut tournament = data::world_cup_2026();
    let mut applied = 0;
    for event in EventLog::read(&log_path).unwrap() {
        if let EventKind::FullTime { score } = event.kind {
            if let Some(m) = tournament
                .matches
                .iter_mut()
                .find(|m| m.id == event.match_id)
            {
                m.status = oracle_domain::MatchStatus::Finished;
                m.score = score;
                applied += 1;
            }
        }
    }
    assert_eq!(applied, 6, "every logged result found its fixture");

    let settled = settle(&records, &tournament);
    assert_eq!(settled.len(), 6, "every call settled");
    let tr = track_record(&records, &tournament, 0);
    assert_eq!(tr.calls, 6);
    assert_eq!(tr.settled, 6);
    assert_eq!(tr.matches, 6);
    assert_eq!(tr.models.len(), 1);
    // Alternating results, a 60% home lean: three right, three wrong.
    assert_eq!(tr.models[0].winners_called, 3);

    let _ = std::fs::remove_file(&journal_path);
    let _ = std::fs::remove_file(&log_path);
}

#[test]
fn a_journal_and_log_that_disagree_settle_only_their_overlap() {
    // The files are written independently, so they can legitimately diverge: a call whose match has
    // not finished, and a result for a match nobody forecast. Neither may be scored.
    let journal_path = scratch("overlap_journal");
    let log_path = scratch("overlap_log");
    let all = results_for(6);

    {
        let journal = ForecastJournal::create(&journal_path).unwrap();
        // Calls on the first four fixtures.
        for (id, _) in all.iter().take(4) {
            journal
                .append_if_new(&ForecastRecord::new(
                    *id,
                    "H",
                    "A",
                    ENSEMBLE_MODEL,
                    oracle_domain::Probabilities::new(0.5, 0.3, 0.2),
                ))
                .unwrap();
        }
        // Results for the last four - so fixtures 3 and 4 overlap.
        let log = EventLog::create(&log_path).unwrap();
        for (id, score) in all.iter().skip(2) {
            log.append(&MatchEvent::new(
                *id,
                90,
                EventKind::FullTime { score: *score },
            ))
            .unwrap();
        }
    }

    let records = ForecastJournal::read(&journal_path).unwrap();
    let mut tournament = data::world_cup_2026();
    for event in EventLog::read(&log_path).unwrap() {
        if let EventKind::FullTime { score } = event.kind {
            if let Some(m) = tournament
                .matches
                .iter_mut()
                .find(|m| m.id == event.match_id)
            {
                m.status = oracle_domain::MatchStatus::Finished;
                m.score = score;
            }
        }
    }

    let tr = track_record(&records, &tournament, 0);
    assert_eq!(tr.calls, 4, "four calls journaled");
    assert_eq!(tr.settled, 2, "only the two that also have a result");
    assert_eq!(tr.matches, 2);

    let _ = std::fs::remove_file(&journal_path);
    let _ = std::fs::remove_file(&log_path);
}
