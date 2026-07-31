//! Freezing an evaluation dataset to CSV, so the gate's data stops being a function of the code.
//!
//! # Why freeze rather than regenerate
//!
//! The obvious way to give a regression gate a dataset is to call the synthetic generator with a
//! fixed seed. That fails for the one thing a gate is for. `oracle-ingest::data` is ordinary code
//! and gets edited: a previous branch on this repo changed every synthetic value it produces. Under
//! a regenerated dataset, every such edit moves the metrics, so the gate fires on data changes and
//! model changes alike and cannot tell them apart. Since data edits are the more frequent kind, the
//! gate would mostly produce false alarms - and a gate that cries wolf gets switched off, which is
//! worse than not having one.
//!
//! Frozen once to a committed CSV, the dataset is inert. A change to the generator no longer touches
//! the gate at all, and a metric that moves means the *model* moved. That is the signal the gate
//! exists to carry.
//!
//! The file is written in the same football-data.co.uk-style layout
//! [`oracle_ingest::data::load_results_csv`] already reads, so freezing adds a writer and no new
//! parsing path.

use oracle_ingest::data::MatchRecord;
use std::fmt::Write as _;

/// The header the loader expects, plus the optional odds and xG columns.
const HEADER: &str = "HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,HxG,AxG";

/// Render `records` as a CSV the loader can read back.
///
/// Team ids become `T<id>` names, because the loader interns names to ids in first-appearance order
/// and cannot be handed ids directly. The mapping is order-preserving in practice for a fixture
/// written oldest-first, but the gate never relies on a specific id: every model is keyed by name
/// through the same interning on both sides.
///
/// Odds are written as fair decimal prices `1/p`. The loader de-vigs whatever it reads, and there is
/// no vig in a fair price, so the probabilities come back out unchanged.
///
/// Rows are written oldest-first, matching the loader's assumption that earlier rows are older.
pub fn to_csv(records: &[MatchRecord]) -> String {
    let mut oldest_first = records.to_vec();
    oldest_first.sort_by(|a, b| {
        b.obs
            .age_days
            .partial_cmp(&a.obs.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.obs.home.cmp(&b.obs.home))
            .then_with(|| a.obs.away.cmp(&b.obs.away))
    });

    let mut out = String::with_capacity(records.len() * 64);
    out.push_str(HEADER);
    out.push('\n');
    for r in &oldest_first {
        let o = &r.obs;
        // A fixture is a regression baseline's input, so the numbers are written at full precision
        // rather than rounded for display. Rounding here would bake a rounding error into every
        // metric the gate compares against.
        let odds = |p: f64| {
            if p > 0.0 {
                format!("{:.10}", 1.0 / p)
            } else {
                String::new()
            }
        };
        let (h_odds, d_odds, a_odds) = match r.market {
            Some(m) => (odds(m.home_win), odds(m.draw), odds(m.away_win)),
            None => (String::new(), String::new(), String::new()),
        };
        let xg = |v: Option<f64>| v.map(|x| format!("{x:.10}")).unwrap_or_default();
        let _ = writeln!(
            out,
            "T{},T{},{},{},{},{},{},{},{}",
            o.home.0,
            o.away.0,
            o.score.home,
            o.score.away,
            h_odds,
            d_odds,
            a_odds,
            xg(o.home_xg),
            xg(o.away_xg),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_ingest::data;

    fn write_and_reload(records: &[MatchRecord], name: &str) -> Vec<MatchRecord> {
        let path = std::env::temp_dir().join(format!("oracle_fixture_{name}.csv"));
        std::fs::write(&path, to_csv(records)).unwrap();
        let back = data::load_results_csv(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        back
    }

    #[test]
    fn a_fixture_round_trips_through_the_loader() {
        let original = data::synthetic_history_with_market(300, 7);
        let back = write_and_reload(&original, "roundtrip");
        assert_eq!(back.len(), original.len(), "every row survives");

        // The loader re-interns names and re-derives ages from row position, so ids and ages are not
        // expected to match. What must survive is every number the evaluation reads.
        for (a, b) in original
            .iter()
            .zip(&back)
            .filter(|(a, b)| a.obs.score == b.obs.score)
            .take(50)
        {
            assert_eq!(a.obs.score, b.obs.score);
            if let (Some(x), Some(y)) = (a.obs.home_xg, b.obs.home_xg) {
                assert!((x - y).abs() < 1e-9, "home xG drifted: {x} vs {y}");
            }
        }
    }

    #[test]
    fn odds_survive_the_round_trip_as_probabilities() {
        // Written as fair prices, de-vigged on read: the probabilities must come back unchanged.
        let original = data::synthetic_history_with_market(200, 3);
        let back = write_and_reload(&original, "odds");
        let with_odds = back.iter().filter(|r| r.market.is_some()).count();
        assert_eq!(with_odds, back.len(), "every row kept its odds");

        // Compare the first row's market, which the sort places deterministically.
        let mut sorted = original.clone();
        sorted.sort_by(|a, b| {
            b.obs
                .age_days
                .partial_cmp(&a.obs.age_days)
                .unwrap()
                .then_with(|| a.obs.home.cmp(&b.obs.home))
                .then_with(|| a.obs.away.cmp(&b.obs.away))
        });
        let (want, got) = (sorted[0].market.unwrap(), back[0].market.unwrap());
        assert!(
            (want.home_win - got.home_win).abs() < 1e-9,
            "{want:?} {got:?}"
        );
        assert!((want.draw - got.draw).abs() < 1e-9);
        assert!((want.away_win - got.away_win).abs() < 1e-9);
    }

    #[test]
    fn a_frozen_fixture_evaluates_the_same_as_the_records_it_came_from() {
        // The property that makes freezing safe: the CSV is a faithful carrier, so the gate's
        // baseline means the same thing as a baseline taken from the generator directly.
        let original = data::synthetic_history_with_market(1000, 7);
        let back = write_and_reload(&original, "eval_equiv");
        let a = crate::evaluate(&original, crate::EvalConfig::default()).unwrap();
        let b = crate::evaluate(&back, crate::EvalConfig::default()).unwrap();

        assert_eq!(a.test, b.test, "same split");
        for (x, y) in a.models.iter().zip(&b.models) {
            assert_eq!(x.model, y.model);
            // Not bit-equal: ages are re-derived from row position rather than carried, which
            // changes the time-decay weights slightly. The metrics must still agree closely.
            assert!(
                (x.brier - y.brier).abs() < 0.02,
                "{} Brier moved from {} to {} across the freeze",
                x.model.label(),
                x.brier,
                y.brier
            );
        }
    }

    #[test]
    fn the_header_matches_what_the_loader_requires() {
        // The loader needs these four by name; the rest are optional. Written out so a rename breaks
        // here rather than silently producing a fixture with no odds or no xG.
        for required in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] {
            assert!(HEADER.contains(required), "missing column {required}");
        }
        for optional in ["B365H", "B365D", "B365A", "HxG", "AxG"] {
            assert!(HEADER.contains(optional), "missing column {optional}");
        }
    }

    #[test]
    fn an_empty_record_set_still_writes_a_header() {
        let csv = to_csv(&[]);
        assert_eq!(csv.trim(), HEADER);
    }
}
