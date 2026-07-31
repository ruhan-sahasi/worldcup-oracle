//! Regenerate the committed skill fixture.
//!
//! ```text
//! cargo run --release -p oracle-eval --example freeze_fixture \
//!     > crates/oracle-eval/fixtures/skill_v1.csv
//! ```
//!
//! This exists as executable provenance: the fixture is a committed data file, and a data file whose
//! origin is described only in prose eventually stops matching its description. Running this is how
//! you check.
//!
//! It reproduces `skill_v1.csv` byte-for-byte only against the version of `oracle-ingest::data` the
//! fixture was frozen from. That is not a defect - it is the reason the fixture is frozen at all. If
//! the generator has since changed this writes a *different* dataset, which is why regenerating over
//! the committed file is never part of updating a baseline. Freeze a new `skill_v2.csv` and retire
//! the old one deliberately instead.

/// Matches in the frozen fixture. 4000 leaves an 800-match test split after the 60/20/20 division,
/// which keeps the gate's bootstrap intervals tight enough to catch a regression worth catching.
const MATCHES: usize = 4000;
/// The generator seed the fixture was frozen from.
const SEED: u64 = 7;

fn main() {
    let records = oracle_ingest::data::synthetic_history_with_market(MATCHES, SEED);
    print!("{}", oracle_eval::fixture::to_csv(&records));
}
