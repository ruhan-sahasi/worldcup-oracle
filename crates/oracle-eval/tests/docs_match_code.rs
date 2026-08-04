//! Assert the architecture docs still describe the actual workspace.
//!
//! Every documentation defect this branch fixed was the same failure: a change was described by
//! editing the sentences someone remembered writing, rather than by checking the document against the
//! code. The crate graph had lost `sim --> numeric` and `ingest --> numeric` - the two edges the
//! numerics work existed to create - along with `oracle-eval` entirely, and the crate count was a
//! release behind.
//!
//! Fixing those by hand does nothing to stop the next one, so the parts that are mechanically
//! checkable are checked here. This is not an attempt to verify the prose; it verifies the claims
//! that are really assertions about the repository - which crates exist, and which depend on which.
//!
//! It lives in `oracle-eval` because that crate already owns "checking that a claim this project
//! makes is still true".

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/oracle-eval.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root is two levels above the crate")
        .to_path_buf()
}

/// Every workspace member's directory name.
fn crate_names() -> BTreeSet<String> {
    std::fs::read_dir(repo_root().join("crates"))
        .expect("crates/ is readable")
        .filter_map(Result::ok)
        .filter(|e| e.path().join("Cargo.toml").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

/// Each crate's internal dependencies, read from its manifest. Keys and values are the short names
/// used in the docs' graph (`oracle-sim` becomes `sim`).
fn internal_deps() -> BTreeMap<String, BTreeSet<String>> {
    let mut out = BTreeMap::new();
    for name in crate_names() {
        let manifest =
            std::fs::read_to_string(repo_root().join("crates").join(&name).join("Cargo.toml"))
                .expect("a member manifest is readable");
        let deps: BTreeSet<String> = manifest
            .lines()
            .filter_map(|l| l.trim().strip_prefix("oracle-"))
            .filter_map(|l| l.split(['.', ' ', '=']).next())
            .map(str::to_string)
            .collect();
        out.insert(
            name.strip_prefix("oracle-").unwrap_or(&name).to_string(),
            deps,
        );
    }
    out
}

fn architecture() -> String {
    std::fs::read_to_string(repo_root().join("docs/ARCHITECTURE.md")).expect("ARCHITECTURE.md")
}

#[test]
fn the_architecture_crate_count_matches_the_workspace() {
    let n = crate_names().len();
    let word = match n {
        12 => "twelve",
        13 => "thirteen",
        14 => "fourteen",
        15 => "fifteen",
        16 => "sixteen",
        other => panic!("no spelling for {other} crates; extend this test with the change"),
    };
    let doc = architecture();
    assert!(
        doc.contains(&format!("workspace** of {word} focused crates")),
        "ARCHITECTURE says a different number of crates than the {n} in crates/"
    );
}

#[test]
fn every_crate_appears_in_the_architecture_graph() {
    let doc = architecture();
    let graph = doc
        .split("graph TD")
        .nth(1)
        .expect("a mermaid crate graph")
        .split("```")
        .next()
        .expect("the graph is fenced");

    for name in crate_names() {
        let short = name.strip_prefix("oracle-").unwrap_or(&name).to_string();
        assert!(
            graph.contains(&format!("{short}[")),
            "{name} has no node in the crate graph; a reader tracing dependencies would not find it"
        );
    }
}

#[test]
fn the_graph_records_every_internal_dependency() {
    let doc = architecture();
    let graph = doc
        .split("graph TD")
        .nth(1)
        .expect("a mermaid crate graph")
        .split("```")
        .next()
        .expect("the graph is fenced");

    // The CLI depends on nearly every crate, and the graph says in prose that its edges are
    // abbreviated. Exempting it here keeps the test honest about what the document promises.
    assert!(
        doc.contains("The CLI's edges are abbreviated"),
        "the graph exempts oracle-cli, so the document has to say so"
    );

    let mut missing = Vec::new();
    for (crate_short, deps) in internal_deps() {
        if crate_short == "cli" {
            continue;
        }
        for dep in deps {
            if !graph.contains(&format!("{crate_short} --> {dep}")) {
                missing.push(format!("{crate_short} --> {dep}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "the crate graph is missing real dependencies: {missing:?}"
    );
}

#[test]
fn the_graph_claims_no_dependency_that_does_not_exist() {
    // The other direction. A stale edge left behind after a dependency is dropped is just as
    // misleading as a missing one, and harder to notice.
    let doc = architecture();
    let graph = doc
        .split("graph TD")
        .nth(1)
        .expect("a mermaid crate graph")
        .split("```")
        .next()
        .expect("the graph is fenced");
    let deps = internal_deps();

    let mut phantom = Vec::new();
    for line in graph.lines() {
        let line = line.trim();
        let Some((from, to)) = line.split_once(" --> ") else {
            continue;
        };
        // `cli` edges are a deliberate subset, so a listed one still has to be real.
        match deps.get(from) {
            Some(actual) if actual.contains(to) => {}
            Some(_) => phantom.push(format!("{from} --> {to}")),
            None => phantom.push(format!("{from} (unknown crate) --> {to}")),
        }
    }
    assert!(
        phantom.is_empty(),
        "the crate graph claims dependencies that do not exist: {phantom:?}"
    );
}

#[test]
fn the_readme_crate_table_lists_every_crate() {
    let readme = std::fs::read_to_string(repo_root().join("README.md")).expect("README.md");
    for name in crate_names() {
        assert!(
            readme.contains(&format!("| `{name}` |")),
            "{name} is missing from the README's crate table"
        );
    }
}
