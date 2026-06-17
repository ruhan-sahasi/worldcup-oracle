//! Criterion benchmark for Monte-Carlo tournament throughput.
//!
//! Run with `cargo bench -p oracle-sim`. Reports wall-clock for a fixed number of
//! full-tournament simulations, which is the metric that matters when the engine
//! refreshes champion odds live.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use oracle_domain::{Confederation, Group, Team, TeamId, Tournament};
use oracle_sim::{simulate, MatchSampler, SimConfig};

struct RankSampler;
impl MatchSampler for RankSampler {
    fn xg(&self, home: TeamId, away: TeamId) -> (f64, f64) {
        let strength = |t: TeamId| 2.0 - 0.02 * t.0 as f64;
        let h = (1.2 + strength(home) - strength(away)).max(0.2);
        let a = (1.2 + strength(away) - strength(home)).max(0.2);
        (h, a)
    }
}

/// A full 48-team, 12-group World-Cup-shaped tournament with nothing played yet.
fn world_cup_shaped() -> Tournament {
    let mut t = Tournament::new("Bench Cup");
    for i in 0..48u32 {
        t.teams.push(Team::new(
            i,
            format!("T{i}"),
            format!("{i:03}"),
            Confederation::Uefa,
        ));
    }
    for g in 0..12u32 {
        let base = g * 4;
        t.groups.push(Group {
            name: (b'A' + g as u8) as char,
            teams: (base..base + 4).map(TeamId).collect(),
        });
    }
    t
}

fn bench_simulate(c: &mut Criterion) {
    let tournament = world_cup_shaped();
    let mut group = c.benchmark_group("tournament");
    for &iters in &[1_000u64, 10_000] {
        group.bench_function(format!("simulate_{iters}"), |b| {
            b.iter_batched(
                || SimConfig {
                    iterations: iters,
                    seed: 7,
                    ..Default::default()
                },
                |cfg| simulate(&tournament, &RankSampler, cfg),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_simulate);
criterion_main!(benches);
