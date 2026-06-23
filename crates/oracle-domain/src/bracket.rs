//! The fixed 2026 knockout bracket template.
//!
//! The 2026 format sends 32 teams to a single-elimination knockout: the 12 group winners, the
//! 12 runners-up, and the 8 best third-placed teams. This module encodes which finishing
//! position fills which Round-of-32 slot, shared by the Monte-Carlo simulator (which fills the
//! slots from each simulated universe's group results) and the ingest layer (which materializes
//! the real bracket once the group stage is complete).
//!
//! Documented modelling choice: the best-third -> slot assignment is a fixed deterministic rule,
//! not FIFA's full 495-row lookup table (which keys off *which* groups the eight thirds come
//! from), and the team-to-group draw is itself synthetic offline.

/// A Round-of-32 slot, referencing a finishing position by group index (0 = A .. 11 = L) or a
/// best-third by rank (0 = best).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketSlot {
    Winner(usize),
    RunnerUp(usize),
    Third(usize),
}

/// The fixed 2026 Round-of-32 pairings: 16 matches, each `(top, bottom)`, using all 12 group
/// winners, 12 runners-up, and 8 best thirds exactly once and forming a stable bracket tree
/// (R16 pairs adjacent matches). No group winner meets its own runner-up.
pub const FIXED_R32: [(BracketSlot, BracketSlot); 16] = {
    use BracketSlot::{RunnerUp as R, Third as T, Winner as W};
    [
        (W(0), R(1)),
        (W(2), T(4)),
        (W(4), R(5)),
        (W(6), T(5)),
        (W(8), R(9)),
        (W(10), T(6)),
        (W(1), R(0)),
        (W(3), T(7)),
        (W(5), R(4)),
        (W(7), R(3)),
        (W(9), R(8)),
        (W(11), R(7)),
        (T(0), R(2)),
        (T(1), R(6)),
        (T(2), R(10)),
        (T(3), R(11)),
    ]
};

/// Resolve a bracket slot to the qualifier filling it, given the ranked group winners,
/// runners-up, and best thirds. Generic over the element type so it serves both the simulator
/// (team indices) and ingest (team ids).
pub fn resolve_slot<T: Copy>(slot: &BracketSlot, winners: &[T], runners: &[T], thirds: &[T]) -> T {
    match *slot {
        BracketSlot::Winner(g) => winners[g],
        BracketSlot::RunnerUp(g) => runners[g],
        BracketSlot::Third(r) => thirds[r],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_qualifier_slot_used_exactly_once() {
        let (mut w, mut r, mut t) = (
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        );
        for (a, b) in FIXED_R32 {
            for slot in [a, b] {
                match slot {
                    BracketSlot::Winner(g) => assert!(w.insert(g), "winner {g} twice"),
                    BracketSlot::RunnerUp(g) => assert!(r.insert(g), "runner {g} twice"),
                    BracketSlot::Third(rank) => assert!(t.insert(rank), "third {rank} twice"),
                }
            }
        }
        assert_eq!(w.len(), 12, "all 12 winners placed");
        assert_eq!(r.len(), 12, "all 12 runners-up placed");
        assert_eq!(t.len(), 8, "all 8 best thirds placed");
    }

    #[test]
    fn no_winner_meets_own_runner_up() {
        for (a, b) in FIXED_R32 {
            if let (BracketSlot::Winner(g1), BracketSlot::RunnerUp(g2)) = (a, b) {
                assert_ne!(g1, g2, "a group winner should not face its own runner-up");
            }
            if let (BracketSlot::RunnerUp(g1), BracketSlot::Winner(g2)) = (a, b) {
                assert_ne!(g1, g2, "a group winner should not face its own runner-up");
            }
        }
    }
}
