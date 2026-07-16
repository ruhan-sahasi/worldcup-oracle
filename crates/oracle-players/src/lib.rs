//! # oracle-players
//!
//! The player-level layer of the oracle: who scores the goals. Every other part of the project
//! reasons about *teams*; this reasons about the individuals inside them. A team's expected goals in
//! a match are shared out among its on-pitch players by attacking weight, and from those per-player
//! expected goals fall the goalscorer markets a fan actually cares about (anytime scorer, brace,
//! hat-trick, first goal) and, simulated over the whole tournament, the **Golden Boot** race.
//!
//! Like the market crate, it is deliberately small and pure: plain calculations over `f64`s plus a
//! self-contained, seeded random generator for the Monte-Carlo, with no I/O, so every layer is
//! unit-testable in isolation. The player scoring weights come from the squad model upstream; this
//! crate never invents data, it only turns weights and expected goals into probabilities.
//!
//! This first layer is the Monte-Carlo's foundation: a **SplitMix64** generator, chosen so the
//! Golden Boot simulation is fully reproducible from a seed with no external `rand` dependency.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// A small, fast, fully seeded pseudo-random generator (SplitMix64). Deterministic given its seed,
/// so a Monte-Carlo run is exactly reproducible; adequate for simulation (not for cryptography).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. The same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in `[0, 1)`, from the top 53 bits (one full `f64` mantissa).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A player's identity in a market or race: their name and their team's name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerRef {
    pub name: String,
    pub team: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generator_is_deterministic_for_a_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn uniform_draws_stay_in_range_and_average_near_a_half() {
        let mut rng = Rng::new(7);
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        assert!(
            (sum / n as f64 - 0.5).abs() < 0.01,
            "mean {}",
            sum / n as f64
        );
    }
}
