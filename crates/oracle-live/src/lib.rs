//! # oracle-live
//!
//! The in-play layer: what the odds do *during* a match, and what a trader can do about it. Every
//! other part of the project prices a match before kickoff; this one follows the score as it
//! changes, re-derives the win probability minute by minute, and simulates **in-play trading** with
//! the tools a real exchange gives you: **hedging** and **cash-out**. The honest question it answers
//! is whether actively trading a live position beats simply holding a pre-match bet.
//!
//! Like the market and players crates it is small and pure: plain calculations plus a self-contained
//! seeded generator for the simulation, no I/O, every layer unit-tested in isolation. The pre-match
//! goal rates come from the goal model upstream; this crate turns them into a live win-probability
//! path and a settled trading ledger.
//!
//! This first layer is the vocabulary and the Monte-Carlo's random source: a [`MatchState`] (the
//! score at a minute) and a **SplitMix64** generator with a Poisson sampler, so a simulated match is
//! reproducible from a seed with no external `rand` dependency.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// The score at a point in a match: minutes elapsed and goals for each side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchState {
    pub minute: u16,
    pub home: u8,
    pub away: u8,
}

impl MatchState {
    /// Kickoff: 0-0 at minute zero.
    pub fn kickoff() -> Self {
        Self {
            minute: 0,
            home: 0,
            away: 0,
        }
    }
}

/// A small, fully seeded pseudo-random generator (SplitMix64). Deterministic given its seed, so a
/// simulated match is exactly reproducible; adequate for simulation, not cryptography.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A Poisson-distributed count with mean `lambda`, by Knuth's method (rates here are small).
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        let threshold = (-lambda).exp();
        let mut product = 1.0;
        let mut k = 0u32;
        loop {
            product *= self.next_f64();
            if product <= threshold {
                return k;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_state_starts_goalless_at_kickoff() {
        let s = MatchState::kickoff();
        assert_eq!((s.minute, s.home, s.away), (0, 0, 0));
    }

    #[test]
    fn the_generator_is_deterministic_and_uniform() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut rng = Rng::new(7);
        let mut sum = 0.0;
        let n = 100_000;
        for _ in 0..n {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.01);
    }

    #[test]
    fn poisson_has_the_right_mean() {
        assert_eq!(Rng::new(1).poisson(0.0), 0);
        let mut rng = Rng::new(11);
        let n = 200_000;
        let total: u64 = (0..n).map(|_| u64::from(rng.poisson(1.4))).sum();
        assert!((total as f64 / n as f64 - 1.4).abs() < 0.02);
    }
}
