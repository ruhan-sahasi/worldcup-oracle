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

/// Share a team's expected goals `team_xg` out among its on-pitch players in proportion to their
/// attacking `weights`, so the returned per-player expected goals sum back to `team_xg`. Negative
/// weights are floored at zero; if every weight is zero (or there are no players) the goals are
/// split evenly, so the total is always conserved.
pub fn allocate(weights: &[f64], team_xg: f64) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let xg = team_xg.max(0.0);
    let clamped: Vec<f64> = weights.iter().map(|w| w.max(0.0)).collect();
    let total: f64 = clamped.iter().sum();
    if total <= 0.0 {
        return vec![xg / n as f64; n];
    }
    clamped.iter().map(|w| xg * w / total).collect()
}

/// Poisson probability mass `P(X = k)` for rate `lambda >= 0`.
fn poisson_pmf(lambda: f64, k: u32) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let mut p = (-lambda).exp();
    for i in 1..=k {
        p *= lambda / i as f64;
    }
    p
}

/// P(the player scores **at least one** goal), `1 - e^(-xg)`: the anytime-scorer market.
pub fn anytime_scorer(xg: f64) -> f64 {
    1.0 - (-xg.max(0.0)).exp()
}

/// P(the player scores **exactly** `k` goals), from a Poisson on their expected goals.
pub fn exactly_goals(xg: f64, k: u32) -> f64 {
    poisson_pmf(xg.max(0.0), k)
}

/// P(the player scores **at least** `k` goals): `k = 2` is a brace, `k = 3` a hat-trick.
pub fn at_least_goals(xg: f64, k: u32) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let x = xg.max(0.0);
    (1.0 - (0..k).map(|i| poisson_pmf(x, i)).sum::<f64>()).clamp(0.0, 1.0)
}

/// A player available in a match: their name and attacking weight (the same weight the allocation
/// shares goals by).
#[derive(Debug, Clone)]
pub struct MatchPlayer {
    pub name: String,
    pub weight: f64,
}

impl MatchPlayer {
    pub fn new(name: impl Into<String>, weight: f64) -> Self {
        Self {
            name: name.into(),
            weight,
        }
    }
}

/// One player's line in a match scorer market: their expected goals and the derived markets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorerLine {
    pub player: PlayerRef,
    pub expected_goals: f64,
    pub anytime: f64,
    pub brace: f64,
    pub hat_trick: f64,
}

/// The goalscorer market for a match: every player from both teams, ranked by anytime probability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScorerMarket {
    pub lines: Vec<ScorerLine>,
}

/// Build a match scorer market: allocate each team's expected goals to its players, then read off
/// the anytime, brace, and hat-trick markets, ranked by anytime probability (likeliest scorer first).
pub fn scorer_market(
    home_team: &str,
    home_players: &[MatchPlayer],
    home_xg: f64,
    away_team: &str,
    away_players: &[MatchPlayer],
    away_xg: f64,
) -> ScorerMarket {
    let mut lines = Vec::with_capacity(home_players.len() + away_players.len());
    for (team, players, xg) in [
        (home_team, home_players, home_xg),
        (away_team, away_players, away_xg),
    ] {
        let weights: Vec<f64> = players.iter().map(|p| p.weight).collect();
        for (p, &goals) in players.iter().zip(allocate(&weights, xg).iter()) {
            lines.push(ScorerLine {
                player: PlayerRef {
                    name: p.name.clone(),
                    team: team.to_string(),
                },
                expected_goals: goals,
                anytime: anytime_scorer(goals),
                brace: at_least_goals(goals, 2),
                hat_trick: at_least_goals(goals, 3),
            });
        }
    }
    lines.sort_by(|a, b| {
        b.anytime
            .partial_cmp(&a.anytime)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ScorerMarket { lines }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn allocation_shares_expected_goals_in_proportion_to_weight() {
        let a = allocate(&[2.0, 1.0, 1.0], 2.0);
        approx(a[0], 1.0);
        approx(a[1], 0.5);
        approx(a[2], 0.5);
        approx(a.iter().sum::<f64>(), 2.0);
    }

    #[test]
    fn allocation_conserves_the_total_and_handles_degenerate_input() {
        // Zero weights split evenly.
        let even = allocate(&[0.0, 0.0, 0.0, 0.0], 3.0);
        for e in &even {
            approx(*e, 0.75);
        }
        // Negative weights are floored, and the total is still conserved.
        let floored = allocate(&[-5.0, 1.0, 3.0], 2.0);
        approx(floored[0], 0.0);
        approx(floored.iter().sum::<f64>(), 2.0);
        // No players, no goals.
        assert!(allocate(&[], 5.0).is_empty());
    }

    #[test]
    fn scorer_probabilities_follow_a_poisson() {
        // Anytime equals "at least one", and is 0 at no threat, near 1 at heavy threat.
        approx(anytime_scorer(0.0), 0.0);
        approx(anytime_scorer(1.0), at_least_goals(1.0, 1));
        assert!(anytime_scorer(3.0) > 0.94);
        // Known Poisson values at lambda = 1: P(0) = P(1) = e^-1.
        approx(exactly_goals(1.0, 0), (-1.0_f64).exp());
        approx(exactly_goals(1.0, 1), (-1.0_f64).exp());
        // More goals is always less likely; a brace beats a hat-trick.
        assert!(at_least_goals(1.2, 1) > at_least_goals(1.2, 2));
        assert!(at_least_goals(1.2, 2) > at_least_goals(1.2, 3));
        // Exact masses plus the tail sum to one.
        let tail = at_least_goals(0.8, 4);
        let head: f64 = (0..4).map(|k| exactly_goals(0.8, k)).sum();
        approx(head + tail, 1.0);
    }

    #[test]
    fn match_market_ranks_scorers_and_conserves_each_team_total() {
        let home = vec![
            MatchPlayer::new("Star", 1.0),
            MatchPlayer::new("Mid", 0.5),
            MatchPlayer::new("Back", 0.2),
        ];
        let away = vec![
            MatchPlayer::new("Away FW", 0.8),
            MatchPlayer::new("Away MF", 0.4),
        ];
        let m = scorer_market("Home", &home, 1.8, "Away", &away, 0.9);

        assert_eq!(m.lines.len(), 5);
        // Ranked by anytime probability, descending.
        assert!(m.lines.windows(2).all(|w| w[0].anytime >= w[1].anytime));
        // The likeliest scorer is the home talisman.
        assert_eq!(m.lines[0].player.name, "Star");
        assert_eq!(m.lines[0].player.team, "Home");
        // Every probability is coherent: brace <= anytime, hat-trick <= brace.
        for l in &m.lines {
            assert!((0.0..=1.0).contains(&l.anytime));
            assert!(l.brace <= l.anytime && l.hat_trick <= l.brace);
        }
        // Allocation is conserved per team.
        let home_goals: f64 = m
            .lines
            .iter()
            .filter(|l| l.player.team == "Home")
            .map(|l| l.expected_goals)
            .sum();
        approx(home_goals, 1.8);
    }

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
