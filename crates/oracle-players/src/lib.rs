//! # oracle-players
//!
//! The player-level layer of the oracle: who scores the goals. Every other part of the project
//! reasons about *teams*; this reasons about the individuals inside them. A team's expected goals in
//! a match are shared out among its on-pitch players by attacking weight, and from those per-player
//! expected goals fall the goalscorer markets a fan actually cares about (anytime scorer, brace,
//! hat-trick, first goal) and, simulated over the whole tournament, the **Golden Boot** race.
//!
//! Like the market crate, it is deliberately small and pure: plain calculations over `f64`s, with
//! no I/O, so every layer is unit-testable in isolation. The player scoring weights come from the
//! squad model upstream; this crate never invents data, it only turns weights and expected goals
//! into probabilities.
//!
//! The Monte-Carlo draws and the Poisson masses both come from [`oracle_numeric`], so a Golden Boot
//! race is fully reproducible from its seed.
#![forbid(unsafe_code)]

use oracle_numeric::poisson_pmf;
use serde::{Deserialize, Serialize};

/// The seeded generator a Golden Boot race draws from, re-exported so callers can build one without
/// naming [`oracle_numeric`] themselves.
pub use oracle_numeric::Rng;

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

/// P(the player scores **at least one** goal), `1 - e^(-xg)`: the anytime-scorer market.
pub fn anytime_scorer(xg: f64) -> f64 {
    1.0 - (-xg.max(0.0)).exp()
}

/// P(the player scores **exactly** `k` goals), from a Poisson on their expected goals.
pub fn exactly_goals(xg: f64, k: u32) -> f64 {
    poisson_pmf(k, xg.max(0.0))
}

/// P(the player scores **at least** `k` goals): `k = 2` is a brace, `k = 3` a hat-trick.
pub fn at_least_goals(xg: f64, k: u32) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let x = xg.max(0.0);
    (1.0 - (0..k).map(|i| poisson_pmf(i, x)).sum::<f64>()).clamp(0.0, 1.0)
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

/// Per-player expected goals for both teams as `(identity, expected goals)`, the shared basis for
/// the scorer markets.
fn per_player_goals(
    home_team: &str,
    home_players: &[MatchPlayer],
    home_xg: f64,
    away_team: &str,
    away_players: &[MatchPlayer],
    away_xg: f64,
) -> Vec<(PlayerRef, f64)> {
    let mut out = Vec::with_capacity(home_players.len() + away_players.len());
    for (team, players, xg) in [
        (home_team, home_players, home_xg),
        (away_team, away_players, away_xg),
    ] {
        let weights: Vec<f64> = players.iter().map(|p| p.weight).collect();
        for (p, &goals) in players.iter().zip(allocate(&weights, xg).iter()) {
            out.push((
                PlayerRef {
                    name: p.name.clone(),
                    team: team.to_string(),
                },
                goals,
            ));
        }
    }
    out
}

/// One player's first-goalscorer probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstScorerLine {
    pub player: PlayerRef,
    pub prob: f64,
}

/// The first-goalscorer market, plus the probability the match has **no** goal at all. The line
/// probabilities and `no_goal` together sum to one.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FirstScorerMarket {
    pub no_goal: f64,
    pub lines: Vec<FirstScorerLine>,
}

/// The first-goalscorer market, from competing Poisson processes: if goals arrive independently at
/// each player's rate, the first goal is theirs with probability `rate / total`, and a goal happens
/// at all with probability `1 - e^(-total)`. So `P(first = i) = (lambda_i / total)(1 - e^-total)`,
/// and the leftover `e^-total` is a goalless match. Ranked by probability.
pub fn first_scorer_market(
    home_team: &str,
    home_players: &[MatchPlayer],
    home_xg: f64,
    away_team: &str,
    away_players: &[MatchPlayer],
    away_xg: f64,
) -> FirstScorerMarket {
    let lambdas = per_player_goals(
        home_team,
        home_players,
        home_xg,
        away_team,
        away_players,
        away_xg,
    );
    let total: f64 = lambdas.iter().map(|(_, l)| l).sum();
    let any_goal = 1.0 - (-total).exp();
    let mut lines: Vec<FirstScorerLine> = lambdas
        .into_iter()
        .map(|(player, lambda)| FirstScorerLine {
            player,
            prob: if total > 0.0 {
                lambda / total * any_goal
            } else {
                0.0
            },
        })
        .collect();
    lines.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    FirstScorerMarket {
        no_goal: (-total).exp(),
        lines,
    }
}

/// A Golden Boot contender: a player and their expected goals across the rest of the tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenBootContender {
    pub player: PlayerRef,
    pub expected_goals: f64,
}

/// Monte-Carlo settings for the Golden Boot race.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GoldenBootConfig {
    pub iters: u32,
    pub seed: u64,
}

impl Default for GoldenBootConfig {
    fn default() -> Self {
        Self {
            iters: 20_000,
            seed: 42,
        }
    }
}

/// One contender's Golden Boot odds: their expected goals, the chance of finishing top scorer
/// (`p_top`, ties split), and the chance of a top-three finish (`p_top3`, ties inclusive).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenBootOdds {
    pub player: PlayerRef,
    pub expected_goals: f64,
    pub p_top: f64,
    pub p_top3: f64,
}

/// Simulate the Golden Boot race: each iteration draws every contender's goal count from a Poisson
/// on their expected tournament goals, then credits the leader (ties split, so the win probabilities
/// sum to one) and everyone finishing in the top three (ties inclusive). Reproducible from the
/// config seed, ranked by the chance of finishing top scorer.
pub fn golden_boot(
    contenders: &[GoldenBootContender],
    config: &GoldenBootConfig,
) -> Vec<GoldenBootOdds> {
    let n = contenders.len();
    if n == 0 {
        return Vec::new();
    }
    let iters = config.iters.max(1);
    let mut rng = Rng::new(config.seed);
    let mut tally = vec![0u32; n];
    let mut wins = vec![0.0f64; n];
    let mut podium = vec![0.0f64; n];

    for _ in 0..iters {
        for (i, c) in contenders.iter().enumerate() {
            tally[i] = rng.poisson(c.expected_goals);
        }
        // The three largest tallies (with multiplicity), in one pass.
        let (mut v1, mut v2, mut v3) = (0u32, 0u32, 0u32);
        for &t in &tally {
            if t > v1 {
                v3 = v2;
                v2 = v1;
                v1 = t;
            } else if t > v2 {
                v3 = v2;
                v2 = t;
            } else if t > v3 {
                v3 = t;
            }
        }
        let leaders = tally.iter().filter(|&&t| t == v1).count();
        let win_share = 1.0 / leaders as f64;
        let podium_cut = v3.max(1);
        for i in 0..n {
            if tally[i] == v1 {
                wins[i] += win_share;
            }
            if tally[i] >= podium_cut {
                podium[i] += 1.0;
            }
        }
    }

    let iters_f = iters as f64;
    let mut odds: Vec<GoldenBootOdds> = contenders
        .iter()
        .enumerate()
        .map(|(i, c)| GoldenBootOdds {
            player: c.player.clone(),
            expected_goals: c.expected_goals,
            p_top: wins[i] / iters_f,
            p_top3: podium[i] / iters_f,
        })
        .collect();
    odds.sort_by(|a, b| {
        b.p_top
            .partial_cmp(&a.p_top)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.expected_goals
                    .partial_cmp(&a.expected_goals)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    odds
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
    fn first_scorer_market_is_a_proper_distribution_over_scorers_and_no_goal() {
        let home = vec![MatchPlayer::new("Star", 1.0), MatchPlayer::new("Mid", 0.5)];
        let away = vec![MatchPlayer::new("Away FW", 0.6)];
        let m = first_scorer_market("Home", &home, 1.6, "Away", &away, 0.8);

        // Everything (every scorer + a goalless match) sums to one.
        let total: f64 = m.lines.iter().map(|l| l.prob).sum::<f64>() + m.no_goal;
        approx(total, 1.0);
        // The highest-rate player leads, and no_goal = e^-(total xg) = e^-2.4.
        assert_eq!(m.lines[0].player.name, "Star");
        approx(m.no_goal, (-2.4_f64).exp());
        assert!(m.lines.windows(2).all(|w| w[0].prob >= w[1].prob));

        // With no expected goals, a goalless match is certain and nobody scores first.
        let none = first_scorer_market("H", &[MatchPlayer::new("A", 1.0)], 0.0, "A", &[], 0.0);
        approx(none.no_goal, 1.0);
        approx(none.lines[0].prob, 0.0);
    }

    fn contender(name: &str, xg: f64) -> GoldenBootContender {
        GoldenBootContender {
            player: PlayerRef {
                name: name.into(),
                team: "T".into(),
            },
            expected_goals: xg,
        }
    }

    #[test]
    fn golden_boot_race_is_a_ranked_distribution_and_reproducible() {
        let field = vec![
            contender("Ace", 7.0),
            contender("Second", 5.0),
            contender("Third", 5.0),
            contender("Squad", 3.0),
            contender("Fringe", 1.0),
        ];
        let cfg = GoldenBootConfig {
            iters: 20_000,
            seed: 42,
        };
        let odds = golden_boot(&field, &cfg);

        assert_eq!(odds.len(), 5);
        // Win probabilities are a distribution (ties split, so they sum to one).
        let total: f64 = odds.iter().map(|o| o.p_top).sum();
        assert!((total - 1.0).abs() < 1e-9, "sum {total}");
        // Ranked by chance of finishing top scorer, and the clear favourite leads.
        assert!(odds.windows(2).all(|w| w[0].p_top >= w[1].p_top));
        assert_eq!(odds[0].player.name, "Ace");
        // Finishing top implies finishing top three, per player.
        for o in &odds {
            assert!(o.p_top3 >= o.p_top - 1e-12);
            assert!((0.0..=1.0).contains(&o.p_top) && (0.0..=1.0).contains(&o.p_top3));
        }
        // Same seed, identical odds.
        let again = golden_boot(&field, &cfg);
        assert!((again[0].p_top - odds[0].p_top).abs() < 1e-12);

        assert!(golden_boot(&[], &cfg).is_empty());
    }
}
