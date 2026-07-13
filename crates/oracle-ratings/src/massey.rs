//! Massey least-squares ratings.
//!
//! A rating built from a completely different idea than Elo or the state-space filter: rather than
//! updating a belief game by game, it treats the whole set of results as one **linear system** and
//! solves it in closed form. Each game says "team i beat team j by this margin", and the least-
//! squares fit that best explains every margin at once is, by construction, **strength-of-schedule
//! adjusted** - beating a strong team counts for more than beating a weak one, because the same
//! margin against a higher-rated opponent implies a higher rating.
//!
//! Concretely, with `T` the diagonal matrix of games played and `P` the matrix of pairwise games,
//! the Massey matrix is `M = T - P` and the ratings `r` solve `M r = p`, where `p` is each team's
//! cumulative goal margin. The system is singular (its rows sum to zero), so a small **ridge** is
//! added to the diagonal: this makes the fit well-posed even when the results graph is thin or
//! disconnected early in a tournament, and shrinks sparsely-observed teams toward the mean.
//!
//! The rating also splits cleanly into **offense** and **defense** (`r = offense + defense`): the
//! defensive ratings solve `(T + P) d = T r - f` for the points-scored vector `f`, and the offense
//! is the remainder. So the same fit says both how strong a team is and whether that strength comes
//! from scoring or from stopping goals.
//!
//! This is deliberately a *within-tournament* rating: fed only the matches actually played, it is a
//! fresh, prior-free read on who has been strongest here, to sit alongside the prior-anchored
//! models.

use oracle_domain::{Scoreline, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tunable parameters for the Massey fit (ratings are in goal-difference units).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MasseyConfig {
    /// Ridge added to each linear system's diagonal, so the fit is well-posed when the results graph
    /// is thin or disconnected (and sparsely-observed teams shrink toward the mean rating of zero).
    pub ridge: f64,
    /// Cap on the per-game goal margin, so a single blowout cannot dominate the least-squares fit.
    pub max_margin: f64,
}

impl Default for MasseyConfig {
    fn default() -> Self {
        Self {
            ridge: 1.0,
            max_margin: 5.0,
        }
    }
}

/// One team's Massey rating and its offense/defense decomposition (all in goal-difference units,
/// centered so the field averages zero). `games` is how many results fed the fit for this team.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MasseyRating {
    pub team: TeamId,
    /// Overall rating: `offense + defense`. Higher is stronger.
    pub rating: f64,
    /// Offensive component: higher means the team's strength comes more from scoring.
    pub offense: f64,
    /// Defensive component: higher means it comes more from conceding fewer than expected.
    pub defense: f64,
    pub games: u32,
}

/// The fitted Massey ratings for a set of results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MasseyRatings {
    ratings: Vec<MasseyRating>,
}

impl MasseyRatings {
    /// Fit with the default configuration. `results` are `(home, away, score)`; only the goal margin
    /// and who played whom matter (venue is irrelevant to this symmetric method).
    pub fn fit(results: &[(TeamId, TeamId, Scoreline)]) -> Self {
        Self::fit_with(results, MasseyConfig::default())
    }

    /// Fit with an explicit configuration.
    pub fn fit_with(results: &[(TeamId, TeamId, Scoreline)], config: MasseyConfig) -> Self {
        // Index the teams that actually appear, in order of first appearance.
        let mut index: HashMap<TeamId, usize> = HashMap::new();
        let mut teams: Vec<TeamId> = Vec::new();
        for (h, a, _) in results {
            for &t in &[*h, *a] {
                if let std::collections::hash_map::Entry::Vacant(slot) = index.entry(t) {
                    slot.insert(teams.len());
                    teams.push(t);
                }
            }
        }
        let n = teams.len();
        if n == 0 {
            return Self::default();
        }

        let mut games = vec![0.0f64; n]; // T diagonal
        let mut pair = vec![vec![0.0f64; n]; n]; // P: pairwise game counts
        let mut net = vec![0.0f64; n]; // p: cumulative goal margin
        let mut scored = vec![0.0f64; n]; // f: total goals scored
        let mut game_count = vec![0u32; n];
        for (h, a, s) in results {
            let (i, j) = (index[h], index[a]);
            let margin =
                ((s.home as f64) - (s.away as f64)).clamp(-config.max_margin, config.max_margin);
            games[i] += 1.0;
            games[j] += 1.0;
            game_count[i] += 1;
            game_count[j] += 1;
            pair[i][j] += 1.0;
            pair[j][i] += 1.0;
            net[i] += margin;
            net[j] -= margin;
            scored[i] += s.home as f64;
            scored[j] += s.away as f64;
        }

        // M = T - P, ridged on the diagonal; solve M r = p, then center to sum zero.
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = if i == j {
                    games[i] + config.ridge
                } else {
                    -pair[i][j]
                };
            }
        }
        let mut r = solve(m, net);
        center(&mut r);

        // Offense/defense split: (T + P) d = T r - f, ridged; then offense = r - d. Both are then
        // centered (the split has a gauge freedom, so centering picks the natural zero-mean gauge).
        let mut tp = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                tp[i][j] = if i == j {
                    games[i] + config.ridge
                } else {
                    pair[i][j]
                };
            }
        }
        let rhs: Vec<f64> = (0..n).map(|i| games[i] * r[i] - scored[i]).collect();
        let mut defense = solve(tp, rhs);
        center(&mut defense);
        let offense: Vec<f64> = (0..n).map(|i| r[i] - defense[i]).collect();

        let mut ratings: Vec<MasseyRating> = (0..n)
            .map(|i| MasseyRating {
                team: teams[i],
                rating: r[i],
                offense: offense[i],
                defense: defense[i],
                games: game_count[i],
            })
            .collect();
        ratings.sort_by(|a, b| {
            b.rating
                .partial_cmp(&a.rating)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { ratings }
    }

    /// The ratings, ranked by overall rating (strongest first).
    pub fn ranked(&self) -> &[MasseyRating] {
        &self.ratings
    }

    /// One team's rating, if it appeared in the fitted results.
    pub fn rating(&self, team: TeamId) -> Option<MasseyRating> {
        self.ratings.iter().find(|r| r.team == team).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.ratings.is_empty()
    }
}

/// Subtract the mean so the vector sums to zero (the natural gauge for a difference-based rating).
fn center(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

/// Solve `a x = b` for a small dense system by Gauss-Jordan elimination with partial pivoting. The
/// systems here are ridged, so the diagonal never vanishes; a defensively-handled zero pivot yields
/// a zero for that component rather than a panic.
// Gaussian elimination is naturally index-based (each row references the pivot row by column).
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let diag = a[col][col];
        if diag.abs() < 1e-12 {
            continue;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }
    (0..n)
        .map(|i| {
            if a[i][i].abs() > 1e-12 {
                b[i] / a[i][i]
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tid(n: u32) -> TeamId {
        TeamId(n)
    }

    fn score(h: u8, a: u8) -> Scoreline {
        Scoreline::new(h, a)
    }

    #[test]
    fn empty_results_give_no_ratings() {
        assert!(MasseyRatings::fit(&[]).is_empty());
    }

    #[test]
    fn a_transitive_result_set_orders_teams_correctly() {
        // A beats B, B beats C, and A beats C: the strict pecking order A > B > C.
        let results = [
            (tid(1), tid(2), score(2, 0)),
            (tid(2), tid(3), score(2, 0)),
            (tid(1), tid(3), score(3, 0)),
        ];
        let m = MasseyRatings::fit(&results);
        let ranked = m.ranked();
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].team, tid(1));
        assert_eq!(ranked[1].team, tid(2));
        assert_eq!(ranked[2].team, tid(3));
        // Ratings are centered, so they sum to (about) zero.
        let sum: f64 = ranked.iter().map(|r| r.rating).sum();
        assert!(sum.abs() < 1e-9, "centered ratings sum to zero, got {sum}");
        // The decomposition reconstructs the overall rating exactly.
        for r in ranked {
            assert!((r.rating - (r.offense + r.defense)).abs() < 1e-9);
        }
    }

    #[test]
    fn strength_of_schedule_rewards_beating_a_stronger_opponent() {
        // Two teams each win by the same 1-0 margin, but one (2) beat the clearly strong team (1)
        // while the other (4) beat a weak team (3). The one who beat the stronger side rates higher.
        let results = [
            (tid(1), tid(3), score(5, 0)), // 1 is strong
            (tid(1), tid(4), score(4, 0)),
            (tid(2), tid(1), score(1, 0)), // 2 beat the strong side by one
            (tid(4), tid(3), score(1, 0)), // 4 beat the weak side by one
        ];
        let m = MasseyRatings::fit(&results);
        let r2 = m.rating(tid(2)).unwrap().rating;
        let r4 = m.rating(tid(4)).unwrap().rating;
        assert!(
            r2 > r4,
            "beating the stronger opponent should rate higher: r2={r2}, r4={r4}"
        );
    }

    #[test]
    fn offense_and_defense_capture_how_a_team_wins() {
        // Team 1 wins by scoring a lot (shoot-out wins); team 2 wins by shutting teams out. Same kind
        // of dominance, different source: 1 should out-rate 2 on offense, 2 on defense.
        let results = [
            (tid(1), tid(3), score(5, 3)),
            (tid(1), tid(4), score(4, 2)),
            (tid(2), tid(3), score(1, 0)),
            (tid(2), tid(4), score(2, 0)),
        ];
        let m = MasseyRatings::fit(&results);
        let a = m.rating(tid(1)).unwrap();
        let b = m.rating(tid(2)).unwrap();
        assert!(
            a.offense > b.offense,
            "the high-scoring team has the stronger offense: {} vs {}",
            a.offense,
            b.offense
        );
        assert!(
            b.defense > a.defense,
            "the shut-out team has the stronger defense: {} vs {}",
            b.defense,
            a.defense
        );
    }
}
