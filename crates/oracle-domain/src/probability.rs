//! Outcomes, win/draw/win probabilities, and exact-score grids.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The three mutually-exclusive results of a match from the home team's view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    HomeWin,
    Draw,
    AwayWin,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Outcome::HomeWin => "home win",
            Outcome::Draw => "draw",
            Outcome::AwayWin => "away win",
        };
        f.write_str(s)
    }
}

/// A normalized win/draw/win probability distribution. Construction always
/// normalizes, so by invariant `home_win + draw + away_win == 1` (within fp error).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Probabilities {
    pub home_win: f64,
    pub draw: f64,
    pub away_win: f64,
}

impl Probabilities {
    /// Build from raw (possibly unnormalized, possibly negative) weights.
    /// Negative inputs are clamped to zero; an all-zero input falls back to uniform.
    pub fn new(home_win: f64, draw: f64, away_win: f64) -> Self {
        let h = home_win.max(0.0);
        let d = draw.max(0.0);
        let a = away_win.max(0.0);
        let sum = h + d + a;
        if sum <= f64::EPSILON {
            return Self::uniform();
        }
        Self {
            home_win: h / sum,
            draw: d / sum,
            away_win: a / sum,
        }
    }

    pub fn uniform() -> Self {
        let third = 1.0 / 3.0;
        Self {
            home_win: third,
            draw: third,
            away_win: third,
        }
    }

    /// Probability assigned to a specific outcome.
    pub fn of(&self, outcome: Outcome) -> f64 {
        match outcome {
            Outcome::HomeWin => self.home_win,
            Outcome::Draw => self.draw,
            Outcome::AwayWin => self.away_win,
        }
    }

    /// The single most likely outcome.
    pub fn most_likely(&self) -> Outcome {
        if self.home_win >= self.draw && self.home_win >= self.away_win {
            Outcome::HomeWin
        } else if self.away_win >= self.draw {
            Outcome::AwayWin
        } else {
            Outcome::Draw
        }
    }

    /// Should always be ~1.0 by construction; used by tests/assertions.
    pub fn sum(&self) -> f64 {
        self.home_win + self.draw + self.away_win
    }

    /// Convert a probability to fair decimal odds (1/p). Useful for display.
    pub fn fair_odds(&self, outcome: Outcome) -> f64 {
        let p = self.of(outcome);
        if p <= f64::EPSILON {
            f64::INFINITY
        } else {
            1.0 / p
        }
    }
}

/// A full joint distribution over exact scorelines, indexed `grid[home][away]`.
///
/// This is the richest output of the goal model: collapsing it gives win/draw/win
/// probabilities, the modal scoreline, clean-sheet and over/under markets, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreGrid {
    /// Goal counts `0..=max_goals` are modelled exactly; the tail is folded in.
    pub max_goals: usize,
    /// `grid[h][a]` = P(home scores h, away scores a). Rows = home goals.
    pub grid: Vec<Vec<f64>>,
}

impl ScoreGrid {
    /// Build a grid from a closure giving the (unnormalized) probability of each
    /// `(home_goals, away_goals)` cell, then normalize so the whole grid sums to 1.
    pub fn from_fn<F>(max_goals: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> f64,
    {
        let mut grid = vec![vec![0.0; max_goals + 1]; max_goals + 1];
        let mut sum = 0.0;
        for (h, row) in grid.iter_mut().enumerate() {
            for (a, cell) in row.iter_mut().enumerate() {
                let v = f(h, a).max(0.0);
                *cell = v;
                sum += v;
            }
        }
        if sum > f64::EPSILON {
            for row in &mut grid {
                for cell in row {
                    *cell /= sum;
                }
            }
        }
        Self { max_goals, grid }
    }

    /// Collapse the joint distribution into win/draw/win probabilities.
    pub fn outcome_probabilities(&self) -> Probabilities {
        let mut home = 0.0;
        let mut draw = 0.0;
        let mut away = 0.0;
        for (h, row) in self.grid.iter().enumerate() {
            for (a, &p) in row.iter().enumerate() {
                match h.cmp(&a) {
                    std::cmp::Ordering::Greater => home += p,
                    std::cmp::Ordering::Equal => draw += p,
                    std::cmp::Ordering::Less => away += p,
                }
            }
        }
        Probabilities::new(home, draw, away)
    }

    /// The single most likely exact scoreline and its probability.
    pub fn most_likely_score(&self) -> (u8, u8, f64) {
        let mut best = (0u8, 0u8, 0.0f64);
        for (h, row) in self.grid.iter().enumerate() {
            for (a, &p) in row.iter().enumerate() {
                if p > best.2 {
                    best = (h as u8, a as u8, p);
                }
            }
        }
        best
    }

    /// The `n` most likely exact scorelines as `(home_goals, away_goals, probability)`, most likely
    /// first. Fewer than `n` are returned only if the grid holds fewer cells than that.
    ///
    /// This is [`most_likely_score`](Self::most_likely_score) generalized to a ranking - the usual
    /// way a forecast is actually presented, since "2-1 at 9%" alone hides that 1-1 was 8.7%. Ties
    /// break towards the lower scoreline in row-major order (the sort is stable), so the ranking is
    /// deterministic for a given grid rather than dependent on the sort's internals.
    pub fn top_scorelines(&self, n: usize) -> Vec<(usize, usize, f64)> {
        let mut cells: Vec<(usize, usize, f64)> = self
            .grid
            .iter()
            .enumerate()
            .flat_map(|(h, row)| row.iter().enumerate().map(move |(a, &p)| (h, a, p)))
            .collect();
        cells.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
        cells.truncate(n);
        cells
    }

    /// P(total goals strictly greater than `line`), e.g. `line = 2.5` → over 2.5.
    pub fn prob_over(&self, line: f64) -> f64 {
        let mut p = 0.0;
        for (h, row) in self.grid.iter().enumerate() {
            for (a, &cell) in row.iter().enumerate() {
                if (h + a) as f64 > line {
                    p += cell;
                }
            }
        }
        p
    }

    /// P(both teams score).
    pub fn prob_btts(&self) -> f64 {
        let mut p = 0.0;
        for (h, row) in self.grid.iter().enumerate() {
            for (a, &cell) in row.iter().enumerate() {
                if h >= 1 && a >= 1 {
                    p += cell;
                }
            }
        }
        p
    }

    /// Total probability mass (≈ 1.0 by construction).
    pub fn sum(&self) -> f64 {
        self.grid.iter().flatten().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_normalize() {
        let p = Probabilities::new(2.0, 1.0, 1.0);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert_eq!(p.most_likely(), Outcome::HomeWin);
    }

    #[test]
    fn all_zero_falls_back_to_uniform() {
        let p = Probabilities::new(0.0, 0.0, 0.0);
        assert!((p.home_win - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn negative_weights_are_clamped() {
        let p = Probabilities::new(-5.0, 1.0, 1.0);
        assert!((p.home_win).abs() < 1e-9);
        assert!((p.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn score_grid_collapses_consistently() {
        // Symmetric grid → home and away win probs must match, sum to 1.
        let g = ScoreGrid::from_fn(5, |h, a| 1.0 / ((1 + h + a) as f64));
        assert!((g.sum() - 1.0).abs() < 1e-9);
        let p = g.outcome_probabilities();
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!((p.home_win - p.away_win).abs() < 1e-9);
    }

    #[test]
    fn top_scorelines_are_ranked_and_truncated() {
        // A grid whose single most likely cell is 2-1.
        let g = ScoreGrid::from_fn(4, |h, a| if (h, a) == (2, 1) { 10.0 } else { 1.0 });
        let top = g.top_scorelines(3);
        assert_eq!(top.len(), 3, "truncated to n");
        assert_eq!((top[0].0, top[0].1), (2, 1), "modal scoreline first");
        assert!(top[0].2 >= top[1].2 && top[1].2 >= top[2].2, "descending");
    }

    #[test]
    fn top_scorelines_agrees_with_most_likely_score() {
        let g = ScoreGrid::from_fn(6, |h, a| 1.0 / ((1 + 2 * h + 3 * a) as f64));
        let (h, a, p) = g.most_likely_score();
        let top = g.top_scorelines(1);
        assert_eq!((top[0].0, top[0].1), (h as usize, a as usize));
        assert!((top[0].2 - p).abs() < 1e-15);
    }

    #[test]
    fn top_scorelines_breaks_ties_towards_the_lower_scoreline() {
        // A flat grid: every cell is equally likely, so only the tie-break orders them.
        let g = ScoreGrid::from_fn(3, |_, _| 1.0);
        let top = g.top_scorelines(3);
        assert_eq!(
            [
                (top[0].0, top[0].1),
                (top[1].0, top[1].1),
                (top[2].0, top[2].1)
            ],
            [(0, 0), (0, 1), (0, 2)],
            "ties keep row-major order"
        );
    }

    #[test]
    fn asking_for_more_scorelines_than_exist_returns_them_all() {
        let g = ScoreGrid::from_fn(2, |_, _| 1.0);
        assert_eq!(g.top_scorelines(500).len(), 9, "a 3x3 grid has nine cells");
        assert!(g.top_scorelines(0).is_empty());
    }
}
