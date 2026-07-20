//! # oracle-derivatives
//!
//! The derivative markets a book actually prices, all read **exactly** off the model's bivariate
//! score grid. The forecasting core gives win/draw/win and the players crate gives goalscorers; this
//! crate turns the joint distribution over scorelines into the rest of the board: **totals**,
//! **both teams to score**, **clean sheets**, **double chance** and **draw-no-bet**, the **winning
//! margin**, **Asian handicaps** (with the quarter-line stake split and push refunds), and the
//! **correct-score** market.
//!
//! Everything is a closed-form sum over the grid, no Monte-Carlo needed, so it is small, pure, and
//! unit-testable against grids whose answers can be worked out by hand. The grid comes from the goal
//! model upstream; this crate only re-expresses it as prices.
//!
//! This first layer is the shared vocabulary: the joint grid's **marginal** goal distributions (how
//! many each side scores) and its total mass, the primitives the market modules are built from.
#![forbid(unsafe_code)]

use oracle_domain::ScoreGrid;

/// The width of the grid (goal counts `0..=max` modelled per side).
fn width(grid: &ScoreGrid) -> usize {
    grid.grid.first().map_or(0, |row| row.len())
}

/// P(home team scores exactly `h`) for each `h`, marginalizing over the away score.
pub fn home_goal_distribution(grid: &ScoreGrid) -> Vec<f64> {
    grid.grid.iter().map(|row| row.iter().sum()).collect()
}

/// P(away team scores exactly `a`) for each `a`, marginalizing over the home score.
pub fn away_goal_distribution(grid: &ScoreGrid) -> Vec<f64> {
    (0..width(grid))
        .map(|a| grid.grid.iter().map(|row| row[a]).sum())
        .collect()
}

/// The total probability mass in the grid (about 1 for a proper distribution).
pub fn total_probability(grid: &ScoreGrid) -> f64 {
    grid.grid.iter().flat_map(|row| row.iter()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid straight from a matrix of `grid[h][a]` cells (already summing to one in the tests).
    pub(crate) fn grid(rows: Vec<Vec<f64>>) -> ScoreGrid {
        ScoreGrid {
            max_goals: rows.len().saturating_sub(1),
            grid: rows,
        }
    }

    pub(crate) fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn marginals_and_total_are_correct() {
        // rows = home goals, cols = away goals.
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        approx(total_probability(&g), 1.0);
        let home = home_goal_distribution(&g);
        approx(home[0], 0.3); // 0.1 + 0.2
        approx(home[1], 0.7); // 0.3 + 0.4
        let away = away_goal_distribution(&g);
        approx(away[0], 0.4); // 0.1 + 0.3
        approx(away[1], 0.6); // 0.2 + 0.4
    }
}
