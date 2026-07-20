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
use serde::{Deserialize, Serialize};

/// The width of the grid (goal counts `0..=max` modelled per side).
fn width(grid: &ScoreGrid) -> usize {
    grid.grid.first().map_or(0, |row| row.len())
}

/// The distribution over the total goals in the match: `dist[k] = P(home + away == k)`.
pub fn total_goals_distribution(grid: &ScoreGrid) -> Vec<f64> {
    let (rows, cols) = (grid.grid.len(), width(grid));
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let mut dist = vec![0.0; (rows - 1) + (cols - 1) + 1];
    for (h, row) in grid.grid.iter().enumerate() {
        for (a, &p) in row.iter().enumerate() {
            dist[h + a] += p;
        }
    }
    dist
}

/// The expected number of goals in the match.
pub fn expected_total_goals(grid: &ScoreGrid) -> f64 {
    total_goals_distribution(grid)
        .iter()
        .enumerate()
        .map(|(k, &p)| k as f64 * p)
        .sum()
}

/// One over/under line: the probability the total is over the line, and under it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TotalsLine {
    pub line: f64,
    pub over: f64,
    pub under: f64,
}

/// The totals market: the expected total, the over/under ladder for the given lines, and the full
/// total-goals distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Totals {
    pub expected: f64,
    pub lines: Vec<TotalsLine>,
    pub distribution: Vec<f64>,
}

/// The standard over/under lines a book quotes.
pub const STANDARD_TOTALS: [f64; 5] = [0.5, 1.5, 2.5, 3.5, 4.5];

/// Price the totals market over `lines` (goal lines are half-integers, so there is no push).
pub fn totals(grid: &ScoreGrid, lines: &[f64]) -> Totals {
    let dist = total_goals_distribution(grid);
    let priced = lines
        .iter()
        .map(|&line| {
            let over = dist
                .iter()
                .enumerate()
                .filter(|(k, _)| *k as f64 > line)
                .map(|(_, &p)| p)
                .sum();
            let under = dist
                .iter()
                .enumerate()
                .filter(|(k, _)| (*k as f64) < line)
                .map(|(_, &p)| p)
                .sum();
            TotalsLine { line, over, under }
        })
        .collect();
    Totals {
        expected: dist.iter().enumerate().map(|(k, &p)| k as f64 * p).sum(),
        lines: priced,
        distribution: dist,
    }
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

/// The goal-based side markets: both teams to score, clean sheets, and win-to-nil.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GoalMarkets {
    pub btts_yes: f64,
    pub btts_no: f64,
    /// Home keeps a clean sheet (the away side fails to score), and vice versa.
    pub clean_sheet_home: f64,
    pub clean_sheet_away: f64,
    /// Home wins without conceding, and vice versa.
    pub win_to_nil_home: f64,
    pub win_to_nil_away: f64,
}

/// Price the goal-based side markets off the grid.
pub fn goal_markets(grid: &ScoreGrid) -> GoalMarkets {
    let mut btts_yes = 0.0;
    let mut win_to_nil_home = 0.0;
    let mut win_to_nil_away = 0.0;
    for (h, row) in grid.grid.iter().enumerate() {
        for (a, &p) in row.iter().enumerate() {
            if h >= 1 && a >= 1 {
                btts_yes += p;
            }
            if h >= 1 && a == 0 {
                win_to_nil_home += p;
            }
            if a >= 1 && h == 0 {
                win_to_nil_away += p;
            }
        }
    }
    let total = total_probability(grid);
    let home_goals = home_goal_distribution(grid);
    let away_goals = away_goal_distribution(grid);
    GoalMarkets {
        btts_yes,
        btts_no: total - btts_yes,
        // Home's clean sheet means the away side scored zero.
        clean_sheet_home: away_goals.first().copied().unwrap_or(0.0),
        clean_sheet_away: home_goals.first().copied().unwrap_or(0.0),
        win_to_nil_home,
        win_to_nil_away,
    }
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
    fn totals_ladder_and_distribution_are_exact() {
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        let dist = total_goals_distribution(&g);
        approx(dist[0], 0.1); // 0-0
        approx(dist[1], 0.5); // 0-1 and 1-0
        approx(dist[2], 0.4); // 1-1
        approx(expected_total_goals(&g), 1.3);

        let t = totals(&g, &[0.5, 1.5, 2.5]);
        approx(t.lines[0].over, 0.9);
        approx(t.lines[0].under, 0.1);
        approx(t.lines[1].over, 0.4);
        approx(t.lines[1].under, 0.6);
        approx(t.lines[2].over, 0.0);
        approx(t.lines[2].under, 1.0);
        // Over and under partition the mass on a half line.
        for l in &t.lines {
            approx(l.over + l.under, 1.0);
        }
    }

    #[test]
    fn goal_markets_price_off_the_grid() {
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        let m = goal_markets(&g);
        approx(m.btts_yes, 0.4); // only 1-1
        approx(m.btts_no, 0.6);
        approx(m.clean_sheet_home, 0.4); // away scored 0: 0-0 and 1-0
        approx(m.clean_sheet_away, 0.3); // home scored 0: 0-0 and 0-1
        approx(m.win_to_nil_home, 0.3); // 1-0
        approx(m.win_to_nil_away, 0.2); // 0-1
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
