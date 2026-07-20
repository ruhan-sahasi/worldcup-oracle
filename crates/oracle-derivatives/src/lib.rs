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

/// The double-chance market: two of the three outcomes each.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DoubleChance {
    pub home_or_draw: f64,
    pub home_or_away: f64,
    pub draw_or_away: f64,
}

/// Double chance, straight from the 1x2 probabilities.
pub fn double_chance(grid: &ScoreGrid) -> DoubleChance {
    let p = grid.outcome_probabilities();
    DoubleChance {
        home_or_draw: p.home_win + p.draw,
        home_or_away: p.home_win + p.away_win,
        draw_or_away: p.draw + p.away_win,
    }
}

/// The draw-no-bet market: the draw is void (stake refunded), so the win probabilities are
/// renormalized over just the two sides.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DrawNoBet {
    pub home: f64,
    pub away: f64,
}

/// Draw-no-bet, renormalizing the 1x2 probabilities with the draw removed.
pub fn draw_no_bet(grid: &ScoreGrid) -> DrawNoBet {
    let p = grid.outcome_probabilities();
    let decisive = p.home_win + p.away_win;
    if decisive <= 0.0 {
        return DrawNoBet {
            home: 0.5,
            away: 0.5,
        };
    }
    DrawNoBet {
        home: p.home_win / decisive,
        away: p.away_win / decisive,
    }
}

/// The distribution over the winning margin `home - away`: `probs[i] = P(margin == min + i)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginDistribution {
    /// The most negative margin represented (a heavy away win).
    pub min: i32,
    pub probs: Vec<f64>,
    pub expected: f64,
}

/// The winning-margin distribution summed off the grid.
pub fn margin_distribution(grid: &ScoreGrid) -> MarginDistribution {
    let (rows, cols) = (grid.grid.len(), width(grid));
    if rows == 0 || cols == 0 {
        return MarginDistribution {
            min: 0,
            probs: Vec::new(),
            expected: 0.0,
        };
    }
    let min = -((cols - 1) as i32);
    let mut probs = vec![0.0; (rows - 1) + (cols - 1) + 1];
    for (h, row) in grid.grid.iter().enumerate() {
        for (a, &p) in row.iter().enumerate() {
            let margin = h as i32 - a as i32;
            probs[(margin - min) as usize] += p;
        }
    }
    let expected = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (min + i as i32) as f64 * p)
        .sum();
    MarginDistribution {
        min,
        probs,
        expected,
    }
}

/// P(the winning margin equals exactly `d`), where `d = home - away`.
pub fn prob_margin(grid: &ScoreGrid, d: i32) -> f64 {
    let dist = margin_distribution(grid);
    let idx = d - dist.min;
    if idx < 0 || idx as usize >= dist.probs.len() {
        0.0
    } else {
        dist.probs[idx as usize]
    }
}

/// One Asian-handicap line, from the home side's perspective. `home_win` / `push` / `away_win` are
/// the effective settlement fractions of a unit stake (a push refunds the stake), and the fair odds
/// price each side accounting for the push refund.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HandicapLine {
    pub line: f64,
    pub home_win: f64,
    pub push: f64,
    pub away_win: f64,
    pub fair_home_odds: f64,
    pub fair_away_odds: f64,
}

/// Settle a half/whole handicap line against the margin distribution: the adjusted margin is the
/// real margin plus the home line, and it wins/pushes/loses as it is positive/zero/negative.
fn settle_line(dist: &MarginDistribution, line: f64) -> (f64, f64, f64) {
    let (mut win, mut push, mut lose) = (0.0, 0.0, 0.0);
    for (i, &p) in dist.probs.iter().enumerate() {
        let adjusted = (dist.min + i as i32) as f64 + line;
        if adjusted > 1e-9 {
            win += p;
        } else if adjusted < -1e-9 {
            lose += p;
        } else {
            push += p;
        }
    }
    (win, push, lose)
}

/// Fair odds for a side that wins with probability `win` when a fraction `push` is refunded:
/// `(1 - push) / win`, clamped to a sane range.
fn fair_odds(win: f64, push: f64) -> f64 {
    if win <= 1e-9 {
        return 1000.0;
    }
    ((1.0 - push) / win).clamp(1.0, 1000.0)
}

/// Price a single Asian-handicap line. A half or whole line (its double is an integer) settles
/// directly; a **quarter** line splits the stake evenly across the two adjacent half/whole lines, so
/// its effective win/push/lose fractions are the average of theirs (the correct expected
/// settlement, capturing half-win and half-loss). Fair odds follow from those fractions.
pub fn handicap_line(grid: &ScoreGrid, line: f64) -> HandicapLine {
    let dist = margin_distribution(grid);
    let (home_win, push, away_win) = if (2.0 * line).fract().abs() < 1e-9 {
        settle_line(&dist, line)
    } else {
        let (w1, p1, l1) = settle_line(&dist, line - 0.25);
        let (w2, p2, l2) = settle_line(&dist, line + 0.25);
        (0.5 * (w1 + w2), 0.5 * (p1 + p2), 0.5 * (l1 + l2))
    };
    HandicapLine {
        line,
        home_win,
        push,
        away_win,
        fair_home_odds: fair_odds(home_win, push),
        fair_away_odds: fair_odds(away_win, push),
    }
}

/// The standard Asian-handicap ladder a book quotes around the level line.
pub const STANDARD_HANDICAPS: [f64; 11] = [
    -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5,
];

/// Price a ladder of handicap lines.
pub fn handicap_ladder(grid: &ScoreGrid, lines: &[f64]) -> Vec<HandicapLine> {
    lines
        .iter()
        .map(|&line| handicap_line(grid, line))
        .collect()
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
    fn double_chance_and_draw_no_bet_follow_the_1x2() {
        // outcome: home 0.3, draw 0.5, away 0.2.
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        let dc = double_chance(&g);
        approx(dc.home_or_draw, 0.8);
        approx(dc.home_or_away, 0.5);
        approx(dc.draw_or_away, 0.7);
        // The three double chances cover every pair, so they sum to twice the whole.
        approx(dc.home_or_draw + dc.home_or_away + dc.draw_or_away, 2.0);

        let dnb = draw_no_bet(&g);
        approx(dnb.home, 0.6); // 0.3 / (0.3 + 0.2)
        approx(dnb.away, 0.4);
        approx(dnb.home + dnb.away, 1.0);
    }

    #[test]
    fn margin_distribution_is_exact() {
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        let m = margin_distribution(&g);
        assert_eq!(m.min, -1);
        approx(m.probs[0], 0.2); // margin -1 (0-1)
        approx(m.probs[1], 0.5); // margin 0 (0-0, 1-1)
        approx(m.probs[2], 0.3); // margin +1 (1-0)
        approx(m.expected, 0.1); // -0.2 + 0 + 0.3
        approx(prob_margin(&g, 0), 0.5);
        approx(prob_margin(&g, 1), 0.3);
        approx(prob_margin(&g, -1), 0.2);
        approx(prob_margin(&g, 5), 0.0);
    }

    #[test]
    fn whole_and_half_handicap_lines_settle_correctly() {
        // margins: -1 -> 0.2, 0 -> 0.5, +1 -> 0.3.
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);

        // Level line (0.0): a level margin pushes.
        let level = handicap_line(&g, 0.0);
        approx(level.home_win, 0.3);
        approx(level.push, 0.5);
        approx(level.away_win, 0.2);
        approx(level.fair_home_odds, 0.5 / 0.3); // (1 - push) / win

        // Home -0.5: home must win outright.
        let minus = handicap_line(&g, -0.5);
        approx(minus.home_win, 0.3);
        approx(minus.push, 0.0);
        approx(minus.away_win, 0.7);

        // Home +0.5: home wins or draws covers.
        let plus = handicap_line(&g, 0.5);
        approx(plus.home_win, 0.8);
        approx(plus.push, 0.0);
        approx(plus.away_win, 0.2);
    }

    #[test]
    fn quarter_handicap_splits_across_the_two_adjacent_lines() {
        let g = grid(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        // Home -0.25 splits between 0.0 (0.3 / 0.5 / 0.2) and -0.5 (0.3 / 0.0 / 0.7):
        // averages to 0.30 / 0.25 / 0.45.
        let q = handicap_line(&g, -0.25);
        approx(q.home_win, 0.30);
        approx(q.push, 0.25);
        approx(q.away_win, 0.45);
        approx(q.home_win + q.push + q.away_win, 1.0);

        let ladder = handicap_ladder(&g, &STANDARD_HANDICAPS);
        assert_eq!(ladder.len(), STANDARD_HANDICAPS.len());
        // Every line's fractions form a distribution.
        for l in &ladder {
            approx(l.home_win + l.push + l.away_win, 1.0);
        }
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
