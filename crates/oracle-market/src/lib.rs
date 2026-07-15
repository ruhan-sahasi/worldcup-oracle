//! # oracle-market
//!
//! Betting-market math for the oracle, and a paper-trading backtest built on top of it. The
//! question it answers is the honest one every forecaster should ask: **does the model actually
//! beat the price?** A model can be well-calibrated and still lose money to the vig, so this crate
//! turns model probabilities and a bookmaker's line into edges, stakes, and a settled bankroll, and
//! reports the result without spin.
//!
//! It is deliberately small and pure: every function is a plain calculation over `f64`s and
//! [`oracle_domain::Probabilities`], with no I/O, so the numbers are unit-testable in isolation.
//!
//! This first layer is the vocabulary: **decimal odds** and their relationship to probability. A
//! decimal price `d` pays `d` per unit staked on a winner (profit `d - 1`); its *raw* implied
//! probability is `1 / d`, and because a book prices in a margin those raw implied probabilities
//! sum to `1 + overround` rather than to one (removing that margin is [`devig`](crate) territory,
//! the next layer up).
#![forbid(unsafe_code)]

use oracle_domain::Probabilities;
use serde::{Deserialize, Serialize};

/// Decimal (European) odds for the three match outcomes: home win, draw, away win.
///
/// A decimal price `d` returns `d` times the stake on a win (profit `d - 1`) and 0 on a loss. The
/// smallest sensible price is `1.0` (an outcome paying nothing above stake), which is enforced as a
/// floor so the reciprocals below never blow up.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Odds {
    pub home: f64,
    pub draw: f64,
    pub away: f64,
}

impl Odds {
    /// A line from three decimal prices, each floored at `1.0`.
    pub fn new(home: f64, draw: f64, away: f64) -> Self {
        Self {
            home: home.max(1.0),
            draw: draw.max(1.0),
            away: away.max(1.0),
        }
    }

    /// The raw implied probabilities `1 / d` for `[home, draw, away]`. These sum to
    /// `1 + overround`, not to one, because the price includes the bookmaker's margin.
    pub fn implied(&self) -> [f64; 3] {
        [1.0 / self.home, 1.0 / self.draw, 1.0 / self.away]
    }

    /// The bookmaker's **overround** (margin): the raw implied probabilities' sum minus one. A fair
    /// (marginless) book has overround `0`; a typical football book is around `0.05` to `0.08`.
    pub fn overround(&self) -> f64 {
        self.implied().iter().sum::<f64>() - 1.0
    }

    /// Price a synthetic book from fair probabilities by shortening each fair price `1 / p`
    /// proportionally by `1 + margin`, so the resulting line carries exactly that overround. The
    /// inverse of proportional (multiplicative) de-vigging, and how the backtest turns a vig-free
    /// market estimate into the vigged prices a bettor would actually face.
    pub fn from_fair(fair: Probabilities, margin: f64) -> Self {
        let m = 1.0 + margin.max(0.0);
        let price = |p: f64| 1.0 / (m * p.clamp(1e-9, 1.0));
        Self::new(price(fair.home_win), price(fair.draw), price(fair.away_win))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn decimal_prices_are_floored_at_one() {
        let o = Odds::new(0.5, -3.0, 1.0);
        assert_eq!(o.home, 1.0);
        assert_eq!(o.draw, 1.0);
        assert_eq!(o.away, 1.0);
    }

    #[test]
    fn implied_probabilities_are_reciprocals() {
        let o = Odds::new(2.0, 4.0, 4.0);
        let imp = o.implied();
        approx(imp[0], 0.5);
        approx(imp[1], 0.25);
        approx(imp[2], 0.25);
    }

    #[test]
    fn a_fair_book_has_no_overround_and_a_priced_book_carries_its_margin() {
        // 1/2 + 1/4 + 1/4 = 1 exactly: no margin.
        approx(Odds::new(2.0, 4.0, 4.0).overround(), 0.0);
        // A book priced from fair probabilities with a 6% margin carries a 6% overround.
        let fair = Probabilities::new(0.5, 0.25, 0.25);
        approx(Odds::from_fair(fair, 0.06).overround(), 0.06);
    }

    #[test]
    fn from_fair_with_no_margin_round_trips_to_the_probabilities() {
        let fair = Probabilities::new(0.55, 0.25, 0.20);
        let imp = Odds::from_fair(fair, 0.0).implied();
        approx(imp[0], 0.55);
        approx(imp[1], 0.25);
        approx(imp[2], 0.20);
    }
}
