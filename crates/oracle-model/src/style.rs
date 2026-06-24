//! Playing-style matchups: a non-additive (bilinear) correction to expected goals.
//!
//! Every other strength signal in the model is *additive*: a team's rating minus its opponent's.
//! But football has matchups that defy ratings - a disciplined low block frustrates a possession
//! side, a relentless press rattles a team that wants to build slowly. Those effects are
//! inherently an *interaction* between the two teams' styles, which an additive model cannot
//! represent.
//!
//! We give each team a low-dimensional **style embedding** and score a matchup with a bilinear
//! form `sₕᵀ M sₐ`. With an antisymmetric `M` the interaction is *non-transitive* - a
//! rock-paper-scissors cycle where style A troubles style B, B troubles C, and C troubles A -
//! which is exactly the structure additive ratings miss. The scalar tilt is applied to the goal
//! rates (home up, away down, or vice versa) on top of the usual strength terms.
//!
//! Here the embeddings are unit vectors (a "style angle"), so the antisymmetric form reduces to
//! `K·sin(θₐ − θₕ)`: orthogonal styles produce the maximum tilt, identical styles none, and
//! swapping the two teams flips the sign. The embeddings themselves are assigned synthetically
//! offline; on real data they would be fit from match residuals (a low-rank factorization of the
//! part of the result that strength alone does not explain).

/// Dimensionality of a team's style embedding.
pub const STYLE_DIM: usize = 2;

/// Strength of the style interaction, in log goal-rate units (the maximum matchup tilt).
const STYLE_K: f64 = 0.08;

/// A team's playing-style embedding - a unit vector in style space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StyleProfile {
    pub axes: [f64; STYLE_DIM],
}

impl StyleProfile {
    pub fn new(axes: [f64; STYLE_DIM]) -> Self {
        Self { axes }
    }

    /// A style-neutral profile (zero vector), which produces no matchup tilt against anyone.
    pub fn neutral() -> Self {
        Self {
            axes: [0.0; STYLE_DIM],
        }
    }
}

/// The bilinear style tilt in favour of the home side, `K · (hₓ·aᵧ − hᵧ·aₓ)` (the antisymmetric
/// form `hᵀ M a` with `M = [[0, K], [-K, 0]]`). Positive = home's style troubles the away side;
/// negative = the reverse. Antisymmetric: swapping the teams negates it; identical styles give 0.
pub fn style_tilt(home: &StyleProfile, away: &StyleProfile) -> f64 {
    let (h, a) = (home.axes, away.axes);
    STYLE_K * (h[0] * a[1] - h[1] * a[0])
}

/// The style matchup as a per-team log-space `((home_attack, home_defense), (away_attack,
/// away_defense))` adjustment, matching the shape used by venue and lineup adjustments so it
/// flows through [`crate::GoalModel::expected_goals_adjusted`] unchanged. The tilt is applied to
/// the attacking output of each side (a pure goal-difference effect).
pub fn style_adjustment(home: &StyleProfile, away: &StyleProfile) -> ((f64, f64), (f64, f64)) {
    let tilt = style_tilt(home, away);
    ((tilt, 0.0), (-tilt, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn at_angle(theta: f64) -> StyleProfile {
        StyleProfile::new([theta.cos(), theta.sin()])
    }

    #[test]
    fn identical_styles_have_no_edge() {
        let s = at_angle(0.7);
        assert!(style_tilt(&s, &s).abs() < 1e-12);
    }

    #[test]
    fn the_matchup_is_antisymmetric() {
        let a = at_angle(0.3);
        let b = at_angle(1.9);
        assert!((style_tilt(&a, &b) + style_tilt(&b, &a)).abs() < 1e-12);
    }

    #[test]
    fn orthogonal_styles_give_the_maximum_tilt() {
        let home = at_angle(0.0); // [1, 0]
        let away = at_angle(FRAC_PI_2); // [0, 1]
                                        // tilt = K * (1*1 - 0*0) = K.
        assert!((style_tilt(&home, &away) - STYLE_K).abs() < 1e-12);
        // A non-orthogonal matchup is strictly smaller in magnitude.
        let near = at_angle(0.3);
        assert!(style_tilt(&home, &near).abs() < STYLE_K);
    }

    #[test]
    fn adjustment_tilts_the_goal_difference() {
        let home = at_angle(0.0);
        let away = at_angle(FRAC_PI_2);
        let ((ha, hd), (aa, ad)) = style_adjustment(&home, &away);
        assert!(ha > 0.0 && aa < 0.0, "home favoured by the matchup");
        assert!((ha + aa).abs() < 1e-12, "a pure goal-difference tilt");
        assert_eq!((hd, ad), (0.0, 0.0), "defenses untouched");
    }

    #[test]
    fn a_neutral_profile_never_tilts() {
        let n = StyleProfile::neutral();
        assert_eq!(style_tilt(&n, &at_angle(1.2)), 0.0);
        assert_eq!(style_tilt(&at_angle(1.2), &n), 0.0);
    }
}
