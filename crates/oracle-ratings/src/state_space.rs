//! A state-space (Kalman / Glicko-style) rating.
//!
//! Each team carries a Gaussian belief about its latent strength, `N(mean, var)`. This is
//! the principled version of "learn during the tournament and track how sure we are":
//!
//! - **Between matches** the strength performs a random walk: variance grows with elapsed
//!   time (we become less certain the longer since we last saw a team).
//! - **At a match** the goal margin is a noisy linear measurement of the strength gap, and a
//!   two-state **Kalman update** moves both teams' means toward the surprise and shrinks
//!   their variances (we learned something).
//!
//! Unlike a point-estimate rating, this yields a per-team *uncertainty* (the variance), which
//! the Monte-Carlo consumes to resample team strength and the win-probability output inflates
//! its spread by, so a thinly-observed team is correctly treated as less certain.

use oracle_domain::{Probabilities, Scoreline, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tunable parameters for the state-space rating (strength is in goal-difference units).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StateSpaceConfig {
    /// Prior mean strength for an unseen team.
    pub initial_mean: f64,
    /// Prior variance for an unseen team (how unsure we start).
    pub initial_var: f64,
    /// Random-walk variance added per day since a team last played.
    pub process_var_per_day: f64,
    /// Random-walk variance injected per *in-tournament* match (see [`observe_tournament`]).
    /// Tournament matches are only days apart, so the day-based walk barely moves the filter and
    /// lets it grow overconfident and sluggish; a fixed per-match bump keeps it reactive to
    /// tournament form (a team's true level shifts with form, injuries, and momentum in a
    /// tournament, faster than the calendar gap alone implies).
    ///
    /// [`observe_tournament`]: StateSpaceRatings::observe_tournament
    pub tournament_process_var: f64,
    /// Measurement noise on a single match's goal margin.
    pub obs_var: f64,
    /// Maps a one-unit strength gap to expected goal-margin (here strength *is* in margin
    /// units, so 1.0); kept explicit for clarity and tuning.
    pub scale: f64,
    /// Home-advantage in goal-margin units (0 at a neutral venue / the World Cup default).
    pub home_advantage: f64,
    /// Half-width of the draw band, in goal-margin units, for the three-way split.
    pub draw_band: f64,
}

impl Default for StateSpaceConfig {
    fn default() -> Self {
        Self {
            initial_mean: 0.0,
            initial_var: 1.0,
            process_var_per_day: 0.0015,
            tournament_process_var: 0.03,
            obs_var: 1.8,
            scale: 1.0,
            home_advantage: 0.0,
            draw_band: 0.75,
        }
    }
}

/// A Gaussian rating store updated by a Kalman filter.
#[derive(Debug, Clone)]
pub struct StateSpaceRatings {
    config: StateSpaceConfig,
    /// `(mean, var)` per team.
    state: HashMap<TeamId, (f64, f64)>,
    /// Age (in days-before-now units; smaller = more recent) at which each team last played,
    /// used to inflate variance for the elapsed gap.
    last_age: HashMap<TeamId, f64>,
}

impl StateSpaceRatings {
    pub fn new(config: StateSpaceConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
            last_age: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(StateSpaceConfig::default())
    }

    fn get(&self, team: TeamId) -> (f64, f64) {
        self.state
            .get(&team)
            .copied()
            .unwrap_or((self.config.initial_mean, self.config.initial_var))
    }

    /// Seed a team's prior mean (variance left at the configured initial).
    pub fn seed(&mut self, team: TeamId, mean: f64) {
        self.state
            .entry(team)
            .or_insert((mean, self.config.initial_var))
            .0 = mean;
    }

    pub fn mean(&self, team: TeamId) -> f64 {
        self.get(team).0
    }

    pub fn stddev(&self, team: TeamId) -> f64 {
        self.get(team).1.max(0.0).sqrt()
    }

    /// Random-walk predict step: inflate a team's variance for the days elapsed since it last
    /// played (passed as the difference in `age_days`, which decreases over time).
    fn predict_to(&mut self, team: TeamId, age_days: f64) {
        let (mean, mut var) = self.get(team);
        if let Some(&prev) = self.last_age.get(&team) {
            let dt = (prev - age_days).max(0.0); // older record -> larger age -> positive gap
            var += self.config.process_var_per_day * dt;
        }
        self.state.insert(team, (mean, var));
        self.last_age.insert(team, age_days);
    }

    /// Observe a finished match (`age_days` = how long ago; smaller is more recent): predict
    /// both teams forward by the elapsed-time random walk, then a two-state Kalman update from
    /// the goal margin.
    pub fn observe(
        &mut self,
        home: TeamId,
        away: TeamId,
        score: Scoreline,
        age_days: f64,
        neutral: bool,
    ) {
        self.predict_to(home, age_days);
        self.predict_to(away, age_days);
        self.kalman_update(home, away, score, neutral);
    }

    /// Observe an **in-tournament** result. Rather than the day-based random walk (tournament
    /// matches are only days apart, so it would barely move the filter and let it grow
    /// overconfident and sluggish over the competition), inject a fixed process-noise bump per
    /// match before the Kalman update. That keeps the Kalman gain up, so recent tournament results
    /// move the estimate faster and the per-team uncertainty the Monte-Carlo consumes does not
    /// collapse as the tournament wears on.
    pub fn observe_tournament(
        &mut self,
        home: TeamId,
        away: TeamId,
        score: Scoreline,
        neutral: bool,
    ) {
        self.bump_variance(home);
        self.bump_variance(away);
        self.kalman_update(home, away, score, neutral);
    }

    /// Inject the per-match tournament process noise and mark the team as just-played (age 0).
    fn bump_variance(&mut self, team: TeamId) {
        let (mean, var) = self.get(team);
        self.state.insert(
            team,
            (mean, var + self.config.tournament_process_var.max(0.0)),
        );
        self.last_age.insert(team, 0.0);
    }

    /// The two-state Kalman measurement update from a match's goal margin, shared by [`observe`]
    /// and [`observe_tournament`] (which differ only in the predict step that precedes it).
    ///
    /// [`observe`]: Self::observe
    /// [`observe_tournament`]: Self::observe_tournament
    fn kalman_update(&mut self, home: TeamId, away: TeamId, score: Scoreline, neutral: bool) {
        let (mh, ph) = self.get(home);
        let (ma, pa) = self.get(away);
        let adv = if neutral {
            0.0
        } else {
            self.config.home_advantage
        };
        let c = self.config.scale;

        let z = f64::from(score.home) - f64::from(score.away); // observed margin
        let predicted = c * (mh - ma) + adv;
        let innovation = z - predicted;
        let s = c * c * (ph + pa) + self.config.obs_var; // innovation variance
        let k_h = c * ph / s;
        let k_a = -c * pa / s;

        self.state
            .insert(home, (mh + k_h * innovation, ph * (1.0 - c * c * ph / s)));
        self.state
            .insert(away, (ma + k_a * innovation, pa * (1.0 - c * c * pa / s)));
    }

    /// Win/draw/win probabilities from the strength gap, with the predictive uncertainty
    /// (state variance + measurement noise) widening the distribution.
    pub fn win_probabilities(&self, home: TeamId, away: TeamId, neutral: bool) -> Probabilities {
        let (mh, ph) = self.get(home);
        let (ma, pa) = self.get(away);
        let adv = if neutral {
            0.0
        } else {
            self.config.home_advantage
        };
        let mu = self.config.scale * (mh - ma) + adv;
        let sigma = (self.config.scale * self.config.scale * (ph + pa) + self.config.obs_var)
            .max(1e-6)
            .sqrt();
        let d = self.config.draw_band;
        // P(margin > d) = home win; P(margin < -d) = away win; the band between is a draw.
        let p_home = 1.0 - normal_cdf((d - mu) / sigma);
        let p_away = normal_cdf((-d - mu) / sigma);
        let p_draw = 1.0 - p_home - p_away;
        Probabilities::new(p_home, p_draw, p_away)
    }
}

/// Standard normal CDF via the Abramowitz-Stegun `erf` approximation (max abs error ~1.5e-7),
/// avoiding a heavyweight stats dependency.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: u32) -> TeamId {
        TeamId(n)
    }

    #[test]
    fn observing_shrinks_variance() {
        let mut r = StateSpaceRatings::with_defaults();
        let before = r.get(t(1)).1;
        r.observe(t(1), t(2), Scoreline::new(2, 0), 10.0, true);
        let after = r.get(t(1)).1;
        assert!(
            after < before,
            "a match should reduce variance ({before} -> {after})"
        );
    }

    #[test]
    fn a_time_gap_grows_variance() {
        let mut r = StateSpaceRatings::with_defaults();
        // First match at 100 days ago, second at 0 days ago (a 100-day gap before it).
        r.observe(t(1), t(2), Scoreline::new(1, 0), 100.0, true);
        let after_first = r.get(t(1)).1;
        r.observe(t(1), t(3), Scoreline::new(1, 0), 0.0, true);
        // The predict step before the second match injected process noise; even after the
        // update the team has been re-inflated relative to back-to-back matches.
        let mut back_to_back = StateSpaceRatings::with_defaults();
        back_to_back.observe(t(1), t(2), Scoreline::new(1, 0), 1.0, true);
        back_to_back.observe(t(1), t(3), Scoreline::new(1, 0), 0.0, true);
        assert!(
            r.get(t(1)).1 > back_to_back.get(t(1)).1,
            "a long gap should leave more residual variance"
        );
        let _ = after_first;
    }

    #[test]
    fn tournament_observations_stay_reactive_where_a_zero_gap_goes_stale() {
        // The same six results fed two ways: as in-tournament matches (fixed process-noise bump)
        // vs as back-to-back matches at age 0 (no random walk, the old engine behaviour). The
        // tournament filter keeps more variance (stays reactive) and tracks the winning team
        // higher and faster.
        let mut tourney = StateSpaceRatings::with_defaults();
        let mut zero_gap = StateSpaceRatings::with_defaults();
        for _ in 0..6 {
            tourney.observe_tournament(t(1), t(2), Scoreline::new(2, 0), true);
            zero_gap.observe(t(1), t(2), Scoreline::new(2, 0), 0.0, true);
        }
        assert!(
            tourney.stddev(t(1)) > zero_gap.stddev(t(1)),
            "tournament process noise should keep the filter reactive ({} vs {})",
            tourney.stddev(t(1)),
            zero_gap.stddev(t(1))
        );
        assert!(
            tourney.mean(t(1)) > zero_gap.mean(t(1)),
            "a higher Kalman gain should track the rising team faster ({} vs {})",
            tourney.mean(t(1)),
            zero_gap.mean(t(1))
        );
    }

    #[test]
    fn stronger_team_is_favoured_and_probs_sum_to_one() {
        let mut r = StateSpaceRatings::with_defaults();
        r.seed(t(1), 1.0); // strong (a goal better, in margin units)
        r.seed(t(2), -1.0); // weak
        let p = r.win_probabilities(t(1), t(2), true);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!(p.home_win > p.away_win, "stronger team should be favoured");
        assert!(p.home_win > 0.0 && p.draw > 0.0 && p.away_win > 0.0);
    }

    #[test]
    fn repeated_wins_raise_the_mean() {
        let mut r = StateSpaceRatings::with_defaults();
        let before = r.mean(t(1));
        for i in 0..6 {
            r.observe(t(1), t(2), Scoreline::new(3, 0), 6.0 - i as f64, true);
        }
        assert!(
            r.mean(t(1)) > before,
            "repeated big wins should raise the rating"
        );
        assert!(r.mean(t(2)) < 0.0, "the beaten team's rating should fall");
    }

    #[test]
    fn normal_cdf_is_sane() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!(normal_cdf(3.0) > 0.99 && normal_cdf(-3.0) < 0.01);
    }
}
