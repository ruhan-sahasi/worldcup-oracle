//! A **Bradley-Terry-Davidson** paired-comparison model: the second, alternative forecaster.
//!
//! Where [`dixon_coles`](crate::dixon_coles) models *goals* (a bivariate Poisson), this models the
//! **match outcome directly** - win, draw, or loss - as a function of a single latent *strength*
//! per team. It is a different model family, fit on who-beat-whom rather than how-many-goals, and
//! it comes into its own **deep in a tournament**, when a large set of real results is available: a
//! time-weighted fit lets those recent games dominate the pre-tournament priors.
//!
//! For two teams with log-strengths `βᵢ`, `βⱼ` (`πᵢ = exp βᵢ`), Davidson's (1970) tie extension is
//!
//! ```text
//! P(i beats j) = πᵢ / (πᵢ + πⱼ + ν·√(πᵢπⱼ))
//! P(draw)      = ν·√(πᵢπⱼ) / (πᵢ + πⱼ + ν·√(πᵢπⱼ))
//! P(j beats i) = πⱼ / (πᵢ + πⱼ + ν·√(πᵢπⱼ))
//! ```
//!
//! with a tie parameter `ν ≥ 0` (larger = more draws) and an additive home term dropped at a
//! neutral venue (the World Cup default). Dropping the tie term recovers plain Bradley-Terry,
//! `πᵢ/(πᵢ+πⱼ)`, the natural "who advances" probability for a knockout tie.

use crate::dixon_coles::Observation;
use oracle_domain::{Outcome, Probabilities, TeamId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hyper-parameters for [`BradleyTerry::fit`].
#[derive(Debug, Clone, Copy)]
pub struct BradleyTerryConfig {
    /// Time-decay rate: a match is weighted `exp(-xi · age_days)`, so recent (tournament) games
    /// dominate.
    pub xi: f64,
    /// L2 shrinkage on the strengths (toward the average), the regularization a sparse,
    /// unbalanced international schedule needs.
    pub ridge: f64,
    pub iterations: usize,
    pub learning_rate: f64,
}

impl Default for BradleyTerryConfig {
    fn default() -> Self {
        Self {
            xi: 0.003,
            ridge: 0.05,
            iterations: 400,
            learning_rate: 0.5,
        }
    }
}

/// A fitted Bradley-Terry-Davidson model. Cheap to clone and query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BradleyTerry {
    /// Log-strength `β` per team (centered around 0; higher = stronger).
    strength: HashMap<TeamId, f64>,
    /// Davidson tie parameter `ν ≥ 0` (larger = more draws).
    tie: f64,
    /// Additive home advantage on the strength scale, applied to non-neutral matches.
    home_advantage: f64,
}

impl Default for BradleyTerry {
    fn default() -> Self {
        Self {
            strength: HashMap::new(),
            tie: 1.0,
            home_advantage: 0.0,
        }
    }
}

impl BradleyTerry {
    /// Fit by time-weighted maximum likelihood (analytic-gradient ascent on the penalized
    /// log-likelihood). Returns a neutral default for empty data.
    pub fn fit(observations: &[Observation], config: BradleyTerryConfig) -> Self {
        if observations.is_empty() {
            return Self::default();
        }
        // Team universe and time-decay weights.
        let mut teams: Vec<TeamId> = Vec::new();
        let mut seen: HashMap<TeamId, usize> = HashMap::new();
        for o in observations {
            for t in [o.home, o.away] {
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(t) {
                    e.insert(teams.len());
                    teams.push(t);
                }
            }
        }
        let weights: Vec<f64> = observations
            .iter()
            .map(|o| (-config.xi * o.age_days).exp())
            .collect();
        let total_w: f64 = weights.iter().sum::<f64>().max(1e-9);

        let n = teams.len();
        let mut beta = vec![0.0f64; n];
        let mut tau = 0.0f64; // ln(ν); ν starts at 1
        let mut home = 0.0f64;

        for _ in 0..config.iterations {
            let mut g_beta = vec![0.0f64; n];
            let mut g_tau = 0.0f64;
            let mut g_home = 0.0f64;
            for (o, &w) in observations.iter().zip(&weights) {
                let i = seen[&o.home];
                let j = seen[&o.away];
                // Training games are treated as home vs away (home carries the edge); at prediction
                // time the home term is dropped for neutral fixtures.
                let d = beta[i] - beta[j] + home;
                let nu = tau.exp();
                let ph = d.exp();
                let g = nu * (0.5 * d).exp();
                let denom = ph + 1.0 + g;
                let shared = (ph + 0.5 * g) / denom; // ∂ log(denom) / ∂d
                let (dl_dd, dl_dtau) = match o.score.outcome() {
                    Outcome::HomeWin => (1.0 - shared, -g / denom),
                    Outcome::Draw => (0.5 - shared, 1.0 - g / denom),
                    Outcome::AwayWin => (-shared, -g / denom),
                };
                g_beta[i] += w * dl_dd;
                g_beta[j] -= w * dl_dd;
                g_home += w * dl_dd;
                g_tau += w * dl_dtau;
            }
            // Average the data gradient, then apply the ridge (toward 0) and step.
            for k in 0..n {
                let grad = g_beta[k] / total_w - config.ridge * beta[k];
                beta[k] += config.learning_rate * grad;
            }
            tau += config.learning_rate * (g_tau / total_w);
            home += config.learning_rate * (g_home / total_w);
            tau = tau.clamp(-4.0, 4.0);
            // Identifiability: strengths are relative, so center them each step.
            let mean = beta.iter().sum::<f64>() / n as f64;
            for b in &mut beta {
                *b -= mean;
            }
        }

        BradleyTerry {
            strength: teams.into_iter().zip(beta).collect(),
            tie: tau.exp(),
            home_advantage: home,
        }
    }

    fn strength_of(&self, team: TeamId) -> f64 {
        self.strength.get(&team).copied().unwrap_or(0.0)
    }

    /// Win/draw/loss probabilities for a matchup (Davidson). `neutral` drops the home term.
    pub fn outcome_probabilities(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
    ) -> Probabilities {
        let adv = if neutral { 0.0 } else { self.home_advantage };
        let d = self.strength_of(home) - self.strength_of(away) + adv;
        let ph = d.exp();
        let g = self.tie * (0.5 * d).exp();
        // `Probabilities::new` renormalizes, so the raw (πₕ, g, 1) weights suffice.
        Probabilities::new(ph, g, 1.0)
    }

    /// Probability `a` advances past `b` in a knockout tie: plain Bradley-Terry with the tie
    /// dropped (someone must go through), at a neutral venue - `πₐ / (πₐ + π_b)`.
    pub fn advance_probability(&self, a: TeamId, b: TeamId) -> f64 {
        let d = self.strength_of(a) - self.strength_of(b);
        let ea = d.exp();
        ea / (ea + 1.0)
    }

    /// One online gradient step from a finished result, updating just the two teams' strengths
    /// (the tie parameter is held), so the model tracks tournament form. `lr` 0 disables it.
    pub fn update_with_result(
        &mut self,
        home: TeamId,
        away: TeamId,
        outcome: Outcome,
        neutral: bool,
        lr: f64,
    ) {
        if lr == 0.0 {
            return;
        }
        let adv = if neutral { 0.0 } else { self.home_advantage };
        let d = self.strength_of(home) - self.strength_of(away) + adv;
        let g = self.tie * (0.5 * d).exp();
        let denom = d.exp() + 1.0 + g;
        let shared = (d.exp() + 0.5 * g) / denom;
        let dl_dd = match outcome {
            Outcome::HomeWin => 1.0 - shared,
            Outcome::Draw => 0.5 - shared,
            Outcome::AwayWin => -shared,
        };
        *self.strength.entry(home).or_insert(0.0) += lr * dl_dd;
        *self.strength.entry(away).or_insert(0.0) -= lr * dl_dd;
    }

    /// The Davidson tie parameter `ν` (larger = more draws).
    pub fn tie(&self) -> f64 {
        self.tie
    }

    /// Teams ranked by strength, strongest first.
    pub fn strength_ranking(&self) -> Vec<(TeamId, f64)> {
        let mut v: Vec<(TeamId, f64)> = self.strength.iter().map(|(&t, &b)| (t, b)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_domain::Scoreline;

    fn t(n: u32) -> TeamId {
        TeamId(n)
    }

    /// Build a round-robin history where strength decreases with team id: lower ids beat higher.
    fn laddered_history() -> Vec<Observation> {
        let mut obs = Vec::new();
        for i in 0..6u32 {
            for j in 0..6u32 {
                if i == j {
                    continue;
                }
                // The stronger (lower-id) side wins; play both home and away for balance.
                let score = if i < j {
                    Scoreline::new(2, 0)
                } else {
                    Scoreline::new(0, 2)
                };
                obs.push(Observation::new(t(i), t(j), score, 5.0));
            }
        }
        obs
    }

    #[test]
    fn fit_recovers_the_strength_ranking() {
        let model = BradleyTerry::fit(&laddered_history(), BradleyTerryConfig::default());
        let ranking = model.strength_ranking();
        let ids: Vec<u32> = ranking.iter().map(|(t, _)| t.0).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2, 3, 4, 5],
            "strength should track the ladder"
        );
    }

    #[test]
    fn outcome_probabilities_normalize_and_favour_the_stronger_side() {
        let model = BradleyTerry::fit(&laddered_history(), BradleyTerryConfig::default());
        let p = model.outcome_probabilities(t(0), t(5), true);
        assert!((p.sum() - 1.0).abs() < 1e-9);
        assert!(
            p.home_win > p.away_win,
            "the strongest team should be favoured over the weakest"
        );
        assert!(p.home_win > 0.0 && p.draw > 0.0 && p.away_win > 0.0);
    }

    #[test]
    fn advance_probability_favours_the_stronger_side_and_is_bounded() {
        let model = BradleyTerry::fit(&laddered_history(), BradleyTerryConfig::default());
        let p = model.advance_probability(t(0), t(5));
        assert!(p > 0.5 && p < 1.0);
        // Antisymmetric: P(a beats b) + P(b beats a) = 1.
        assert!((p + model.advance_probability(t(5), t(0)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tie_parameter_grows_with_draw_heavy_data() {
        // Two histories over the same teams: one all decisive, one all draws.
        let decisive: Vec<Observation> = (0..40)
            .map(|k| Observation::new(t(k % 4), t((k + 1) % 4), Scoreline::new(2, 0), 5.0))
            .collect();
        let drawish: Vec<Observation> = (0..40)
            .map(|k| Observation::new(t(k % 4), t((k + 1) % 4), Scoreline::new(1, 1), 5.0))
            .collect();
        let cfg = BradleyTerryConfig::default();
        let lo = BradleyTerry::fit(&decisive, cfg).tie();
        let hi = BradleyTerry::fit(&drawish, cfg).tie();
        assert!(
            hi > lo,
            "draw-heavy data should raise the tie parameter ({lo} -> {hi})"
        );
    }

    #[test]
    fn online_update_moves_the_upset_winner_up() {
        let mut model = BradleyTerry::fit(&laddered_history(), BradleyTerryConfig::default());
        let before = model.strength_ranking();
        let weak = before.last().unwrap().0; // weakest team
        let strong = before.first().unwrap().0; // strongest team
                                                // The weak side pulls off a shock win; its strength should rise.
        let s_before = model.outcome_probabilities(weak, strong, true).home_win;
        for _ in 0..20 {
            model.update_with_result(weak, strong, Outcome::HomeWin, true, 0.1);
        }
        let s_after = model.outcome_probabilities(weak, strong, true).home_win;
        assert!(
            s_after > s_before,
            "repeated shocks should lift the underdog"
        );
    }
}
