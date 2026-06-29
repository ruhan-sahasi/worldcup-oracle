//! On-demand model queries for the interactive explorer.
//!
//! The live [`Engine`](crate::Engine) tracks one running tournament; this is the complement - a
//! fit-once, read-only view of the model that answers *ad-hoc* questions a human (or the web
//! explorer) might ask: predict any matchup, draw its HMC posterior credible interval, run a
//! custom Monte-Carlo simulation, or browse the ratings. Keeping it separate from the engine means
//! exploration never perturbs the live state, and one server can offer both.
//!
//! Everything is fit once in [`Explorer::new`] and then queried; the type is cheap to share
//! (`Arc<Explorer>`) across request handlers.

use oracle_domain::{Confederation, Probabilities, ScoreGrid, TeamId, Tournament};
use oracle_ingest::data;
use oracle_model::hmc::HmcConfig;
use oracle_model::{implied_probabilities, Ensemble, GoalModel, LiveConfig, Observation};
use oracle_ratings::{RatingStore, StateSpaceRatings};
use oracle_sim::{simulate_with_live, LiveInputs, SimConfig};
use serde::Serialize;
use std::collections::HashMap;

/// Hard ceilings so a single request cannot tie the server up indefinitely.
const SIM_MAX_ITERS: u64 = 100_000;
const POSTERIOR_MAX_SAMPLES: usize = 1500;

/// A fitted-once view of the model for interactive, on-demand queries.
pub struct Explorer {
    tournament: Tournament,
    model: GoalModel,
    ratings: RatingStore,
    state_space: StateSpaceRatings,
    ensemble: Ensemble,
    /// The training history, needed to reconstruct the likelihood for the HMC posterior.
    observations: Vec<Observation>,
    confederations: HashMap<TeamId, Confederation>,
    shootout_rating: HashMap<TeamId, f64>,
    knockout_pedigree: HashMap<TeamId, f64>,
}

impl Default for Explorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Explorer {
    /// Fit the offline baseline once (a few seconds) and hold everything needed to answer queries.
    pub fn new() -> Self {
        let tournament = data::world_cup_2026();
        let baseline = data::fit_baseline(7);
        let observations: Vec<Observation> = data::synthetic_history_with_market(4000, 7)
            .into_iter()
            .map(|r| r.obs)
            .collect();
        // Elo: seed from the strength priors, then replay the training history so the ratings
        // browser shows learned (not just prior) values.
        let mut ratings = RatingStore::with_defaults();
        for (team, rating) in &baseline.elo_seeds {
            ratings.seed(*team, *rating);
        }
        for o in &observations {
            ratings.record(o.home, o.away, o.score, true);
        }
        Self {
            tournament,
            model: baseline.model,
            ratings,
            state_space: baseline.state_space,
            ensemble: baseline.ensemble,
            observations,
            confederations: data::confederations(),
            shootout_rating: data::shootout_ratings(),
            knockout_pedigree: data::knockout_pedigree(),
        }
    }

    /// Resolve a team by FIFA code, full name, or a name substring (case-insensitive).
    pub fn resolve(&self, query: &str) -> Option<TeamId> {
        let q = query.trim().to_lowercase();
        self.tournament
            .teams
            .iter()
            .find(|t| t.code.to_lowercase() == q || t.name.to_lowercase() == q)
            .or_else(|| {
                self.tournament
                    .teams
                    .iter()
                    .find(|t| t.name.to_lowercase().contains(&q))
            })
            .map(|t| t.id)
    }

    fn name(&self, id: TeamId) -> String {
        self.tournament
            .teams
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.name.clone())
            .unwrap_or_else(|| id.to_string())
    }

    /// Pre-match forecast for a matchup: the stacked ensemble, each member, expected goals, the
    /// exact-score grid, and the usual derived markets. Optional bookmaker `odds` add the market
    /// member and anchor the ensemble to it.
    pub fn predict(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
        odds: Option<(f64, f64, f64)>,
    ) -> MatchupForecast {
        let grid = self.model.score_grid(home, away, neutral);
        let dixon_coles = grid.outcome_probabilities();
        let elo = self.ratings.win_probabilities(home, away, neutral);
        let state_space = self.state_space.win_probabilities(home, away, neutral);
        let market = odds.map(|(h, d, a)| implied_probabilities(h, d, a));
        let mut members = vec![dixon_coles, elo, state_space];
        if let Some(m) = market {
            members.push(m);
        }
        let ensemble = self.ensemble.blend(&members);
        let (expected_home, expected_away) = self.model.expected_goals(home, away, neutral);
        let (mh, ma, mp) = grid.most_likely_score();
        MatchupForecast {
            home: self.name(home),
            away: self.name(away),
            ensemble,
            dixon_coles,
            elo,
            state_space,
            market,
            expected_home,
            expected_away,
            most_likely: ScoreLine {
                home: mh,
                away: ma,
                prob: mp,
            },
            over_2_5: grid.prob_over(2.5),
            btts: grid.prob_btts(),
            top_scorelines: top_scorelines(&grid, 6),
            grid,
        }
    }

    /// The HMC posterior over a matchup's win/draw/win probabilities, reduced to a mean plus a 90%
    /// credible interval per outcome - the model's uncertainty about its own forecast.
    pub fn posterior(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
        samples: usize,
    ) -> PosteriorForecast {
        let n = samples.clamp(100, POSTERIOR_MAX_SAMPLES);
        let draws = self.model.posterior_outcome_samples(
            &self.observations,
            &self.confederations,
            home,
            away,
            neutral,
            HmcConfig {
                n_samples: n,
                n_warmup: (n / 4).max(100),
                step_size: 0.2,
                n_leapfrog: 16,
                seed: 7,
            },
        );
        let ci = |pick: fn(&Probabilities) -> f64| -> Interval {
            let mut v: Vec<f64> = draws.iter().map(pick).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if v.is_empty() {
                return Interval::default();
            }
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let pct = |q: f64| v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            Interval {
                mean,
                lo: pct(0.05),
                hi: pct(0.95),
            }
        };
        PosteriorForecast {
            home: self.name(home),
            away: self.name(away),
            samples: draws.len(),
            home_win: ci(|p| p.home_win),
            draw: ci(|p| p.draw),
            away_win: ci(|p| p.away_win),
        }
    }

    /// Run the full Monte-Carlo tournament simulation (with all the context and knockout factors a
    /// live forecast uses) and return the ranked champion-odds table with Monte-Carlo error bars.
    pub fn simulate(&self, iters: u64, seed: u64) -> SimForecast {
        let iters = iters.clamp(1000, SIM_MAX_ITERS);
        let inputs = LiveInputs {
            venue: data::matchup_adjustments(&self.tournament),
            shootout_rating: self.shootout_rating.clone(),
            knockout_pedigree: self.knockout_pedigree.clone(),
            ..Default::default()
        };
        let forecast = simulate_with_live(
            &self.tournament,
            &self.model,
            SimConfig {
                iterations: iters,
                seed,
                ..SimConfig::default()
            },
            &inputs,
            LiveConfig::default(),
        );
        let n = forecast.iterations.max(1) as f64;
        let mut teams: Vec<SimRow> = forecast
            .teams
            .iter()
            .map(|t| SimRow {
                team: self.name(t.team),
                p_advance: t.p_advance_group,
                p_round_of_16: t.p_round_of_16,
                p_quarter: t.p_quarter_final,
                p_semi: t.p_semi_final,
                p_final: t.p_final,
                p_champion: t.p_champion,
                champion_stderr: (t.p_champion * (1.0 - t.p_champion) / n).sqrt(),
            })
            .collect();
        teams.sort_by(|a, b| {
            b.p_champion
                .partial_cmp(&a.p_champion)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        SimForecast {
            iterations: forecast.iterations,
            seed,
            teams,
        }
    }

    /// Team ratings (Elo, state-space mean, goal-model strength) and the fitted confederation
    /// strength levels, for the ratings browser.
    pub fn ratings(&self) -> RatingsView {
        let strength: HashMap<TeamId, f64> = self.model.strength_ranking().into_iter().collect();
        let mut teams: Vec<RatingRow> = self
            .tournament
            .teams
            .iter()
            .map(|t| RatingRow {
                team: t.name.clone(),
                code: t.code.clone(),
                confederation: t.confederation,
                elo: self.ratings.rating(t.id),
                state_space: self.state_space.mean(t.id),
                strength: strength.get(&t.id).copied().unwrap_or(0.0),
            })
            .collect();
        teams.sort_by(|a, b| {
            b.elo
                .partial_cmp(&a.elo)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut confederations: Vec<ConfRow> = self
            .model
            .confederation_levels()
            .into_iter()
            .map(|(confederation, level)| ConfRow {
                confederation,
                level,
            })
            .collect();
        confederations.sort_by(|a, b| {
            b.level
                .partial_cmp(&a.level)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        RatingsView {
            teams,
            confederations,
        }
    }
}

fn top_scorelines(grid: &ScoreGrid, n: usize) -> Vec<ScoreLine> {
    let mut cells: Vec<ScoreLine> = grid
        .grid
        .iter()
        .enumerate()
        .flat_map(|(h, row)| {
            row.iter().enumerate().map(move |(a, &p)| ScoreLine {
                home: h as u8,
                away: a as u8,
                prob: p,
            })
        })
        .collect();
    cells.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    cells.truncate(n);
    cells
}

// ----- serializable result types (the API forwards these as JSON) -----

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ScoreLine {
    pub home: u8,
    pub away: u8,
    pub prob: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchupForecast {
    pub home: String,
    pub away: String,
    pub ensemble: Probabilities,
    pub dixon_coles: Probabilities,
    pub elo: Probabilities,
    pub state_space: Probabilities,
    pub market: Option<Probabilities>,
    pub expected_home: f64,
    pub expected_away: f64,
    pub most_likely: ScoreLine,
    pub over_2_5: f64,
    pub btts: f64,
    pub top_scorelines: Vec<ScoreLine>,
    pub grid: ScoreGrid,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Interval {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PosteriorForecast {
    pub home: String,
    pub away: String,
    pub samples: usize,
    pub home_win: Interval,
    pub draw: Interval,
    pub away_win: Interval,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimRow {
    pub team: String,
    pub p_advance: f64,
    pub p_round_of_16: f64,
    pub p_quarter: f64,
    pub p_semi: f64,
    pub p_final: f64,
    pub p_champion: f64,
    pub champion_stderr: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimForecast {
    pub iterations: u64,
    pub seed: u64,
    pub teams: Vec<SimRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RatingRow {
    pub team: String,
    pub code: String,
    pub confederation: Confederation,
    pub elo: f64,
    pub state_space: f64,
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfRow {
    pub confederation: Confederation,
    pub level: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RatingsView {
    pub teams: Vec<RatingRow>,
    pub confederations: Vec<ConfRow>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_simulate_and_ratings_are_coherent() {
        let ex = Explorer::new();
        let arg = ex.resolve("Argentina").unwrap();
        let nzl = ex.resolve("NZL").unwrap();
        assert!(ex.resolve("Notateam").is_none());

        // Prediction: probabilities normalize and the strong side is favoured.
        let f = ex.predict(arg, nzl, true, None);
        assert!((f.ensemble.sum() - 1.0).abs() < 1e-9);
        assert!((f.dixon_coles.sum() - 1.0).abs() < 1e-9);
        assert!(
            f.ensemble.home_win > f.ensemble.away_win,
            "Argentina favoured over NZ"
        );
        assert_eq!(f.market, None);
        assert_eq!(f.top_scorelines.len(), 6);
        assert_eq!(f.grid.grid.len(), 11);

        // Odds add the market member.
        let fm = ex.predict(arg, nzl, true, Some((1.2, 7.0, 12.0)));
        assert!(fm.market.is_some());

        // Simulation: champion mass sums to ~1, strongest team leads.
        let s = ex.simulate(3000, 42);
        let mass: f64 = s.teams.iter().map(|t| t.p_champion).sum();
        assert!((mass - 1.0).abs() < 0.02, "champion mass = {mass}");
        assert!(s.teams[0].champion_stderr > 0.0);

        // Ratings: every team present, six confederations, sorted by Elo descending.
        let r = ex.ratings();
        assert_eq!(r.teams.len(), 48);
        assert_eq!(r.confederations.len(), 6);
        assert!(r.teams[0].elo >= r.teams[1].elo);
    }

    #[test]
    fn posterior_brackets_the_point_estimate() {
        let ex = Explorer::new();
        let (a, b) = (ex.resolve("Brazil").unwrap(), ex.resolve("Japan").unwrap());
        let point = ex.predict(a, b, true, None);
        let post = ex.posterior(a, b, true, 300);
        assert!(post.samples >= 100);
        // The credible interval contains the point estimate, with real width.
        assert!(post.home_win.lo <= point.dixon_coles.home_win + 0.05);
        assert!(post.home_win.hi >= point.dixon_coles.home_win - 0.05);
        assert!(post.home_win.hi > post.home_win.lo);
    }
}
