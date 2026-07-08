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

use oracle_domain::{
    Confederation, MatchId, MatchStatus, Probabilities, ScoreGrid, Scoreline, TeamId, Tournament,
};
use oracle_ingest::data::{self, SignalMask, VenueAdj};
use oracle_model::hmc::HmcConfig;
use oracle_model::{
    implied_probabilities, BradleyTerry, DixonColesConfig, Ensemble, GoalModel, LiveConfig,
    Observation,
};
use oracle_ratings::{RatingStore, StateSpaceRatings};
use oracle_sim::{meeting_probabilities, simulate_with_live, LiveInputs, SimConfig};
use serde::Serialize;
use std::collections::HashMap;

/// Hard ceilings so a single request cannot tie the server up indefinitely.
const SIM_MAX_ITERS: u64 = 100_000;
const POSTERIOR_MAX_SAMPLES: usize = 1500;
/// The sensitivity analysis runs ten simulations, so cap its per-variant iterations lower.
const SENSITIVITY_MAX_ITERS: u64 = 50_000;
/// Kingmaker runs a baseline plus several conditional simulations, so cap iterations similarly.
const KINGMAKER_MAX_ITERS: u64 = 40_000;

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
    /// The second, outcome-based forecaster (Bradley-Terry-Davidson), an alternative to the goal
    /// model that predicts win/draw/loss directly and yields knockout advance probabilities.
    bradley_terry: BradleyTerry,
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
            bradley_terry: data::fit_bradley_terry(7),
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

    /// Explain a matchup forecast: the named factors driving the goal model's edge (attacking and
    /// defensive strength, home advantage, and the style matchup), the resulting expected goals,
    /// and how the ensemble blends its members (with their learned weights). Answers *why* the
    /// model favours one side, not just by how much.
    pub fn explain(
        &self,
        home: TeamId,
        away: TeamId,
        neutral: bool,
        odds: Option<(f64, f64, f64)>,
    ) -> Explanation {
        let f = self.predict(home, away, neutral, odds);
        let rb = self.model.rate_breakdown(home, away, neutral);
        let style_tilt = data::matchup_style_tilt(home, away);
        let (hn, an) = (f.home.clone(), f.away.clone());
        let strength_factors = vec![
            Factor {
                name: "Attacking strength".into(),
                effect: rb.attack_edge,
                note: format!("{hn}'s attack vs {an}'s"),
            },
            Factor {
                name: "Defensive strength".into(),
                effect: rb.defense_edge,
                note: format!("{hn}'s defence vs {an}'s"),
            },
            Factor {
                name: "Home advantage".into(),
                effect: rb.home_advantage,
                note: if neutral {
                    "neutral venue".into()
                } else {
                    "host-side edge".into()
                },
            },
            Factor {
                name: "Style matchup".into(),
                effect: 2.0 * style_tilt,
                note: "rock-paper-scissors style edge (applies at real fixtures)".into(),
            },
        ];
        let w = |i: usize| self.ensemble.weights.get(i).copied().unwrap_or(0.0);
        let mut members = vec![
            MemberView {
                name: "Dixon-Coles".into(),
                probabilities: f.dixon_coles,
                weight: w(0),
            },
            MemberView {
                name: "Elo".into(),
                probabilities: f.elo,
                weight: w(1),
            },
            MemberView {
                name: "State-space".into(),
                probabilities: f.state_space,
                weight: w(2),
            },
        ];
        if let Some(m) = f.market {
            members.push(MemberView {
                name: "Market".into(),
                probabilities: m,
                weight: w(3),
            });
        }
        Explanation {
            home: f.home,
            away: f.away,
            ensemble: f.ensemble,
            expected_home: rb.expected_home,
            expected_away: rb.expected_away,
            strength_factors,
            members,
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

    /// Per-team champion probabilities from a Monte-Carlo run on a given tournament state (which
    /// may have some matches fixed to a result). Shared by [`kingmaker`](Self::kingmaker).
    fn champ_probs(&self, tournament: &Tournament, iters: u64, seed: u64) -> HashMap<TeamId, f64> {
        let inputs = LiveInputs {
            venue: data::matchup_adjustments(tournament),
            shootout_rating: self.shootout_rating.clone(),
            knockout_pedigree: self.knockout_pedigree.clone(),
            ..Default::default()
        };
        let config = SimConfig {
            iterations: iters,
            seed,
            ..SimConfig::default()
        };
        simulate_with_live(
            tournament,
            &self.model,
            config,
            &inputs,
            LiveConfig::default(),
        )
        .teams
        .iter()
        .map(|t| (t.team, t.p_champion))
        .collect()
    }

    /// Rooting interest for a team: how much each of its group rivals' matches would swing its
    /// championship odds. Conditions the tournament on each result (a representative scoreline) and
    /// re-simulates, diffing the team's champion probability against the baseline. All runs share
    /// the seed, so a swing reflects the result rather than Monte-Carlo noise.
    pub fn kingmaker(&self, team: TeamId, iters: u64, seed: u64) -> KingmakerReport {
        let iters = iters.clamp(2000, KINGMAKER_MAX_ITERS);
        let base_champion = self
            .champ_probs(&self.tournament, iters, seed)
            .get(&team)
            .copied()
            .unwrap_or(0.0);
        // The team's group rivals' matches (not involving the team) shape whether it advances.
        let group_teams: Vec<TeamId> = self
            .tournament
            .groups
            .iter()
            .find(|g| g.teams.contains(&team))
            .map(|g| g.teams.clone())
            .unwrap_or_default();
        let cand_ids: Vec<MatchId> = self
            .tournament
            .matches
            .iter()
            .filter(|m| {
                !m.is_finished()
                    && m.home != team
                    && m.away != team
                    && group_teams.contains(&m.home)
                    && group_teams.contains(&m.away)
            })
            .map(|m| m.id)
            .collect();
        let swing = |id: MatchId, score: Scoreline| -> f64 {
            let mut t = self.tournament.clone();
            if let Some(mm) = t.matches.iter_mut().find(|x| x.id == id) {
                mm.status = MatchStatus::Finished;
                mm.score = score;
            }
            self.champ_probs(&t, iters, seed)
                .get(&team)
                .copied()
                .unwrap_or(0.0)
                - base_champion
        };
        let mut matches: Vec<KingmakerRow> = cand_ids
            .into_iter()
            .map(|id| {
                let m = self.tournament.matches.iter().find(|x| x.id == id).unwrap();
                KingmakerRow {
                    match_id: id,
                    home_name: self.name(m.home),
                    away_name: self.name(m.away),
                    home_win_swing: swing(id, Scoreline::new(1, 0)),
                    draw_swing: swing(id, Scoreline::new(1, 1)),
                    away_win_swing: swing(id, Scoreline::new(0, 1)),
                }
            })
            .collect();
        let magnitude = |r: &KingmakerRow| {
            r.home_win_swing
                .abs()
                .max(r.draw_swing.abs())
                .max(r.away_win_swing.abs())
        };
        matches.sort_by(|a, b| {
            magnitude(b)
                .partial_cmp(&magnitude(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        KingmakerReport {
            team: self.name(team),
            base_champion,
            matches,
        }
    }

    /// Collision course: how likely two teams are to meet in the knockouts, and at which round.
    pub fn collision(&self, a: TeamId, b: TeamId, iters: u64, seed: u64) -> CollisionForecast {
        let iters = iters.clamp(2000, SIM_MAX_ITERS);
        let inputs = LiveInputs {
            venue: data::matchup_adjustments(&self.tournament),
            shootout_rating: self.shootout_rating.clone(),
            knockout_pedigree: self.knockout_pedigree.clone(),
            ..Default::default()
        };
        let config = SimConfig {
            iterations: iters,
            seed,
            ..SimConfig::default()
        };
        let m = meeting_probabilities(
            &self.tournament,
            &self.model,
            config,
            &inputs,
            LiveConfig::default(),
            a,
            b,
        );
        // The round they are most likely to meet at (if they meet at all).
        let most_likely_round = m
            .by_round
            .iter()
            .filter(|(_, p)| *p > 0.0)
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label.to_string());
        CollisionForecast {
            team_a: self.name(a),
            team_b: self.name(b),
            meet_probability: m.meet_probability,
            most_likely_round,
            by_round: m
                .by_round
                .into_iter()
                .map(|(round, probability)| RoundProb {
                    round: round.to_string(),
                    probability,
                })
                .collect(),
        }
    }

    /// The second model's take on a matchup: Bradley-Terry-Davidson win/draw/loss (outcome-based,
    /// not goal-based), an alternative to the ensemble forecast.
    pub fn bt_predict(&self, home: TeamId, away: TeamId, neutral: bool) -> BtMatchup {
        BtMatchup {
            home: self.name(home),
            away: self.name(away),
            probabilities: self
                .bradley_terry
                .outcome_probabilities(home, away, neutral),
        }
    }

    /// The second model's winner prediction: exact champion odds over a projected knockout bracket
    /// (top two per group plus the eight strongest third-placed teams, by Bradley-Terry strength,
    /// seeded so the strongest sides start apart), computed by bracket dynamic programming from the
    /// model's pairwise advance probabilities - no Monte-Carlo needed.
    pub fn bt_champion_odds(&self) -> BtChampions {
        let leaves = self.projected_bracket();
        if leaves.len() != 32 {
            return BtChampions { teams: Vec::new() };
        }
        let odds =
            bracket_champion_odds(&leaves, |a, b| self.bradley_terry.advance_probability(a, b));
        BtChampions {
            teams: odds
                .into_iter()
                .map(|(team, champion)| BtTeamOdds {
                    team: self.name(team),
                    champion,
                })
                .collect(),
        }
    }

    /// The 32 knockout qualifiers projected by Bradley-Terry strength (top two per group plus the
    /// best eight thirds), returned as bracket leaves in standard seeded order.
    fn projected_bracket(&self) -> Vec<TeamId> {
        let strength: HashMap<TeamId, f64> =
            self.bradley_terry.strength_ranking().into_iter().collect();
        let key = |t: &TeamId| strength.get(t).copied().unwrap_or(0.0);
        let cmp = |a: &TeamId, b: &TeamId| {
            key(b)
                .partial_cmp(&key(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        let mut qualifiers: Vec<TeamId> = Vec::new();
        let mut thirds: Vec<TeamId> = Vec::new();
        for g in &self.tournament.groups {
            let mut ts = g.teams.clone();
            ts.sort_by(cmp);
            if ts.len() < 2 {
                return Vec::new();
            }
            qualifiers.push(ts[0]);
            qualifiers.push(ts[1]);
            if ts.len() > 2 {
                thirds.push(ts[2]);
            }
        }
        thirds.sort_by(cmp);
        qualifiers.extend(thirds.into_iter().take(8));
        if qualifiers.len() != 32 {
            return Vec::new();
        }
        qualifiers.sort_by(cmp); // strongest first (seed 1..32)
        seed_order(32)
            .into_iter()
            .map(|s| qualifiers[s - 1])
            .collect()
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

    /// Signal sensitivity: disable each unconventional signal in turn, re-simulate the whole
    /// tournament on a shared seed, and report how far each one moves the championship distribution
    /// (and which teams move most). The same analysis the `wc-oracle sensitivity` command prints,
    /// served for the explorer.
    pub fn sensitivity(&self, iters: u64, seed: u64) -> SensitivityForecast {
        let iters = iters.clamp(2000, SENSITIVITY_MAX_ITERS);
        // The "no confederation pooling" variant: refit globally on the same history (an empty
        // confederations map reduces to the plain fit).
        let unpooled = GoalModel::fit(&self.observations, DixonColesConfig::default());
        let signals = signal_sensitivity(
            &self.tournament,
            &self.model,
            &unpooled,
            &self.shootout_rating,
            &self.knockout_pedigree,
            iters,
            seed,
            3,
        )
        .into_iter()
        .map(|c| SignalRow {
            signal: c.signal.to_string(),
            title_shift: c.title_shift,
            movers: c
                .movers
                .into_iter()
                .map(|(team, delta)| MoverRow {
                    team: self.name(team),
                    delta,
                })
                .collect(),
        })
        .collect();
        SensitivityForecast {
            iterations: iters,
            seed,
            signals,
        }
    }
}

/// One signal's measured contribution to the championship distribution.
pub struct SignalContribution {
    pub signal: &'static str,
    /// Total variation distance between the full-model and ablated champion distributions.
    pub title_shift: f64,
    /// Teams with the largest signed change in champion probability (most affected first).
    pub movers: Vec<(TeamId, f64)>,
}

/// Ablate each unconventional signal in turn and measure how far disabling it moves the
/// championship distribution. Shared by the `wc-oracle sensitivity` command and the web explorer.
/// All variants share `seed` (common random numbers), so each delta reflects the signal rather
/// than Monte-Carlo noise. `unpooled` is a model refit without confederation pooling; `iters` is
/// used as-is (callers clamp).
#[allow(clippy::too_many_arguments)]
pub fn signal_sensitivity(
    tournament: &Tournament,
    model: &GoalModel,
    unpooled: &GoalModel,
    shootout: &HashMap<TeamId, f64>,
    pedigree: &HashMap<TeamId, f64>,
    iters: u64,
    seed: u64,
    top: usize,
) -> Vec<SignalContribution> {
    let full_venue = data::matchup_adjustments(tournament);
    let no_teams: HashMap<TeamId, f64> = HashMap::new();
    let fat_on = SimConfig::default().ko_fatigue_penalty;
    let mut model_no_dispersion = model.clone();
    model_no_dispersion.set_dispersion(0.0);

    // Run one variant and return its per-team champion probabilities.
    let champ = |venue: &HashMap<MatchId, VenueAdj>,
                 shoot: &HashMap<TeamId, f64>,
                 ped: &HashMap<TeamId, f64>,
                 sampler: &GoalModel,
                 fatigue: f64|
     -> HashMap<TeamId, f64> {
        let inputs = LiveInputs {
            venue: venue.clone(),
            shootout_rating: shoot.clone(),
            knockout_pedigree: ped.clone(),
            ..Default::default()
        };
        let config = SimConfig {
            iterations: iters,
            seed,
            ko_fatigue_penalty: fatigue,
            ..SimConfig::default()
        };
        simulate_with_live(tournament, sampler, config, &inputs, LiveConfig::default())
            .ranked()
            .into_iter()
            .map(|t| (t.team, t.p_champion))
            .collect()
    };
    let masked = |m: SignalMask| data::matchup_adjustments_masked(tournament, m);

    let base = champ(&full_venue, shootout, pedigree, model, fat_on);

    // Each entry disables exactly one signal; everything else is the full model.
    let variants: Vec<(&'static str, HashMap<TeamId, f64>)> = vec![
        (
            "Crowd composition",
            champ(
                &masked(SignalMask {
                    crowd: false,
                    ..Default::default()
                }),
                shootout,
                pedigree,
                model,
                fat_on,
            ),
        ),
        (
            "Travel & circadian load",
            champ(
                &masked(SignalMask {
                    travel: false,
                    ..Default::default()
                }),
                shootout,
                pedigree,
                model,
                fat_on,
            ),
        ),
        (
            "Heat suppression",
            champ(
                &masked(SignalMask {
                    heat: false,
                    ..Default::default()
                }),
                shootout,
                pedigree,
                model,
                fat_on,
            ),
        ),
        (
            "Style matchup",
            champ(
                &masked(SignalMask {
                    style: false,
                    ..Default::default()
                }),
                shootout,
                pedigree,
                model,
                fat_on,
            ),
        ),
        (
            "Shootout skill",
            champ(&full_venue, &no_teams, pedigree, model, fat_on),
        ),
        (
            "Knockout pedigree",
            champ(&full_venue, shootout, &no_teams, model, fat_on),
        ),
        (
            "Overdispersion (NB margins)",
            champ(
                &full_venue,
                shootout,
                pedigree,
                &model_no_dispersion,
                fat_on,
            ),
        ),
        (
            "Extra-time fatigue",
            champ(&full_venue, shootout, pedigree, model, 0.0),
        ),
        (
            "Confederation pooling",
            champ(&full_venue, shootout, pedigree, unpooled, fat_on),
        ),
    ];

    let mut out: Vec<SignalContribution> = variants
        .into_iter()
        .map(|(signal, ablated)| {
            let mut tvd = 0.0;
            let mut deltas: Vec<(TeamId, f64)> = Vec::with_capacity(base.len());
            for (&team, &pb) in &base {
                let pa = ablated.get(&team).copied().unwrap_or(0.0);
                tvd += (pa - pb).abs();
                deltas.push((team, pa - pb));
            }
            tvd *= 0.5;
            deltas.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            deltas.truncate(top);
            SignalContribution {
                signal,
                title_shift: tvd,
                movers: deltas,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.title_shift
            .partial_cmp(&a.title_shift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Standard single-elimination seed order for `n` (a power of two): 1 plays `n`, 2 plays `n-1`,
/// and top seeds are kept in opposite halves. Returns 1-based seed positions in bracket-leaf order.
fn seed_order(n: usize) -> Vec<usize> {
    let mut seeds = vec![1usize];
    while seeds.len() < n {
        let rounds = seeds.len() * 2;
        let mut next = Vec::with_capacity(rounds);
        for &s in &seeds {
            next.push(s);
            next.push(rounds + 1 - s);
        }
        seeds = next;
    }
    seeds
}

/// Exact champion probability per team over a single-elimination bracket, by dynamic programming:
/// merge sibling sub-brackets bottom-up, where a team wins a node by winning its own side and then
/// beating the (probability-weighted) field from the other side. `leaves` are the teams in bracket
/// order (a power-of-two length); `advance(a, b)` is `P(a beats b)`. Champion probabilities sum to 1.
fn bracket_champion_odds(
    leaves: &[TeamId],
    advance: impl Fn(TeamId, TeamId) -> f64,
) -> Vec<(TeamId, f64)> {
    let mut layer: Vec<HashMap<TeamId, f64>> = leaves
        .iter()
        .map(|&t| {
            let mut m = HashMap::new();
            m.insert(t, 1.0);
            m
        })
        .collect();
    while layer.len() > 1 {
        let mut next: Vec<HashMap<TeamId, f64>> = Vec::with_capacity(layer.len() / 2);
        let mut k = 0;
        while k + 1 < layer.len() {
            let (left, right) = (&layer[k], &layer[k + 1]);
            let mut merged: HashMap<TeamId, f64> = HashMap::new();
            for (&a, &pa) in left {
                let beats: f64 = right.iter().map(|(&b, &pb)| pb * advance(a, b)).sum();
                *merged.entry(a).or_insert(0.0) += pa * beats;
            }
            for (&b, &pb) in right {
                let beats: f64 = left.iter().map(|(&a, &pa)| pa * advance(b, a)).sum();
                *merged.entry(b).or_insert(0.0) += pb * beats;
            }
            next.push(merged);
            k += 2;
        }
        layer = next;
    }
    let mut v: Vec<(TeamId, f64)> = layer
        .into_iter()
        .next()
        .unwrap_or_default()
        .into_iter()
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
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

/// One named driver of a forecast, with its signed contribution to the home side's log
/// expected-goal edge (positive favours home).
#[derive(Debug, Clone, Serialize)]
pub struct Factor {
    pub name: String,
    pub effect: f64,
    pub note: String,
}

/// One ensemble member's view of a matchup, with its learned blend weight.
#[derive(Debug, Clone, Serialize)]
pub struct MemberView {
    pub name: String,
    pub probabilities: Probabilities,
    pub weight: f64,
}

/// Why the model forecasts a matchup the way it does: the strength/style factors behind the goal
/// model's edge, and how the ensemble blends its members.
#[derive(Debug, Clone, Serialize)]
pub struct Explanation {
    pub home: String,
    pub away: String,
    pub ensemble: Probabilities,
    pub expected_home: f64,
    pub expected_away: f64,
    pub strength_factors: Vec<Factor>,
    pub members: Vec<MemberView>,
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

/// One rooting-interest row: how a group rival's match result would swing the team's title odds.
#[derive(Debug, Clone, Serialize)]
pub struct KingmakerRow {
    pub match_id: MatchId,
    pub home_name: String,
    pub away_name: String,
    /// Signed change (in probability) to the team's championship odds under each result.
    pub home_win_swing: f64,
    pub draw_swing: f64,
    pub away_win_swing: f64,
}

/// A team's rooting interest: how much each of its group rivals' matches moves its title odds,
/// ranked by the biggest swing.
#[derive(Debug, Clone, Serialize)]
pub struct KingmakerReport {
    pub team: String,
    pub base_champion: f64,
    pub matches: Vec<KingmakerRow>,
}

/// The probability two teams meet at a given knockout round.
#[derive(Debug, Clone, Serialize)]
pub struct RoundProb {
    pub round: String,
    pub probability: f64,
}

/// The second model's (Bradley-Terry-Davidson) win/draw/loss for a matchup.
#[derive(Debug, Clone, Serialize)]
pub struct BtMatchup {
    pub home: String,
    pub away: String,
    pub probabilities: Probabilities,
}

/// One team's champion probability from the second model's bracket dynamic programming.
#[derive(Debug, Clone, Serialize)]
pub struct BtTeamOdds {
    pub team: String,
    pub champion: f64,
}

/// The second model's winner prediction: champion odds over the projected knockout bracket.
#[derive(Debug, Clone, Serialize)]
pub struct BtChampions {
    pub teams: Vec<BtTeamOdds>,
}

/// Collision course: how likely two teams are to meet in the knockouts, overall and by round.
#[derive(Debug, Clone, Serialize)]
pub struct CollisionForecast {
    pub team_a: String,
    pub team_b: String,
    pub meet_probability: f64,
    pub most_likely_round: Option<String>,
    pub by_round: Vec<RoundProb>,
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

#[derive(Debug, Clone, Serialize)]
pub struct MoverRow {
    pub team: String,
    /// Signed change in championship probability when the signal is disabled.
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignalRow {
    pub signal: String,
    /// Total variation distance between the full-model and ablated champion distributions.
    pub title_shift: f64,
    pub movers: Vec<MoverRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensitivityForecast {
    pub iterations: u64,
    pub seed: u64,
    /// Signals ranked by how far disabling each one moves the title picture.
    pub signals: Vec<SignalRow>,
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

    #[test]
    fn explain_decomposes_a_matchup() {
        let ex = Explorer::new();
        let (a, b) = (
            ex.resolve("Argentina").unwrap(),
            ex.resolve("New Zealand").unwrap(),
        );
        let e = ex.explain(a, b, true, None);
        assert_eq!(e.strength_factors.len(), 4);
        assert_eq!(e.members.len(), 3, "no odds -> DC, Elo, state-space");
        assert!(e.expected_home > 0.0 && e.expected_away > 0.0);
        // The strong side's attacking + defensive + home-advantage edges net positive.
        let edge: f64 = e
            .strength_factors
            .iter()
            .filter(|f| f.name != "Style matchup")
            .map(|f| f.effect)
            .sum();
        assert!(
            edge > 0.0,
            "Argentina should carry a positive strength edge"
        );
        // Odds add the market as a fourth member.
        let em = ex.explain(a, b, true, Some((1.2, 7.0, 12.0)));
        assert_eq!(em.members.len(), 4);
        assert_eq!(em.members[3].name, "Market");
    }

    #[test]
    fn kingmaker_reports_group_rooting_interest() {
        let ex = Explorer::new();
        let team = ex.resolve("Brazil").unwrap();
        let k = ex.kingmaker(team, 3000, 1);
        assert_eq!(k.team, "Brazil");
        assert!((0.0..=1.0).contains(&k.base_champion));
        // A four-team group has three rival matches that do not involve the team.
        assert_eq!(k.matches.len(), 3);
        for r in &k.matches {
            for s in [r.home_win_swing, r.draw_swing, r.away_win_swing] {
                assert!(s.is_finite() && s.abs() <= 1.0);
            }
        }
    }

    #[test]
    fn collision_reports_meeting_probabilities() {
        let ex = Explorer::new();
        let (a, b) = (
            ex.resolve("Brazil").unwrap(),
            ex.resolve("Argentina").unwrap(),
        );
        let c = ex.collision(a, b, 4000, 3);
        assert_eq!(c.by_round.len(), 5);
        assert!((0.0..=1.0).contains(&c.meet_probability));
        // A meeting is counted at exactly one round, so the per-round probs sum to the total.
        let sum: f64 = c.by_round.iter().map(|r| r.probability).sum();
        assert!((c.meet_probability - sum).abs() < 1e-9);
    }

    #[test]
    fn bracket_dp_sums_to_one_and_favours_the_top_seed() {
        let leaves = [TeamId(0), TeamId(3), TeamId(2), TeamId(1)];
        // Higher id = stronger, via a logistic on the id difference.
        let advance = |a: TeamId, b: TeamId| {
            let (x, y) = (a.0 as f64, b.0 as f64);
            x.exp() / (x.exp() + y.exp())
        };
        let odds = bracket_champion_odds(&leaves, advance);
        let total: f64 = odds.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "champion odds sum to 1: {total}"
        );
        assert_eq!(
            odds[0].0,
            TeamId(3),
            "the strongest team leads the champion odds"
        );
    }

    #[test]
    fn bradley_terry_second_model_predicts_and_picks_a_winner() {
        let ex = Explorer::new();
        let (a, b) = (
            ex.resolve("Argentina").unwrap(),
            ex.resolve("New Zealand").unwrap(),
        );
        let m = ex.bt_predict(a, b, true);
        assert!((m.probabilities.sum() - 1.0).abs() < 1e-9);
        assert!(
            m.probabilities.home_win > m.probabilities.away_win,
            "the second model should favour Argentina over NZ"
        );
        let champs = ex.bt_champion_odds();
        assert_eq!(champs.teams.len(), 32, "a projected 32-team bracket");
        let total: f64 = champs.teams.iter().map(|t| t.champion).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "champion odds sum to 1: {total}"
        );
        assert!(champs.teams[0].champion >= champs.teams[1].champion);
    }

    #[test]
    fn sensitivity_reports_nine_ranked_signals() {
        let ex = Explorer::new();
        let s = ex.sensitivity(2000, 42);
        assert_eq!(s.signals.len(), 9);
        // Ranked by title shift descending, each a valid total-variation distance in [0, 1].
        for w in s.signals.windows(2) {
            assert!(w[0].title_shift >= w[1].title_shift);
        }
        for sig in &s.signals {
            assert!((0.0..=1.0).contains(&sig.title_shift));
            assert!(sig.movers.len() <= 3);
        }
    }
}
