//! # oracle-engine
//!
//! The orchestrator that turns a stream of [`MatchEvent`]s into a continuously
//! updated set of predictions. It is the only place the prediction pieces
//! (`oracle-ratings`, `oracle-model`, `oracle-sim`) and a data source
//! (`oracle-ingest`) are wired together.
//!
//! ## Runtime shape (event-driven, single-writer)
//! ```text
//!   DataProvider.run ──mpsc──▶ event loop ──▶ EngineState (sole owner, no locks)
//!                                  │                 │
//!                                  │            arc-swap latest  ◀── lock-free REST reads
//!                                  └──broadcast──▶ subscribers   ◀── WebSocket / TUI
//! ```
//! A single task owns all mutable state and applies events serially, so there are
//! no data races and no locks on the hot path. Readers get a consistent immutable
//! [`Snapshot`] either by loading the lock-free `arc-swap` cell (REST) or by
//! subscribing to the broadcast channel (push). Expensive Monte-Carlo forecasts are
//! recomputed on a throttle and whenever a result lands, not on every tick.

#![forbid(unsafe_code)]

pub mod presets;
mod snapshot;

pub use snapshot::{MatchPrediction, Metrics, RatingEntry, Snapshot};

use arc_swap::ArcSwap;
use oracle_domain::{
    EventKind, MatchEvent, MatchId, MatchStatus, Outcome, Probabilities, Scoreline, TeamId,
    Tournament,
};
use oracle_ingest::DataProvider;
use oracle_model::{
    implied_probabilities, live_score_grid, Ensemble, GoalModel, LiveConfig, LiveState,
};
use oracle_ratings::{EloConfig, RatingStore};
use oracle_sim::{simulate_with_live, InProgress, LiveInputs, SimConfig, VenueAdj};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Everything the engine needs to start: a data source plus the prediction
/// components (with sensible defaults you can override fluently).
pub struct EngineDeps {
    pub provider: Arc<dyn DataProvider>,
    pub model: GoalModel,
    pub elo_seeds: Vec<(TeamId, f64)>,
    pub elo_config: EloConfig,
    pub ensemble: Ensemble,
    pub live_config: LiveConfig,
    pub sim_config: SimConfig,
}

impl EngineDeps {
    pub fn new(provider: Arc<dyn DataProvider>) -> Self {
        Self {
            provider,
            model: GoalModel::default(),
            elo_seeds: Vec::new(),
            elo_config: EloConfig::default(),
            ensemble: Ensemble::default(),
            live_config: LiveConfig::default(),
            // Live recomputes want to be quick, so a lighter Monte-Carlo by default.
            sim_config: SimConfig {
                iterations: 20_000,
                ..SimConfig::default()
            },
        }
    }

    pub fn with_model(mut self, model: GoalModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_elo_seeds(mut self, seeds: Vec<(TeamId, f64)>) -> Self {
        self.elo_seeds = seeds;
        self
    }

    pub fn with_sim_config(mut self, cfg: SimConfig) -> Self {
        self.sim_config = cfg;
        self
    }

    pub fn with_ensemble(mut self, ensemble: Ensemble) -> Self {
        self.ensemble = ensemble;
        self
    }
}

/// Tuning for the engine runtime.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Bounded capacity of the ingest→engine channel (back-pressure on the source).
    pub channel_capacity: usize,
    /// Broadcast buffer for subscribers (WebSocket/TUI).
    pub broadcast_capacity: usize,
    /// How often the tournament forecast is recomputed regardless of events.
    pub forecast_every: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 1024,
            broadcast_capacity: 256,
            forecast_every: Duration::from_secs(3),
        }
    }
}

/// The shared, read-only handle to a running engine.
pub struct Engine {
    latest: ArcSwap<Snapshot>,
    updates: broadcast::Sender<Arc<Snapshot>>,
    metrics: Arc<Metrics>,
    tournament_name: String,
    provider_name: String,
}

impl Engine {
    /// The current snapshot (lock-free).
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.latest.load_full()
    }

    /// Subscribe to pushed snapshot updates.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Snapshot>> {
        self.updates.subscribe()
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn subscriber_count(&self) -> usize {
        self.updates.receiver_count()
    }

    pub fn tournament_name(&self) -> &str {
        &self.tournament_name
    }

    pub fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// Metrics rendered in Prometheus exposition format.
    pub fn metrics_prometheus(&self) -> String {
        self.metrics.prometheus(self.subscriber_count())
    }
}

/// Start the engine: load the tournament, compute an initial forecast, then spawn the
/// data source and the event loop. Returns the shared handle and the loop's join
/// handle (which completes when the feed ends or `cancel` fires).
pub async fn spawn(
    deps: EngineDeps,
    config: EngineConfig,
    cancel: CancellationToken,
) -> anyhow::Result<(Arc<Engine>, JoinHandle<()>)> {
    let provider = deps.provider.clone();
    let provider_name = provider.name().to_string();
    let tournament = provider.load_tournament().await?;
    tracing::info!(
        provider = %provider_name,
        teams = tournament.teams.len(),
        matches = tournament.matches.len(),
        "loaded tournament"
    );

    let mut state = EngineState::new(tournament, deps);
    state.recompute_forecast();
    let initial = Arc::new(state.build_snapshot(&provider_name));

    let (updates, _rx) = broadcast::channel(config.broadcast_capacity);
    let engine = Arc::new(Engine {
        latest: ArcSwap::new(initial),
        updates,
        metrics: Arc::new(Metrics::default()),
        tournament_name: state.tournament.name.clone(),
        provider_name: provider_name.clone(),
    });

    // Ingest → engine channel, with the provider driving events in its own task.
    let (tx, rx) = mpsc::channel(config.channel_capacity);
    let provider_cancel = cancel.clone();
    let provider_handle = tokio::spawn(async move {
        if let Err(e) = provider.run(tx, provider_cancel).await {
            tracing::warn!(error = %e, "data provider stopped with error");
        }
    });

    let loop_engine = engine.clone();
    let join = tokio::spawn(async move {
        event_loop(state, loop_engine, rx, cancel, config).await;
        provider_handle.abort();
    });

    Ok((engine, join))
}

/// The single-writer event loop.
async fn event_loop(
    mut state: EngineState,
    engine: Arc<Engine>,
    mut rx: mpsc::Receiver<MatchEvent>,
    cancel: CancellationToken,
    config: EngineConfig,
) {
    let mut forecast_timer = tokio::time::interval(config.forecast_every);
    forecast_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    forecast_timer.tick().await; // consume the immediate first tick

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = forecast_timer.tick() => {
                state.recompute_forecast();
                engine.metrics.forecasts_computed.fetch_add(1, Ordering::Relaxed);
                publish(&engine, &state);
            }
            msg = rx.recv() => match msg {
                None => break, // feed exhausted
                Some(event) => {
                    let landed = state.apply_event(&event, &engine.metrics);
                    engine.metrics.events_processed.fetch_add(1, Ordering::Relaxed);
                    // Recompute when a result lands, a score-changing event arrives, or a
                    // lineup is confirmed, so the forecast reflects live scores and lineups.
                    if landed
                        || event.kind.is_material()
                        || matches!(event.kind, EventKind::Lineup { .. })
                    {
                        state.recompute_forecast();
                        engine.metrics.forecasts_computed.fetch_add(1, Ordering::Relaxed);
                    }
                    publish(&engine, &state);
                }
            }
        }
    }
    tracing::info!("engine event loop shutting down");
}

fn publish(engine: &Engine, state: &EngineState) {
    let snap = Arc::new(state.build_snapshot(&engine.provider_name));
    engine.latest.store(snap.clone());
    let _ = engine.updates.send(snap); // Err only means no subscribers; fine.
    engine
        .metrics
        .snapshots_published
        .fetch_add(1, Ordering::Relaxed);
}

/// Live runtime state of one match (overlaying the static fixture).
#[derive(Clone, Copy)]
struct LiveMatch {
    status: MatchStatus,
    score: Scoreline,
    minute: u16,
    home_reds: u8,
    away_reds: u8,
    /// Log-space attack/defense adjustments from a confirmed lineup (positive = stronger).
    /// All zero until a `Lineup` event arrives.
    home_attack_adj: f64,
    home_defense_adj: f64,
    away_attack_adj: f64,
    away_defense_adj: f64,
    /// Bookmaker-implied probabilities, set by an `Odds` event. Folded into the pre-match
    /// ensemble as a third member when present.
    market: Option<Probabilities>,
}

impl LiveMatch {
    fn from_fixture(m: &oracle_domain::Match) -> Self {
        let minute = match m.status {
            MatchStatus::Live { minute } => minute,
            _ => 0,
        };
        Self {
            status: m.status,
            score: m.score,
            minute,
            home_reds: 0,
            away_reds: 0,
            home_attack_adj: 0.0,
            home_defense_adj: 0.0,
            away_attack_adj: 0.0,
            away_defense_adj: 0.0,
            market: None,
        }
    }

    fn has_lineup(&self) -> bool {
        self.home_attack_adj != 0.0
            || self.home_defense_adj != 0.0
            || self.away_attack_adj != 0.0
            || self.away_defense_adj != 0.0
    }

    fn home_adj(&self) -> (f64, f64) {
        (self.home_attack_adj, self.home_defense_adj)
    }

    fn away_adj(&self) -> (f64, f64) {
        (self.away_attack_adj, self.away_defense_adj)
    }
}

/// All mutable engine state - owned exclusively by the event loop.
struct EngineState {
    tournament: Tournament,
    names: HashMap<TeamId, String>,
    match_index: HashMap<MatchId, usize>,
    ratings: RatingStore,
    model: GoalModel,
    ensemble: Ensemble,
    live_config: LiveConfig,
    sim_config: SimConfig,
    live: HashMap<MatchId, LiveMatch>,
    last_forecast: oracle_domain::TournamentForecast,
    /// Per-match venue/travel adjustments (host, altitude, rest), precomputed once.
    venue_adj: HashMap<MatchId, VenueAdj>,
    /// Whether the data feed is currently healthy (updated by `SourceStatus` events).
    source_healthy: bool,
    /// Wall-clock time the last event was processed - surfaces feed staleness.
    last_update: chrono::DateTime<chrono::Utc>,
}

impl EngineState {
    fn new(tournament: Tournament, deps: EngineDeps) -> Self {
        let names = tournament
            .teams
            .iter()
            .map(|t| (t.id, t.name.clone()))
            .collect();
        let match_index = tournament
            .matches
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id, i))
            .collect();
        let mut ratings = RatingStore::new(deps.elo_config);
        for (team, rating) in deps.elo_seeds {
            ratings.seed(team, rating);
        }
        // Venue/travel context is static for the tournament, so precompute it once.
        let venue_adj = oracle_ingest::data::venue_adjustments(&tournament);
        Self {
            tournament,
            names,
            match_index,
            ratings,
            model: deps.model,
            ensemble: deps.ensemble,
            live_config: deps.live_config,
            sim_config: deps.sim_config,
            live: HashMap::new(),
            last_forecast: oracle_domain::TournamentForecast {
                iterations: 0,
                teams: Vec::new(),
            },
            venue_adj,
            source_healthy: true,
            last_update: chrono::Utc::now(),
        }
    }

    /// Combined venue + lineup `(home_adj, away_adj)` deltas for a match (summed in log
    /// space), feeding the adjusted goal model and the Monte-Carlo.
    fn match_adjustments(&self, id: MatchId) -> VenueAdj {
        let ((vh_a, vh_d), (va_a, va_d)) = self.venue_adj.get(&id).copied().unwrap_or_default();
        let ((lh_a, lh_d), (la_a, la_d)) = self
            .live
            .get(&id)
            .map(|l| (l.home_adj(), l.away_adj()))
            .unwrap_or_default();
        ((vh_a + lh_a, vh_d + lh_d), (va_a + la_a, va_d + la_d))
    }

    /// Apply an event to mutable state. Returns `true` when a result landed (the
    /// tournament forecast should be recomputed).
    fn apply_event(&mut self, event: &MatchEvent, metrics: &Metrics) -> bool {
        self.last_update = chrono::Utc::now();
        // Feed-health heartbeats aren't tied to a match - handle before the lookup.
        if let EventKind::SourceStatus { healthy } = event.kind {
            self.source_healthy = healthy;
            return false;
        }
        let Some(&pos) = self.match_index.get(&event.match_id) else {
            return false;
        };
        let home = self.tournament.matches[pos].home;
        let away = self.tournament.matches[pos].away;
        let base = LiveMatch::from_fixture(&self.tournament.matches[pos]);

        let mut final_score: Option<Scoreline> = None;
        {
            let lm = self.live.entry(event.match_id).or_insert(base);
            match &event.kind {
                EventKind::KickOff => {
                    lm.status = MatchStatus::Live {
                        minute: event.minute,
                    };
                    lm.minute = event.minute;
                }
                EventKind::Tick => {
                    lm.minute = event.minute;
                    lm.status = MatchStatus::Live {
                        minute: event.minute,
                    };
                }
                EventKind::Goal { team, .. } => {
                    if *team == home {
                        lm.score.home += 1;
                    } else {
                        lm.score.away += 1;
                    }
                    lm.minute = event.minute;
                    lm.status = MatchStatus::Live {
                        minute: event.minute,
                    };
                    metrics.goals_seen.fetch_add(1, Ordering::Relaxed);
                }
                EventKind::RedCard { team } => {
                    if *team == home {
                        lm.home_reds += 1;
                    } else {
                        lm.away_reds += 1;
                    }
                    lm.minute = event.minute;
                }
                EventKind::FullTime { score } => {
                    lm.score = *score;
                    lm.status = MatchStatus::Finished;
                    lm.minute = 90;
                    final_score = Some(*score);
                }
                EventKind::ScoreSync { score } => {
                    // Authoritative correction: set, don't increment (self-heals drift).
                    lm.score = *score;
                    lm.status = MatchStatus::Live {
                        minute: event.minute,
                    };
                    lm.minute = event.minute;
                }
                EventKind::Lineup {
                    home: home_xi,
                    away: away_xi,
                } => {
                    // A confirmed lineup adjusts each team's effective attack/defense.
                    let (h_atk, h_def) = oracle_ingest::data::lineup_adjustment(home, home_xi);
                    let (a_atk, a_def) = oracle_ingest::data::lineup_adjustment(away, away_xi);
                    lm.home_attack_adj = h_atk;
                    lm.home_defense_adj = h_def;
                    lm.away_attack_adj = a_atk;
                    lm.away_defense_adj = a_def;
                }
                EventKind::Odds { home, draw, away } => {
                    lm.market = Some(implied_probabilities(*home, *draw, *away));
                }
                // SourceStatus is handled above the match lookup; the rest are no-ops.
                EventKind::SourceStatus { .. }
                | EventKind::HalfTime
                | EventKind::YellowCard { .. } => {}
            }
        }

        if let Some(score) = final_score {
            // World Cup matches are at neutral venues.
            self.ratings.record(home, away, score, true);
            self.tournament.matches[pos].status = MatchStatus::Finished;
            self.tournament.matches[pos].score = score;
            return true;
        }
        false
    }

    fn recompute_forecast(&mut self) {
        // Condition the forecast on live state: matches in play (sample the remainder) and
        // matches with a confirmed lineup (adjusted goal rates), instead of treating every
        // unfinished match as a fresh, full-strength 0-0. Venue/travel context applies to
        // every fixture.
        let live: HashMap<MatchId, InProgress> = self
            .live
            .iter()
            .filter(|(_, lm)| matches!(lm.status, MatchStatus::Live { .. }) || lm.has_lineup())
            .map(|(&id, lm)| {
                (
                    id,
                    InProgress {
                        score: lm.score,
                        minute: lm.minute,
                        home_reds: lm.home_reds,
                        away_reds: lm.away_reds,
                        home_attack_adj: lm.home_attack_adj,
                        home_defense_adj: lm.home_defense_adj,
                        away_attack_adj: lm.away_attack_adj,
                        away_defense_adj: lm.away_defense_adj,
                    },
                )
            })
            .collect();
        let inputs = LiveInputs {
            live,
            venue: self.venue_adj.clone(),
        };
        self.last_forecast = simulate_with_live(
            &self.tournament,
            &self.model,
            self.sim_config,
            &inputs,
            self.live_config,
        );
    }

    fn name_of(&self, id: TeamId) -> String {
        self.names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| id.to_string())
    }

    fn predict_match(&self, m: &oracle_domain::Match) -> MatchPrediction {
        let lm = self.live.get(&m.id).copied();
        let (status, score, minute, home_reds, away_reds) = match lm {
            Some(l) => (l.status, l.score, l.minute, l.home_reds, l.away_reds),
            None => (m.status, m.score, 0, 0, 0),
        };
        const NEUTRAL: bool = true;
        // Apply venue/travel + confirmed-lineup adjustments to this match's goal rates.
        let (home_adj, away_adj) = self.match_adjustments(m.id);
        let (lambda, mu) = self
            .model
            .expected_goals_adjusted(m.home, m.away, NEUTRAL, home_adj, away_adj);

        let (probabilities, score_grid, most_likely_score) = match status {
            MatchStatus::Live { .. } => {
                let st = LiveState {
                    current: score,
                    minute,
                    home_red_cards: home_reds,
                    away_red_cards: away_reds,
                };
                let grid = live_score_grid(lambda, mu, &st, &self.live_config);
                let probs = grid.outcome_probabilities();
                let mls = grid.most_likely_score();
                (probs, Some(grid), Some(mls))
            }
            MatchStatus::Finished => (one_hot(score.outcome()), None, None),
            MatchStatus::Scheduled | MatchStatus::Postponed => {
                // Pre-match: blend the (lineup-adjusted) Dixon-Coles grid with Elo, plus the
                // bookmaker market as a third member when odds are present. The ensemble has
                // three weights; `blend` renormalizes, so two members is a clean fallback.
                let grid = self
                    .model
                    .score_grid_adjusted(m.home, m.away, NEUTRAL, home_adj, away_adj);
                let dc = grid.outcome_probabilities();
                let elo = self.ratings.win_probabilities(m.home, m.away, NEUTRAL);
                let mut members = vec![dc, elo];
                if let Some(market) = lm.and_then(|l| l.market) {
                    members.push(market);
                }
                let blended = self.ensemble.blend(&members);
                let mls = grid.most_likely_score();
                (blended, Some(grid), Some(mls))
            }
        };

        MatchPrediction {
            match_id: m.id,
            home: m.home,
            away: m.away,
            home_name: self.name_of(m.home),
            away_name: self.name_of(m.away),
            stage: m.stage,
            status,
            score,
            minute,
            probabilities,
            score_grid,
            most_likely_score,
        }
    }

    fn build_snapshot(&self, provider_name: &str) -> Snapshot {
        let matches = self
            .tournament
            .matches
            .iter()
            .map(|m| self.predict_match(m))
            .collect();

        let mut ratings: Vec<RatingEntry> = self
            .tournament
            .teams
            .iter()
            .map(|t| RatingEntry {
                team: t.id,
                name: t.name.clone(),
                rating: self.ratings.rating(t.id),
            })
            .collect();
        ratings.sort_by(|a, b| {
            b.rating
                .partial_cmp(&a.rating)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Snapshot {
            generated_at: chrono::Utc::now(),
            tournament: self.tournament.name.clone(),
            provider: provider_name.to_string(),
            source_healthy: self.source_healthy,
            last_update: self.last_update,
            matches,
            forecast: self.last_forecast.clone(),
            ratings,
        }
    }
}

fn one_hot(outcome: Outcome) -> Probabilities {
    match outcome {
        Outcome::HomeWin => Probabilities::new(1.0, 0.0, 0.0),
        Outcome::Draw => Probabilities::new(0.0, 1.0, 0.0),
        Outcome::AwayWin => Probabilities::new(0.0, 0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_ingest::{data, SimProvider};
    use std::time::Duration;

    fn simulated_deps() -> EngineDeps {
        let provider = Arc::new(SimProvider::new().with_minute_delay(Duration::from_millis(0)));
        EngineDeps::new(provider)
            .with_model(data::fit_baseline_model(7))
            .with_elo_seeds(data::team_strengths())
    }

    #[tokio::test]
    async fn initial_snapshot_is_well_formed() {
        let cancel = CancellationToken::new();
        let (engine, join) = spawn(simulated_deps(), EngineConfig::default(), cancel.clone())
            .await
            .unwrap();

        let snap = engine.snapshot();
        assert_eq!(snap.matches.len(), 72);
        assert_eq!(snap.forecast.teams.len(), 48);
        for m in &snap.matches {
            assert!((m.probabilities.sum() - 1.0).abs() < 1e-6);
        }
        // Champion probabilities should sum to ~1 across the field.
        let total: f64 = snap.forecast.teams.iter().map(|t| t.p_champion).sum();
        assert!((total - 1.0).abs() < 0.05, "champion mass = {total}");

        cancel.cancel();
        let _ = join.await;
    }

    #[tokio::test]
    async fn processes_live_events_and_pushes_updates() {
        let cancel = CancellationToken::new();
        let (engine, join) = spawn(simulated_deps(), EngineConfig::default(), cancel.clone())
            .await
            .unwrap();

        let mut sub = engine.subscribe();
        // We should receive pushed snapshots as the sim feed produces events.
        let pushed = tokio::time::timeout(Duration::from_secs(5), sub.recv()).await;
        assert!(pushed.is_ok(), "expected at least one pushed snapshot");

        // Give the feed a moment, then confirm events were processed.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(engine.metrics().events_processed.load(Ordering::Relaxed) > 0);

        cancel.cancel();
        let _ = join.await;
    }

    fn fresh_state() -> EngineState {
        let deps = EngineDeps::new(Arc::new(SimProvider::new()));
        EngineState::new(data::world_cup_2026(), deps)
    }

    #[test]
    fn score_sync_authoritatively_corrects_drift() {
        let mut state = fresh_state();
        let metrics = Metrics::default();
        let m = state.tournament.matches[0].clone();

        // A goal nudges the running tally to 1-0...
        state.apply_event(
            &MatchEvent::new(
                m.id,
                10,
                EventKind::Goal {
                    team: m.home,
                    scorer: None,
                },
            ),
            &metrics,
        );
        // ...but the authoritative feed says it's actually 0-3 (we missed events).
        state.apply_event(
            &MatchEvent::new(
                m.id,
                70,
                EventKind::ScoreSync {
                    score: Scoreline::new(0, 3),
                },
            ),
            &metrics,
        );

        assert_eq!(
            state.live.get(&m.id).expect("live state").score,
            Scoreline::new(0, 3),
            "ScoreSync should set, not add to, the score"
        );
    }

    #[test]
    fn source_status_toggles_health() {
        let mut state = fresh_state();
        let metrics = Metrics::default();
        assert!(state.source_healthy, "feed starts healthy");

        state.apply_event(
            &MatchEvent::new(MatchId(0), 0, EventKind::SourceStatus { healthy: false }),
            &metrics,
        );
        assert!(!state.source_healthy, "an outage marks the feed unhealthy");

        state.apply_event(
            &MatchEvent::new(MatchId(0), 0, EventKind::SourceStatus { healthy: true }),
            &metrics,
        );
        assert!(state.source_healthy, "recovery marks it healthy again");
    }

    #[test]
    fn benching_the_star_lowers_a_teams_win_probability() {
        let mut state = fresh_state();
        let metrics = Metrics::default();
        let m = state.tournament.matches[0].clone();

        let baseline = state.predict_match(&m).probabilities.home_win;

        // Confirmed lineup: the home side benches its star, the away side is full strength.
        state.apply_event(
            &MatchEvent::new(
                m.id,
                0,
                EventKind::Lineup {
                    home: data::starting_lineup(m.home, true),
                    away: data::starting_lineup(m.away, false),
                },
            ),
            &metrics,
        );

        let weakened = state.predict_match(&m).probabilities.home_win;
        assert!(
            weakened < baseline,
            "benching the home star should lower home win probability ({baseline:.3} -> {weakened:.3})"
        );
    }

    #[test]
    fn odds_anchor_a_scheduled_match_toward_the_market() {
        let mut state = fresh_state();
        let metrics = Metrics::default();
        let m = state.tournament.matches[0].clone();
        let baseline = state.predict_match(&m).probabilities.home_win;

        // A bookmaker line heavily favouring the away team.
        state.apply_event(
            &MatchEvent::new(
                m.id,
                0,
                EventKind::Odds {
                    home: 8.0,
                    draw: 5.0,
                    away: 1.3,
                },
            ),
            &metrics,
        );
        let anchored = state.predict_match(&m).probabilities.home_win;
        assert!(
            anchored < baseline,
            "away-favouring odds should pull home win probability down ({baseline:.3} -> {anchored:.3})"
        );
    }
}
