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

pub mod event_log;
pub mod presets;
pub mod query;
mod snapshot;

pub use event_log::EventLog;
pub use query::{signal_sensitivity, Explorer, SignalContribution};
pub use snapshot::{
    AdaptiveState, BtChampion, Call, MatchPrediction, Metrics, ModelScore, RatingEntry, ReportCard,
    Snapshot, Upset, WinProbSample,
};

use arc_swap::ArcSwap;
use oracle_domain::{
    EventKind, Match, MatchEvent, MatchId, MatchStatus, Outcome, Probabilities, Scoreline, Stage,
    TeamId, Tournament,
};
use oracle_ingest::DataProvider;
use oracle_model::{
    apply_temperature, fit_gain_toward_one, fit_temperature, implied_probabilities,
    live_score_grid, score, BradleyTerry, CalibrationReport, Ensemble, GoalModel, LiveConfig,
    LiveState,
};
use oracle_ratings::{EloConfig, RatingStore, StateSpaceRatings};
use oracle_sim::{simulate_with_live, InProgress, LiveInputs, SimConfig, VenueAdj};
use std::collections::HashMap;
use std::path::PathBuf;
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
    /// Learning rate for the online goal-model update applied to each finished match, so the
    /// model learns from in-tournament results. 0 disables in-tournament learning.
    pub tournament_lr: f64,
    /// Yellow cards that suspend a player for their team's next match. 0 disables suspension
    /// tracking. The World Cup group-stage rule is 2.
    pub suspension_threshold: u8,
    /// State-space (Kalman) rating: a dynamic in-tournament rating that also supplies the
    /// per-team uncertainty the Monte-Carlo resamples from.
    pub state_space: StateSpaceRatings,
    /// The second, outcome-based forecaster (Bradley-Terry-Davidson), scored head-to-head with the
    /// goal-model ensemble and updated online from results.
    pub bradley_terry: BradleyTerry,
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
            // In-tournament results are the most relevant data the model sees, so the online
            // goal-model update leans in a bit harder than a routine friendly would.
            tournament_lr: 0.05,
            suspension_threshold: 2,
            state_space: StateSpaceRatings::with_defaults(),
            bradley_terry: BradleyTerry::default(),
        }
    }

    pub fn with_model(mut self, model: GoalModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_tournament_lr(mut self, lr: f64) -> Self {
        self.tournament_lr = lr;
        self
    }

    pub fn with_suspension_threshold(mut self, threshold: u8) -> Self {
        self.suspension_threshold = threshold;
        self
    }

    pub fn with_state_space(mut self, ratings: StateSpaceRatings) -> Self {
        self.state_space = ratings;
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

    pub fn with_bradley_terry(mut self, bt: BradleyTerry) -> Self {
        self.bradley_terry = bt;
        self
    }
}

/// Tuning for the engine runtime.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Bounded capacity of the ingest→engine channel (back-pressure on the source).
    pub channel_capacity: usize,
    /// Broadcast buffer for subscribers (WebSocket/TUI).
    pub broadcast_capacity: usize,
    /// How often the tournament forecast is recomputed regardless of events.
    pub forecast_every: Duration,
    /// Optional append-only event log. When set, every event is recorded and the log is
    /// replayed on startup to recover state across restarts.
    pub event_log: Option<PathBuf>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 1024,
            broadcast_capacity: 256,
            forecast_every: Duration::from_secs(3),
            event_log: None,
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
    let metrics = Arc::new(Metrics::default());

    // Recover from a prior run: replay the event log before the live feed resumes.
    let event_log = match &config.event_log {
        Some(path) => {
            let prior = EventLog::read(path)?;
            if !prior.is_empty() {
                tracing::info!(events = prior.len(), "replaying event log to recover state");
                for ev in &prior {
                    state.apply_event(ev, &metrics);
                    metrics.events_processed.fetch_add(1, Ordering::Relaxed);
                }
            }
            Some(Arc::new(EventLog::create(path)?))
        }
        None => None,
    };

    state.recompute_forecast();
    let initial = Arc::new(state.build_snapshot(&provider_name));

    let (updates, _rx) = broadcast::channel(config.broadcast_capacity);
    let engine = Arc::new(Engine {
        latest: ArcSwap::new(initial),
        updates,
        metrics: metrics.clone(),
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
        event_loop(state, loop_engine, rx, cancel, config, event_log).await;
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
    event_log: Option<Arc<EventLog>>,
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
                    // Durably record the event before acting on it (crash recovery).
                    if let Some(log) = &event_log {
                        if let Err(e) = log.append(&event) {
                            tracing::warn!(error = %e, "event-log append failed");
                        }
                    }
                    let landed = state.apply_event(&event, &engine.metrics);
                    engine.metrics.events_processed.fetch_add(1, Ordering::Relaxed);
                    // Record the live win-probability timeline (the fan drama graph).
                    state.sample_live_history();
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
    state_space: StateSpaceRatings,
    model: GoalModel,
    ensemble: Ensemble,
    live_config: LiveConfig,
    sim_config: SimConfig,
    /// Online learning rate applied to the goal model on each finished match (0 = off).
    tournament_lr: f64,
    /// Yellow-card threshold for a suspension (0 = off).
    suspension_threshold: u8,
    /// Yellow-card count per (team, player), reset when a suspension is triggered.
    yellows: HashMap<(TeamId, String), u8>,
    /// Players suspended for a given match (its team's next unplayed match when carded).
    suspended: HashMap<MatchId, Vec<(TeamId, String)>>,
    live: HashMap<MatchId, LiveMatch>,
    last_forecast: oracle_domain::TournamentForecast,
    /// Per-match **context** adjustments (host/crowd/travel/heat/altitude/rest), precomputed once
    /// (`data::venue_adjustments`). Kept separate from style so the live context gain scales only
    /// this part.
    context_adj: HashMap<MatchId, VenueAdj>,
    /// Per-match **style matchup** adjustments, precomputed once (`data::style_adjustments`).
    style_adj: HashMap<MatchId, VenueAdj>,
    /// Aggregate multiplier on the context adjustment, recalibrated in-tournament: the reasoned
    /// priors (host/crowd/travel/heat) are scaled toward what the tournament actually shows. `1.0`
    /// (the prior) until enough results accumulate. One gain rather than per-signal, since a
    /// tournament's matches cannot reliably separate the correlated context effects.
    context_gain: f64,
    /// `(context contribution to predicted margin, residual vs observed margin)` per finished
    /// match, the regression that fits `context_gain`.
    context_calib: Vec<(f64, f64)>,
    /// Each finished match's leak-free pre-match forecast, kept so the snapshot can flag the
    /// biggest upsets (favourite-failed-to-win) for the fan-facing upset radar.
    pre_match_forecast: HashMap<MatchId, Probabilities>,
    /// Per-live-match win-probability timeline (one sample per minute), for the fan drama graph.
    /// Dropped when a match finishes.
    live_history: HashMap<MatchId, Vec<WinProbSample>>,
    /// The second forecaster (Bradley-Terry-Davidson), online-updated from each result.
    bradley_terry: BradleyTerry,
    /// Each finished match's pre-match Bradley-Terry forecast (leak-free), for the head-to-head.
    bt_pre_match_forecast: HashMap<MatchId, Probabilities>,
    /// Per-team knockout factors precomputed once: penalty-shootout skill and knockout pedigree.
    shootout_rating: HashMap<TeamId, f64>,
    knockout_pedigree: HashMap<TeamId, f64>,
    /// Whether the data feed is currently healthy (updated by `SourceStatus` events).
    source_healthy: bool,
    /// Wall-clock time the last event was processed - surfaces feed staleness.
    last_update: chrono::DateTime<chrono::Utc>,
    /// Live calibration: each finished match's pre-match forecast paired with its realized
    /// outcome (leak-free - the forecast is recorded before the result updates the model).
    calib_pairs: Vec<(Probabilities, Outcome)>,
    /// The temperature fitted on `calib_pairs` and applied to remaining pre-match forecasts:
    /// post-hoc **temperature scaling**, recalibrating the model against the tournament as it
    /// plays. `1.0` (the identity) until enough matches have finished.
    calib_temperature: f64,
}

/// Finished matches needed before the live temperature recalibration kicks in (below this it stays
/// at the identity, since a handful of results cannot pin down a calibration correction).
const MIN_CALIB_SAMPLES: usize = 12;

/// Finished matches needed before the context gain moves off 1.0.
const CONTEXT_CALIB_MIN: usize = 20;
/// Prior precision (shrinkage toward gain 1) for the context recalibration. Deliberately strong:
/// a tournament is a small, noisy sample, so the gain should barely move without clear evidence.
const CONTEXT_PRIOR_PRECISION: f64 = 8.0;

/// Cap on a single match's win-probability timeline (a match is ~90-120 minutes, one sample each).
const MAX_LIVE_SAMPLES: usize = 130;

/// Extra weight on the bookmaker market for **knockout** ties, where the single-match closing line
/// is an especially sharp signal (the ensemble weights are learned mostly on group/league play, so
/// they under-weight it). Applied only when odds are present for a knockout-stage match.
const KNOCKOUT_MARKET_BOOST: f64 = 1.75;

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
        // Match context (venue/crowd/travel/heat) and the style matchup are static for the
        // tournament, so precompute each once. They are kept separate so the live context gain
        // can scale the context part without touching style.
        let context_adj = oracle_ingest::data::venue_adjustments(&tournament);
        let style_adj = oracle_ingest::data::style_adjustments(&tournament);
        let shootout_rating = oracle_ingest::data::shootout_ratings();
        let knockout_pedigree = oracle_ingest::data::knockout_pedigree();
        Self {
            tournament,
            names,
            match_index,
            ratings,
            state_space: deps.state_space,
            model: deps.model,
            ensemble: deps.ensemble,
            live_config: deps.live_config,
            sim_config: deps.sim_config,
            tournament_lr: deps.tournament_lr,
            suspension_threshold: deps.suspension_threshold,
            yellows: HashMap::new(),
            suspended: HashMap::new(),
            live: HashMap::new(),
            last_forecast: oracle_domain::TournamentForecast {
                iterations: 0,
                teams: Vec::new(),
            },
            context_adj,
            style_adj,
            context_gain: 1.0,
            context_calib: Vec::new(),
            pre_match_forecast: HashMap::new(),
            live_history: HashMap::new(),
            bradley_terry: deps.bradley_terry,
            bt_pre_match_forecast: HashMap::new(),
            shootout_rating,
            knockout_pedigree,
            source_healthy: true,
            last_update: chrono::Utc::now(),
            calib_pairs: Vec::new(),
            calib_temperature: 1.0,
        }
    }

    /// Combined `(home_adj, away_adj)` deltas for a match (summed in log space): the
    /// context (scaled by the live-recalibrated `context_gain`), the style matchup, and player
    /// availability. Feeds the adjusted goal model and the Monte-Carlo.
    fn match_adjustments(&self, id: MatchId) -> VenueAdj {
        let g = self.context_gain;
        let ((ch_a, ch_d), (ca_a, ca_d)) = self.context_adj.get(&id).copied().unwrap_or_default();
        let ((sh_a, sh_d), (sa_a, sa_d)) = self.style_adj.get(&id).copied().unwrap_or_default();
        // Player availability: announced lineup if known, else a suspension-derived estimate.
        let ((lh_a, lh_d), (la_a, la_d)) = self.availability_adj(id);
        (
            (g * ch_a + sh_a + lh_a, g * ch_d + sh_d + lh_d),
            (g * ca_a + sa_a + la_a, g * ca_d + sa_d + la_d),
        )
    }

    /// The static per-match adjustment (no availability): the context scaled by `context_gain`
    /// plus the style matchup. This is the base the Monte-Carlo's `venue` map is built from.
    fn scaled_context_plus_style(&self) -> HashMap<MatchId, VenueAdj> {
        let g = self.context_gain;
        let mut out: HashMap<MatchId, VenueAdj> = HashMap::new();
        for (&id, adj) in &self.context_adj {
            out.insert(
                id,
                ((g * adj.0 .0, g * adj.0 .1), (g * adj.1 .0, g * adj.1 .1)),
            );
        }
        for (&id, adj) in &self.style_adj {
            let e = out.entry(id).or_default();
            e.0 .0 += adj.0 .0;
            e.0 .1 += adj.0 .1;
            e.1 .0 += adj.1 .0;
            e.1 .1 += adj.1 .1;
        }
        out
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
        let mut booking: Option<(TeamId, String)> = None;
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
                EventKind::YellowCard { team, player } => {
                    // Recorded for booking accumulation; resolved after the borrow ends.
                    if let Some(name) = player {
                        booking = Some((*team, name.clone()));
                    }
                }
                // SourceStatus is handled above the match lookup; the rest are no-ops.
                EventKind::SourceStatus { .. } | EventKind::HalfTime => {}
            }
        }

        if let Some(score) = final_score {
            // Live calibration: pair this match's *pre-match* forecast with its realized outcome
            // before the model learns from it (so the pair is leak-free), then refit the
            // temperature applied to the remaining forecasts.
            let m = self.tournament.matches[pos].clone();
            let pred = self.pre_match_probs(&m);
            self.calib_pairs.push((pred, score.outcome()));
            self.pre_match_forecast.insert(event.match_id, pred);
            // Second model's pre-match call, recorded before it learns from the result (leak-free).
            let bt_pred = self.bradley_terry.outcome_probabilities(home, away, true);
            self.bt_pre_match_forecast.insert(event.match_id, bt_pred);
            self.live_history.remove(&event.match_id); // done; drop its live timeline
            self.refit_calibration();
            // Context recalibration: how much of this match's margin the context signals explain
            // (residual vs observed), to shrink the aggregate context gain toward what the
            // tournament shows. Uses the pre-result model, so it is leak-free.
            let z = f64::from(score.home) - f64::from(score.away);
            let (lb, mb) = self.model.expected_goals(home, away, true);
            let base_margin = lb - mb;
            let ((ch_a, ch_d), (ca_a, ca_d)) = self
                .context_adj
                .get(&event.match_id)
                .copied()
                .unwrap_or_default();
            let (lc, mc) =
                self.model
                    .expected_goals_adjusted(home, away, true, (ch_a, ch_d), (ca_a, ca_d));
            self.context_calib
                .push(((lc - mc) - base_margin, z - base_margin));
            self.refit_context_gain();
            // World Cup matches are at neutral venues. Both ratings learn from the result:
            // Elo, and the Dixon-Coles goal model (online, so the forecast tracks tournament
            // form instead of staying frozen at the offline fit).
            self.ratings.record(home, away, score, true);
            self.model
                .update_with_result(home, away, score, true, self.tournament_lr);
            // The state-space rating learns too, updating both its mean and its per-team variance
            // (which feeds the forecast's parameter uncertainty). Using the tournament-specific
            // observe injects a per-match process-noise bump so the filter keeps tracking form
            // fast instead of growing overconfident across the competition.
            self.state_space.observe_tournament(home, away, score, true);
            // The second model learns from the same result (an online step on the two strengths).
            self.bradley_terry.update_with_result(
                home,
                away,
                score.outcome(),
                true,
                self.tournament_lr,
            );
            self.tournament.matches[pos].status = MatchStatus::Finished;
            self.tournament.matches[pos].score = score;
            // If that was the last group result, the knockout participants are now known:
            // materialize the real bracket so the forecast plays it (and live knockout matches
            // are conditioned) instead of re-deriving it from a fresh group simulation.
            self.materialize_knockout_if_ready();
            return true;
        }

        // Booking accumulation: on reaching the threshold, suspend the player for the
        // team's next unplayed match (reset the count, since the suspension is then served).
        if let Some((team, name)) = booking {
            if self.suspension_threshold > 0 {
                let count = self.yellows.entry((team, name.clone())).or_insert(0);
                *count += 1;
                if *count >= self.suspension_threshold {
                    *count = 0;
                    if let Some(next) = self.next_unplayed_match(team) {
                        let slot = self.suspended.entry(next).or_default();
                        if !slot.iter().any(|(t, n)| *t == team && *n == name) {
                            slot.push((team, name));
                            return true; // the next match's forecast changed
                        }
                    }
                }
            }
        }
        false
    }

    /// When the final group result lands the 32 knockout qualifiers are known: materialize the
    /// Round-of-32 fixtures (the real 2026 bracket), append them, and re-index so they enter the
    /// forecast and can be conditioned live as they play. A no-op when the bracket already exists,
    /// the group stage is unfinished, or the tournament is not the 2026 shape (see
    /// [`oracle_ingest::data::materialize_knockout`]).
    fn materialize_knockout_if_ready(&mut self) {
        // Round of 32, derived from the completed group stage.
        let r32 = oracle_ingest::data::materialize_knockout(&self.tournament);
        if !r32.is_empty() {
            tracing::info!(
                fixtures = r32.len(),
                "group stage complete - materialized the Round of 32"
            );
            self.append_fixtures(r32);
        }
        // Then each subsequent round, once the prior one is fully decided. This only fires when the
        // bracket was derived here and its ties get played; a provider that supplies the full
        // bracket up front leaves it a no-op (the next round already exists).
        self.materialize_next_knockout_round();
    }

    /// Append fixtures to the live tournament, re-index them, and refresh the per-match context and
    /// style adjustments so they cover the new fixtures.
    fn append_fixtures(&mut self, fixtures: Vec<Match>) {
        let start = self.tournament.matches.len();
        self.tournament.matches.extend(fixtures);
        for i in start..self.tournament.matches.len() {
            self.match_index.insert(self.tournament.matches[i].id, i);
        }
        self.context_adj = oracle_ingest::data::venue_adjustments(&self.tournament);
        self.style_adj = oracle_ingest::data::style_adjustments(&self.tournament);
    }

    /// Materialize the next knockout round once the current one is fully decided: pair the adjacent
    /// ties' winners up the bracket (the Round-of-32 fixtures are stored in bracket order, so
    /// adjacent ties feed one next-round tie). A level tie resolves to the home side, matching the
    /// simulator's documented penalty-shootout rule. One round per call.
    fn materialize_next_knockout_round(&mut self) {
        const PATH: [Stage; 5] = [
            Stage::RoundOf32,
            Stage::RoundOf16,
            Stage::QuarterFinal,
            Stage::SemiFinal,
            Stage::Final,
        ];
        for w in PATH.windows(2) {
            let (cur, next) = (w[0], w[1]);
            let ties: Vec<(TeamId, TeamId, Scoreline, bool)> = self
                .tournament
                .matches
                .iter()
                .filter(|m| m.stage == cur)
                .map(|m| (m.home, m.away, m.score, m.is_finished()))
                .collect();
            let next_exists = self.tournament.matches.iter().any(|m| m.stage == next);
            if ties.len() < 2 || next_exists || !ties.iter().all(|(_, _, _, fin)| *fin) {
                continue;
            }
            let winners: Vec<TeamId> = ties
                .iter()
                .map(|(h, a, s, _)| if s.away > s.home { *a } else { *h })
                .collect();
            let base_id = self
                .tournament
                .matches
                .iter()
                .map(|m| m.id.0)
                .max()
                .unwrap_or(0);
            let last_kickoff = self
                .tournament
                .matches
                .iter()
                .map(|m| m.kickoff)
                .max()
                .unwrap_or_else(chrono::Utc::now);
            let fixtures: Vec<Match> = winners
                .chunks(2)
                .enumerate()
                .filter(|(_, pair)| pair.len() == 2)
                .map(|(i, pair)| Match {
                    id: MatchId(base_id + 1 + i as u32),
                    home: pair[0],
                    away: pair[1],
                    stage: next,
                    kickoff: last_kickoff + chrono::Duration::hours(24 + (i as i64) * 3),
                    status: MatchStatus::Scheduled,
                    score: Scoreline::new(0, 0),
                })
                .collect();
            if fixtures.is_empty() {
                continue;
            }
            tracing::info!(round = %next, fixtures = fixtures.len(), "knockout round decided - materialized the next round");
            self.append_fixtures(fixtures);
            break;
        }
    }

    /// The second model's live champion odds over the current knockout bracket: run the bracket
    /// dynamic program from the deepest materialized round (a decided tie enters as its winner, an
    /// undecided tie as the Bradley-Terry advance split), so it conditions on knockout results
    /// already played and projects the rest. Empty until the Round of 32 is materialized.
    fn bt_champions(&self) -> Vec<BtChampion> {
        const PATH: [Stage; 5] = [
            Stage::RoundOf32,
            Stage::RoundOf16,
            Stage::QuarterFinal,
            Stage::SemiFinal,
            Stage::Final,
        ];
        let Some(&stage) = PATH
            .iter()
            .rev()
            .find(|s| self.tournament.matches.iter().any(|m| m.stage == **s))
        else {
            return Vec::new();
        };
        let layer: Vec<HashMap<TeamId, f64>> = self
            .tournament
            .matches
            .iter()
            .filter(|m| m.stage == stage)
            .map(|m| {
                let mut d = HashMap::new();
                if m.is_finished() {
                    let winner = if m.score.away > m.score.home {
                        m.away
                    } else {
                        m.home
                    };
                    d.insert(winner, 1.0);
                } else {
                    let p = self.bradley_terry.advance_probability(m.home, m.away);
                    d.insert(m.home, p);
                    d.insert(m.away, 1.0 - p);
                }
                d
            })
            .collect();
        if layer.is_empty() {
            return Vec::new();
        }
        crate::query::champion_odds_from_layer(layer, |a, b| {
            self.bradley_terry.advance_probability(a, b)
        })
        .into_iter()
        .map(|(team, champion)| BtChampion {
            team: self.name_of(team),
            champion,
        })
        .collect()
    }

    /// The team's next unplayed match by kickoff (where a suspension would be served).
    fn next_unplayed_match(&self, team: TeamId) -> Option<MatchId> {
        self.tournament
            .matches
            .iter()
            .filter(|m| !m.is_finished() && (m.home == team || m.away == team))
            .min_by_key(|m| m.kickoff)
            .map(|m| m.id)
    }

    /// Player-availability lineup delta for a match: the announced lineup if one was
    /// received (it already reflects reality), else a suspension-derived estimate (a
    /// suspended starter missing the strongest XI), else none.
    fn availability_adj(&self, id: MatchId) -> VenueAdj {
        if let Some(lm) = self.live.get(&id) {
            if lm.has_lineup() {
                return (lm.home_adj(), lm.away_adj());
            }
        }
        let Some(out) = self.suspended.get(&id) else {
            return ((0.0, 0.0), (0.0, 0.0));
        };
        let Some(m) = self
            .match_index
            .get(&id)
            .map(|&p| &self.tournament.matches[p])
        else {
            return ((0.0, 0.0), (0.0, 0.0));
        };
        let side_adj = |team: TeamId| -> (f64, f64) {
            let banned: Vec<&String> = out
                .iter()
                .filter(|(t, _)| *t == team)
                .map(|(_, n)| n)
                .collect();
            if banned.is_empty() {
                return (0.0, 0.0);
            }
            let present: Vec<String> = oracle_ingest::data::starting_lineup(team, false)
                .into_iter()
                .filter(|n| !banned.contains(&n))
                .collect();
            oracle_ingest::data::lineup_adjustment(team, &present)
        };
        (side_adj(m.home), side_adj(m.away))
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
        // Context (scaled by the recalibrated gain) + style apply to every fixture; suspensions
        // add a player-availability penalty to scheduled matches that have no announced lineup yet
        // (lineup/live matches already carry their delta in `live`, so we skip them to avoid
        // double count).
        let mut venue = self.scaled_context_plus_style();
        for &id in self.suspended.keys() {
            if live.contains_key(&id) {
                continue;
            }
            let ((ha, hd), (aa, ad)) = self.availability_adj(id);
            let e = venue.entry(id).or_default();
            e.0 .0 += ha;
            e.0 .1 += hd;
            e.1 .0 += aa;
            e.1 .1 += ad;
        }
        // Dynamic parameter uncertainty: map each team's state-space strength SD into the
        // Monte-Carlo's log-rate units. This shrinks as a team plays more and grows with
        // time, the principled replacement for the goal model's static fit-based uncertainty.
        const STRENGTH_SD_TO_LOGRATE: f64 = 0.25;
        let rating_sigma: HashMap<TeamId, f64> = self
            .tournament
            .teams
            .iter()
            .map(|t| (t.id, STRENGTH_SD_TO_LOGRATE * self.state_space.stddev(t.id)))
            .collect();
        let inputs = LiveInputs {
            live,
            venue,
            rating_sigma,
            shootout_rating: self.shootout_rating.clone(),
            knockout_pedigree: self.knockout_pedigree.clone(),
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

    /// The raw (pre-calibration) pre-match ensemble blend for a matchup, given its score grid:
    /// [Dixon-Coles, Elo, state-space, (market)] through the learned ensemble. Shared by the
    /// scheduled-match prediction and the live-calibration capture so they stay identical.
    fn blend_pre_match(
        &self,
        m: &oracle_domain::Match,
        grid: &oracle_domain::ScoreGrid,
    ) -> Probabilities {
        const NEUTRAL: bool = true;
        let dc = grid.outcome_probabilities();
        let elo = self.ratings.win_probabilities(m.home, m.away, NEUTRAL);
        let kalman = self.state_space.win_probabilities(m.home, m.away, NEUTRAL);
        let mut members = vec![dc, elo, kalman];
        let market_present = if let Some(market) = self.live.get(&m.id).and_then(|l| l.market) {
            members.push(market);
            true
        } else {
            false
        };
        // Knockout ties lean harder on the market: the single-match closing line is the sharpest
        // signal there, so boost its weight when odds are present for a knockout-stage fixture.
        let boost = if market_present && m.stage.is_knockout() {
            KNOCKOUT_MARKET_BOOST
        } else {
            1.0
        };
        self.ensemble.blend_with_market_boost(&members, boost)
    }

    /// The full raw pre-match forecast for a matchup (its own grid, then [`blend_pre_match`]).
    /// Used to record a leak-free `(prediction, outcome)` pair for live calibration.
    fn pre_match_probs(&self, m: &oracle_domain::Match) -> Probabilities {
        let (home_adj, away_adj) = self.match_adjustments(m.id);
        let grid = self
            .model
            .score_grid_adjusted(m.home, m.away, true, home_adj, away_adj);
        self.blend_pre_match(m, &grid)
    }

    /// Refit the calibration temperature once enough matches have finished. Post-hoc temperature
    /// scaling: the single temperature that minimizes log-loss over the accumulated pre-match
    /// forecasts and their realized outcomes, then applied to the remaining forecasts.
    fn refit_calibration(&mut self) {
        if self.calib_pairs.len() < MIN_CALIB_SAMPLES {
            return;
        }
        let t = fit_temperature(&self.calib_pairs);
        if (t - self.calib_temperature).abs() > 1e-6 {
            self.calib_temperature = t;
            tracing::info!(
                temperature = t,
                samples = self.calib_pairs.len(),
                "recalibrated remaining forecasts against tournament results"
            );
        }
    }

    /// Recalibrate the aggregate context gain once enough matches have finished: a strongly shrunk
    /// 1-D ridge of the observed margin residual on the context's predicted contribution, clamped
    /// so a noisy tournament cannot swing it wildly.
    fn refit_context_gain(&mut self) {
        if self.context_calib.len() < CONTEXT_CALIB_MIN {
            return;
        }
        let g = fit_gain_toward_one(&self.context_calib, CONTEXT_PRIOR_PRECISION).clamp(0.5, 1.5);
        if (g - self.context_gain).abs() > 1e-6 {
            self.context_gain = g;
            tracing::info!(
                context_gain = g,
                samples = self.context_calib.len(),
                "recalibrated context effects against tournament results"
            );
        }
    }

    /// Sample every currently-live match's win probabilities into its timeline, one point per new
    /// minute (the fan drama graph). Cheap: only in-play matches are touched.
    fn sample_live_history(&mut self) {
        let samples: Vec<(MatchId, WinProbSample)> = self
            .tournament
            .matches
            .iter()
            .filter(|m| {
                let status = self.live.get(&m.id).map(|l| l.status).unwrap_or(m.status);
                matches!(status, MatchStatus::Live { .. })
            })
            .map(|m| {
                let p = self.predict_match(m);
                (
                    m.id,
                    WinProbSample {
                        minute: p.minute,
                        probabilities: p.probabilities,
                    },
                )
            })
            .collect();
        for (id, sample) in samples {
            let hist = self.live_history.entry(id).or_default();
            // One sample per minute (ticks can repeat a minute); overwrite the latest within a
            // minute so a goal mid-minute is reflected.
            if hist.last().map(|s| s.minute) == Some(sample.minute) {
                *hist.last_mut().unwrap() = sample;
            } else {
                hist.push(sample);
                if hist.len() > MAX_LIVE_SAMPLES {
                    hist.remove(0);
                }
            }
        }
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
                // Pre-match: blend the (lineup-adjusted) Dixon-Coles grid, Elo, and the
                // state-space rating, plus the bookmaker market as a fourth member when odds
                // are present. Member order matches `fit_baseline`: [DC, Elo, StateSpace,
                // Market]; `blend` renormalizes its weights, so a missing market is a clean
                // fallback to three members.
                let grid = self
                    .model
                    .score_grid_adjusted(m.home, m.away, NEUTRAL, home_adj, away_adj);
                // Raw ensemble blend, then the live calibration correction (identity until enough
                // tournament results have accumulated to fit a temperature).
                let blended =
                    apply_temperature(self.blend_pre_match(m, &grid), self.calib_temperature);
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
            history: Vec::new(),
        }
    }

    fn build_snapshot(&self, provider_name: &str) -> Snapshot {
        let matches = self
            .tournament
            .matches
            .iter()
            .map(|m| {
                let mut p = self.predict_match(m);
                // Attach the live win-probability timeline for the drama graph (live matches only).
                if matches!(p.status, MatchStatus::Live { .. }) {
                    if let Some(h) = self.live_history.get(&m.id) {
                        p.history = h.clone();
                    }
                }
                p
            })
            .collect();

        let mut ratings: Vec<RatingEntry> = self
            .tournament
            .teams
            .iter()
            .map(|t| RatingEntry {
                team: t.id,
                name: t.name.clone(),
                code: t.code.clone(),
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
            adaptive: AdaptiveState {
                results_seen: self.calib_pairs.len(),
                calibration_temperature: self.calib_temperature,
                context_gain: self.context_gain,
            },
            shocks: self.biggest_shocks(8),
            report_card: self.report_card(),
            bt_champions: self.bt_champions(),
        }
    }

    /// The finished matches the model was most surprised by: the pre-match favourite failed to win.
    /// Ranked by shock (`1 - P(actual outcome)` from the pre-match forecast), top `n`.
    fn biggest_shocks(&self, n: usize) -> Vec<Upset> {
        let mut shocks: Vec<Upset> = self
            .tournament
            .matches
            .iter()
            .filter(|m| m.is_finished())
            .filter_map(|m| {
                let p = self.pre_match_forecast.get(&m.id)?;
                let outcome = m.score.outcome();
                let fav_is_home = p.home_win >= p.away_win;
                let fav_won = matches!(
                    (fav_is_home, outcome),
                    (true, Outcome::HomeWin) | (false, Outcome::AwayWin)
                );
                if fav_won {
                    return None; // the favourite came through - not an upset
                }
                Some(Upset {
                    match_id: m.id,
                    home_name: self.name_of(m.home),
                    away_name: self.name_of(m.away),
                    stage: m.stage,
                    score: m.score,
                    favorite_name: self.name_of(if fav_is_home { m.home } else { m.away }),
                    favorite_prob: if fav_is_home { p.home_win } else { p.away_win },
                    shock: 1.0 - p.of(outcome),
                })
            })
            .collect();
        shocks.sort_by(|a, b| {
            b.shock
                .partial_cmp(&a.shock)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        shocks.truncate(n);
        shocks
    }

    /// Score the model's own pre-match calls on the finished matches: accuracy, Brier / log-loss
    /// (against the uniform baseline), and its most confident correct calls and misses. Almost free
    /// - the pre-match forecasts and results are already on hand.
    fn report_card(&self) -> ReportCard {
        let mut pairs: Vec<(Probabilities, Outcome)> = Vec::new();
        let mut calls: Vec<Call> = Vec::new();
        for m in self.tournament.matches.iter().filter(|m| m.is_finished()) {
            let Some(&p) = self.pre_match_forecast.get(&m.id) else {
                continue;
            };
            let actual = m.score.outcome();
            let predicted = p.most_likely();
            pairs.push((p, actual));
            calls.push(Call {
                match_id: m.id,
                home_name: self.name_of(m.home),
                away_name: self.name_of(m.away),
                score: m.score,
                confidence: p.of(predicted),
                correct: predicted == actual,
            });
        }
        let report = score(&pairs);
        let scored = pairs.len();
        let winners_called = calls.iter().filter(|c| c.correct).count();
        let by_confidence = |a: &Call, b: &Call| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        let mut best: Vec<Call> = calls.iter().filter(|c| c.correct).cloned().collect();
        best.sort_by(by_confidence);
        best.truncate(3);
        let mut worst: Vec<Call> = calls.iter().filter(|c| !c.correct).cloned().collect();
        worst.sort_by(by_confidence);
        worst.truncate(3);
        let head_to_head = vec![
            self.score_model("Dixon-Coles ensemble", &self.pre_match_forecast),
            self.score_model("Bradley-Terry", &self.bt_pre_match_forecast),
        ];
        ReportCard {
            scored,
            winners_called,
            accuracy: report.accuracy,
            brier: report.brier,
            log_loss: report.log_loss,
            baseline_brier: CalibrationReport::uniform_baseline(scored.max(1)).brier,
            best_calls: best,
            worst_calls: worst,
            head_to_head,
        }
    }

    /// Score one forecaster's pre-match calls against the finished matches (accuracy, Brier,
    /// log-loss), for the head-to-head between the two models.
    fn score_model(&self, name: &str, forecasts: &HashMap<MatchId, Probabilities>) -> ModelScore {
        let pairs: Vec<(Probabilities, Outcome)> = self
            .tournament
            .matches
            .iter()
            .filter(|m| m.is_finished())
            .filter_map(|m| forecasts.get(&m.id).map(|&p| (p, m.score.outcome())))
            .collect();
        let r = score(&pairs);
        ModelScore {
            model: name.to_string(),
            scored: pairs.len(),
            accuracy: r.accuracy,
            brier: r.brier,
            log_loss: r.log_loss,
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

    fn fresh_state_bt() -> EngineState {
        let deps = EngineDeps::new(Arc::new(SimProvider::new()))
            .with_bradley_terry(data::fit_bradley_terry(7));
        EngineState::new(data::world_cup_2026(), deps)
    }

    fn state_with_lr(lr: f64) -> EngineState {
        let deps = EngineDeps::new(Arc::new(SimProvider::new()))
            .with_model(data::fit_baseline_model(7))
            .with_elo_seeds(data::team_strengths())
            .with_tournament_lr(lr);
        EngineState::new(data::world_cup_2026(), deps)
    }

    #[test]
    fn in_tournament_learning_lifts_an_overperformer() {
        // Pick a team and finish two of its group matches as 5-0 thrashings, then compare a
        // learning engine (lr > 0) against an identical no-learning one (lr = 0).
        let pick = |state: &EngineState| -> (TeamId, Vec<MatchId>) {
            let team = state.tournament.matches[0].home;
            let ids: Vec<MatchId> = state
                .tournament
                .matches
                .iter()
                .filter(|m| m.home == team)
                .take(2)
                .map(|m| m.id)
                .collect();
            (team, ids)
        };

        let run = |lr: f64| -> f64 {
            let mut state = state_with_lr(lr);
            let metrics = Metrics::default();
            let (team, ids) = pick(&state);
            for id in ids {
                state.apply_event(
                    &MatchEvent::new(
                        id,
                        90,
                        EventKind::FullTime {
                            score: Scoreline::new(5, 0),
                        },
                    ),
                    &metrics,
                );
            }
            state.recompute_forecast();
            state
                .last_forecast
                .teams
                .iter()
                .find(|t| t.team == team)
                .unwrap()
                .p_champion
        };

        let learned = run(0.08);
        let frozen = run(0.0);
        assert!(
            learned > frozen,
            "two 5-0 wins should raise the team's title odds via learning ({frozen:.4} -> {learned:.4})"
        );
    }

    #[test]
    fn group_completion_materializes_the_knockout_bracket() {
        let mut state = fresh_state();
        let metrics = Metrics::default();

        // Before: only the 72 group fixtures, no knockout matches.
        assert_eq!(state.tournament.matches.len(), 72);
        assert!(!state
            .tournament
            .matches
            .iter()
            .any(|m| m.stage.is_knockout()));

        // Finish every group match (the stronger seed, the lower id, wins).
        let group: Vec<(MatchId, TeamId, TeamId)> = state
            .tournament
            .matches
            .iter()
            .map(|m| (m.id, m.home, m.away))
            .collect();
        for (id, home, away) in group {
            let score = if home.0 < away.0 {
                Scoreline::new(2, 0)
            } else {
                Scoreline::new(0, 2)
            };
            state.apply_event(
                &MatchEvent::new(id, 90, EventKind::FullTime { score }),
                &metrics,
            );
        }

        // After the last group result the engine materialized the real Round of 32.
        let r32 = state
            .tournament
            .matches
            .iter()
            .filter(|m| m.stage == oracle_domain::Stage::RoundOf32)
            .count();
        assert_eq!(r32, 16, "16 R32 fixtures materialized");
        assert_eq!(state.tournament.matches.len(), 88);

        // The new fixtures are indexed, so live events addressed to them resolve.
        for m in state
            .tournament
            .matches
            .iter()
            .filter(|m| m.stage.is_knockout())
        {
            assert!(
                state.match_index.contains_key(&m.id),
                "knockout fixture {} should be indexed",
                m.id
            );
        }

        // The forecast now plays the real bracket; champion mass stays coherent.
        state.recompute_forecast();
        let total: f64 = state.last_forecast.teams.iter().map(|t| t.p_champion).sum();
        assert!((total - 1.0).abs() < 0.05, "champion mass = {total}");
    }

    #[test]
    fn knockout_rounds_materialize_round_by_round_and_drive_bt_champions() {
        let mut state = fresh_state_bt();
        let metrics = Metrics::default();

        // Before the group stage finishes there is no bracket, so the second model publishes no
        // live champion odds.
        assert!(state.bt_champions().is_empty());

        // Finish a set of ties decisively: the lower-id (stronger seed) side wins, matching the
        // engine's level-tie-goes-to-home materialization rule.
        let finish = |state: &mut EngineState, ties: &[(MatchId, TeamId, TeamId)]| {
            for &(id, home, away) in ties {
                let score = if home.0 < away.0 {
                    Scoreline::new(2, 0)
                } else {
                    Scoreline::new(0, 2)
                };
                state.apply_event(
                    &MatchEvent::new(id, 90, EventKind::FullTime { score }),
                    &metrics,
                );
            }
        };
        let ties_of = |state: &EngineState, stage: Stage| -> Vec<(MatchId, TeamId, TeamId)> {
            state
                .tournament
                .matches
                .iter()
                .filter(|m| m.stage == stage)
                .map(|m| (m.id, m.home, m.away))
                .collect()
        };

        let groups: Vec<(MatchId, TeamId, TeamId)> = state
            .tournament
            .matches
            .iter()
            .filter(|m| matches!(m.stage, Stage::Group(_)))
            .map(|m| (m.id, m.home, m.away))
            .collect();
        finish(&mut state, &groups);

        // The Round of 32 is materialized, so the live champion odds now span all 32 bracket teams
        // and are a proper distribution.
        let r32 = ties_of(&state, Stage::RoundOf32);
        assert_eq!(r32.len(), 16, "16 Round-of-32 fixtures materialized");
        let champs = state.bt_champions();
        assert_eq!(champs.len(), 32, "all 32 bracket teams carry title odds");
        let total: f64 = champs.iter().map(|c| c.champion).sum();
        assert!((total - 1.0).abs() < 1e-6, "champion mass = {total}");

        // Finish the whole Round of 32; note who goes out.
        let eliminated: Vec<String> = r32
            .iter()
            .map(|&(_, h, a)| state.name_of(if h.0 < a.0 { a } else { h }))
            .collect();
        finish(&mut state, &r32);

        // The Round of 16 is now materialized off the winners, and the live odds condition on it:
        // only the 16 survivors carry title probability, and the eliminated sides are gone.
        let r16 = ties_of(&state, Stage::RoundOf16);
        assert_eq!(r16.len(), 8, "8 Round-of-16 fixtures materialized");
        let champs = state.bt_champions();
        assert_eq!(champs.len(), 16, "only the 16 survivors carry title odds");
        let total: f64 = champs.iter().map(|c| c.champion).sum();
        assert!((total - 1.0).abs() < 1e-6, "champion mass = {total}");
        for name in &eliminated {
            assert!(
                !champs.iter().any(|c| &c.team == name),
                "eliminated team {name} should carry no title odds"
            );
        }
    }

    fn state_with_threshold(threshold: u8) -> EngineState {
        let deps = EngineDeps::new(Arc::new(SimProvider::new()))
            .with_model(data::fit_baseline_model(7))
            .with_elo_seeds(data::team_strengths())
            .with_suspension_threshold(threshold);
        EngineState::new(data::world_cup_2026(), deps)
    }

    #[test]
    fn two_yellows_suspend_a_starter_and_weaken_the_next_match() {
        // A team's probability of winning its next match should drop once a starter is
        // suspended (threshold 2) versus suspension tracking off (threshold 0).
        let win_prob_next = |threshold: u8| -> f64 {
            let mut state = state_with_threshold(threshold);
            let metrics = Metrics::default();
            let team = state.tournament.matches[0].home;
            let starter = oracle_ingest::data::starting_lineup(team, false)[1].clone();
            let carrier = state.tournament.matches[0].id;
            for _ in 0..2 {
                state.apply_event(
                    &MatchEvent::new(
                        carrier,
                        30,
                        EventKind::YellowCard {
                            team,
                            player: Some(starter.clone()),
                        },
                    ),
                    &metrics,
                );
            }
            let next = state.next_unplayed_match(team).expect("a next match");
            let m = state
                .tournament
                .matches
                .iter()
                .find(|m| m.id == next)
                .unwrap()
                .clone();
            let p = state.predict_match(&m).probabilities;
            if m.home == team {
                p.home_win
            } else {
                p.away_win
            }
        };
        let suspended = win_prob_next(2);
        let available = win_prob_next(0);
        assert!(
            suspended < available,
            "a suspended starter should lower the team's next-match win probability ({available:.3} -> {suspended:.3})"
        );
    }

    #[test]
    fn one_yellow_or_unnamed_card_creates_no_suspension() {
        let mut state = state_with_threshold(2);
        let metrics = Metrics::default();
        let team = state.tournament.matches[0].home;
        let carrier = state.tournament.matches[0].id;
        let starter = oracle_ingest::data::starting_lineup(team, false)[1].clone();

        // One yellow: under threshold, no suspension.
        state.apply_event(
            &MatchEvent::new(
                carrier,
                10,
                EventKind::YellowCard {
                    team,
                    player: Some(starter),
                },
            ),
            &metrics,
        );
        // Two unnamed yellows: untrackable, no suspension.
        for _ in 0..2 {
            state.apply_event(
                &MatchEvent::new(carrier, 20, EventKind::YellowCard { team, player: None }),
                &metrics,
            );
        }
        assert!(
            state.suspended.is_empty(),
            "no suspension should have been created"
        );
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

    #[test]
    fn event_log_replay_rebuilds_state() {
        let path = std::env::temp_dir().join("oracle_engine_recovery_test.jsonl");
        let _ = std::fs::remove_file(&path);
        let metrics = Metrics::default();

        // State A: apply a full match and record every event to the log.
        let mut a = fresh_state();
        let m = a.tournament.matches[0].clone();
        let events = vec![
            MatchEvent::new(m.id, 0, EventKind::KickOff),
            MatchEvent::new(
                m.id,
                30,
                EventKind::Goal {
                    team: m.home,
                    scorer: None,
                },
            ),
            MatchEvent::new(
                m.id,
                90,
                EventKind::FullTime {
                    score: Scoreline::new(2, 1),
                },
            ),
        ];
        let log = EventLog::create(&path).unwrap();
        for e in &events {
            log.append(e).unwrap();
            a.apply_event(e, &metrics);
        }
        drop(log);

        // State B: rebuild purely by replaying the recorded log.
        let mut b = fresh_state();
        for e in EventLog::read(&path).unwrap() {
            b.apply_event(&e, &metrics);
        }

        // The recovered finished result and rating change must match the original.
        assert_eq!(b.tournament.matches[0].score, Scoreline::new(2, 1));
        assert_eq!(
            a.tournament.matches[0].status,
            b.tournament.matches[0].status
        );
        assert!((a.ratings.rating(m.home) - b.ratings.rating(m.home)).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn state_space_rating_learns_from_results() {
        let mut state = fresh_state();
        let metrics = Metrics::default();
        let team = state.tournament.matches[0].home;
        let mean_before = state.state_space.mean(team);
        let sd_before = state.state_space.stddev(team);

        // Win two of the team's matches comfortably.
        let ids: Vec<MatchId> = state
            .tournament
            .matches
            .iter()
            .filter(|m| m.home == team)
            .take(2)
            .map(|m| m.id)
            .collect();
        for id in ids {
            state.apply_event(
                &MatchEvent::new(
                    id,
                    90,
                    EventKind::FullTime {
                        score: Scoreline::new(3, 0),
                    },
                ),
                &metrics,
            );
        }

        assert!(
            state.state_space.mean(team) > mean_before,
            "wins should raise the Kalman mean"
        );
        assert!(
            state.state_space.stddev(team) < sd_before,
            "observing matches should reduce the Kalman uncertainty"
        );
    }

    #[test]
    fn live_recalibration_fits_and_applies_a_temperature() {
        // Online learning off (to isolate calibration). Finish enough matches so each one's
        // favoured outcome always wins: the pre-match forecasts are then systematically
        // under-confident, so the fitted temperature sharpens them (> 1).
        let mut state = state_with_lr(0.0);
        let metrics = Metrics::default();
        assert!(
            (state.calib_temperature - 1.0).abs() < 1e-9,
            "starts at the identity"
        );

        let ids: Vec<MatchId> = state
            .tournament
            .matches
            .iter()
            .take(30)
            .map(|m| m.id)
            .collect();
        for id in ids {
            let m = state
                .tournament
                .matches
                .iter()
                .find(|x| x.id == id)
                .unwrap()
                .clone();
            let score = match state.pre_match_probs(&m).most_likely() {
                Outcome::HomeWin => Scoreline::new(2, 0),
                Outcome::Draw => Scoreline::new(1, 1),
                Outcome::AwayWin => Scoreline::new(0, 2),
            };
            state.apply_event(
                &MatchEvent::new(id, 90, EventKind::FullTime { score }),
                &metrics,
            );
        }

        assert_eq!(state.calib_pairs.len(), 30);
        assert!(
            state.calib_temperature > 1.0,
            "an under-confident model should be sharpened, got {}",
            state.calib_temperature
        );

        // A scheduled match's shown forecast is exactly the raw blend after the fitted temperature.
        let sched = state
            .tournament
            .matches
            .iter()
            .find(|m| !m.is_finished())
            .unwrap()
            .clone();
        let expected = apply_temperature(state.pre_match_probs(&sched), state.calib_temperature);
        let shown = state.predict_match(&sched).probabilities;
        assert!((shown.home_win - expected.home_win).abs() < 1e-9);
        assert!((shown.draw - expected.draw).abs() < 1e-9);
        assert!((shown.away_win - expected.away_win).abs() < 1e-9);
    }

    #[test]
    fn context_recalibration_accumulates_and_scales_only_the_context_part() {
        // Finish enough matches to move the gain off its prior, then confirm a scheduled match's
        // adjustment is exactly context * gain + style (no lineup/suspension in play here).
        let mut state = state_with_lr(0.0);
        let metrics = Metrics::default();
        let ids: Vec<MatchId> = state
            .tournament
            .matches
            .iter()
            .take(25)
            .map(|m| m.id)
            .collect();
        for id in ids {
            state.apply_event(
                &MatchEvent::new(
                    id,
                    90,
                    EventKind::FullTime {
                        score: Scoreline::new(2, 0),
                    },
                ),
                &metrics,
            );
        }
        assert_eq!(state.context_calib.len(), 25);
        assert!(
            (0.5..=1.5).contains(&state.context_gain),
            "gain stays in the clamped range: {}",
            state.context_gain
        );

        let sched = state
            .tournament
            .matches
            .iter()
            .find(|m| !m.is_finished())
            .unwrap()
            .clone();
        let ctx = state
            .context_adj
            .get(&sched.id)
            .copied()
            .unwrap_or_default();
        let sty = state.style_adj.get(&sched.id).copied().unwrap_or_default();
        let g = state.context_gain;
        let adj = state.match_adjustments(sched.id);
        assert!((adj.0 .0 - (g * ctx.0 .0 + sty.0 .0)).abs() < 1e-9);
        assert!((adj.1 .0 - (g * ctx.1 .0 + sty.1 .0)).abs() < 1e-9);
    }

    #[test]
    fn knockout_ties_lean_harder_on_the_market() {
        // Finish the group stage so the real knockout bracket is materialized.
        let mut state = state_with_lr(0.0);
        let metrics = Metrics::default();
        let groups: Vec<(MatchId, TeamId, TeamId)> = state
            .tournament
            .matches
            .iter()
            .map(|m| (m.id, m.home, m.away))
            .collect();
        for (id, home, away) in groups {
            let score = if home.0 < away.0 {
                Scoreline::new(2, 0)
            } else {
                Scoreline::new(0, 2)
            };
            state.apply_event(
                &MatchEvent::new(id, 90, EventKind::FullTime { score }),
                &metrics,
            );
        }
        let ko = state
            .tournament
            .matches
            .iter()
            .find(|m| m.stage.is_knockout())
            .unwrap()
            .clone();
        // Strongly away-favouring bookmaker odds on the tie.
        state.apply_event(
            &MatchEvent::new(
                ko.id,
                0,
                EventKind::Odds {
                    home: 8.0,
                    draw: 5.0,
                    away: 1.3,
                },
            ),
            &metrics,
        );

        let shown = state.predict_match(&ko).probabilities;

        // Reference: the same members and temperature, but WITHOUT the knockout market boost.
        let (home_adj, away_adj) = state.match_adjustments(ko.id);
        let grid = state
            .model
            .score_grid_adjusted(ko.home, ko.away, true, home_adj, away_adj);
        let members = [
            grid.outcome_probabilities(),
            state.ratings.win_probabilities(ko.home, ko.away, true),
            state.state_space.win_probabilities(ko.home, ko.away, true),
            implied_probabilities(8.0, 5.0, 1.3),
        ];
        let plain = apply_temperature(state.ensemble.blend(&members), state.calib_temperature);

        assert!(
            shown.away_win > plain.away_win + 1e-6,
            "the knockout market boost should pull toward the away-favouring line ({} vs {})",
            shown.away_win,
            plain.away_win
        );
    }

    #[test]
    fn a_favoured_side_losing_registers_as_a_shock() {
        let mut state = state_with_lr(0.0);
        let metrics = Metrics::default();
        // A match where the home side is the stronger (lower-id) team, so it is the favourite.
        let m = state
            .tournament
            .matches
            .iter()
            .find(|m| m.home.0 < m.away.0)
            .unwrap()
            .clone();
        // The underdog away side wins 2-0: the favourite failed to win.
        state.apply_event(
            &MatchEvent::new(
                m.id,
                90,
                EventKind::FullTime {
                    score: Scoreline::new(0, 2),
                },
            ),
            &metrics,
        );
        let snap = state.build_snapshot("test");
        let shock = snap
            .shocks
            .iter()
            .find(|s| s.match_id == m.id)
            .expect("a favoured side losing should be flagged as a shock");
        assert!(shock.shock > 0.0 && shock.favorite_prob > 0.0);
        assert_eq!(shock.favorite_name, state.name_of(m.home));
    }

    #[test]
    fn live_history_records_the_win_probability_timeline() {
        let mut state = fresh_state();
        let metrics = Metrics::default();
        let m = state.tournament.matches[0].clone();
        for (minute, kind) in [
            (1u16, EventKind::KickOff),
            (10, EventKind::Tick),
            (
                25,
                EventKind::Goal {
                    team: m.home,
                    scorer: None,
                },
            ),
            (40, EventKind::Tick),
        ] {
            state.apply_event(&MatchEvent::new(m.id, minute, kind), &metrics);
            state.sample_live_history();
        }
        let hist = state
            .live_history
            .get(&m.id)
            .expect("a live match should accumulate a timeline");
        assert!(hist.len() >= 2, "timeline should grow: {}", hist.len());
        for s in hist {
            assert!((s.probabilities.sum() - 1.0).abs() < 1e-6);
        }
        // The snapshot exposes the timeline on the live match.
        let snap = state.build_snapshot("test");
        let mp = snap.matches.iter().find(|p| p.match_id == m.id).unwrap();
        assert_eq!(mp.history.len(), hist.len());
    }

    #[test]
    fn report_card_scores_the_models_own_calls() {
        let mut state = state_with_lr(0.0);
        let metrics = Metrics::default();
        // Finish 10 matches with the stronger (lower-id) side winning - the model's favourite - so
        // most calls come off and it comfortably beats the uniform baseline.
        let group: Vec<(MatchId, TeamId, TeamId)> = state
            .tournament
            .matches
            .iter()
            .take(10)
            .map(|m| (m.id, m.home, m.away))
            .collect();
        for (id, home, away) in group {
            let score = if home.0 < away.0 {
                Scoreline::new(2, 0)
            } else {
                Scoreline::new(0, 2)
            };
            state.apply_event(
                &MatchEvent::new(id, 90, EventKind::FullTime { score }),
                &metrics,
            );
        }
        let rc = state.build_snapshot("test").report_card;
        assert_eq!(rc.scored, 10);
        assert!((0.0..=1.0).contains(&rc.accuracy));
        assert!(
            rc.brier < rc.baseline_brier,
            "the model should beat the uniform baseline ({} vs {})",
            rc.brier,
            rc.baseline_brier
        );
        assert!(rc.best_calls.iter().all(|c| c.correct));
        assert!(rc.worst_calls.iter().all(|c| !c.correct));
    }

    #[test]
    fn head_to_head_scores_both_models() {
        let mut state = state_with_lr(0.05);
        let metrics = Metrics::default();
        let group: Vec<(MatchId, TeamId, TeamId)> = state
            .tournament
            .matches
            .iter()
            .take(12)
            .map(|m| (m.id, m.home, m.away))
            .collect();
        for (id, home, away) in group {
            let score = if home.0 < away.0 {
                Scoreline::new(2, 0)
            } else {
                Scoreline::new(0, 2)
            };
            state.apply_event(
                &MatchEvent::new(id, 90, EventKind::FullTime { score }),
                &metrics,
            );
        }
        // The second model recorded a leak-free pre-match call for every finished match...
        assert_eq!(state.bt_pre_match_forecast.len(), 12);
        // ...and the report card scores both forecasters on the same results.
        let rc = state.build_snapshot("test").report_card;
        assert_eq!(rc.head_to_head.len(), 2);
        let names: Vec<&str> = rc.head_to_head.iter().map(|m| m.model.as_str()).collect();
        assert!(names.contains(&"Dixon-Coles ensemble") && names.contains(&"Bradley-Terry"));
        for m in &rc.head_to_head {
            assert_eq!(m.scored, 12);
            assert!((0.0..=1.0).contains(&m.accuracy) && m.brier >= 0.0);
        }
    }
}
