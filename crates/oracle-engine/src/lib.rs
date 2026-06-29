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
mod snapshot;

pub use event_log::EventLog;
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
            tournament_lr: 0.03,
            suspension_threshold: 2,
            state_space: StateSpaceRatings::with_defaults(),
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
    /// Per-match adjustments precomputed once: venue/crowd/travel/heat context plus the style
    /// matchup (`data::matchup_adjustments`).
    venue_adj: HashMap<MatchId, VenueAdj>,
    /// Per-team knockout factors precomputed once: penalty-shootout skill and knockout pedigree.
    shootout_rating: HashMap<TeamId, f64>,
    knockout_pedigree: HashMap<TeamId, f64>,
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
        // Match context (venue/crowd/travel/heat) plus the style matchup is static for the
        // tournament, so precompute it once.
        let venue_adj = oracle_ingest::data::matchup_adjustments(&tournament);
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
            venue_adj,
            shootout_rating,
            knockout_pedigree,
            source_healthy: true,
            last_update: chrono::Utc::now(),
        }
    }

    /// Combined venue + lineup `(home_adj, away_adj)` deltas for a match (summed in log
    /// space), feeding the adjusted goal model and the Monte-Carlo.
    fn match_adjustments(&self, id: MatchId) -> VenueAdj {
        let ((vh_a, vh_d), (va_a, va_d)) = self.venue_adj.get(&id).copied().unwrap_or_default();
        // Player availability: announced lineup if known, else a suspension-derived estimate.
        let ((lh_a, lh_d), (la_a, la_d)) = self.availability_adj(id);
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
            // World Cup matches are at neutral venues. Both ratings learn from the result:
            // Elo, and the Dixon-Coles goal model (online, so the forecast tracks tournament
            // form instead of staying frozen at the offline fit).
            self.ratings.record(home, away, score, true);
            self.model
                .update_with_result(home, away, score, true, self.tournament_lr);
            // The state-space rating learns too (age 0 = just now), updating both its mean
            // and its per-team variance, which feeds the forecast's parameter uncertainty.
            self.state_space.observe(home, away, score, 0.0, true);
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
        let ko = oracle_ingest::data::materialize_knockout(&self.tournament);
        if ko.is_empty() {
            return;
        }
        tracing::info!(
            fixtures = ko.len(),
            "group stage complete - materialized the knockout bracket"
        );
        let start = self.tournament.matches.len();
        self.tournament.matches.extend(ko);
        for i in start..self.tournament.matches.len() {
            self.match_index.insert(self.tournament.matches[i].id, i);
        }
        // Match context + style matchup now cover the knockout fixtures too.
        self.venue_adj = oracle_ingest::data::matchup_adjustments(&self.tournament);
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
        // Venue context applies to every fixture; suspensions add a player-availability
        // penalty to scheduled matches that have no announced lineup yet (lineup/live
        // matches already carry their delta in `live`, so we skip them to avoid double count).
        let mut venue = self.venue_adj.clone();
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
                let dc = grid.outcome_probabilities();
                let elo = self.ratings.win_probabilities(m.home, m.away, NEUTRAL);
                let kalman = self.state_space.win_probabilities(m.home, m.away, NEUTRAL);
                let mut members = vec![dc, elo, kalman];
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
}
