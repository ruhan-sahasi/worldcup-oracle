//! # oracle-api
//!
//! The transport layer: an [`axum`] application exposing the engine and the on-demand model over
//! HTTP and WebSocket. It is a thin shell - the live handlers read an immutable
//! [`Snapshot`](oracle_engine::Snapshot) from the engine (lock-free) or relay the broadcast
//! stream; the `/api/*` query handlers just forward an [`Explorer`](oracle_engine::Explorer)
//! result as JSON. No prediction logic lives here.
//!
//! ## Endpoints
//! | Method | Path | Purpose |
//! |--------|------|---------|
//! | GET | `/` | live tournament dashboard |
//! | GET | `/explore` | interactive model explorer |
//! | GET | `/team` | fan "your team" hub (page) |
//! | GET | `/card` | shareable prediction card (team or matchup) |
//! | GET | `/api/team?q=` | one team's journey odds, rank, and next match |
//! | GET | `/health` | liveness probe |
//! | GET | `/teams` | current Elo ratings |
//! | GET | `/matches` | all match predictions (compact) |
//! | GET | `/predict/match/{id}` | one tournament fixture, full exact-score grid |
//! | GET | `/predict/tournament` | live champion-odds table |
//! | GET | `/upsets` | fan upset radar: upcoming shock-ripe matches + biggest shocks so far |
//! | GET | `/report` | the model's self-scored report card on its own pre-match calls |
//! | GET | `/bt/champions` | second model's live champion odds over the current knockout bracket |
//! | GET | `/consensus` | consensus title forecast blending both models + their live divergence (JSD) |
//! | GET | `/calibration` | the model's live reliability curve + expected calibration error |
//! | GET | `/power` | prior-free Massey power ranking over this tournament's results (offense/defense) |
//! | GET | `/form` | biggest over- and under-performers versus their pre-tournament seeding |
//! | GET | `/road` | each contender's road to the final: expected strength of remaining opponents |
//! | GET | `/bracket` | the model's single most likely completion of the knockout bracket |
//! | GET | `/leverage` | current-round ties ranked by how much each would reshape the title race |
//! | GET | `/openness` | how open the title race is: champion-odds entropy + effective contenders |
//! | GET | `/history` | championship-odds time series for the current top contenders |
//! | GET | `/momentum` | biggest recent movers in the title race (rising and falling odds) |
//! | GET | `/lead-changes` | when the title favourite changed hands over the tournament |
//! | GET | `/api/predict?home=&away=` | on-demand matchup forecast (any two teams) |
//! | GET | `/api/explain?home=&away=` | factor attribution: why the model favours a side |
//! | GET | `/api/bt?home=&away=` | second model (Bradley-Terry-Davidson) win/draw/loss |
//! | GET | `/api/bt/champions` | second model's champion odds (bracket DP) |
//! | GET | `/api/posterior?home=&away=` | HMC posterior credible intervals for a matchup |
//! | GET | `/api/simulate?iters=&seed=` | custom Monte-Carlo champion-odds run |
//! | GET | `/api/sensitivity?iters=&seed=` | per-signal ablation: how much each signal moves the title |
//! | GET | `/api/kingmaker?q=` | rooting interest: which group results swing a team's title odds |
//! | GET | `/api/collision?home=&away=` | probability two teams meet in the knockouts, by round |
//! | GET | `/api/ratings` | team ratings + confederation strength levels |
//! | GET | `/metrics` | Prometheus metrics |
//! | GET | `/live` | WebSocket: pushes a compact live view on every update |
#![forbid(unsafe_code)]

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        FromRef, Path, Query, State,
    },
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use oracle_domain::{MatchId, MatchStatus, Probabilities, Scoreline, Stage, TeamId};
use oracle_engine::query::{
    BtChampions, BtMatchup, CollisionForecast, Explanation, KingmakerReport, MatchupForecast,
    PosteriorForecast, RatingsView, SensitivityForecast, SimForecast,
};
use oracle_engine::{AdaptiveState, Engine, Explorer, ReportCard, Snapshot, Upset, WinProbSample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// A lazily-fit explorer: empty until the background fit completes (the `/api/*` endpoints return
/// 503 until then), so the server starts serving the live dashboard and engine endpoints at once.
type ExplorerSlot = Arc<OnceLock<Explorer>>;

/// Shared application state: the live [`Engine`] and the (lazily-fit) [`Explorer`] for on-demand
/// queries. Handlers extract whichever they need via [`FromRef`].
#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
    explorer: ExplorerSlot,
}

impl FromRef<AppState> for Arc<Engine> {
    fn from_ref(state: &AppState) -> Self {
        state.engine.clone()
    }
}

impl FromRef<AppState> for ExplorerSlot {
    fn from_ref(state: &AppState) -> Self {
        state.explorer.clone()
    }
}

/// Create an explorer handle and fit its baseline in a background blocking task, so the server can
/// start serving immediately. The `/api/*` query endpoints return 503 until the fit (a few
/// seconds) completes; the live dashboard and engine endpoints are available at once.
pub fn spawn_explorer() -> ExplorerSlot {
    let slot: ExplorerSlot = Arc::new(OnceLock::new());
    let fill = slot.clone();
    tokio::task::spawn_blocking(move || {
        let _ = fill.set(Explorer::new());
        tracing::info!("model explorer ready");
    });
    slot
}

/// Build the application router. The live tournament is served from the `Engine`; the on-demand
/// `/explore` page and `/api/*` query endpoints from the `Explorer`.
pub fn router(engine: Arc<Engine>, explorer: ExplorerSlot) -> Router {
    Router::new()
        .route("/", get(dashboard))
        .route("/explore", get(explorer_page))
        .route("/team", get(team_page))
        .route("/card", get(card_page))
        .route("/api", get(api_info))
        .route("/health", get(health))
        .route("/teams", get(teams))
        .route("/matches", get(matches))
        .route("/predict/match/:id", get(predict_match))
        .route("/predict/tournament", get(predict_tournament))
        .route("/upsets", get(upsets))
        .route("/report", get(report))
        .route("/bt/champions", get(bt_champions))
        .route("/consensus", get(consensus))
        .route("/calibration", get(calibration))
        .route("/power", get(power))
        .route("/form", get(form))
        .route("/road", get(road))
        .route("/bracket", get(bracket))
        .route("/leverage", get(leverage))
        .route("/openness", get(openness))
        .route("/history", get(history))
        .route("/momentum", get(momentum))
        .route("/lead-changes", get(lead_changes))
        .route("/api/team", get(team_hub))
        // On-demand model queries (any matchup, posterior, custom simulation, ratings).
        .route("/api/predict", get(api_predict))
        .route("/api/explain", get(api_explain))
        .route("/api/bt", get(api_bt))
        .route("/api/bt/champions", get(api_bt_champions))
        .route("/api/posterior", get(api_posterior))
        .route("/api/simulate", get(api_simulate))
        .route("/api/sensitivity", get(api_sensitivity))
        .route("/api/kingmaker", get(api_kingmaker))
        .route("/api/collision", get(api_collision))
        .route("/api/ratings", get(api_ratings))
        .route("/metrics", get(metrics))
        .route("/live", get(live_ws))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(AppState { engine, explorer })
}

/// Resolve the address to listen on from the environment, for portable deploys. `$PORT` wins (the
/// platform-as-a-service convention: Render, Heroku, and friends inject it), then an explicit
/// `$ORACLE_ADDR`, then the default `0.0.0.0:8080`. Pass the raw env values in (so it stays pure
/// and testable).
pub fn resolve_listen_addr(
    port: Option<String>,
    oracle_addr: Option<String>,
) -> anyhow::Result<SocketAddr> {
    if let Some(p) = port.filter(|p| !p.trim().is_empty()) {
        return Ok(format!("0.0.0.0:{}", p.trim()).parse()?);
    }
    Ok(oracle_addr
        .unwrap_or_else(|| "0.0.0.0:8080".to_string())
        .parse()?)
}

/// Serve the API until `shutdown` resolves (graceful shutdown).
pub async fn serve<F>(
    engine: Arc<Engine>,
    explorer: ExplorerSlot,
    addr: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "oracle-api listening");
    axum::serve(listener, router(engine, explorer))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

// ----- handlers -----

async fn health() -> &'static str {
    "ok"
}

/// The live dashboard, a self-contained page that consumes the `/live` WebSocket.
async fn dashboard() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn api_info(
    State(engine): State<Arc<Engine>>,
    State(explorer): State<ExplorerSlot>,
) -> Json<serde_json::Value> {
    let snap = engine.snapshot();
    Json(serde_json::json!({
        "service": "worldcup-oracle",
        "tournament": engine.tournament_name(),
        "provider": engine.provider_name(),
        "source_healthy": snap.source_healthy,
        "last_update": snap.last_update.to_rfc3339(),
        // The explorer fits in the background; the /api/* query endpoints 503 until it is ready.
        "explorer_ready": explorer.get().is_some(),
        // The engine's live self-recalibration state (updates as results arrive).
        "adaptive": {
            "results_seen": snap.adaptive.results_seen,
            "calibration_temperature": snap.adaptive.calibration_temperature,
            "context_gain": snap.adaptive.context_gain,
        },
        "endpoints": [
            "/ (live dashboard)", "/explore (model explorer)", "/team (fan team hub)",
            "/card (shareable prediction card)", "/health", "/teams", "/matches",
            "/predict/match/{id}", "/predict/tournament", "/upsets", "/report", "/bt/champions",
            "/consensus", "/calibration", "/power", "/form", "/road", "/bracket", "/leverage",
            "/openness", "/history", "/momentum", "/lead-changes", "/api/team?q=",
            "/api/predict?home=&away=", "/api/explain?home=&away=", "/api/posterior?home=&away=",
            "/api/bt?home=&away=", "/api/bt/champions",
            "/api/simulate?iters=&seed=", "/api/sensitivity?iters=&seed=",
            "/api/kingmaker?q=", "/api/collision?home=&away=", "/api/ratings",
            "/metrics", "/live (websocket)"
        ],
    }))
}

/// The interactive model explorer, a self-contained page that calls the `/api/*` endpoints.
async fn explorer_page() -> Html<&'static str> {
    Html(include_str!("../static/explore.html"))
}

/// Query for an on-demand matchup prediction. `neutral` defaults to true (the World Cup default);
/// supply all three odds to add the market member and anchor the ensemble to it.
#[derive(Deserialize)]
struct MatchupParams {
    home: String,
    away: String,
    neutral: Option<bool>,
    home_odds: Option<f64>,
    draw_odds: Option<f64>,
    away_odds: Option<f64>,
}

/// The fit explorer, or 503 while it is still warming up.
fn ready(explorer: &OnceLock<Explorer>) -> Result<&Explorer, StatusCode> {
    explorer.get().ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

async fn api_predict(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<MatchupParams>,
) -> Result<Json<MatchupForecast>, StatusCode> {
    let ex = ready(&explorer)?;
    let (Some(home), Some(away)) = (ex.resolve(&p.home), ex.resolve(&p.away)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let odds = match (p.home_odds, p.draw_odds, p.away_odds) {
        (Some(h), Some(d), Some(a)) => Some((h, d, a)),
        _ => None,
    };
    Ok(Json(ex.predict(
        home,
        away,
        p.neutral.unwrap_or(true),
        odds,
    )))
}

async fn api_explain(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<MatchupParams>,
) -> Result<Json<Explanation>, StatusCode> {
    let ex = ready(&explorer)?;
    let (Some(home), Some(away)) = (ex.resolve(&p.home), ex.resolve(&p.away)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let odds = match (p.home_odds, p.draw_odds, p.away_odds) {
        (Some(h), Some(d), Some(a)) => Some((h, d, a)),
        _ => None,
    };
    Ok(Json(ex.explain(
        home,
        away,
        p.neutral.unwrap_or(true),
        odds,
    )))
}

/// Second model (Bradley-Terry-Davidson) win/draw/loss for a matchup.
async fn api_bt(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<MatchupParams>,
) -> Result<Json<BtMatchup>, StatusCode> {
    let ex = ready(&explorer)?;
    let (Some(home), Some(away)) = (ex.resolve(&p.home), ex.resolve(&p.away)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    Ok(Json(ex.bt_predict(home, away, p.neutral.unwrap_or(true))))
}

/// Second model's winner prediction: champion odds over its projected knockout bracket.
async fn api_bt_champions(
    State(explorer): State<ExplorerSlot>,
) -> Result<Json<BtChampions>, StatusCode> {
    Ok(Json(ready(&explorer)?.bt_champion_odds()))
}

#[derive(Deserialize)]
struct PosteriorParams {
    home: String,
    away: String,
    neutral: Option<bool>,
    samples: Option<usize>,
}

async fn api_posterior(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<PosteriorParams>,
) -> Result<Json<PosteriorForecast>, StatusCode> {
    let ex = ready(&explorer)?;
    let (Some(home), Some(away)) = (ex.resolve(&p.home), ex.resolve(&p.away)) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let neutral = p.neutral.unwrap_or(true);
    let samples = p.samples.unwrap_or(500);
    // HMC is the slow path; run it off the async runtime so it never stalls other requests.
    tokio::task::spawn_blocking(move || {
        explorer
            .get()
            .unwrap()
            .posterior(home, away, neutral, samples)
    })
    .await
    .map(Json)
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[derive(Deserialize)]
struct SimParams {
    iters: Option<u64>,
    seed: Option<u64>,
}

async fn api_simulate(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<SimParams>,
) -> Result<Json<SimForecast>, StatusCode> {
    if explorer.get().is_none() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let iters = p.iters.unwrap_or(20_000);
    let seed = p.seed.unwrap_or(42);
    tokio::task::spawn_blocking(move || explorer.get().unwrap().simulate(iters, seed))
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn api_ratings(
    State(explorer): State<ExplorerSlot>,
) -> Result<Json<RatingsView>, StatusCode> {
    Ok(Json(ready(&explorer)?.ratings()))
}

#[derive(Deserialize)]
struct KingmakerParams {
    q: String,
    iters: Option<u64>,
    seed: Option<u64>,
}

async fn api_kingmaker(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<KingmakerParams>,
) -> Result<Json<KingmakerReport>, StatusCode> {
    let Some(team) = ready(&explorer)?.resolve(&p.q) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let iters = p.iters.unwrap_or(12_000);
    let seed = p.seed.unwrap_or(42);
    // A baseline plus several conditional simulations; keep it off the async runtime.
    tokio::task::spawn_blocking(move || explorer.get().unwrap().kingmaker(team, iters, seed))
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[derive(Deserialize)]
struct CollisionParams {
    home: String,
    away: String,
    iters: Option<u64>,
    seed: Option<u64>,
}

async fn api_collision(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<CollisionParams>,
) -> Result<Json<CollisionForecast>, StatusCode> {
    let (Some(a), Some(b)) = (
        ready(&explorer)?.resolve(&p.home),
        ready(&explorer)?.resolve(&p.away),
    ) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let iters = p.iters.unwrap_or(20_000);
    let seed = p.seed.unwrap_or(42);
    tokio::task::spawn_blocking(move || explorer.get().unwrap().collision(a, b, iters, seed))
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[derive(Deserialize)]
struct SensitivityParams {
    iters: Option<u64>,
    seed: Option<u64>,
}

async fn api_sensitivity(
    State(explorer): State<ExplorerSlot>,
    Query(p): Query<SensitivityParams>,
) -> Result<Json<SensitivityForecast>, StatusCode> {
    if explorer.get().is_none() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let iters = p.iters.unwrap_or(20_000);
    let seed = p.seed.unwrap_or(42);
    // The ablation runs ten simulations; keep it off the async runtime.
    tokio::task::spawn_blocking(move || explorer.get().unwrap().sensitivity(iters, seed))
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn teams(State(engine): State<Arc<Engine>>) -> Json<Vec<oracle_engine::RatingEntry>> {
    Json(engine.snapshot().ratings.clone())
}

async fn matches(State(engine): State<Arc<Engine>>) -> Json<Vec<MatchSummary>> {
    let snap = engine.snapshot();
    Json(
        snap.matches
            .iter()
            .map(MatchSummary::from_prediction)
            .collect(),
    )
}

async fn predict_match(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<u32>,
) -> Result<Json<oracle_engine::MatchPrediction>, StatusCode> {
    engine
        .snapshot()
        .match_prediction(MatchId(id))
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn predict_tournament(State(engine): State<Arc<Engine>>) -> Json<TournamentView> {
    Json(TournamentView::from_snapshot(&engine.snapshot()))
}

/// An upcoming match with a clear favourite the underdog still has a real chance against.
#[derive(Debug, Clone, Serialize)]
pub struct UpcomingUpset {
    pub match_id: MatchId,
    pub stage: Stage,
    pub status: MatchStatus,
    pub favorite_name: String,
    pub underdog_name: String,
    pub favorite_prob: f64,
    pub underdog_prob: f64,
}

/// The fan-facing upset radar: upcoming matches ripe for a shock, and the biggest shocks so far.
#[derive(Debug, Clone, Serialize)]
pub struct UpsetBoard {
    pub upcoming: Vec<UpcomingUpset>,
    pub shocks: Vec<Upset>,
}

async fn upsets(State(engine): State<Arc<Engine>>) -> Json<UpsetBoard> {
    let snap = engine.snapshot();
    // Upcoming: unfinished matches with a clear favourite but a live underdog, ranked by the
    // underdog's chance (the most "upset-ripe" first).
    let mut upcoming: Vec<UpcomingUpset> = snap
        .matches
        .iter()
        .filter(|m| !matches!(m.status, MatchStatus::Finished))
        .filter_map(|m| {
            let p = m.probabilities;
            let fav_is_home = p.home_win >= p.away_win;
            let (favorite_prob, underdog_prob) = if fav_is_home {
                (p.home_win, p.away_win)
            } else {
                (p.away_win, p.home_win)
            };
            // A clear favourite, but the underdog still has a real shot.
            if favorite_prob < 0.55 || underdog_prob < 0.18 {
                return None;
            }
            Some(UpcomingUpset {
                match_id: m.match_id,
                stage: m.stage,
                status: m.status,
                favorite_name: if fav_is_home {
                    m.home_name.clone()
                } else {
                    m.away_name.clone()
                },
                underdog_name: if fav_is_home {
                    m.away_name.clone()
                } else {
                    m.home_name.clone()
                },
                favorite_prob,
                underdog_prob,
            })
        })
        .collect();
    upcoming.sort_by(|a, b| {
        b.underdog_prob
            .partial_cmp(&a.underdog_prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    upcoming.truncate(8);
    Json(UpsetBoard {
        upcoming,
        shocks: snap.shocks.clone(),
    })
}

/// The model's self-scored report card on its own pre-match calls (from the live snapshot).
async fn report(State(engine): State<Arc<Engine>>) -> Json<ReportCard> {
    Json(engine.snapshot().report_card.clone())
}

/// The second model's (Bradley-Terry) live champion odds over the current knockout bracket, from the
/// running tournament. Conditions on the knockout ties already decided; empty until the Round of 32
/// is materialized. Distinct from `/api/bt/champions`, which is the explorer's static projection.
async fn bt_champions(State(engine): State<Arc<Engine>>) -> Json<Vec<oracle_engine::BtChampion>> {
    Json(engine.snapshot().bt_champions.clone())
}

/// The consensus title forecast blending the two models, with a live Jensen-Shannon divergence
/// measuring how far they disagree. Empty until the knockout bracket is materialized.
async fn consensus(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::Consensus> {
    Json(engine.snapshot().consensus.clone())
}

/// The headline forecaster's live reliability curve + expected calibration error over its own
/// leak-free pre-match calls (empty bins until matches finish).
async fn calibration(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::ReliabilityReport> {
    Json(engine.snapshot().reliability.clone())
}

/// A prior-free Massey power ranking over only this tournament's results: who has been strongest
/// here, strength-of-schedule adjusted, with the offense/defense split. Empty until matches finish.
async fn power(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::PowerRanking> {
    Json(engine.snapshot().power_ranking.clone())
}

/// The tournament's biggest over- and under-performers versus their pre-tournament seeding (the gap
/// between the strength prior's ranking and the live power ranking). Empty until matches finish.
async fn form(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::TournamentForm> {
    Json(engine.snapshot().form.clone())
}

/// Each surviving contender's road to the final: the expected strength of the opponents it still has
/// to get through, round by round. Empty until the knockout bracket is materialized.
async fn road(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::RoadBoard> {
    Json(engine.snapshot().road.clone())
}

/// The model's single most likely completion of the knockout bracket, round by round to a projected
/// champion, with the chance this exact bracket occurs. Empty until the bracket is materialized.
async fn bracket(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::PredictedBracket> {
    Json(engine.snapshot().predicted_bracket.clone())
}

/// The current knockout round's ties ranked by how much each result would reshape the title race
/// (the total-variation swing in champion odds). Empty until the bracket is materialized.
async fn leverage(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::MatchLeverage> {
    Json(engine.snapshot().leverage.clone())
}

/// How open the title race is: the entropy of the champion odds, the effective number of contenders,
/// and a normalized openness in [0, 1]. Meaningful from the group stage onward.
async fn openness(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::Openness> {
    Json(engine.snapshot().openness.clone())
}

/// The championship-odds time series for the current top contenders over the tournament so far (the
/// title race's trajectory, persisted server-side rather than only in the browser).
async fn history(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::ChampionTimeline> {
    Json(engine.snapshot().timeline.clone())
}

/// The biggest recent movers in the title race: teams whose championship odds have risen or fallen
/// most over the last several forecast recomputes. Empty until enough history has accumulated.
async fn momentum(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::Momentum> {
    Json(engine.snapshot().momentum.clone())
}

/// When the title favourite changed hands over the tournament, with the current leader. Empty until
/// a forecast has been recorded.
async fn lead_changes(State(engine): State<Arc<Engine>>) -> Json<oracle_engine::LeadChanges> {
    Json(engine.snapshot().lead_changes.clone())
}

#[derive(Deserialize)]
struct TeamParams {
    /// Team name or FIFA code (case-insensitive).
    q: String,
}

/// A team's next unfinished fixture, from that team's perspective.
#[derive(Debug, Clone, Serialize)]
pub struct TeamNextMatch {
    pub match_id: MatchId,
    pub opponent: String,
    pub is_home: bool,
    pub stage: Stage,
    pub status: MatchStatus,
    pub win: f64,
    pub draw: f64,
    pub loss: f64,
    pub most_likely_score: Option<(u8, u8, f64)>,
}

/// The fan "your team" hub: the team's stage-by-stage journey odds, championship rank, and next
/// match, all read from the live tournament snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct TeamHub {
    pub team: String,
    pub code: String,
    pub rank: usize,
    pub p_advance: f64,
    pub p_round_of_16: f64,
    pub p_quarter: f64,
    pub p_semi: f64,
    pub p_final: f64,
    pub p_champion: f64,
    pub next_match: Option<TeamNextMatch>,
}

async fn team_hub(
    State(engine): State<Arc<Engine>>,
    Query(p): Query<TeamParams>,
) -> Result<Json<TeamHub>, StatusCode> {
    let snap = engine.snapshot();
    let q = p.q.trim().to_lowercase();
    let entry = snap
        .ratings
        .iter()
        .find(|r| r.code.to_lowercase() == q || r.name.to_lowercase() == q)
        .or_else(|| {
            snap.ratings
                .iter()
                .find(|r| r.name.to_lowercase().contains(&q))
        })
        .ok_or(StatusCode::NOT_FOUND)?;
    let id = entry.team;

    let ranked = snap.forecast.ranked();
    let rank = ranked
        .iter()
        .position(|t| t.team == id)
        .map(|i| i + 1)
        .unwrap_or(0);
    let f = ranked
        .iter()
        .find(|t| t.team == id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let next_match = snap
        .matches
        .iter()
        .find(|m| (m.home == id || m.away == id) && !matches!(m.status, MatchStatus::Finished))
        .map(|m| {
            let is_home = m.home == id;
            let pr = m.probabilities;
            TeamNextMatch {
                match_id: m.match_id,
                opponent: if is_home {
                    m.away_name.clone()
                } else {
                    m.home_name.clone()
                },
                is_home,
                stage: m.stage,
                status: m.status,
                win: if is_home { pr.home_win } else { pr.away_win },
                draw: pr.draw,
                loss: if is_home { pr.away_win } else { pr.home_win },
                most_likely_score: m.most_likely_score,
            }
        });

    Ok(Json(TeamHub {
        team: entry.name.clone(),
        code: entry.code.clone(),
        rank,
        p_advance: f.p_advance_group,
        p_round_of_16: f.p_round_of_16,
        p_quarter: f.p_quarter_final,
        p_semi: f.p_semi_final,
        p_final: f.p_final,
        p_champion: f.p_champion,
        next_match,
    }))
}

/// The fan "your team" page, a self-contained page that calls `/api/team`.
async fn team_page() -> Html<&'static str> {
    Html(include_str!("../static/team.html"))
}

#[derive(Deserialize)]
struct CardParams {
    team: Option<String>,
    home: Option<String>,
    away: Option<String>,
}

/// A shareable prediction card (`/card?team=` or `/card?home=&away=`). The visible card is rendered
/// client-side from `/api/team` and `/api/predict`, but the OpenGraph/Twitter meta is filled in
/// **server-side** from the live snapshot, so a pasted link shows a rich preview (crawlers do not
/// run the page's JavaScript).
async fn card_page(State(engine): State<Arc<Engine>>, Query(p): Query<CardParams>) -> Html<String> {
    let (title, desc) = card_social(&engine.snapshot(), &p);
    let esc = |s: &str| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    };
    let (t, d) = (esc(&title), esc(&desc));
    let meta = format!(
        "<meta property=\"og:title\" content=\"{t}\">\n\
         <meta property=\"og:description\" content=\"{d}\">\n\
         <meta property=\"og:type\" content=\"website\">\n\
         <meta name=\"twitter:card\" content=\"summary\">\n\
         <meta name=\"description\" content=\"{d}\">"
    );
    Html(include_str!("../static/card.html").replace("<!--SOCIAL-->", &meta))
}

/// The `(og:title, og:description)` for a card, computed from the live snapshot. Team mode is rich
/// (championship odds + rank, always available from the engine); a matchup is enriched when the two
/// sides meet in a known fixture, else a clean generic line.
fn card_social(snap: &Snapshot, p: &CardParams) -> (String, String) {
    if let (Some(h), Some(a)) = (&p.home, &p.away) {
        let (hl, al) = (h.to_lowercase(), a.to_lowercase());
        let fixture = snap.matches.iter().find(|m| {
            let (mh, ma) = (m.home_name.to_lowercase(), m.away_name.to_lowercase());
            (mh.contains(&hl) && ma.contains(&al)) || (mh.contains(&al) && ma.contains(&hl))
        });
        let desc = match fixture {
            Some(m) => {
                let pr = m.probabilities;
                let (fav, favp) = if pr.home_win >= pr.away_win {
                    (&m.home_name, pr.home_win)
                } else {
                    (&m.away_name, pr.away_win)
                };
                format!(
                    "The oracle's call: {} favoured ({:.0}%).",
                    fav,
                    favp * 100.0
                )
            }
            None => "The oracle's call for this matchup.".to_string(),
        };
        return (format!("{h} vs {a} - worldcup-oracle"), desc);
    }
    if let Some(q) = &p.team {
        let ql = q.trim().to_lowercase();
        let entry = snap
            .ratings
            .iter()
            .find(|r| r.code.to_lowercase() == ql || r.name.to_lowercase() == ql)
            .or_else(|| {
                snap.ratings
                    .iter()
                    .find(|r| r.name.to_lowercase().contains(&ql))
            });
        if let Some(e) = entry {
            let ranked = snap.forecast.ranked();
            let rank = ranked
                .iter()
                .position(|t| t.team == e.team)
                .map(|i| i + 1)
                .unwrap_or(0);
            let champ = ranked
                .iter()
                .find(|t| t.team == e.team)
                .map(|t| t.p_champion)
                .unwrap_or(0.0);
            return (
                format!(
                    "{} - {:.1}% to win the 2026 World Cup",
                    e.name,
                    champ * 100.0
                ),
                format!("worldcup-oracle rates {} the #{} favourite.", e.name, rank),
            );
        }
    }
    (
        "worldcup-oracle - the call".to_string(),
        "Share the model's 2026 World Cup predictions.".to_string(),
    )
}

async fn metrics(State(engine): State<Arc<Engine>>) -> impl IntoResponse {
    (
        [("content-type", "text/plain; version=0.0.4")],
        engine.metrics_prometheus(),
    )
}

async fn live_ws(State(engine): State<Arc<Engine>>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| live_socket(socket, engine))
}

/// Push a compact live view to one WebSocket client on every engine update.
async fn live_socket(socket: WebSocket, engine: Arc<Engine>) {
    let (mut sender, mut receiver) = socket.split();
    let mut updates = engine.subscribe();

    // Send the current state immediately so a fresh client isn't blank.
    let initial = LiveView::from_snapshot(&engine.snapshot());
    if send_json(&mut sender, &initial).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            update = updates.recv() => match update {
                Ok(snap) => {
                    let view = LiveView::from_snapshot(&snap);
                    if send_json(&mut sender, &view).await.is_err() {
                        break;
                    }
                }
                // Slow client fell behind; skip ahead rather than disconnect.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            },
            incoming = receiver.next() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                Some(Ok(_)) => {} // ignore pings / client chatter
            }
        }
    }
}

async fn send_json<T: Serialize>(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    value: &T,
) -> Result<(), axum::Error> {
    let text = serde_json::to_string(value).unwrap_or_else(|_| "{}".into());
    sender.send(Message::Text(text)).await
}

// ----- view types (compact, transport-friendly projections of a Snapshot) -----

/// A match prediction without the (large) exact-score grid.
#[derive(Debug, Clone, Serialize)]
pub struct MatchSummary {
    pub match_id: MatchId,
    pub home: TeamId,
    pub away: TeamId,
    pub home_name: String,
    pub away_name: String,
    pub stage: Stage,
    pub status: MatchStatus,
    pub score: Scoreline,
    pub minute: u16,
    pub probabilities: Probabilities,
    /// Win-probability timeline for a live match (the drama graph); empty for others.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<WinProbSample>,
}

impl MatchSummary {
    fn from_prediction(p: &oracle_engine::MatchPrediction) -> Self {
        Self {
            match_id: p.match_id,
            home: p.home,
            away: p.away,
            home_name: p.home_name.clone(),
            away_name: p.away_name.clone(),
            stage: p.stage,
            status: p.status,
            score: p.score,
            minute: p.minute,
            probabilities: p.probabilities,
            history: p.history.clone(),
        }
    }
}

/// One row of the champion-odds table, with the team name joined in.
#[derive(Debug, Clone, Serialize)]
pub struct ForecastRow {
    pub team: TeamId,
    pub name: String,
    pub p_advance_group: f64,
    pub p_round_of_16: f64,
    pub p_quarter_final: f64,
    pub p_semi_final: f64,
    pub p_final: f64,
    pub p_champion: f64,
}

/// The champion-odds table, ranked by championship probability.
#[derive(Debug, Clone, Serialize)]
pub struct TournamentView {
    pub tournament: String,
    pub iterations: u64,
    pub teams: Vec<ForecastRow>,
}

fn name_map(snap: &Snapshot) -> HashMap<TeamId, String> {
    snap.ratings
        .iter()
        .map(|r| (r.team, r.name.clone()))
        .collect()
}

impl TournamentView {
    pub fn from_snapshot(snap: &Snapshot) -> Self {
        let names = name_map(snap);
        let teams = snap
            .forecast
            .ranked()
            .into_iter()
            .map(|t| ForecastRow {
                team: t.team,
                name: names
                    .get(&t.team)
                    .cloned()
                    .unwrap_or_else(|| t.team.to_string()),
                p_advance_group: t.p_advance_group,
                p_round_of_16: t.p_round_of_16,
                p_quarter_final: t.p_quarter_final,
                p_semi_final: t.p_semi_final,
                p_final: t.p_final,
                p_champion: t.p_champion,
            })
            .collect();
        Self {
            tournament: snap.tournament.clone(),
            iterations: snap.forecast.iterations,
            teams,
        }
    }
}

/// The payload pushed over the WebSocket: only what a live dashboard needs.
#[derive(Debug, Clone, Serialize)]
pub struct LiveView {
    pub generated_at: String,
    /// Whether the data feed is healthy; `false` means these figures may be stale.
    pub source_healthy: bool,
    /// When the engine last processed an event from the feed.
    pub last_update: String,
    pub live_matches: Vec<MatchSummary>,
    pub top_contenders: Vec<ForecastRow>,
    /// The engine's live self-recalibration state (temperature, context gain, results seen).
    pub adaptive: AdaptiveState,
}

impl LiveView {
    pub fn from_snapshot(snap: &Snapshot) -> Self {
        let live_matches = snap
            .matches
            .iter()
            .filter(|m| matches!(m.status, MatchStatus::Live { .. }))
            .map(MatchSummary::from_prediction)
            .collect();
        let top_contenders = TournamentView::from_snapshot(snap)
            .teams
            .into_iter()
            .take(10)
            .collect();
        Self {
            generated_at: snap.generated_at.to_rfc3339(),
            source_healthy: snap.source_healthy,
            last_update: snap.last_update.to_rfc3339(),
            live_matches,
            top_contenders,
            adaptive: snap.adaptive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{router, AppState};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use oracle_engine::{presets, spawn, EngineConfig, Explorer};
    use std::sync::{Arc, OnceLock};
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt; // for `oneshot`

    /// One shared, already-fit explorer slot for the whole test module (fitting the baseline takes
    /// a few seconds). Tests that want the warming-up state use an empty `OnceLock` instead.
    fn shared_explorer() -> Arc<OnceLock<Explorer>> {
        static EX: OnceLock<Arc<OnceLock<Explorer>>> = OnceLock::new();
        EX.get_or_init(|| {
            let slot = Arc::new(OnceLock::new());
            let _ = slot.set(Explorer::new());
            slot
        })
        .clone()
    }

    /// Spawn an engine over the deterministic simulation feed (no network), pair it with the
    /// shared explorer, and return the app state plus a cancel token to wind the engine down.
    async fn test_state() -> (AppState, CancellationToken) {
        let cancel = CancellationToken::new();
        let (engine, _join) = spawn(
            presets::simulated(),
            EngineConfig::default(),
            cancel.clone(),
        )
        .await
        .expect("engine spawns");
        (
            AppState {
                engine,
                explorer: shared_explorer(),
            },
            cancel,
        )
    }

    /// Fire one GET request at a fresh router and return `(status, body bytes)`.
    async fn get(state: &AppState, uri: &str) -> (StatusCode, Vec<u8>) {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let res = router(state.engine.clone(), state.explorer.clone())
            .oneshot(req)
            .await
            .unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
        (status, bytes.to_vec())
    }

    fn json(bytes: &[u8]) -> serde_json::Value {
        serde_json::from_slice(bytes).expect("valid JSON body")
    }

    #[test]
    fn listen_addr_prefers_port_then_oracle_addr() {
        use super::resolve_listen_addr;
        // $PORT wins (the PaaS convention).
        assert_eq!(
            resolve_listen_addr(Some("3000".into()), Some("0.0.0.0:8080".into()))
                .unwrap()
                .port(),
            3000
        );
        // Then $ORACLE_ADDR.
        assert_eq!(
            resolve_listen_addr(None, Some("127.0.0.1:9000".into()))
                .unwrap()
                .port(),
            9000
        );
        // Then the default. An empty $PORT is ignored.
        assert_eq!(
            resolve_listen_addr(None, None).unwrap().to_string(),
            "0.0.0.0:8080"
        );
        assert_eq!(
            resolve_listen_addr(Some("  ".into()), None).unwrap().port(),
            8080
        );
    }

    #[tokio::test]
    async fn rest_routes_return_expected_shapes() {
        let (state, cancel) = test_state().await;

        let (status, body) = get(&state, "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, b"ok");

        let (status, body) = get(&state, "/api").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json(&body)["service"], "worldcup-oracle");
        // The live self-recalibration state is surfaced (neutral before any results).
        let adaptive = &json(&body)["adaptive"];
        assert!(adaptive["calibration_temperature"].is_number());
        assert!(adaptive["context_gain"].is_number());
        assert_eq!(adaptive["results_seen"].as_u64(), Some(0));

        let (status, body) = get(&state, "/teams").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json(&body).as_array().unwrap().len(), 48);

        let (status, body) = get(&state, "/matches").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json(&body).as_array().unwrap().len(), 72);

        let (status, body) = get(&state, "/predict/tournament").await;
        assert_eq!(status, StatusCode::OK);
        let champ_mass: f64 = json(&body)["teams"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["p_champion"].as_f64().unwrap())
            .sum();
        assert!(
            (champ_mass - 1.0).abs() < 0.05,
            "champion mass ~1: {champ_mass}"
        );

        let (status, body) = get(&state, "/predict/match/1").await;
        assert_eq!(status, StatusCode::OK);
        let m = json(&body);
        assert!(m["home_name"].is_string() && m["away_name"].is_string());
        let p = &m["probabilities"];
        let sum = p["home_win"].as_f64().unwrap()
            + p["draw"].as_f64().unwrap()
            + p["away_win"].as_f64().unwrap();
        assert!((sum - 1.0).abs() < 1e-6, "match probs sum to 1");

        let (status, body) = get(&state, "/upsets").await;
        assert_eq!(status, StatusCode::OK);
        let board = json(&body);
        assert!(board["upcoming"].is_array() && board["shocks"].is_array());

        let (status, body) = get(&state, "/report").await;
        assert_eq!(status, StatusCode::OK);
        let rc = json(&body);
        assert!(rc["scored"].is_number() && rc["brier"].is_number());
        assert!(rc["best_calls"].is_array() && rc["worst_calls"].is_array());

        // The second model's live champion odds are an array (empty before the bracket is set).
        let (status, body) = get(&state, "/bt/champions").await;
        assert_eq!(status, StatusCode::OK);
        assert!(json(&body).is_array());

        // The consensus forecast carries a numeric divergence and a team array.
        let (status, body) = get(&state, "/consensus").await;
        assert_eq!(status, StatusCode::OK);
        let cons = json(&body);
        assert!(cons["jsd"].is_number() && cons["teams"].is_array());

        // The calibration report carries an expected calibration error and ten reliability bins.
        let (status, body) = get(&state, "/calibration").await;
        assert_eq!(status, StatusCode::OK);
        let cal = json(&body);
        assert!(cal["ece"].is_number());
        assert_eq!(cal["bins"].as_array().unwrap().len(), 10);

        // The Massey power ranking carries a match count and a teams array.
        let (status, body) = get(&state, "/power").await;
        assert_eq!(status, StatusCode::OK);
        let pow = json(&body);
        assert!(pow["matches"].is_number() && pow["teams"].is_array());

        // The form board carries riser and faller arrays.
        let (status, body) = get(&state, "/form").await;
        assert_eq!(status, StatusCode::OK);
        let form = json(&body);
        assert!(form["risers"].is_array() && form["fallers"].is_array());

        // The road board carries a teams array.
        let (status, body) = get(&state, "/road").await;
        assert_eq!(status, StatusCode::OK);
        assert!(json(&body)["teams"].is_array());

        // The predicted bracket carries a champion, a probability, and a rounds array.
        let (status, body) = get(&state, "/bracket").await;
        assert_eq!(status, StatusCode::OK);
        let br = json(&body);
        assert!(
            br["champion"].is_string() && br["probability"].is_number() && br["rounds"].is_array()
        );

        // The match-leverage board carries a ties array.
        let (status, body) = get(&state, "/leverage").await;
        assert_eq!(status, StatusCode::OK);
        assert!(json(&body)["ties"].is_array());

        // The openness readout carries entropy and effective-contenders numbers.
        let (status, body) = get(&state, "/openness").await;
        assert_eq!(status, StatusCode::OK);
        let op = json(&body);
        assert!(op["entropy_bits"].is_number() && op["effective_contenders"].is_number());

        // The champion timeline carries a sample count and a series array.
        let (status, body) = get(&state, "/history").await;
        assert_eq!(status, StatusCode::OK);
        let hist = json(&body);
        assert!(hist["samples"].is_number() && hist["series"].is_array());

        // The momentum board carries riser and faller arrays.
        let (status, body) = get(&state, "/momentum").await;
        assert_eq!(status, StatusCode::OK);
        let mom = json(&body);
        assert!(mom["risers"].is_array() && mom["fallers"].is_array());

        // The lead-changes board carries a current leader and a changes array.
        let (status, body) = get(&state, "/lead-changes").await;
        assert_eq!(status, StatusCode::OK);
        let lc = json(&body);
        assert!(lc["current_leader"].is_string() && lc["changes"].is_array());

        let (status, body) = get(&state, "/api/team?q=Brazil").await;
        assert_eq!(status, StatusCode::OK);
        let hub = json(&body);
        assert_eq!(hub["team"], "Brazil");
        assert!(hub["p_champion"].is_number() && hub["rank"].as_u64().unwrap() >= 1);
        let (status, _) = get(&state, "/api/team?q=Atlantis").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, body) = get(&state, "/team").await;
        assert_eq!(status, StatusCode::OK);
        assert!(String::from_utf8_lossy(&body).contains("<title>"));

        let (status, body) = get(&state, "/card?team=Brazil").await;
        assert_eq!(status, StatusCode::OK);
        let card = String::from_utf8_lossy(&body);
        // Server-rendered social preview: og:title carries the team and its championship odds.
        assert!(card.contains("og:title") && card.contains("Brazil"));
        assert!(card.contains("% to win the 2026 World Cup"));

        let (status, body) = get(&state, "/metrics").await;
        assert_eq!(status, StatusCode::OK);
        assert!(String::from_utf8_lossy(&body).contains("oracle_events_processed_total"));

        cancel.cancel();
    }

    #[tokio::test]
    async fn unknown_match_is_404_and_dashboard_is_served() {
        let (state, cancel) = test_state().await;

        let (status, _) = get(&state, "/predict/match/999999").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, body) = get(&state, "/").await;
        assert_eq!(status, StatusCode::OK);
        assert!(String::from_utf8_lossy(&body).contains("<title>"));

        cancel.cancel();
    }

    #[tokio::test]
    async fn api_is_503_while_warming_but_live_endpoints_stay_up() {
        // An empty explorer slot models the warmup window (before the background fit completes).
        let cancel = CancellationToken::new();
        let (engine, _join) = spawn(
            presets::simulated(),
            EngineConfig::default(),
            cancel.clone(),
        )
        .await
        .expect("engine spawns");
        let state = AppState {
            engine,
            explorer: Arc::new(OnceLock::new()),
        };

        // The on-demand query endpoints report "warming up".
        for uri in [
            "/api/ratings",
            "/api/predict?home=Brazil&away=Japan",
            "/api/simulate?iters=2000",
            "/api/sensitivity?iters=2000",
        ] {
            let (status, _) = get(&state, uri).await;
            assert_eq!(
                status,
                StatusCode::SERVICE_UNAVAILABLE,
                "{uri} should 503 while warming"
            );
        }
        // The live dashboard, health, and engine endpoints are unaffected by explorer readiness.
        for uri in ["/health", "/", "/predict/tournament"] {
            let (status, _) = get(&state, uri).await;
            assert_eq!(status, StatusCode::OK, "{uri} should be up during warmup");
        }
        cancel.cancel();
    }

    #[tokio::test]
    async fn on_demand_query_endpoints_work() {
        let (state, cancel) = test_state().await;

        // Predict any matchup: ensemble probabilities normalize, the full grid is present.
        let (status, body) = get(&state, "/api/predict?home=Brazil&away=Japan").await;
        assert_eq!(status, StatusCode::OK);
        let v = json(&body);
        let e = &v["ensemble"];
        let sum = e["home_win"].as_f64().unwrap()
            + e["draw"].as_f64().unwrap()
            + e["away_win"].as_f64().unwrap();
        assert!((sum - 1.0).abs() < 1e-6, "ensemble probs sum to 1");
        assert_eq!(v["grid"]["grid"].as_array().unwrap().len(), 11);

        // Unknown team -> 404.
        let (status, _) = get(&state, "/api/predict?home=Brazil&away=Atlantis").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Explain: named factors + the ensemble member breakdown.
        let (status, body) = get(&state, "/api/explain?home=Brazil&away=Japan").await;
        assert_eq!(status, StatusCode::OK);
        let ex = json(&body);
        assert_eq!(ex["strength_factors"].as_array().unwrap().len(), 4);
        assert!(ex["members"].as_array().unwrap().len() >= 3);
        assert!(ex["expected_home"].as_f64().unwrap() > 0.0);

        // Second model (Bradley-Terry): a matchup W/D/L and champion odds.
        let (status, body) = get(&state, "/api/bt?home=Brazil&away=Japan").await;
        assert_eq!(status, StatusCode::OK);
        let bt = json(&body);
        let s = bt["probabilities"]["home_win"].as_f64().unwrap()
            + bt["probabilities"]["draw"].as_f64().unwrap()
            + bt["probabilities"]["away_win"].as_f64().unwrap();
        assert!((s - 1.0).abs() < 1e-6, "BT probabilities sum to 1");
        let (status, body) = get(&state, "/api/bt/champions").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json(&body)["teams"].as_array().unwrap().len(), 32);

        // Custom simulation: champion mass ~1.
        let (status, body) = get(&state, "/api/simulate?iters=2000&seed=1").await;
        assert_eq!(status, StatusCode::OK);
        let mass: f64 = json(&body)["teams"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["p_champion"].as_f64().unwrap())
            .sum();
        assert!((mass - 1.0).abs() < 0.05, "champion mass ~1: {mass}");

        // Ratings: all teams + six confederations.
        let (status, body) = get(&state, "/api/ratings").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json(&body)["teams"].as_array().unwrap().len(), 48);
        assert_eq!(json(&body)["confederations"].as_array().unwrap().len(), 6);

        // Sensitivity: nine signals, each a valid total-variation distance, ranked descending.
        let (status, body) = get(&state, "/api/sensitivity?iters=2000&seed=1").await;
        assert_eq!(status, StatusCode::OK);
        let signals = json(&body)["signals"].as_array().unwrap().clone();
        assert_eq!(signals.len(), 9);
        let shifts: Vec<f64> = signals
            .iter()
            .map(|s| s["title_shift"].as_f64().unwrap())
            .collect();
        assert!(shifts.iter().all(|&s| (0.0..=1.0).contains(&s)));
        assert!(shifts.windows(2).all(|w| w[0] >= w[1]), "ranked descending");

        // Posterior: a 90% credible interval that brackets its own mean.
        let (status, body) = get(&state, "/api/posterior?home=Brazil&away=Japan&samples=200").await;
        assert_eq!(status, StatusCode::OK);
        let hw = &json(&body)["home_win"];
        assert!(hw["lo"].as_f64().unwrap() <= hw["mean"].as_f64().unwrap());
        assert!(hw["mean"].as_f64().unwrap() <= hw["hi"].as_f64().unwrap());

        // Kingmaker: rooting-interest rows for a team.
        let (status, body) = get(&state, "/api/kingmaker?q=Brazil&iters=2000").await;
        assert_eq!(status, StatusCode::OK);
        let km = json(&body);
        assert_eq!(km["team"], "Brazil");
        assert!(km["matches"].is_array() && km["base_champion"].is_number());

        // Collision course: two teams' knockout meeting probability, by round.
        let (status, body) = get(
            &state,
            "/api/collision?home=Brazil&away=Argentina&iters=3000",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let col = json(&body);
        assert!(col["meet_probability"].is_number());
        assert_eq!(col["by_round"].as_array().unwrap().len(), 5);

        // The explorer page is served.
        let (status, body) = get(&state, "/explore").await;
        assert_eq!(status, StatusCode::OK);
        assert!(String::from_utf8_lossy(&body).contains("<title>"));

        cancel.cancel();
    }
}
