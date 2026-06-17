//! # oracle-api
//!
//! The transport layer: an [`axum`] application exposing the engine over HTTP and
//! WebSocket. It is a thin shell — every handler just reads an immutable
//! [`Snapshot`](oracle_engine::Snapshot) from the engine (lock-free) or relays the
//! broadcast stream. No prediction logic lives here.
//!
//! ## Endpoints
//! | Method | Path | Purpose |
//! |--------|------|---------|
//! | GET | `/health` | liveness probe |
//! | GET | `/teams` | current Elo ratings |
//! | GET | `/matches` | all match predictions (compact) |
//! | GET | `/predict/match/{id}` | one match, full exact-score grid |
//! | GET | `/predict/tournament` | champion-odds table |
//! | GET | `/metrics` | Prometheus metrics |
//! | GET | `/live` | WebSocket: pushes a compact live view on every update |
#![forbid(unsafe_code)]

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use oracle_domain::{MatchId, MatchStatus, Probabilities, Scoreline, Stage, TeamId};
use oracle_engine::{Engine, Snapshot};
use serde::Serialize;
use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Build the application router with the engine as shared state.
pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/teams", get(teams))
        .route("/matches", get(matches))
        .route("/predict/match/:id", get(predict_match))
        .route("/predict/tournament", get(predict_tournament))
        .route("/metrics", get(metrics))
        .route("/live", get(live_ws))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(engine)
}

/// Serve the API until `shutdown` resolves (graceful shutdown).
pub async fn serve<F>(engine: Arc<Engine>, addr: SocketAddr, shutdown: F) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "oracle-api listening");
    axum::serve(listener, router(engine))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

// ----- handlers -----

async fn health() -> &'static str {
    "ok"
}

async fn index(State(engine): State<Arc<Engine>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "service": "worldcup-oracle",
        "tournament": engine.tournament_name(),
        "provider": engine.provider_name(),
        "endpoints": [
            "/health", "/teams", "/matches",
            "/predict/match/{id}", "/predict/tournament",
            "/metrics", "/live (websocket)"
        ],
    }))
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
    pub live_matches: Vec<MatchSummary>,
    pub top_contenders: Vec<ForecastRow>,
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
            live_matches,
            top_contenders,
        }
    }
}
