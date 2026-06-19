//! Live adapter for the [football-data.org](https://www.football-data.org) v4 API.
//!
//! Loads the real competition (teams, groups, fixtures) and then polls for changes,
//! translating them into the engine's normalized [`MatchEvent`] stream. Every request
//! passes through the [`RateLimiter`] (free tier ≈ 10 req/min) and a short-lived
//! [`TtlCache`], so a chatty poll loop never blows the request budget.
//!
//! Set `FOOTBALL_DATA_API_KEY` to use it; without a key the engine falls back to the
//! simulation/replay providers, so this is an optional upgrade path, never a hard
//! requirement.
//!
//! What the feed provides, by tier: **results** (scores, status, stage, group) work on the
//! free tier and drive `Goal`/`FullTime`/`ScoreSync` events. **Line-ups** are emitted as
//! `Lineup` events when the match payload includes them (a paid tier), and are simply absent
//! on the free tier (no adjustment, no error). **Odds** and **expected goals** are not
//! offered by this provider at any tier: use the CSV path (`backtest --data`) for real odds,
//! and a dedicated xG source to populate `Observation` xG.

use crate::cache::TtlCache;
use crate::error::{IngestError, Result};
use crate::provider::DataProvider;
use crate::rate_limit::RateLimiter;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use oracle_domain::{
    EventKind, Group, Match, MatchEvent, MatchId, MatchStatus, Scoreline, Stage, Team, TeamId,
    Tournament,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc::Sender;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE: &str = "https://api.football-data.org/v4";
const ENV_KEY: &str = "FOOTBALL_DATA_API_KEY";

/// Live football-data.org provider.
pub struct FootballDataProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    competition: String,
    limiter: RateLimiter,
    cache: TtlCache<String, String>,
    poll_interval: Duration,
}

impl FootballDataProvider {
    /// Construct from the `FOOTBALL_DATA_API_KEY` env var, targeting the World Cup
    /// competition (`WC`).
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var(ENV_KEY).map_err(|_| IngestError::MissingConfig(ENV_KEY))?;
        Ok(Self::new(api_key, "WC"))
    }

    pub fn new(api_key: impl Into<String>, competition: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE.to_string(),
            api_key: api_key.into(),
            competition: competition.into(),
            limiter: RateLimiter::per_minute(10),
            cache: TtlCache::new(Duration::from_secs(15)),
            poll_interval: Duration::from_secs(20),
        }
    }

    /// Fetch JSON for a path, honouring the rate limit and the short-lived cache.
    async fn get(&self, path: &str) -> Result<String> {
        if let Some(hit) = self.cache.get(&path.to_string()) {
            return Ok(hit);
        }
        self.limiter.acquire().await;
        let url = format!("{}/{}", self.base_url, path);
        let text = self
            .client
            .get(&url)
            .header("X-Auth-Token", &self.api_key)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        self.cache.put(path.to_string(), text.clone());
        Ok(text)
    }

    async fn fetch_matches(&self) -> Result<Vec<ApiMatch>> {
        let path = format!("competitions/{}/matches", self.competition);
        let body = self.get(&path).await?;
        let parsed: MatchesResponse = serde_json::from_str(&body)?;
        Ok(parsed.matches)
    }
}

#[async_trait]
impl DataProvider for FootballDataProvider {
    fn name(&self) -> &'static str {
        "football-data.org"
    }

    async fn load_tournament(&self) -> Result<Tournament> {
        let teams_body = self
            .get(&format!("competitions/{}/teams", self.competition))
            .await?;
        let teams_resp: TeamsResponse = serde_json::from_str(&teams_body)?;
        let matches = self.fetch_matches().await?;

        let mut t = Tournament::new("FIFA World Cup");
        t.teams = teams_resp
            .teams
            .iter()
            .map(|at| {
                Team::new(
                    at.id as u32,
                    at.name.clone(),
                    at.tla.clone().unwrap_or_else(|| "???".into()),
                    // The API doesn't expose confederation; it only feeds an optional
                    // prior, so a neutral default is harmless.
                    oracle_domain::Confederation::Uefa,
                )
            })
            .collect();

        // Reconstruct groups from the group-stage fixtures.
        let mut groups: HashMap<char, Vec<TeamId>> = HashMap::new();
        for m in &matches {
            if let (Some(letter), Some(h), Some(a)) =
                (m.group_letter(), m.home_team.id, m.away_team.id)
            {
                let entry = groups.entry(letter).or_default();
                for id in [TeamId(h as u32), TeamId(a as u32)] {
                    if !entry.contains(&id) {
                        entry.push(id);
                    }
                }
            }
            if let Some(parsed) = m.to_domain() {
                t.matches.push(parsed);
            }
        }
        let mut groups: Vec<Group> = groups
            .into_iter()
            .map(|(name, teams)| Group { name, teams })
            .collect();
        groups.sort_by_key(|g| g.name);
        t.groups = groups;

        Ok(t)
    }

    async fn run(&self, tx: Sender<MatchEvent>, cancel: CancellationToken) -> Result<()> {
        // Track the last-seen status/score so we can diff into events.
        let mut last: HashMap<u32, (MatchStatus, Scoreline)> = HashMap::new();
        let mut failures: u32 = 0;

        loop {
            if cancel.is_cancelled() {
                return Ok(());
            }
            match self.fetch_matches().await {
                Ok(matches) => {
                    failures = 0;
                    let _ = tx
                        .send(MatchEvent::new(
                            HEARTBEAT,
                            0,
                            EventKind::SourceStatus { healthy: true },
                        ))
                        .await;
                    for m in &matches {
                        let Some(parsed) = m.to_domain() else {
                            continue;
                        };
                        let id = parsed.id.0;
                        let minute = m.minute.unwrap_or(0);
                        match last.get(&id) {
                            None => {
                                // On first sighting, surface a confirmed line-up if the feed
                                // provides one (tiers that expose it announce it pre-kickoff).
                                if let Some((home, away)) = m.lineups() {
                                    let _ = tx
                                        .send(MatchEvent::new(
                                            parsed.id,
                                            minute,
                                            EventKind::Lineup { home, away },
                                        ))
                                        .await;
                                }
                                if parsed.is_live() {
                                    let _ = tx
                                        .send(MatchEvent::new(
                                            parsed.id,
                                            minute,
                                            EventKind::KickOff,
                                        ))
                                        .await;
                                }
                            }
                            Some((prev_status, prev_score)) => {
                                emit_diff(&tx, &parsed, *prev_status, *prev_score, minute).await;
                            }
                        }
                        // Authoritative reconcile: the engine *sets* the live score to
                        // this, so a dropped/duplicated goal event can't cause drift.
                        if parsed.is_live() {
                            let _ = tx
                                .send(MatchEvent::new(
                                    parsed.id,
                                    minute,
                                    EventKind::ScoreSync {
                                        score: parsed.score,
                                    },
                                ))
                                .await;
                        }
                        last.insert(id, (parsed.status, parsed.score));
                    }
                }
                Err(e) => {
                    failures = failures.saturating_add(1);
                    tracing::warn!(error = %e, failures, "football-data poll failed; backing off");
                    let _ = tx
                        .send(MatchEvent::new(
                            HEARTBEAT,
                            0,
                            EventKind::SourceStatus { healthy: false },
                        ))
                        .await;
                }
            }

            // Exponential backoff after consecutive failures (capped at 8× interval).
            let wait = self.poll_interval * (1u32 << failures.min(3));
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = sleep(wait) => {}
            }
        }
    }
}

/// Sentinel match id for feed-level (non-match) events like [`EventKind::SourceStatus`].
const HEARTBEAT: MatchId = MatchId(0);

/// Emit the events implied by the change between a previous and current match state.
async fn emit_diff(
    tx: &Sender<MatchEvent>,
    cur: &Match,
    prev_status: MatchStatus,
    prev_score: Scoreline,
    minute: u16,
) {
    let was_scheduled = matches!(prev_status, MatchStatus::Scheduled);
    if was_scheduled && cur.is_live() {
        let _ = tx
            .send(MatchEvent::new(cur.id, minute, EventKind::KickOff))
            .await;
    }
    for _ in prev_score.home..cur.score.home {
        let _ = tx
            .send(MatchEvent::new(
                cur.id,
                minute,
                EventKind::Goal {
                    team: cur.home,
                    scorer: None,
                },
            ))
            .await;
    }
    for _ in prev_score.away..cur.score.away {
        let _ = tx
            .send(MatchEvent::new(
                cur.id,
                minute,
                EventKind::Goal {
                    team: cur.away,
                    scorer: None,
                },
            ))
            .await;
    }
    if cur.is_finished() && !matches!(prev_status, MatchStatus::Finished) {
        let _ = tx
            .send(MatchEvent::new(
                cur.id,
                90,
                EventKind::FullTime { score: cur.score },
            ))
            .await;
    }
    // Live matches get an authoritative `ScoreSync` from the caller, which also
    // advances the minute, so no separate tick is emitted here.
}

// ----- football-data.org v4 response DTOs -----

#[derive(Debug, Deserialize)]
struct TeamsResponse {
    teams: Vec<ApiTeam>,
}

#[derive(Debug, Deserialize)]
struct ApiTeam {
    id: u64,
    name: String,
    tla: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MatchesResponse {
    matches: Vec<ApiMatch>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiMatch {
    id: u64,
    utc_date: Option<String>,
    status: String,
    stage: String,
    #[serde(default)]
    group: Option<String>,
    #[serde(default)]
    minute: Option<u16>,
    home_team: ApiSide,
    away_team: ApiSide,
    score: ApiScore,
}

#[derive(Debug, Deserialize)]
struct ApiSide {
    id: Option<u64>,
    /// Starting line-up, present only on API tiers that expose it (absent on the free tier,
    /// where this is simply an empty list and the engine applies no lineup adjustment).
    #[serde(default)]
    lineup: Vec<ApiPlayer>,
}

#[derive(Debug, Deserialize)]
struct ApiPlayer {
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiScore {
    #[serde(rename = "fullTime")]
    full_time: ApiGoals,
}

#[derive(Debug, Deserialize)]
struct ApiGoals {
    #[serde(default)]
    home: Option<u8>,
    #[serde(default)]
    away: Option<u8>,
}

impl ApiMatch {
    /// The group letter, e.g. "GROUP_A" → 'A'.
    fn group_letter(&self) -> Option<char> {
        self.group.as_ref().and_then(|g| g.chars().last())
    }

    /// Confirmed `(home, away)` line-ups as player-name lists, when both sides expose one.
    /// Returns `None` on tiers (or before kickoff) that don't include line-ups.
    fn lineups(&self) -> Option<(Vec<String>, Vec<String>)> {
        let names = |side: &ApiSide| -> Vec<String> {
            side.lineup.iter().filter_map(|p| p.name.clone()).collect()
        };
        let (home, away) = (names(&self.home_team), names(&self.away_team));
        (!home.is_empty() && !away.is_empty()).then_some((home, away))
    }

    /// Translate into the domain `Match`, or `None` if teams aren't assigned yet.
    fn to_domain(&self) -> Option<Match> {
        let (h, a) = (self.home_team.id?, self.away_team.id?);
        let kickoff: DateTime<Utc> = self
            .utc_date
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        Some(Match {
            id: MatchId(self.id as u32),
            home: TeamId(h as u32),
            away: TeamId(a as u32),
            stage: map_stage(&self.stage, self.group_letter()),
            kickoff,
            status: map_status(&self.status, self.minute),
            score: Scoreline::new(
                self.score.full_time.home.unwrap_or(0),
                self.score.full_time.away.unwrap_or(0),
            ),
        })
    }
}

fn map_status(status: &str, minute: Option<u16>) -> MatchStatus {
    match status {
        "FINISHED" | "AWARDED" => MatchStatus::Finished,
        "IN_PLAY" | "PAUSED" | "LIVE" | "HALFTIME" => MatchStatus::Live {
            minute: minute.unwrap_or(0),
        },
        "POSTPONED" | "CANCELLED" | "SUSPENDED" => MatchStatus::Postponed,
        _ => MatchStatus::Scheduled,
    }
}

fn map_stage(stage: &str, group: Option<char>) -> Stage {
    match stage {
        "GROUP_STAGE" => Stage::Group(group.unwrap_or('A')),
        "LAST_32" | "ROUND_OF_32" => Stage::RoundOf32,
        "LAST_16" | "ROUND_OF_16" => Stage::RoundOf16,
        "QUARTER_FINALS" => Stage::QuarterFinal,
        "SEMI_FINALS" => Stage::SemiFinal,
        "THIRD_PLACE" => Stage::ThirdPlace,
        "FINAL" => Stage::Final,
        _ => Stage::Group(group.unwrap_or('A')),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping() {
        assert!(matches!(
            map_status("FINISHED", None),
            MatchStatus::Finished
        ));
        assert!(matches!(
            map_status("IN_PLAY", Some(57)),
            MatchStatus::Live { minute: 57 }
        ));
        assert!(matches!(map_status("TIMED", None), MatchStatus::Scheduled));
    }

    #[test]
    fn stage_mapping() {
        assert!(matches!(
            map_stage("GROUP_STAGE", Some('C')),
            Stage::Group('C')
        ));
        assert!(matches!(map_stage("FINAL", None), Stage::Final));
        assert!(matches!(
            map_stage("QUARTER_FINALS", None),
            Stage::QuarterFinal
        ));
    }

    #[test]
    fn parses_a_sample_match_payload() {
        let json = r#"{
            "matches": [{
                "id": 12345,
                "utcDate": "2026-06-12T19:00:00Z",
                "status": "IN_PLAY",
                "stage": "GROUP_STAGE",
                "group": "GROUP_B",
                "minute": 63,
                "homeTeam": { "id": 770 },
                "awayTeam": { "id": 759 },
                "score": { "fullTime": { "home": 2, "away": 1 } }
            }]
        }"#;
        let resp: MatchesResponse = serde_json::from_str(json).unwrap();
        let m = resp.matches[0].to_domain().unwrap();
        assert_eq!(m.id, MatchId(12345));
        assert_eq!(m.score, Scoreline::new(2, 1));
        assert!(matches!(m.stage, Stage::Group('B')));
        assert!(matches!(m.status, MatchStatus::Live { minute: 63 }));
        // No lineup in this payload (free-tier shape) -> no lineup ingested.
        assert!(resp.matches[0].lineups().is_none());
    }

    #[test]
    fn parses_lineups_when_present() {
        let json = r#"{
            "matches": [{
                "id": 1,
                "status": "TIMED",
                "stage": "GROUP_STAGE",
                "group": "GROUP_A",
                "homeTeam": { "id": 10, "lineup": [{ "name": "A. One" }, { "name": "B. Two" }] },
                "awayTeam": { "id": 11, "lineup": [{ "name": "C. Three" }] },
                "score": { "fullTime": { "home": null, "away": null } }
            }]
        }"#;
        let resp: MatchesResponse = serde_json::from_str(json).unwrap();
        let (home, away) = resp.matches[0].lineups().expect("lineups present");
        assert_eq!(home, vec!["A. One".to_string(), "B. Two".to_string()]);
        assert_eq!(away, vec!["C. Three".to_string()]);
    }
}
