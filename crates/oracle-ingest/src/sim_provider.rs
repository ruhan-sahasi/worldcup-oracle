//! A deterministic synthetic data feed — the default, key-free way to run the engine.
//!
//! `SimProvider` owns a tournament and a fitted goal model, then plays the group
//! stage forward in (accelerated) wall-clock time, emitting a realistic stream of
//! kickoffs, clock ticks, goals, the occasional red card, and full-time results.
//! Goal timings are sampled from the model's expected goals, so favourites really do
//! score more. Seeding the RNG per match keeps a given `seed` perfectly reproducible.

use crate::data;
use crate::error::{IngestError, Result};
use crate::provider::DataProvider;
use async_trait::async_trait;
use oracle_domain::{EventKind, MatchEvent, MatchId, Scoreline, Stage, TeamId, Tournament};
use oracle_model::GoalModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use tokio::sync::mpsc::Sender;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

/// Synthetic live feed over a fixed tournament.
pub struct SimProvider {
    tournament: Tournament,
    model: GoalModel,
    /// Wall-clock time that one simulated minute takes.
    minute_delay: Duration,
    /// Emit a clock tick every this many simulated minutes.
    tick_every: u16,
    seed: u64,
}

impl SimProvider {
    /// The default: the embedded 2026 World Cup, a baseline-fitted model, and a
    /// brisk-but-watchable clock (~20 ms per simulated minute).
    pub fn new() -> Self {
        Self {
            tournament: data::world_cup_2026(),
            model: data::fit_baseline_model(7),
            minute_delay: Duration::from_millis(20),
            tick_every: 5,
            seed: 7,
        }
    }

    /// Override the simulation speed (smaller delay → faster matches).
    pub fn with_minute_delay(mut self, delay: Duration) -> Self {
        self.minute_delay = delay;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Play one match minute-by-minute, emitting its events. Returns the final score.
    async fn play_match(
        &self,
        match_id: MatchId,
        home: TeamId,
        away: TeamId,
        tx: &Sender<MatchEvent>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let (lambda, mu) = self.model.expected_goals(home, away, true);
        let p_home_goal = (lambda / 90.0).min(1.0);
        let p_away_goal = (mu / 90.0).min(1.0);
        const P_RED_CARD: f64 = 0.0004; // per team per minute

        let mut rng = StdRng::seed_from_u64(self.seed ^ (u64::from(match_id.0) << 8));
        let mut score = Scoreline::new(0, 0);

        send(tx, MatchEvent::new(match_id, 0, EventKind::KickOff)).await?;

        for minute in 1..=90u16 {
            if cancel.is_cancelled() {
                return Ok(());
            }
            if rng.gen_bool(p_home_goal) {
                score.home += 1;
                send(
                    tx,
                    MatchEvent::new(
                        match_id,
                        minute,
                        EventKind::Goal {
                            team: home,
                            scorer: None,
                        },
                    ),
                )
                .await?;
            }
            if rng.gen_bool(p_away_goal) {
                score.away += 1;
                send(
                    tx,
                    MatchEvent::new(
                        match_id,
                        minute,
                        EventKind::Goal {
                            team: away,
                            scorer: None,
                        },
                    ),
                )
                .await?;
            }
            if rng.gen_bool(P_RED_CARD) {
                let team = if rng.gen_bool(0.5) { home } else { away };
                send(
                    tx,
                    MatchEvent::new(match_id, minute, EventKind::RedCard { team }),
                )
                .await?;
            } else if minute % self.tick_every == 0 {
                send(tx, MatchEvent::new(match_id, minute, EventKind::Tick)).await?;
            }

            // Wait one simulated minute, or bail out promptly on cancellation.
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = sleep(self.minute_delay) => {}
            }
        }

        send(
            tx,
            MatchEvent::new(match_id, 90, EventKind::FullTime { score }),
        )
        .await?;
        Ok(())
    }
}

impl Default for SimProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataProvider for SimProvider {
    fn name(&self) -> &'static str {
        "simulation"
    }

    async fn load_tournament(&self) -> Result<Tournament> {
        Ok(self.tournament.clone())
    }

    async fn run(&self, tx: Sender<MatchEvent>, cancel: CancellationToken) -> Result<()> {
        // Play group-stage matches in scheduled order.
        let mut fixtures: Vec<_> = self
            .tournament
            .matches
            .iter()
            .filter(|m| matches!(m.stage, Stage::Group(_)))
            .collect();
        fixtures.sort_by_key(|m| m.kickoff);

        for m in fixtures {
            if cancel.is_cancelled() {
                break;
            }
            self.play_match(m.id, m.home, m.away, &tx, &cancel).await?;
        }
        Ok(())
    }
}

/// Send an event, mapping a closed channel to a typed error.
async fn send(tx: &Sender<MatchEvent>, event: MatchEvent) -> Result<()> {
    tx.send(event).await.map_err(|_| IngestError::ChannelClosed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_tournament_returns_full_world_cup() {
        let p = SimProvider::new();
        let t = p.load_tournament().await.unwrap();
        assert_eq!(t.teams.len(), 48);
    }

    #[tokio::test]
    async fn run_emits_kickoff_then_events_and_stops_on_cancel() {
        let p = SimProvider::new().with_minute_delay(Duration::from_millis(0));
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let cancel = CancellationToken::new();
        let handle = {
            let cancel = cancel.clone();
            tokio::spawn(async move { p.run(tx, cancel).await })
        };

        // First event of the feed must be a kickoff.
        let first = rx.recv().await.expect("at least one event");
        assert!(matches!(first.kind, EventKind::KickOff));

        // Drain a few more, then cancel and ensure the task winds down.
        for _ in 0..20 {
            if rx.recv().await.is_none() {
                break;
            }
        }
        cancel.cancel();
        while rx.recv().await.is_some() {}
        let _ = handle.await.unwrap();
    }
}
