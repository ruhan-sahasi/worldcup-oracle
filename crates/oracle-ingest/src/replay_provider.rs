//! Replays a *completed* tournament event-by-event.
//!
//! Where [`crate::SimProvider`] invents results, `ReplayProvider` takes a tournament
//! whose matches already have final scores - a past competition, or one captured
//! from the live API - and re-emits it as a live stream: kickoff, goals scattered
//! across plausible minutes that add up to the real scoreline, then full-time. Handy
//! for demos and for validating the engine against known outcomes.

use crate::error::{IngestError, Result};
use crate::provider::DataProvider;
use async_trait::async_trait;
use oracle_domain::{EventKind, Match, MatchEvent, MatchId, Stage, Tournament};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use tokio::sync::mpsc::Sender;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

/// Replays a tournament's finished results as a live event stream.
pub struct ReplayProvider {
    tournament: Tournament,
    minute_delay: Duration,
    seed: u64,
}

impl ReplayProvider {
    pub fn new(tournament: Tournament) -> Self {
        Self {
            tournament,
            minute_delay: Duration::from_millis(15),
            seed: 1,
        }
    }

    pub fn with_minute_delay(mut self, delay: Duration) -> Self {
        self.minute_delay = delay;
        self
    }

    async fn replay_match(
        &self,
        m: &Match,
        tx: &Sender<MatchEvent>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(self.seed ^ u64::from(m.id.0));

        // Assign each real goal a distinct-ish minute, then replay in order.
        let mut goals: Vec<(u16, bool)> = Vec::new(); // (minute, is_home)
        for _ in 0..m.score.home {
            goals.push((rng.gen_range(1..=90), true));
        }
        for _ in 0..m.score.away {
            goals.push((rng.gen_range(1..=90), false));
        }
        goals.sort_by_key(|(minute, _)| *minute);

        send(tx, MatchEvent::new(m.id, 0, EventKind::KickOff)).await?;
        let mut gi = 0;
        for minute in 1..=90u16 {
            if cancel.is_cancelled() {
                return Ok(());
            }
            while gi < goals.len() && goals[gi].0 == minute {
                let team = if goals[gi].1 { m.home } else { m.away };
                send(
                    tx,
                    MatchEvent::new(m.id, minute, EventKind::Goal { team, scorer: None }),
                )
                .await?;
                gi += 1;
            }
            if minute % 10 == 0 {
                send(tx, MatchEvent::new(m.id, minute, EventKind::Tick)).await?;
            }
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = sleep(self.minute_delay) => {}
            }
        }
        send(
            tx,
            MatchEvent::new(m.id, 90, EventKind::FullTime { score: m.score }),
        )
        .await?;
        Ok(())
    }
}

#[async_trait]
impl DataProvider for ReplayProvider {
    fn name(&self) -> &'static str {
        "replay"
    }

    async fn load_tournament(&self) -> Result<Tournament> {
        Ok(self.tournament.clone())
    }

    async fn run(&self, tx: Sender<MatchEvent>, cancel: CancellationToken) -> Result<()> {
        send(
            &tx,
            MatchEvent::new(MatchId(0), 0, EventKind::SourceStatus { healthy: true }),
        )
        .await?;
        let mut fixtures: Vec<Match> = self
            .tournament
            .matches
            .iter()
            .filter(|m| matches!(m.stage, Stage::Group(_)))
            .cloned()
            .collect();
        fixtures.sort_by_key(|m| m.kickoff);
        for m in &fixtures {
            if cancel.is_cancelled() {
                break;
            }
            self.replay_match(m, &tx, &cancel).await?;
        }
        Ok(())
    }
}

async fn send(tx: &Sender<MatchEvent>, event: MatchEvent) -> Result<()> {
    tx.send(event).await.map_err(|_| IngestError::ChannelClosed)
}
