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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataProvider;
    use oracle_domain::{Confederation, Group, MatchStatus, Scoreline, Team, TeamId};
    use std::time::Duration;

    fn one_match_tournament() -> Tournament {
        let mut t = Tournament::new("Replay Test");
        t.teams = vec![
            Team::new(0, "Alpha", "ALP", Confederation::Uefa),
            Team::new(1, "Beta", "BET", Confederation::Uefa),
        ];
        t.groups.push(Group {
            name: 'A',
            teams: vec![TeamId(0), TeamId(1)],
        });
        t.matches.push(Match {
            id: MatchId(1),
            home: TeamId(0),
            away: TeamId(1),
            stage: Stage::Group('A'),
            kickoff: chrono::DateTime::from_timestamp(0, 0).unwrap(),
            status: MatchStatus::Finished,
            score: Scoreline::new(2, 1),
        });
        t
    }

    #[tokio::test]
    async fn replays_a_finished_match_into_an_event_stream() {
        let provider =
            ReplayProvider::new(one_match_tournament()).with_minute_delay(Duration::from_millis(0));
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let cancel = CancellationToken::new();
        provider.run(tx, cancel).await.unwrap();

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        // Opens with a health heartbeat, then a kickoff for the match.
        assert!(matches!(
            events[0].kind,
            EventKind::SourceStatus { healthy: true }
        ));
        assert!(events.iter().any(|e| matches!(e.kind, EventKind::KickOff)));

        // The three real goals (2-1) are all emitted.
        let goals = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::Goal { .. }))
            .count();
        assert_eq!(goals, 3, "two home + one away goal");

        // Ends with the authoritative full-time score.
        let ft = events
            .iter()
            .rev()
            .find_map(|e| match e.kind {
                EventKind::FullTime { score } => Some(score),
                _ => None,
            })
            .expect("a full-time event");
        assert_eq!(ft, Scoreline::new(2, 1));
    }
}
