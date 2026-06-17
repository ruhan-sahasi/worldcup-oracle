//! `wc-oracle watch` — a live terminal dashboard.
//!
//! Boots the engine over the simulation feed in-process, then renders a [`ratatui`]
//! dashboard that refreshes ~8×/second from the engine's lock-free snapshot: live
//! matches with shifting win/draw/win numbers on the left, the Monte-Carlo champion
//! odds (as bars) on the right. Press `q` to quit.

use crossterm::event::{self, Event, KeyCode};
use oracle_domain::{MatchStatus, TeamId};
use oracle_engine::{presets, Engine, EngineConfig, Snapshot};
use ratatui::prelude::*;
use ratatui::widgets::{Block, List, ListItem, Paragraph};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub async fn run(speed_ms: u64) -> anyhow::Result<()> {
    let cancel = CancellationToken::new();
    let (engine, join) = oracle_engine::spawn(
        presets::simulated_with_speed(Duration::from_millis(speed_ms)),
        EngineConfig::default(),
        cancel.clone(),
    )
    .await?;

    let mut terminal = ratatui::init();
    let result = render_loop(&mut terminal, &engine).await;
    ratatui::restore();

    cancel.cancel();
    let _ = join.await;
    result
}

async fn render_loop(
    terminal: &mut ratatui::DefaultTerminal,
    engine: &Engine,
) -> anyhow::Result<()> {
    loop {
        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                if matches!(
                    key.code,
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc
                ) {
                    break;
                }
            }
        }
        let snapshot = engine.snapshot();
        terminal.draw(|frame| ui(frame, &snapshot, engine))?;
    }
    Ok(())
}

fn ui(frame: &mut Frame, snap: &Snapshot, engine: &Engine) {
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(frame.area());

    let m = engine.metrics();
    let health = if snap.source_healthy {
        String::new()
    } else {
        "    ⚠ STALE FEED".to_string()
    };
    let header = Paragraph::new(format!(
        " {}    provider: {}    events: {}    goals: {}    updated: {}{}",
        snap.tournament,
        snap.provider,
        m.events_processed.load(Ordering::Relaxed),
        m.goals_seen.load(Ordering::Relaxed),
        snap.generated_at.format("%H:%M:%S"),
        health,
    ))
    .block(Block::bordered().title(" worldcup-oracle · LIVE "));
    frame.render_widget(header, chunks[0]);

    let body = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[1]);
    render_live(frame, body[0], snap);
    render_odds(frame, body[1], snap);

    let footer =
        Paragraph::new(" press q to quit ").style(Style::default().add_modifier(Modifier::DIM));
    frame.render_widget(footer, chunks[2]);
}

fn render_live(frame: &mut Frame, area: Rect, snap: &Snapshot) {
    let mut items: Vec<ListItem> = snap
        .live_matches()
        .map(|m| {
            let p = &m.probabilities;
            ListItem::new(format!(
                "{:>12} {}-{} {:<12} {:>3}'   W {:>3.0}%  D {:>3.0}%  L {:>3.0}%",
                truncate(&m.home_name, 12),
                m.score.home,
                m.score.away,
                truncate(&m.away_name, 12),
                m.minute,
                p.home_win * 100.0,
                p.draw * 100.0,
                p.away_win * 100.0,
            ))
        })
        .collect();

    if items.is_empty() {
        items.push(ListItem::new(
            "  (no matches in play — waiting for kickoff…)",
        ));
        let finished: Vec<_> = snap
            .matches
            .iter()
            .filter(|m| matches!(m.status, MatchStatus::Finished))
            .collect();
        for m in finished.iter().rev().take(8) {
            items.push(ListItem::new(format!(
                "  FT  {:>12} {}-{} {:<12}",
                truncate(&m.home_name, 12),
                m.score.home,
                m.score.away,
                truncate(&m.away_name, 12),
            )));
        }
    }

    let list = List::new(items).block(Block::bordered().title(" Live matches "));
    frame.render_widget(list, area);
}

fn render_odds(frame: &mut Frame, area: Rect, snap: &Snapshot) {
    let names: HashMap<TeamId, String> = snap
        .ratings
        .iter()
        .map(|r| (r.team, r.name.clone()))
        .collect();
    let ranked = snap.forecast.ranked();
    let max = ranked
        .first()
        .map(|t| t.p_champion)
        .unwrap_or(1.0)
        .max(1e-6);
    let rows = (area.height as usize).saturating_sub(2).min(ranked.len());

    let items: Vec<ListItem> = ranked
        .iter()
        .take(rows)
        .enumerate()
        .map(|(i, t)| {
            let name = names.get(&t.team).cloned().unwrap_or_default();
            let bar_len = ((t.p_champion / max) * 18.0).round() as usize;
            let bar = "█".repeat(bar_len);
            ListItem::new(format!(
                "{:>2} {:<13} {:<18} {:>5.1}%",
                i + 1,
                truncate(&name, 13),
                bar,
                t.p_champion * 100.0,
            ))
        })
        .collect();

    let list = List::new(items).block(Block::bordered().title(" Champion odds (Monte-Carlo) "));
    frame.render_widget(list, area);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}
