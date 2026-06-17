//! # wc-oracle
//!
//! The command-line front-end to the worldcup-oracle engine.
//!
//! ```text
//! wc-oracle simulate   # Monte-Carlo champion odds for the 2026 World Cup
//! wc-oracle predict    # one-off matchup prediction (ensemble + score grid)
//! wc-oracle backtest   # model calibration vs a naive baseline
//! wc-oracle serve      # run the REST + WebSocket server
//! wc-oracle watch      # live terminal dashboard (TUI)
//! ```
#![forbid(unsafe_code)]

mod watch;

use clap::{Parser, Subcommand};
use oracle_domain::{ScoreGrid, Team, TeamId};
use oracle_ingest::data;
use oracle_model::{Ensemble, GoalModel};
use oracle_ratings::RatingStore;
use oracle_sim::{simulate, SimConfig};
use std::net::SocketAddr;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "wc-oracle",
    version,
    about = "A live, ensemble World Cup prediction engine",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Monte-Carlo simulate the rest of the tournament and print champion odds.
    Simulate {
        /// Number of Monte-Carlo iterations.
        #[arg(long, default_value_t = 50_000)]
        iters: u64,
        /// RNG seed (fixed seed ⇒ reproducible output).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// How many teams to show.
        #[arg(long, default_value_t = 24)]
        top: usize,
    },
    /// Predict a single matchup (pre-match ensemble + exact-score grid).
    Predict {
        /// Home team (name or FIFA code, case-insensitive).
        #[arg(long)]
        home: String,
        /// Away team (name or FIFA code, case-insensitive).
        #[arg(long)]
        away: String,
    },
    /// Backtest the model on synthetic history and report calibration metrics.
    Backtest {
        /// Total synthetic matches (split 80/20 train/test by recency).
        #[arg(long, default_value_t = 4000)]
        matches: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Run the REST + WebSocket server.
    Serve {
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: SocketAddr,
    },
    /// Live terminal dashboard following a simulated tournament.
    Watch {
        /// Wall-clock milliseconds per simulated match-minute (lower = faster).
        #[arg(long, default_value_t = 40)]
        speed: u64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Simulate { iters, seed, top } => cmd_simulate(iters, seed, top),
        Command::Predict { home, away } => cmd_predict(&home, &away),
        Command::Backtest { matches, seed } => cmd_backtest(matches, seed),
        Command::Serve { addr } => cmd_serve(addr).await,
        Command::Watch { speed } => watch::run(speed).await,
    }
}

/// Fit the baseline goal model and strength-seeded Elo store.
fn baseline() -> (GoalModel, RatingStore) {
    let model = data::fit_baseline_model(7);
    let mut ratings = RatingStore::with_defaults();
    for (team, rating) in data::team_strengths() {
        ratings.seed(team, rating);
    }
    (model, ratings)
}

fn resolve_team(query: &str, teams: &[Team]) -> Option<TeamId> {
    let q = query.trim().to_lowercase();
    teams
        .iter()
        .find(|t| t.code.to_lowercase() == q || t.name.to_lowercase() == q)
        .or_else(|| teams.iter().find(|t| t.name.to_lowercase().contains(&q)))
        .map(|t| t.id)
}

fn cmd_simulate(iters: u64, seed: u64, top: usize) -> anyhow::Result<()> {
    let tournament = data::world_cup_2026();
    let model = data::fit_baseline_model(seed);
    let names: std::collections::HashMap<_, _> = tournament
        .teams
        .iter()
        .map(|t| (t.id, t.name.clone()))
        .collect();

    println!(
        "Simulating {} — {} iterations (seed {})…\n",
        tournament.name, iters, seed
    );
    let start = Instant::now();
    let forecast = simulate(
        &tournament,
        &model,
        SimConfig {
            iterations: iters,
            seed,
            ..SimConfig::default()
        },
    );
    let elapsed = start.elapsed();

    println!(
        "{:>3}  {:<16} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "#", "Team", "Champ", "Final", "Semi", "Quart", "R16"
    );
    println!("{}", "-".repeat(68));
    for (i, t) in forecast.ranked().into_iter().take(top).enumerate() {
        let name = names.get(&t.team).cloned().unwrap_or_default();
        println!(
            "{:>3}  {:<16} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            i + 1,
            name,
            t.p_champion * 100.0,
            t.p_final * 100.0,
            t.p_semi_final * 100.0,
            t.p_quarter_final * 100.0,
            t.p_round_of_16 * 100.0,
        );
    }
    println!(
        "\n{} simulations in {:.2}s  ({:.0} tournaments/sec)",
        iters,
        elapsed.as_secs_f64(),
        iters as f64 / elapsed.as_secs_f64().max(1e-9),
    );
    Ok(())
}

fn cmd_predict(home_q: &str, away_q: &str) -> anyhow::Result<()> {
    let teams = data::teams();
    let home =
        resolve_team(home_q, &teams).ok_or_else(|| anyhow::anyhow!("unknown team: {home_q}"))?;
    let away =
        resolve_team(away_q, &teams).ok_or_else(|| anyhow::anyhow!("unknown team: {away_q}"))?;
    let name = |id: TeamId| teams.iter().find(|t| t.id == id).unwrap().name.clone();

    let (model, ratings) = baseline();
    let grid = model.score_grid(home, away, true);
    let dc = grid.outcome_probabilities();
    let elo = ratings.win_probabilities(home, away, true);
    let ensemble = Ensemble::default().blend(&[dc, elo]);
    let (lambda, mu) = model.expected_goals(home, away, true);

    println!("\n  {}  vs  {}   (neutral venue)\n", name(home), name(away));
    println!(
        "  Ensemble :  {:<11} {:>5.1}%    Draw {:>5.1}%    {:<11} {:>5.1}%",
        name(home),
        ensemble.home_win * 100.0,
        ensemble.draw * 100.0,
        name(away),
        ensemble.away_win * 100.0,
    );
    println!(
        "    Dixon-Coles:  {:>5.1}% / {:>4.1}% / {:>5.1}%      Elo:  {:>5.1}% / {:>4.1}% / {:>5.1}%",
        dc.home_win * 100.0,
        dc.draw * 100.0,
        dc.away_win * 100.0,
        elo.home_win * 100.0,
        elo.draw * 100.0,
        elo.away_win * 100.0,
    );
    println!("\n  Expected goals : {lambda:.2} – {mu:.2}");
    let (mh, ma, mp) = grid.most_likely_score();
    println!("  Most likely    : {mh}–{ma}  ({:.1}%)", mp * 100.0);
    println!(
        "  Over 2.5 goals : {:.1}%      Both teams to score : {:.1}%",
        grid.prob_over(2.5) * 100.0,
        grid.prob_btts() * 100.0,
    );
    println!("\n  Top scorelines:");
    for (h, a, p) in top_scorelines(&grid, 5) {
        println!("    {h}–{a}   {:>5.1}%", p * 100.0);
    }
    println!();
    Ok(())
}

fn top_scorelines(grid: &ScoreGrid, n: usize) -> Vec<(usize, usize, f64)> {
    let mut cells: Vec<(usize, usize, f64)> = grid
        .grid
        .iter()
        .enumerate()
        .flat_map(|(h, row)| row.iter().enumerate().map(move |(a, &p)| (h, a, p)))
        .collect();
    cells.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
    cells.truncate(n);
    cells
}

fn cmd_backtest(n_matches: usize, seed: u64) -> anyhow::Result<()> {
    use oracle_model::{score, DixonColesConfig};

    let mut history = data::synthetic_history(n_matches, seed);
    // Temporal split: train on the older 80%, test on the most recent 20%.
    history.sort_by(|a, b| {
        b.age_days
            .partial_cmp(&a.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let split = history.len() * 4 / 5;
    let (train, test) = history.split_at(split);

    // Fit the goal model, and build Elo by replaying the training results.
    let model = GoalModel::fit(train, DixonColesConfig::default());
    let mut ratings = RatingStore::with_defaults();
    for (team, rating) in data::team_strengths() {
        ratings.seed(team, rating);
    }
    for obs in train.iter().rev() {
        ratings.record(obs.home, obs.away, obs.score, true);
    }

    let ensemble = Ensemble::default();
    let mut dc_preds = Vec::new();
    let mut ens_preds = Vec::new();
    for obs in test {
        let actual = obs.score.outcome();
        let dc = model.outcome_probabilities(obs.home, obs.away, true);
        let elo = ratings.win_probabilities(obs.home, obs.away, true);
        dc_preds.push((dc, actual));
        ens_preds.push((ensemble.blend(&[dc, elo]), actual));
    }

    let baseline = oracle_model::CalibrationReport::uniform_baseline(test.len());
    let dc_report = score(&dc_preds);
    let ens_report = score(&ens_preds);

    println!(
        "\nBacktest on {} synthetic matches  (train {} / test {})\n",
        n_matches,
        train.len(),
        test.len()
    );
    println!(
        "  {:<20} {:>8} {:>9} {:>8}",
        "Model", "Brier", "LogLoss", "Acc"
    );
    println!("  {}", "-".repeat(48));
    let row = |label: &str, r: oracle_model::CalibrationReport| {
        println!(
            "  {:<20} {:>8.4} {:>9.4} {:>7.1}%",
            label,
            r.brier,
            r.log_loss,
            r.accuracy * 100.0
        );
    };
    row("Uniform baseline", baseline);
    row("Dixon-Coles", dc_report);
    row("Ensemble (+Elo)", ens_report);
    println!("\n  (lower Brier / log-loss is better)\n");
    Ok(())
}

async fn cmd_serve(addr: SocketAddr) -> anyhow::Result<()> {
    use tokio_util::sync::CancellationToken;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cancel = CancellationToken::new();
    let (engine, join) = oracle_engine::spawn(
        oracle_engine::presets::auto(),
        oracle_engine::EngineConfig::default(),
        cancel.clone(),
    )
    .await?;

    println!("worldcup-oracle serving on http://{addr}  (Ctrl-C to stop)");
    let shutdown_cancel = cancel.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_cancel.cancel();
    };
    oracle_api::serve(engine, addr, shutdown).await?;
    cancel.cancel();
    let _ = join.await;
    Ok(())
}
