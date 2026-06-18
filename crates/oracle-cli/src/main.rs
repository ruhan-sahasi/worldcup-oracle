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
    /// Backtest the model and benchmark it against the bookmaker.
    Backtest {
        /// Total synthetic matches when no --data file is given.
        #[arg(long, default_value_t = 4000)]
        matches: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Path to a football-data.co.uk style results CSV (real data + odds). When set,
        /// overrides the synthetic dataset.
        #[arg(long)]
        data: Option<std::path::PathBuf>,
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
        Command::Backtest {
            matches,
            seed,
            data,
        } => cmd_backtest(matches, seed, data),
        Command::Serve { addr } => cmd_serve(addr).await,
        Command::Watch { speed } => watch::run(speed).await,
    }
}

/// Fit the baseline goal model, strength-seeded Elo store, and the learned ensemble.
fn baseline() -> (GoalModel, RatingStore, Ensemble) {
    let b = data::fit_baseline(7);
    let mut ratings = RatingStore::with_defaults();
    for (team, rating) in &b.elo_seeds {
        ratings.seed(*team, *rating);
    }
    (b.model, ratings, b.ensemble)
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
        "Simulating {} - {} iterations (seed {})...\n",
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

    let (model, ratings, ensemble) = baseline();
    let grid = model.score_grid(home, away, true);
    let dc = grid.outcome_probabilities();
    let elo = ratings.win_probabilities(home, away, true);
    let blended = ensemble.blend(&[dc, elo]);
    let (lambda, mu) = model.expected_goals(home, away, true);

    println!("\n  {}  vs  {}   (neutral venue)\n", name(home), name(away));
    println!(
        "  Ensemble :  {:<11} {:>5.1}%    Draw {:>5.1}%    {:<11} {:>5.1}%",
        name(home),
        blended.home_win * 100.0,
        blended.draw * 100.0,
        name(away),
        blended.away_win * 100.0,
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

fn cmd_backtest(
    n_matches: usize,
    seed: u64,
    data_path: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    use oracle_model::{score, DixonColesConfig, Observation};

    // Real CSV when --data is given, else synthetic history with a synthetic market line.
    let (mut records, source, real_data) = match &data_path {
        Some(p) => (data::load_results_csv(p)?, format!("{}", p.display()), true),
        None => (
            data::synthetic_history_with_market(n_matches, seed),
            format!("{n_matches} synthetic matches"),
            false,
        ),
    };
    if records.len() < 50 {
        anyhow::bail!(
            "need at least 50 matches to backtest (got {})",
            records.len()
        );
    }

    // Three-way temporal split: fit on the oldest 60%, learn the ensemble weights on the
    // next 20% (validation), report on the most recent 20% (test).
    records.sort_by(|a, b| {
        b.obs
            .age_days
            .partial_cmp(&a.obs.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = records.len();
    let train_end = n * 6 / 10;
    let val_end = n * 8 / 10;
    let train_obs: Vec<Observation> = records[..train_end].iter().map(|r| r.obs).collect();
    let validation = &records[train_end..val_end];
    let test = &records[val_end..];

    // Fit the goal model and build Elo by replaying the training results. For synthetic
    // data we can also seed Elo from known strengths; real CSV teams are interned, so Elo
    // simply learns from the training matches.
    let model = GoalModel::fit(&train_obs, DixonColesConfig::default());
    let mut ratings = RatingStore::with_defaults();
    if !real_data {
        for (team, rating) in data::team_strengths() {
            ratings.seed(team, rating);
        }
    }
    for r in &records[..train_end] {
        ratings.record(r.obs.home, r.obs.away, r.obs.score, true);
    }

    // Learn the ensemble weights + temperature on the validation split.
    let mut val_preds = Vec::new();
    let mut val_actuals = Vec::new();
    for r in validation {
        val_preds.push(vec![
            model.outcome_probabilities(r.obs.home, r.obs.away, true),
            ratings.win_probabilities(r.obs.home, r.obs.away, true),
        ]);
        val_actuals.push(r.obs.score.outcome());
    }
    let ensemble = Ensemble::fit(&val_preds, &val_actuals, 2);

    // Evaluate everything (incl. the bookmaker) on the held-out test split.
    let mut dc_preds = Vec::new();
    let mut elo_preds = Vec::new();
    let mut ens_preds = Vec::new();
    let mut market_preds = Vec::new();
    for r in test {
        let actual = r.obs.score.outcome();
        let dc = model.outcome_probabilities(r.obs.home, r.obs.away, true);
        let elo = ratings.win_probabilities(r.obs.home, r.obs.away, true);
        dc_preds.push((dc, actual));
        elo_preds.push((elo, actual));
        ens_preds.push((ensemble.blend(&[dc, elo]), actual));
        if let Some(market) = r.market {
            market_preds.push((market, actual));
        }
    }

    println!(
        "\nBacktest on {}  (train {} / val {} / test {})\n",
        source,
        train_obs.len(),
        validation.len(),
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
    row(
        "Uniform baseline",
        oracle_model::CalibrationReport::uniform_baseline(test.len()),
    );
    row("Dixon-Coles", score(&dc_preds));
    row("Elo", score(&elo_preds));
    row("Ensemble (learned)", score(&ens_preds));
    if market_preds.is_empty() {
        println!("  Market (bookmaker)        (no odds in this dataset)");
    } else {
        row("Market (bookmaker)", score(&market_preds));
    }

    let wsum: f64 = ensemble.weights.iter().sum();
    println!(
        "\n  learned weights: Dixon-Coles {:.2} / Elo {:.2}   temperature {:.2}",
        ensemble.weights.first().copied().unwrap_or(0.0) / wsum,
        ensemble.weights.get(1).copied().unwrap_or(0.0) / wsum,
        ensemble.temperature,
    );
    println!("  (lower Brier / log-loss is better; the market is the bar to beat)\n");
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
