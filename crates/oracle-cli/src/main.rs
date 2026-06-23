//! # wc-oracle
//!
//! The command-line front-end to the worldcup-oracle engine.
//!
//! ```text
//! wc-oracle simulate   # Monte-Carlo champion odds for the 2026 World Cup
//! wc-oracle predict    # one-off matchup prediction (ensemble + score grid)
//! wc-oracle backtest   # calibration + bookmaker benchmark (real or synthetic data)
//! wc-oracle tune       # search goal-model hyperparameters by held-out log-loss
//! wc-oracle serve      # run the REST + WebSocket server
//! wc-oracle watch      # live terminal dashboard (TUI)
//! ```
#![forbid(unsafe_code)]

mod watch;

use clap::{Parser, Subcommand};
use oracle_domain::{ScoreGrid, Team, TeamId};
use oracle_ingest::data;
use oracle_model::{Ensemble, GoalModel, LiveConfig};
use oracle_ratings::RatingStore;
use oracle_sim::{simulate_with_live, LiveInputs, SimConfig};
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
        /// Optional bookmaker decimal odds. Supply all three to anchor the ensemble to
        /// the market line.
        #[arg(long)]
        home_odds: Option<f64>,
        #[arg(long)]
        draw_odds: Option<f64>,
        #[arg(long)]
        away_odds: Option<f64>,
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
    /// Search goal-model hyperparameters, picking the best by held-out log-loss.
    Tune {
        /// Total synthetic matches when no --data file is given.
        #[arg(long, default_value_t = 4000)]
        matches: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Path to a football-data.co.uk style results CSV (tune on real data + odds).
        #[arg(long)]
        data: Option<std::path::PathBuf>,
    },
    /// Run the REST + WebSocket server.
    Serve {
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: SocketAddr,
        /// Append-only event log path. Events are recorded here and replayed on restart
        /// to recover state.
        #[arg(long)]
        event_log: Option<std::path::PathBuf>,
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
        Command::Predict {
            home,
            away,
            home_odds,
            draw_odds,
            away_odds,
        } => cmd_predict(&home, &away, (home_odds, draw_odds, away_odds)),
        Command::Backtest {
            matches,
            seed,
            data,
        } => cmd_backtest(matches, seed, data),
        Command::Tune {
            matches,
            seed,
            data,
        } => cmd_tune(matches, seed, data),
        Command::Serve { addr, event_log } => cmd_serve(addr, event_log).await,
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
    // Apply the venue/travel context (host advantage, altitude, rest) to every fixture.
    let inputs = LiveInputs {
        venue: data::venue_adjustments(&tournament),
        ..Default::default()
    };
    let start = Instant::now();
    let forecast = simulate_with_live(
        &tournament,
        &model,
        SimConfig {
            iterations: iters,
            seed,
            ..SimConfig::default()
        },
        &inputs,
        LiveConfig::default(),
    );
    let elapsed = start.elapsed();

    // Monte-Carlo standard error on a probability from N iterations: sqrt(p(1-p)/N).
    let n = forecast.iterations.max(1) as f64;
    let stderr = |p: f64| (p * (1.0 - p) / n).sqrt();

    println!(
        "{:>3}  {:<16} {:>15} {:>8} {:>8} {:>8} {:>8}",
        "#", "Team", "Champ (±MC err)", "Final", "Semi", "Quart", "R16"
    );
    println!("{}", "-".repeat(74));
    for (i, t) in forecast.ranked().into_iter().take(top).enumerate() {
        let name = names.get(&t.team).cloned().unwrap_or_default();
        println!(
            "{:>3}  {:<16} {:>6.1}% ±{:>4.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            i + 1,
            name,
            t.p_champion * 100.0,
            stderr(t.p_champion) * 100.0,
            t.p_final * 100.0,
            t.p_semi_final * 100.0,
            t.p_quarter_final * 100.0,
            t.p_round_of_16 * 100.0,
        );
    }
    println!(
        "\n{} simulations in {:.2}s  ({:.0} tournaments/sec); ±MC err is the Monte-Carlo \
         standard error on the champion probability",
        iters,
        elapsed.as_secs_f64(),
        iters as f64 / elapsed.as_secs_f64().max(1e-9),
    );
    Ok(())
}

fn cmd_predict(
    home_q: &str,
    away_q: &str,
    odds: (Option<f64>, Option<f64>, Option<f64>),
) -> anyhow::Result<()> {
    let teams = data::teams();
    let home =
        resolve_team(home_q, &teams).ok_or_else(|| anyhow::anyhow!("unknown team: {home_q}"))?;
    let away =
        resolve_team(away_q, &teams).ok_or_else(|| anyhow::anyhow!("unknown team: {away_q}"))?;
    let name = |id: TeamId| {
        teams
            .iter()
            .find(|t| t.id == id)
            .expect("team id was resolved from this list")
            .name
            .clone()
    };

    let (model, ratings, ensemble) = baseline();
    let grid = model.score_grid(home, away, true);
    let dc = grid.outcome_probabilities();
    let elo = ratings.win_probabilities(home, away, true);
    // Anchor to the market when all three odds are supplied.
    let market = match odds {
        (Some(h), Some(d), Some(a)) => Some(oracle_model::implied_probabilities(h, d, a)),
        _ => None,
    };
    let mut members = vec![dc, elo];
    if let Some(m) = market {
        members.push(m);
    }
    let blended = ensemble.blend(&members);
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
    if let Some(m) = market {
        println!(
            "    Market     :  {:>5.1}% / {:>4.1}% / {:>5.1}%   (vig removed; anchored into the ensemble)",
            m.home_win * 100.0,
            m.draw * 100.0,
            m.away_win * 100.0,
        );
    }
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

    let xg_present = train_obs.iter().any(|o| o.home_xg.is_some());
    let market_present = records.iter().any(|r| r.market.is_some());
    // Real club fixtures have a genuine home venue; the synthetic World Cup is neutral.
    let neutral = !real_data;

    // Fit the goal model on xG when present (sharper), and build Elo by replaying training
    // results. For synthetic data we can also seed Elo from known strengths; real CSV teams
    // are interned, so Elo simply learns from the training matches.
    let model = GoalModel::fit(&train_obs, DixonColesConfig::default());
    // A goals-only refit for comparison, to show the xG lever explicitly.
    let model_goals = xg_present.then(|| {
        let stripped: Vec<Observation> = train_obs
            .iter()
            .map(|o| Observation::new(o.home, o.away, o.score, o.age_days))
            .collect();
        GoalModel::fit(&stripped, DixonColesConfig::default())
    });
    let mut ratings = RatingStore::with_defaults();
    if !real_data {
        for (team, rating) in data::team_strengths() {
            ratings.seed(team, rating);
        }
    }
    for r in &records[..train_end] {
        ratings.record(r.obs.home, r.obs.away, r.obs.score, neutral);
    }

    // Learn the ensemble on validation. Include the market as a third member when odds are
    // available, so the ensemble can anchor to the sharpest signal.
    let mut val_preds = Vec::new();
    let mut val_actuals = Vec::new();
    for r in validation {
        let dc = model.outcome_probabilities(r.obs.home, r.obs.away, neutral);
        let elo = ratings.win_probabilities(r.obs.home, r.obs.away, neutral);
        if market_present {
            if let Some(m) = r.market {
                val_preds.push(vec![dc, elo, m]);
                val_actuals.push(r.obs.score.outcome());
            }
        } else {
            val_preds.push(vec![dc, elo]);
            val_actuals.push(r.obs.score.outcome());
        }
    }
    let n_members = if market_present { 3 } else { 2 };
    let ensemble = Ensemble::fit(&val_preds, &val_actuals, n_members);

    // Evaluate everything (incl. the bookmaker) on the held-out test split.
    let mut dc_preds = Vec::new();
    let mut dc_goals_preds = Vec::new();
    let mut elo_preds = Vec::new();
    let mut ens_preds = Vec::new();
    let mut market_preds = Vec::new();
    for r in test {
        let actual = r.obs.score.outcome();
        let dc = model.outcome_probabilities(r.obs.home, r.obs.away, neutral);
        let elo = ratings.win_probabilities(r.obs.home, r.obs.away, neutral);
        dc_preds.push((dc, actual));
        elo_preds.push((elo, actual));
        if let Some(mg) = &model_goals {
            dc_goals_preds.push((
                mg.outcome_probabilities(r.obs.home, r.obs.away, neutral),
                actual,
            ));
        }
        let mut members = vec![dc, elo];
        if let Some(market) = r.market {
            members.push(market);
            market_preds.push((market, actual));
        }
        ens_preds.push((ensemble.blend(&members), actual));
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
    if !dc_goals_preds.is_empty() {
        row("Dixon-Coles (goals)", score(&dc_goals_preds));
    }
    row(
        if xg_present {
            "Dixon-Coles (xG)"
        } else {
            "Dixon-Coles"
        },
        score(&dc_preds),
    );
    row("Elo", score(&elo_preds));
    row(
        if market_present {
            "Ensemble (+Market)"
        } else {
            "Ensemble (DC+Elo)"
        },
        score(&ens_preds),
    );
    if market_preds.is_empty() {
        println!("  Market (bookmaker)        (no odds in this dataset)");
    } else {
        row("Market (bookmaker)", score(&market_preds));
    }

    let wsum: f64 = ensemble.weights.iter().sum::<f64>().max(1e-9);
    let w = |i: usize| ensemble.weights.get(i).copied().unwrap_or(0.0) / wsum;
    if market_present {
        println!(
            "\n  learned weights: DC {:.2} / Elo {:.2} / Market {:.2}   temperature {:.2}",
            w(0),
            w(1),
            w(2),
            ensemble.temperature,
        );
    } else {
        println!(
            "\n  learned weights: DC {:.2} / Elo {:.2}   temperature {:.2}",
            w(0),
            w(1),
            ensemble.temperature,
        );
    }
    println!("  (lower Brier / log-loss is better; the market is the bar to beat)");

    // Reliability (calibration) of the learned ensemble: in each predicted-probability
    // bucket, how often did the outcome actually happen?
    let rel = oracle_model::reliability(&ens_preds, 5);
    println!("\n  Ensemble calibration (ECE {:.3}):", rel.ece);
    println!(
        "    {:>12}   {:>9}   {:>9}   {:>6}",
        "bucket", "predicted", "empirical", "n"
    );
    for b in &rel.bins {
        if b.count == 0 {
            continue;
        }
        println!(
            "    {:>4.0}-{:<3.0}%      {:>7.1}%   {:>7.1}%   {:>6}",
            b.lo * 100.0,
            b.hi * 100.0,
            b.mean_pred * 100.0,
            b.empirical * 100.0,
            b.count,
        );
    }
    println!("    (predicted ≈ empirical in every bucket means well-calibrated)\n");
    Ok(())
}

fn cmd_tune(
    n_matches: usize,
    seed: u64,
    data_path: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    use oracle_model::{score, DixonColesConfig, GoalModel, Observation, ScoreModel};

    let (mut records, source, real_data) = match &data_path {
        Some(p) => (data::load_results_csv(p)?, format!("{}", p.display()), true),
        None => (
            data::synthetic_history_with_market(n_matches, seed),
            format!("{n_matches} synthetic matches"),
            false,
        ),
    };
    if records.len() < 100 {
        anyhow::bail!("need at least 100 matches to tune (got {})", records.len());
    }
    let neutral = !real_data;

    // Temporal split: fit on the oldest 60%, select hyperparameters on the next 20%
    // (validation), and report the winner's honest loss on the most recent 20% (test).
    records.sort_by(|a, b| {
        b.obs
            .age_days
            .partial_cmp(&a.obs.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = records.len();
    let train: Vec<Observation> = records[..n * 6 / 10].iter().map(|r| r.obs).collect();
    let validation: Vec<Observation> = records[n * 6 / 10..n * 8 / 10]
        .iter()
        .map(|r| r.obs)
        .collect();
    let test: Vec<Observation> = records[n * 8 / 10..].iter().map(|r| r.obs).collect();

    // Log-loss of a config's Dixon-Coles model fit on `train`, scored on `eval`.
    let evaluate = |cfg: DixonColesConfig, eval: &[Observation]| -> f64 {
        let model = GoalModel::fit(&train, cfg);
        let preds: Vec<_> = eval
            .iter()
            .map(|o| {
                (
                    model.outcome_probabilities(o.home, o.away, neutral),
                    o.score.outcome(),
                )
            })
            .collect();
        score(&preds).log_loss
    };

    // The hand-picked constants we are replacing with searched values.
    const XIS: &[f64] = &[0.001, 0.002, 0.003, 0.005, 0.008];
    const RIDGES: &[f64] = &[0.0, 0.005, 0.01, 0.02, 0.05];
    const MODELS: &[(ScoreModel, &str)] = &[
        (ScoreModel::Independent, "independent"),
        (ScoreModel::Bivariate, "bivariate"),
    ];

    let base = DixonColesConfig::default();
    let mut best = (base, evaluate(base, &validation));
    let mut evaluated = 0usize;
    for &xi in XIS {
        for &ridge in RIDGES {
            for &(model, _) in MODELS {
                let cfg = DixonColesConfig {
                    xi,
                    ridge,
                    model,
                    ..base
                };
                let ll = evaluate(cfg, &validation);
                evaluated += 1;
                if ll < best.1 {
                    best = (cfg, ll);
                }
            }
        }
    }

    let model_name = |m: ScoreModel| match m {
        ScoreModel::Independent => "independent",
        ScoreModel::Bivariate => "bivariate",
    };
    println!(
        "\nTuning on {}  (train {} / validation {} / test {}); {} configs searched\n",
        source,
        train.len(),
        validation.len(),
        test.len(),
        evaluated
    );
    println!(
        "  {:<26} {:>12} {:>12}",
        "Config", "val logloss", "test logloss"
    );
    println!("  {}", "-".repeat(52));
    println!(
        "  {:<26} {:>12.4} {:>12.4}",
        format!(
            "default (xi {:.3}, ridge {:.3}, {})",
            base.xi,
            base.ridge,
            model_name(base.model)
        ),
        evaluate(base, &validation),
        evaluate(base, &test),
    );
    println!(
        "  {:<26} {:>12.4} {:>12.4}",
        format!(
            "tuned   (xi {:.3}, ridge {:.3}, {})",
            best.0.xi,
            best.0.ridge,
            model_name(best.0.model)
        ),
        best.1,
        evaluate(best.0, &test),
    );
    println!(
        "\n  Searched goal-model hyperparameters (xi, ridge, score model) by held-out\n  \
         log-loss. Lower is better; the test column is the honest out-of-sample number.\n"
    );
    Ok(())
}

async fn cmd_serve(addr: SocketAddr, event_log: Option<std::path::PathBuf>) -> anyhow::Result<()> {
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
        oracle_engine::EngineConfig {
            event_log,
            ..Default::default()
        },
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

#[cfg(test)]
mod tests {
    use super::{resolve_team, top_scorelines};
    use oracle_domain::ScoreGrid;
    use oracle_ingest::data;

    #[test]
    fn resolve_team_matches_name_code_case_and_substring() {
        let teams = data::teams();
        let brazil = resolve_team("Brazil", &teams).expect("full name");
        assert_eq!(resolve_team("BRA", &teams), Some(brazil), "FIFA code");
        assert_eq!(
            resolve_team("brazil", &teams),
            Some(brazil),
            "case-insensitive"
        );
        assert_eq!(resolve_team("  brazil  ", &teams), Some(brazil), "trimmed");
        // Substring fallback (no exact name/code match).
        let usa = resolve_team("United States", &teams);
        assert_eq!(resolve_team("United", &teams), usa, "substring match");
        assert_eq!(resolve_team("Atlantis", &teams), None, "unknown team");
    }

    #[test]
    fn top_scorelines_are_ranked_and_truncated() {
        // A grid whose single most likely cell is 2-1.
        let grid = ScoreGrid::from_fn(4, |h, a| if (h, a) == (2, 1) { 10.0 } else { 1.0 });
        let top = top_scorelines(&grid, 3);
        assert_eq!(top.len(), 3, "truncated to n");
        assert_eq!((top[0].0, top[0].1), (2, 1), "modal scoreline first");
        // Sorted by probability, descending.
        assert!(top[0].2 >= top[1].2 && top[1].2 >= top[2].2);
    }
}
