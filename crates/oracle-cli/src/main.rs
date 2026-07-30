//! # wc-oracle
//!
//! The command-line front-end to the worldcup-oracle engine.
//!
//! ```text
//! wc-oracle simulate   # Monte-Carlo champion odds for the 2026 World Cup
//! wc-oracle predict    # one-off matchup prediction (ensemble + score grid)
//! wc-oracle backtest   # calibration + bookmaker benchmark (real or synthetic data)
//! wc-oracle market-backtest # paper-trade the model against the market (bankroll, ROI, edge)
//! wc-oracle scorers    # goalscorer market for a matchup (anytime / brace / hat-trick)
//! wc-oracle golden-boot # top-scorer race across the tournament
//! wc-oracle in-play    # in-play trading study (cash-out vs hold) for a matchup
//! wc-oracle derivatives # totals, Asian handicap, correct score, and more for a matchup
//! wc-oracle tune       # search goal-model hyperparameters by held-out log-loss
//! wc-oracle serve      # run the REST + WebSocket server
//! wc-oracle watch      # live terminal dashboard (TUI)
//! wc-oracle sensitivity # ablation: how much each unconventional signal moves the title odds
//! ```

mod watch;

use clap::{Parser, Subcommand};
use oracle_domain::{Outcome, Probabilities, Team, TeamId};
use oracle_engine::Explorer;
use oracle_ingest::{actual_2026, data};
use oracle_market::BetPolicy;
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
        /// Condition on the REAL 2026 field still alive at a stage and forecast forward from it
        /// (`round-of-16` | `quarter-final` | `semi-final` | `final`). Omit to simulate the whole
        /// synthetic tournament from scratch.
        #[arg(long)]
        stage: Option<String>,
        /// Target Monte-Carlo precision instead of a fixed iteration count: keep simulating until
        /// no team's champion probability has a standard error above this (e.g. `0.002` for
        /// ±0.2 pp). Overrides `--iters`, which becomes the ceiling.
        #[arg(long)]
        precision: Option<f64>,
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
        /// Also sample the goal model's posterior by HMC and print 90% credible intervals on the
        /// win/draw/win probabilities (the model's uncertainty about its own forecast).
        #[arg(long)]
        posterior: bool,
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
        /// Rolling-origin cross-validation with this many folds (>= 2), reporting each metric
        /// with a bootstrap 95% confidence interval instead of a single train/test split.
        #[arg(long)]
        cv: Option<usize>,
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
    /// Paper-trade the model against a synthetic bookmaker over a held-out season.
    MarketBacktest {
        /// Held-out season seed. Use a value other than the training seed (7) so bets are out of sample.
        #[arg(long, default_value_t = 101)]
        seed: u64,
        /// Number of matches in the held-out season.
        #[arg(long, default_value_t = 2000)]
        matches: usize,
        /// Starting bankroll.
        #[arg(long, default_value_t = 100.0)]
        bankroll: f64,
        /// Minimum edge required to back a side.
        #[arg(long, default_value_t = 0.02)]
        edge: f64,
        /// Fraction of full Kelly to stake.
        #[arg(long, default_value_t = 0.25)]
        kelly: f64,
        /// Cap on any single bet as a fraction of bankroll.
        #[arg(long, default_value_t = 0.05)]
        cap: f64,
        /// Bookmaker margin (overround) applied to the fair line.
        #[arg(long, default_value_t = 0.06)]
        margin: f64,
    },
    /// Goalscorer market for a matchup (anytime, brace, hat-trick per player).
    Scorers {
        /// Home team (name or FIFA code).
        #[arg(long)]
        home: String,
        /// Away team (name or FIFA code).
        #[arg(long)]
        away: String,
        /// How many players to show.
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
    /// Golden Boot race: each player's chance of finishing the tournament's top scorer.
    GoldenBoot {
        /// Monte-Carlo iterations.
        #[arg(long, default_value_t = 20_000)]
        iters: u32,
        /// RNG seed (fixed seed => reproducible output).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// How many contenders to show.
        #[arg(long, default_value_t = 20)]
        top: usize,
    },
    /// In-play trading study for a matchup: a live cash-out backtest versus holding to settlement.
    InPlay {
        /// Home team (name or FIFA code).
        #[arg(long)]
        home: String,
        /// Away team (name or FIFA code).
        #[arg(long)]
        away: String,
        /// Monte-Carlo matches to simulate.
        #[arg(long, default_value_t = 20_000)]
        iters: u32,
        /// RNG seed (fixed seed => reproducible output).
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Derivative markets for a matchup: totals, Asian handicap, correct score, and the side markets.
    Derivatives {
        /// Home team (name or FIFA code).
        #[arg(long)]
        home: String,
        /// Away team (name or FIFA code).
        #[arg(long)]
        away: String,
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
        /// Append-only forecast journal path. Each model's pre-match call is recorded the first time
        /// a match settles, and the track record is scored from those records rather than from
        /// forecasts the current model would recompute. Served at `/track-record`.
        #[arg(long)]
        forecast_journal: Option<std::path::PathBuf>,
    },
    /// Score a forecast journal: how the calls the engine actually published have held up.
    TrackRecord {
        /// Path to the append-only journal written by `serve --forecast-journal`.
        #[arg(long)]
        journal: std::path::PathBuf,
        /// Path to the event log written by `serve --event-log`. Results come from here, so without
        /// it there is nothing to settle the journaled calls against and the command reports only
        /// coverage.
        #[arg(long)]
        event_log: Option<std::path::PathBuf>,
        /// How many of the most recent calls to list. 0 lists none.
        #[arg(long, default_value_t = 10)]
        recent: usize,
    },
    /// Live terminal dashboard following a simulated tournament.
    Watch {
        /// Wall-clock milliseconds per simulated match-minute (lower = faster).
        #[arg(long, default_value_t = 40)]
        speed: u64,
    },
    /// Signal sensitivity: how much each unconventional signal moves the championship odds.
    Sensitivity {
        /// Monte-Carlo iterations per signal variant (nine variants plus a baseline are run).
        #[arg(long, default_value_t = 40_000)]
        iters: u64,
        /// RNG seed, shared across variants so the deltas isolate each signal from MC noise.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// How many top movers to list per signal.
        #[arg(long, default_value_t = 3)]
        top: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Simulate {
            iters,
            seed,
            top,
            stage,
            precision,
        } => cmd_simulate(iters, seed, top, stage, precision),
        Command::Predict {
            home,
            away,
            home_odds,
            draw_odds,
            away_odds,
            posterior,
        } => cmd_predict(&home, &away, (home_odds, draw_odds, away_odds), posterior),
        Command::Backtest {
            matches,
            seed,
            data,
            cv,
        } => match cv {
            Some(folds) if folds >= 2 => cmd_backtest_cv(matches, seed, data, folds),
            _ => cmd_backtest(matches, seed, data),
        },
        Command::Tune {
            matches,
            seed,
            data,
        } => cmd_tune(matches, seed, data),
        Command::MarketBacktest {
            seed,
            matches,
            bankroll,
            edge,
            kelly,
            cap,
            margin,
        } => cmd_market_backtest(seed, matches, bankroll, edge, kelly, cap, margin),
        Command::Scorers { home, away, top } => cmd_scorers(&home, &away, top),
        Command::GoldenBoot { iters, seed, top } => cmd_golden_boot(iters, seed, top),
        Command::InPlay {
            home,
            away,
            iters,
            seed,
        } => cmd_inplay(&home, &away, iters, seed),
        Command::Derivatives { home, away } => cmd_derivatives(&home, &away),
        Command::Serve {
            addr,
            event_log,
            forecast_journal,
        } => cmd_serve(addr, event_log, forecast_journal).await,
        Command::TrackRecord {
            journal,
            event_log,
            recent,
        } => cmd_track_record(&journal, event_log.as_deref(), recent),
        Command::Watch { speed } => watch::run(speed).await,
        Command::Sensitivity { iters, seed, top } => cmd_sensitivity(iters, seed, top),
    }
}

/// `track-record --journal <path>`: score the calls the engine actually published.
///
/// Read-only and offline, and reconstructible from disk alone: the calls come from the journal and
/// the results from the event log, which are the engine's two durable records. Nothing is refit and
/// nothing is recomputed, which is the whole point - these numbers cannot be improved by changing the
/// model, only by playing more matches.
fn cmd_track_record(
    path: &std::path::Path,
    event_log: Option<&std::path::Path>,
    recent: usize,
) -> anyhow::Result<()> {
    let (records, unreadable) = oracle_engine::ForecastJournal::read_reporting(path)?;
    if records.is_empty() && unreadable == 0 {
        println!(
            "No calls in {}.\n  Run `wc-oracle serve --forecast-journal {}` and let some matches \
             settle.",
            path.display(),
            path.display()
        );
        return Ok(());
    }

    // Results live in the event log, not the journal - the journal records what was said, not what
    // happened. Replaying only the full-time events is enough to settle calls, and needs no model.
    let mut tournament = data::world_cup_2026();
    let results = match event_log {
        Some(log) => apply_logged_results(&mut tournament, log)?,
        None => 0,
    };
    let tr = oracle_engine::track_record(&records, &tournament, unreadable);

    println!("Track record from {}\n", path.display());
    println!(
        "  {} calls journaled, {} settled across {} matches",
        tr.calls, tr.settled, tr.matches
    );
    if let (Some(first), Some(last)) = (tr.first_call, tr.last_call) {
        println!(
            "  spanning {} to {}",
            first.format("%Y-%m-%d %H:%M"),
            last.format("%Y-%m-%d %H:%M")
        );
    }
    if tr.unreadable_lines > 0 {
        println!(
            "  WARNING: {} journal lines could not be read; the scores below exclude them",
            tr.unreadable_lines
        );
    }
    match event_log {
        Some(log) => println!("  {results} results applied from {}", log.display()),
        None => println!(
            "  no --event-log given, so no results are available to settle these calls against"
        ),
    }

    if tr.models.is_empty() {
        println!("\n  Nothing has settled yet, so there is nothing to score.");
        return Ok(());
    }

    println!(
        "\n  {:<24} {:>7} {:>7} {:>8} {:>9}",
        "Model", "Scored", "Called", "Brier", "LogLoss"
    );
    println!("  {}", "-".repeat(59));
    for m in &tr.models {
        println!(
            "  {:<24} {:>7} {:>6}  {:>8.4} {:>9.4}",
            m.model, m.scored, m.winners_called, m.brier, m.log_loss
        );
    }
    // The baseline is quoted over the leading model's sample, not `tr.settled` - that counts calls
    // across every model, so with two forecasters it would read as double the matches scored.
    println!(
        "  {:<24} {:>7} {:>6}  {:>8.4} {:>9.4}",
        "Uniform baseline", tr.models[0].scored, "-", tr.baseline_brier, 1.0986
    );
    println!("  (lower Brier / log-loss is better)");

    println!(
        "\n  Calibration of {} (ECE {:.3}):",
        tr.models[0].model, tr.reliability.ece
    );
    for b in tr.reliability.bins.iter().filter(|b| b.count > 0) {
        println!(
            "    {:>5.0}-{:<3.0}%   predicted {:>5.1}%   empirical {:>5.1}%   n={}",
            b.lo * 100.0,
            b.hi * 100.0,
            b.mean_pred * 100.0,
            b.empirical * 100.0,
            b.count
        );
    }

    if recent > 0 {
        let settled = oracle_engine::settle(&records, &tournament);
        let mut latest: Vec<_> = settled
            .iter()
            .filter(|s| s.record.model == tr.models[0].model)
            .collect();
        latest.sort_by_key(|s| std::cmp::Reverse(s.record.made_at));
        if !latest.is_empty() {
            println!("\n  Most recent settled calls by {}:", tr.models[0].model);
            for s in latest.into_iter().take(recent) {
                println!(
                    "    {} {}-{} {}   called {:>5.1}% {}",
                    if s.called_correctly() { "OK  " } else { "MISS" },
                    s.record.home_name,
                    s.record.away_name,
                    s.score,
                    s.confidence() * 100.0,
                    s.record.forecast.most_likely(),
                );
            }
        }
    }

    // A track record is only as strong as its sample, and this one will be small for a long while.
    if tr.settled < 30 {
        println!(
            "\n  Note: {} settled calls is a small sample. Treat the numbers as directional; \
             differences this size are not significant.",
            tr.settled
        );
    }
    Ok(())
}

/// Mark every match the event log reports a full-time score for as finished, and return how many.
///
/// Deliberately the smallest possible replay: only `FullTime` events are read, and no model is
/// touched. Settling a journaled call needs the result and nothing else, and reconstructing engine
/// state here would reintroduce exactly the recomputation the journal exists to avoid.
fn apply_logged_results(
    tournament: &mut oracle_domain::Tournament,
    path: &std::path::Path,
) -> anyhow::Result<usize> {
    use oracle_domain::{EventKind, MatchStatus};
    let mut applied = 0usize;
    for event in oracle_engine::EventLog::read(path)? {
        if let EventKind::FullTime { score } = event.kind {
            if let Some(m) = tournament
                .matches
                .iter_mut()
                .find(|m| m.id == event.match_id)
            {
                m.status = MatchStatus::Finished;
                m.score = score;
                applied += 1;
            }
        }
    }
    Ok(applied)
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

#[allow(clippy::too_many_arguments)]
fn cmd_market_backtest(
    seed: u64,
    matches: usize,
    bankroll: f64,
    edge: f64,
    kelly: f64,
    cap: f64,
    margin: f64,
) -> anyhow::Result<()> {
    println!("Fitting the model, then paper-trading a {matches}-match held-out season (seed {seed})...\n");
    let explorer = Explorer::new();
    let policy = BetPolicy {
        min_edge: edge,
        kelly_fraction: kelly,
        max_fraction: cap,
    };
    let report = explorer.market_backtest(seed, matches, bankroll, policy, margin);
    let s = &report.run.summary;
    let signed = |v: f64| {
        if v >= 0.0 {
            format!("+{:.1}%", v * 100.0)
        } else {
            format!("{:.1}%", v * 100.0)
        }
    };

    println!(
        "Bankroll:     ${:.2} -> ${:.2}   (ROI {})",
        s.start_bankroll,
        s.final_bankroll,
        signed(s.roi)
    );
    println!(
        "Bets:         {} of {} matches   hit rate {:.1}%",
        s.bets,
        report.matches,
        s.hit_rate * 100.0
    );
    println!(
        "Turnover:     ${:.2}   yield {}",
        s.turnover,
        signed(s.yield_pct)
    );
    println!("Max drawdown: {:.1}%\n", s.max_drawdown * 100.0);

    println!("Is the edge real? Accuracy on the same matches (lower is better):");
    println!(
        "  {:<22} Brier {:.4}   log-loss {:.4}",
        "Model (Dixon-Coles)", report.skill.model_brier, report.skill.model_log_loss
    );
    println!(
        "  {:<22} Brier {:.4}   log-loss {:.4}",
        "Market (de-vigged)", report.skill.market_brier, report.skill.market_log_loss
    );

    let verdict = if s.yield_pct > 0.0 {
        "cleared the vig on this season"
    } else {
        "did not clear the vig"
    };
    println!("\nVerdict: the model {verdict} over {} bets.", s.bets);
    Ok(())
}

fn cmd_scorers(home: &str, away: &str, top: usize) -> anyhow::Result<()> {
    let explorer = Explorer::new();
    let h = explorer
        .resolve(home)
        .ok_or_else(|| anyhow::anyhow!("unknown team: {home}"))?;
    let a = explorer
        .resolve(away)
        .ok_or_else(|| anyhow::anyhow!("unknown team: {away}"))?;
    let market = explorer.scorer_market(h, a, true);
    println!("Goalscorer market ({home} v {away})   anytime / brace / hat-trick\n");
    for line in market.lines.iter().take(top) {
        println!(
            "  {:<16} {:<16} xg {:.2}   {:>5.1}%  {:>5.1}%  {:>5.1}%",
            line.player.name,
            line.player.team,
            line.expected_goals,
            line.anytime * 100.0,
            line.brace * 100.0,
            line.hat_trick * 100.0
        );
    }
    Ok(())
}

fn cmd_golden_boot(iters: u32, seed: u64, top: usize) -> anyhow::Result<()> {
    println!("Fitting the model, then simulating the Golden Boot race (seed {seed})...\n");
    let explorer = Explorer::new();
    let race = explorer.golden_boot(iters, seed);
    println!("Golden Boot race   top scorer / top 3\n");
    for (i, o) in race.iter().take(top).enumerate() {
        println!(
            "  {:>2}. {:<16} {:<16} exp {:>4.1}   {:>5.1}%  {:>5.1}%",
            i + 1,
            o.player.name,
            o.player.team,
            o.expected_goals,
            o.p_top * 100.0,
            o.p_top3 * 100.0
        );
    }
    Ok(())
}

fn cmd_derivatives(home: &str, away: &str) -> anyhow::Result<()> {
    let explorer = Explorer::new();
    let h = explorer
        .resolve(home)
        .ok_or_else(|| anyhow::anyhow!("unknown team: {home}"))?;
    let a = explorer
        .resolve(away)
        .ok_or_else(|| anyhow::anyhow!("unknown team: {away}"))?;
    let b = explorer.derivatives(h, a, true);
    let o = &b.outcome;
    println!("Derivative markets: {home} v {away}\n");
    println!(
        "  1X2:  {:.1}% / {:.1}% / {:.1}%    expected goals {:.2}",
        o.home_win * 100.0,
        o.draw * 100.0,
        o.away_win * 100.0,
        b.totals.expected
    );
    println!("\n  Totals:");
    for l in &b.totals.lines {
        println!(
            "    {:.1}   over {:>5.1}%   under {:>5.1}%",
            l.line,
            l.over * 100.0,
            l.under * 100.0
        );
    }
    println!("\n  Asian handicap (home line):");
    for hc in &b.handicap {
        println!(
            "    {:+.2}   home {:>5.1}%  push {:>5.1}%  away {:>5.1}%   fair {:.2} / {:.2}",
            hc.line,
            hc.home_win * 100.0,
            hc.push * 100.0,
            hc.away_win * 100.0,
            hc.fair_home_odds,
            hc.fair_away_odds
        );
    }
    println!("\n  Correct score:");
    for s in &b.correct_score.top {
        println!("    {}-{}   {:>5.1}%", s.home, s.away, s.prob * 100.0);
    }
    println!("    any other   {:>5.1}%", b.correct_score.other * 100.0);
    println!(
        "\n  BTTS {:.0}%   clean sheet H/A {:.0}%/{:.0}%   draw-no-bet H/A {:.0}%/{:.0}%",
        b.goals.btts_yes * 100.0,
        b.goals.clean_sheet_home * 100.0,
        b.goals.clean_sheet_away * 100.0,
        b.draw_no_bet.home * 100.0,
        b.draw_no_bet.away * 100.0
    );
    Ok(())
}

fn cmd_inplay(home: &str, away: &str, iters: u32, seed: u64) -> anyhow::Result<()> {
    println!("Fitting the model, then simulating in-play trading (seed {seed})...\n");
    let explorer = Explorer::new();
    let h = explorer
        .resolve(home)
        .ok_or_else(|| anyhow::anyhow!("unknown team: {home}"))?;
    let a = explorer
        .resolve(away)
        .ok_or_else(|| anyhow::anyhow!("unknown team: {away}"))?;
    let view = explorer.inplay_backtest(h, a, iters, seed);
    let r = &view.report;
    println!(
        "Backing {} at {:.2}   ({} v {})\n",
        view.backed, view.back_odds, view.home, view.away
    );
    println!(
        "  Cash-out:       mean {:+.3}   ROI {:+.1}%   cash-out {:.0}%   max drawdown {:.1}u",
        r.mean_pnl,
        r.roi * 100.0,
        r.cash_out_rate * 100.0,
        r.max_drawdown
    );
    println!(
        "  Hold to settle: mean {:+.3}   (baseline, never cash out)",
        r.hold_mean_pnl
    );
    println!("\nFair odds mean no edge either way; cash-out trades variance, not profit.");
    Ok(())
}

fn cmd_simulate(
    iters: u64,
    seed: u64,
    top: usize,
    stage: Option<String>,
    precision: Option<f64>,
) -> anyhow::Result<()> {
    if let Some(slug) = stage {
        return cmd_simulate_stage(&slug, iters, seed, top);
    }
    let tournament = data::world_cup_2026();
    let model = data::fit_baseline_model(seed);
    let names: std::collections::HashMap<_, _> = tournament
        .teams
        .iter()
        .map(|t| (t.id, t.name.clone()))
        .collect();

    match precision {
        Some(p) => println!(
            "Simulating {} - to ±{:.3} champion standard error, at most {} iterations (seed {})...\n",
            tournament.name, p, iters, seed
        ),
        None => println!(
            "Simulating {} - {} iterations (seed {})...\n",
            tournament.name, iters, seed
        ),
    }
    // Apply the full match context (host, crowd, altitude, rest, travel, heat) plus the style
    // matchup to every fixture, and the per-team knockout factors (shootout skill, pedigree).
    let inputs = LiveInputs {
        venue: data::matchup_adjustments(&tournament),
        shootout_rating: data::shootout_ratings(),
        knockout_pedigree: data::knockout_pedigree(),
        ..Default::default()
    };
    let start = Instant::now();
    let config = SimConfig {
        iterations: iters,
        seed,
        ..SimConfig::default()
    };
    // With a precision target the run decides its own length, up to `--iters` as the ceiling.
    let (forecast, achieved) = match precision {
        Some(p) => {
            let out = oracle_sim::simulate_to_precision(
                &tournament,
                &model,
                config,
                &inputs,
                LiveConfig::default(),
                oracle_sim::PrecisionTarget {
                    champion_std_error: p.max(1e-4),
                    batch: 5_000,
                    max_iterations: iters,
                },
            );
            (
                out.forecast,
                Some((out.worst_champion_std_error, out.target_met)),
            )
        }
        None => (
            simulate_with_live(&tournament, &model, config, &inputs, LiveConfig::default()),
            None,
        ),
    };
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
    let ran = forecast.iterations;
    println!(
        "\n{} simulations in {:.2}s  ({:.0} tournaments/sec); ±MC err is the Monte-Carlo \
         standard error on the champion probability",
        ran,
        elapsed.as_secs_f64(),
        ran as f64 / elapsed.as_secs_f64().max(1e-9),
    );
    if let Some((worst, met)) = achieved {
        if met {
            println!(
                "  precision target reached: worst champion standard error ±{:.4} after {ran} \
                 iterations",
                worst
            );
        } else {
            println!(
                "  precision target NOT reached: stopped at the {ran}-iteration ceiling with worst \
                 champion standard error ±{:.4}",
                worst
            );
        }
    }
    Ok(())
}

/// `simulate --stage <round>`: take the REAL 2026 teams still alive at a knockout stage and
/// simulate the remaining bracket forward over just that field. The real result is printed
/// underneath so the model's pick can be compared with what actually happened.
fn cmd_simulate_stage(slug: &str, iters: u64, seed: u64, top: usize) -> anyhow::Result<()> {
    let stage = actual_2026::parse_stage(slug).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown stage '{slug}' (use round-of-16 | quarter-final | semi-final | final)"
        )
    })?;
    let tournament = actual_2026::stage_tournament(stage).ok_or_else(|| {
        anyhow::anyhow!(
            "stage '{slug}' is not covered by the real dataset (Round of 16 onward only)"
        )
    })?;

    // Real teams the offline fit never saw (e.g. Cape Verde) get a weak floor rating.
    let mut model = data::fit_baseline_model(seed);
    let (weak_atk, weak_def) = model.weakest_coefficients();
    for t in &tournament.teams {
        if !model.contains_team(t.id) {
            model.set_team_coefficients(t.id, weak_atk, weak_def);
        }
    }
    let names: std::collections::HashMap<_, _> = tournament
        .teams
        .iter()
        .map(|t| (t.id, t.name.clone()))
        .collect();

    println!(
        "Real 2026 field still alive at the {stage} ({} teams) - simulating {} tournaments (seed {})...\n",
        tournament.teams.len(),
        iters,
        seed
    );
    let inputs = LiveInputs {
        shootout_rating: data::shootout_ratings(),
        knockout_pedigree: data::knockout_pedigree(),
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
    let n = forecast.iterations.max(1) as f64;
    let stderr = |p: f64| (p * (1.0 - p) / n).sqrt();

    println!(
        "{:>3}  {:<16} {:>15} {:>12}",
        "#", "Team", "Champ (±MC err)", "Reach final"
    );
    println!("{}", "-".repeat(52));
    for (i, t) in forecast.ranked().into_iter().take(top).enumerate() {
        let name = names.get(&t.team).cloned().unwrap_or_default();
        println!(
            "{:>3}  {:<16} {:>6.1}% ±{:>4.1}% {:>11.1}%",
            i + 1,
            name,
            t.p_champion * 100.0,
            stderr(t.p_champion) * 100.0,
            t.p_final * 100.0,
        );
    }

    let o = actual_2026::actual_outcome();
    let full = |code: &str| {
        actual_2026::resolve_code(code)
            .map(|t| t.name)
            .unwrap_or_else(|| code.to_string())
    };
    let fav = forecast
        .ranked()
        .first()
        .and_then(|t| names.get(&t.team).cloned())
        .unwrap_or_default();
    println!(
        "\nModel's pick: {fav}.  Actual 2026: {} won the final over {}; {} beat {} for third.",
        full(o.champion),
        full(o.runner_up),
        full(o.third),
        full(o.fourth)
    );
    println!(
        "{} simulations in {:.2}s ({:.0} tournaments/sec)",
        iters,
        elapsed.as_secs_f64(),
        iters as f64 / elapsed.as_secs_f64().max(1e-9),
    );
    Ok(())
}

/// Ablation study: re-run the tournament with each unconventional signal disabled in turn and
/// report how far it moves the championship distribution. A signal that barely shifts the title
/// picture is honest to surface; one that moves it a lot earns its place. All variants share the
/// RNG seed, so each delta reflects the signal rather than Monte-Carlo noise.
fn cmd_sensitivity(iters: u64, seed: u64, top: usize) -> anyhow::Result<()> {
    use std::collections::HashMap;

    let tournament = data::world_cup_2026();
    let names: HashMap<TeamId, String> = tournament
        .teams
        .iter()
        .map(|t| (t.id, t.name.clone()))
        .collect();

    eprintln!("Fitting the baseline and nine signal variants (this runs several simulations)...");
    let model = data::fit_baseline_model(seed);
    let unpooled = data::fit_model_unpooled(seed);

    // The ablation itself lives in oracle-engine, shared with the web explorer.
    let contributions = oracle_engine::signal_sensitivity(
        &tournament,
        &model,
        &unpooled,
        &data::shootout_ratings(),
        &data::knockout_pedigree(),
        iters,
        seed,
        top,
    );

    // Already ranked by title shift; join in team names for display.
    struct SignalRow {
        label: &'static str,
        tvd: f64,
        movers: Vec<(String, f64)>,
    }
    let rows: Vec<SignalRow> = contributions
        .into_iter()
        .map(|c| SignalRow {
            label: c.signal,
            tvd: c.title_shift,
            movers: c
                .movers
                .into_iter()
                .map(|(t, d)| (names.get(&t).cloned().unwrap_or_default(), d))
                .collect(),
        })
        .collect();

    println!(
        "Signal sensitivity - how much each unconventional signal moves the title picture\n\
         ({iters} iterations per variant, seed {seed}; one signal disabled per row)\n"
    );
    println!(
        "{:<30}{:>12}   Biggest movers",
        "Signal (disabled)", "Title shift"
    );
    println!("{}", "-".repeat(80));
    for row in &rows {
        let movers_str = row
            .movers
            .iter()
            .map(|(n, d)| format!("{} {:+.1}pp", n, d * 100.0))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{:<30}{:>11.2}%   {}",
            row.label,
            row.tvd * 100.0,
            movers_str
        );
    }
    println!(
        "\nTitle shift = total variation distance between the full-model and ablated champion\n\
         distributions (0 = identical). Each row turns off exactly one signal and re-simulates;\n\
         the shared seed couples the runs so the delta reflects the signal, not Monte-Carlo noise."
    );
    Ok(())
}

fn cmd_predict(
    home_q: &str,
    away_q: &str,
    odds: (Option<f64>, Option<f64>, Option<f64>),
    posterior: bool,
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
    for (h, a, p) in grid.top_scorelines(5) {
        println!("    {h}–{a}   {:>5.1}%", p * 100.0);
    }

    if posterior {
        // Sample the goal model's posterior by HMC to show its uncertainty about its own forecast.
        // The model was fit (in `baseline`) on this synthetic history; reconstruct it for the
        // likelihood.
        let obs: Vec<oracle_model::Observation> = data::synthetic_history_with_market(4000, 7)
            .iter()
            .map(|r| r.obs)
            .collect();
        let samples = model.posterior_outcome_samples(
            &obs,
            &data::confederations(),
            home,
            away,
            true,
            oracle_model::hmc::HmcConfig {
                n_samples: 800,
                n_warmup: 200,
                step_size: 0.2,
                n_leapfrog: 16,
                seed: 7,
            },
        );
        // 90% credible interval per outcome from the posterior draws.
        let ci = |pick: fn(&oracle_domain::Probabilities) -> f64| -> (f64, f64, f64) {
            let mut v: Vec<f64> = samples.iter().map(pick).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let pct = |q: f64| v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            (mean, pct(0.05), pct(0.95))
        };
        let (hm, hlo, hhi) = ci(|p| p.home_win);
        let (dm, dlo, dhi) = ci(|p| p.draw);
        let (am, alo, ahi) = ci(|p| p.away_win);
        println!(
            "\n  Posterior (HMC, {} samples) - the model's uncertainty about its own",
            samples.len()
        );
        println!("  forecast, as a 90% credible interval:");
        println!(
            "    {:<11} win  {:>5.1}%  [{:>4.1}%, {:>4.1}%]",
            name(home),
            hm * 100.0,
            hlo * 100.0,
            hhi * 100.0
        );
        println!(
            "    {:<11} draw {:>5.1}%  [{:>4.1}%, {:>4.1}%]",
            "",
            dm * 100.0,
            dlo * 100.0,
            dhi * 100.0
        );
        println!(
            "    {:<11} win  {:>5.1}%  [{:>4.1}%, {:>4.1}%]",
            name(away),
            am * 100.0,
            alo * 100.0,
            ahi * 100.0
        );
    }
    println!();
    Ok(())
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

/// Print one model's Brier and log-loss with bootstrap 95% confidence intervals.
fn cv_row(label: &str, preds: &[(Probabilities, Outcome)], seed: u64) {
    const N_BOOT: usize = 2000;
    if preds.is_empty() {
        println!("  {label:<20}   (no predictions)");
        return;
    }
    let (brier, log_loss, _acc) = oracle_model::bootstrap_score_ci(preds, N_BOOT, seed);
    println!(
        "  {:<20}  {:.4} [{:.4}, {:.4}]   {:.4} [{:.4}, {:.4}]",
        label, brier.point, brier.lo, brier.hi, log_loss.point, log_loss.lo, log_loss.hi
    );
}

/// Rolling-origin (expanding-window) cross-validation. The first half of the chronologically
/// ordered matches is always training; the rest is split into `folds` consecutive evaluation
/// blocks. Each fold refits the goal model, Elo, and ensemble on everything *before* its block
/// and predicts the block, so there is never any look-ahead. The out-of-fold predictions are
/// pooled and each model's skill is reported with a bootstrap 95% confidence interval - a far
/// more honest read on skill (and on whether a change actually helped) than a single split.
fn cmd_backtest_cv(
    n_matches: usize,
    seed: u64,
    data_path: Option<std::path::PathBuf>,
    folds: usize,
) -> anyhow::Result<()> {
    use oracle_model::{DixonColesConfig, Observation};

    let (mut records, source, real_data) = match &data_path {
        Some(p) => (data::load_results_csv(p)?, format!("{}", p.display()), true),
        None => (
            data::synthetic_history_with_market(n_matches, seed),
            format!("{n_matches} synthetic matches"),
            false,
        ),
    };
    if records.len() < 200 {
        anyhow::bail!(
            "need at least 200 matches for cross-validation (got {})",
            records.len()
        );
    }
    // Oldest first (chronological): each fold trains on the past, evaluates on the future.
    records.sort_by(|a, b| {
        b.obs
            .age_days
            .partial_cmp(&a.obs.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = records.len();
    let neutral = !real_data;
    let market_present = records.iter().any(|r| r.market.is_some());

    let anchor = n / 2;
    let block = (n - anchor) / folds;
    if block == 0 {
        anyhow::bail!("too few matches ({n}) for {folds} folds");
    }

    let mut dc_oof: Vec<(Probabilities, Outcome)> = Vec::new();
    let mut elo_oof: Vec<(Probabilities, Outcome)> = Vec::new();
    let mut ens_oof: Vec<(Probabilities, Outcome)> = Vec::new();
    let mut market_oof: Vec<(Probabilities, Outcome)> = Vec::new();
    let mut fold_sizes: Vec<usize> = Vec::new();

    for fold in 0..folds {
        let fit_end = anchor + fold * block;
        let eval_end = if fold + 1 == folds {
            n
        } else {
            anchor + (fold + 1) * block
        };
        // Hold out the last quarter of the training portion to learn the ensemble weights.
        let inner_train_end = (fit_end * 3 / 4).max(1);

        let train_obs: Vec<Observation> =
            records[..inner_train_end].iter().map(|r| r.obs).collect();
        let model = GoalModel::fit(&train_obs, DixonColesConfig::default());
        let mut ratings = RatingStore::with_defaults();
        if !real_data {
            for (team, rating) in data::team_strengths() {
                ratings.seed(team, rating);
            }
        }
        for r in &records[..inner_train_end] {
            ratings.record(r.obs.home, r.obs.away, r.obs.score, neutral);
        }

        let mut val_preds = Vec::new();
        let mut val_actuals = Vec::new();
        for r in &records[inner_train_end..fit_end] {
            let dc = model.outcome_probabilities(r.obs.home, r.obs.away, neutral);
            let elo = ratings.win_probabilities(r.obs.home, r.obs.away, neutral);
            match (market_present, r.market) {
                (true, Some(m)) => {
                    val_preds.push(vec![dc, elo, m]);
                    val_actuals.push(r.obs.score.outcome());
                }
                (false, _) => {
                    val_preds.push(vec![dc, elo]);
                    val_actuals.push(r.obs.score.outcome());
                }
                _ => {}
            }
        }
        let n_members = if market_present { 3 } else { 2 };
        let ensemble = Ensemble::fit(&val_preds, &val_actuals, n_members);

        let mut count = 0;
        for r in &records[fit_end..eval_end] {
            let actual = r.obs.score.outcome();
            let dc = model.outcome_probabilities(r.obs.home, r.obs.away, neutral);
            let elo = ratings.win_probabilities(r.obs.home, r.obs.away, neutral);
            dc_oof.push((dc, actual));
            elo_oof.push((elo, actual));
            let mut members = vec![dc, elo];
            if let Some(market) = r.market {
                members.push(market);
                market_oof.push((market, actual));
            }
            ens_oof.push((ensemble.blend(&members), actual));
            count += 1;
        }
        fold_sizes.push(count);
    }

    println!(
        "\nRolling-origin cross-validation on {}  ({} expanding-window folds, {} out-of-fold predictions)\n",
        source,
        folds,
        ens_oof.len()
    );
    println!("  fold eval sizes: {fold_sizes:?}");
    println!(
        "\n  {:<20}  {:<24}   {:<24}",
        "Model", "Brier [95% CI]", "LogLoss [95% CI]"
    );
    println!("  {}", "-".repeat(72));
    let base = oracle_model::CalibrationReport::uniform_baseline(ens_oof.len());
    println!(
        "  {:<20}  {:.4}                    {:.4}",
        "Uniform baseline", base.brier, base.log_loss
    );
    cv_row(
        if real_data {
            "Dixon-Coles"
        } else {
            "Dixon-Coles (xG)"
        },
        &dc_oof,
        seed,
    );
    cv_row("Elo", &elo_oof, seed);
    cv_row(
        if market_present {
            "Ensemble (+Market)"
        } else {
            "Ensemble (DC+Elo)"
        },
        &ens_oof,
        seed,
    );
    if market_oof.is_empty() {
        println!("  {:<20}   (no odds in this dataset)", "Market (bookmaker)");
    } else {
        cv_row("Market (bookmaker)", &market_oof, seed);
    }
    println!(
        "\n  Out-of-fold over {folds} folds; 95% CI from 2000 bootstrap resamples (seeded, reproducible)."
    );
    println!("  Non-overlapping intervals mean the skill gap is not a single-split fluke.\n");
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

    // Bayesian optimization over the *continuous* (xi, ridge) space, run once per score model.
    // A GP surrogate + Expected-Improvement picks where to look next, so it finds a better optimum
    // in fewer fits than the old grid - and over values the grid could never land on exactly.
    use oracle_model::bayes_opt::{minimize, BoConfig};
    const MODELS: &[ScoreModel] = &[ScoreModel::Independent, ScoreModel::Bivariate];
    // (xi, ridge) search box.
    let bounds = [(0.0005_f64, 0.01_f64), (0.0_f64, 0.08_f64)];

    let base = DixonColesConfig::default();
    let mut best = (base, evaluate(base, &validation));
    let mut evaluated = 0usize;
    for (mi, &model) in MODELS.iter().enumerate() {
        let res = minimize(
            &bounds,
            BoConfig {
                n_init: 6,
                n_iter: 16,
                seed: seed.wrapping_add(mi as u64),
            },
            |x| {
                let cfg = DixonColesConfig {
                    xi: x[0],
                    ridge: x[1],
                    model,
                    ..base
                };
                evaluate(cfg, &validation)
            },
        );
        evaluated += res.evaluations;
        if res.best_value < best.1 {
            best = (
                DixonColesConfig {
                    xi: res.best_x[0],
                    ridge: res.best_x[1],
                    model,
                    ..base
                },
                res.best_value,
            );
        }
    }

    let model_name = |m: ScoreModel| match m {
        ScoreModel::Independent => "independent",
        ScoreModel::Bivariate => "bivariate",
    };
    println!(
        "\nTuning on {}  (train {} / validation {} / test {}); {} fits via Bayesian optimization\n",
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
        "\n  Bayesian optimization (GP surrogate + Expected Improvement) over the continuous\n  \
         (xi, ridge) space per score model, selecting by held-out validation log-loss. Lower is\n  \
         better; the test column is the honest out-of-sample number.\n"
    );
    Ok(())
}

async fn cmd_serve(
    addr: SocketAddr,
    event_log: Option<std::path::PathBuf>,
    forecast_journal: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
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
            forecast_journal,
            ..Default::default()
        },
        cancel.clone(),
    )
    .await?;

    // The on-demand explorer (backs /explore and the /api/* query endpoints); fit in the
    // background so the live dashboard and engine endpoints respond immediately.
    let explorer = oracle_api::spawn_explorer();

    println!("worldcup-oracle serving on http://{addr}  (/ live · /explore interactive · Ctrl-C to stop)");
    let shutdown_cancel = cancel.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_cancel.cancel();
    };
    oracle_api::serve(engine, explorer, addr, shutdown).await?;
    cancel.cancel();
    let _ = join.await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::resolve_team;
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
}
