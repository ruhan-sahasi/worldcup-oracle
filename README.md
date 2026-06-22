<h1 align="center">⚽ worldcup-oracle</h1>

<p align="center">
  <em>A live, ensemble World Cup prediction engine in Rust.</em>
</p>

<p align="center">
  <a href="https://github.com/ruhan-sahasi/worldcup-oracle/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ruhan-sahasi/worldcup-oracle/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Rust" src="https://img.shields.io/badge/rust-stable-orange.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="unsafe forbidden" src="https://img.shields.io/badge/unsafe-forbidden-success.svg">
</p>

`worldcup-oracle` ingests live match events including results, goals, red cards, the running
clock, and **continuously re-computes** every team's odds of winning each match and
lifting the trophy. It pairs a sophisticated statistical model with the
kind of systems engineering a backend role cares about: a modular crate workspace,
a lock-free event-driven core, parallel Monte-Carlo, back-pressured ingestion, a
REST + WebSocket API, and a live terminal dashboard.

It is timed for the **2026 World Cup** and can follow the real tournament via a free
API, but it also ships a deterministic simulator and a replay engine, so it runs
fully offline with **zero keys and zero network**.

---

## ✨ What it does

- **Ensemble predictions** -> a Dixon-Coles bivariate-Poisson goal model + Elo ratings,
  blended in log-space, with **Bayesian in-match updating** that shifts the odds as a
  match plays out.
- **Champion odds** -> a parallel Monte-Carlo simulator plays the rest of the
  tournament tens of thousands of times to estimate each team's chance of advancing,
  reaching each round, and winning it all.
- **Live, event-driven** -> an async engine consumes a stream of match events and
  pushes fresh forecasts to subscribers in real time.
- **Three pluggable data sources** behind one trait -> deterministic simulation,
  replay of a finished tournament, or the live [football-data.org](https://www.football-data.org) feed.
- **Lineup aware** -> a confirmed starting XI adjusts a team's effective attack and
  defense, so resting or losing a key player visibly moves that team's odds.
- **Multiple surfaces** -> a REST API, a WebSocket live stream, a live web dashboard,
  and a polished CLI/TUI.

## 🧠 The model (in one breath)

| Piece | What it contributes |
|-------|---------------------|
| **Dixon-Coles** bivariate Poisson, MLE-fit with time decay, **fit on xG when available**, **updated online from results** | full exact-score distribution that sharpens as the tournament unfolds |
| **Elo** with home edge + margin-of-victory scaling | a complementary strength signal |
| **Log-opinion-pool ensemble** (`[Dixon-Coles, Elo, Market]` weights + temperature **learned by stacking**) | a single sharper forecast, anchored to the bookmaker when odds are present |
| **Bayesian live updater** | conditions on score + minute + red cards for live odds |
| **Lineup adjustment** | a confirmed XI shifts each team's attack and defense |
| **Suspension tracking** | yellow-card accumulation drops a suspended starter from the next match before its lineup is known |
| **Venue & travel context** | host advantage, altitude, and rest-day differential adjust each match |
| **Monte-Carlo** (rayon-parallel, **conditions in-progress matches** on their live score) | tournament-level champion odds that move with live results |

Calibration is measured with proper scoring rules (Brier, log-loss), benchmarked against
the bookmaker's implied odds, and regression-tested. The maths are written up in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 🏗️ Architecture

A Cargo workspace of eight crates with strict downhill dependencies from a pure,
zero-I/O domain core:

```mermaid
flowchart LR
    P["DataProvider<br/>sim · replay · live API"]
    subgraph engine [oracle-engine]
      L["event loop<br/>(single writer)"]
      A["arc-swap snapshot"]
      B["broadcast"]
    end
    P -- "mpsc (bounded)" --> L
    L -- store --> A
    L -- publish --> B
    A -- "lock-free read" --> REST["REST /predict/*"]
    B -- push --> WS["WebSocket /live"]
    B -- push --> TUI["wc-oracle watch"]
```

| Crate | Responsibility |
|-------|----------------|
| `oracle-domain` | pure types (teams, matches, events, probabilities); no I/O |
| `oracle-ratings` | Elo rating system |
| `oracle-model` | Dixon-Coles, Bayesian live model, ensemble, calibration |
| `oracle-sim` | parallel Monte-Carlo tournament simulator |
| `oracle-ingest` | `DataProvider` trait + sim / replay / live adapters, rate-limit + cache |
| `oracle-engine` | event-driven orchestrator, pub/sub, snapshot cache, metrics |
| `oracle-api` | axum REST + WebSocket server (`oracle-server`) |
| `oracle-cli` | `wc-oracle`: CLI commands + live TUI |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for diagrams and the model maths.

## 🚀 Quickstart

```bash
# 1. Install Rust (https://rustup.rs) if needed, then:
git clone https://github.com/ruhan-sahasi/worldcup-oracle
cd worldcup-oracle
cargo build --release

# 2. Champion odds for the 2026 World Cup (reproducible with --seed):
cargo run --release -p oracle-cli -- simulate --iters 50000
```

```text
  #  Team             Champ (±MC err)    Final     Semi    Quart      R16
--------------------------------------------------------------------------
  1  France              8.7% ± 0.2%    14.2%    22.8%    36.5%    57.9%
  2  Argentina           7.5% ± 0.2%    12.2%    20.2%    32.5%    53.6%
  3  Spain               6.7% ± 0.2%    11.3%    18.8%    31.7%    52.4%
  ...
(host advantage, altitude, and rest folded in; ±MC err is the Monte-Carlo standard error)
```

```bash
# 3. Predict a matchup. Pass --*-odds to anchor the ensemble to a bookmaker line:
cargo run --release -p oracle-cli -- predict --home Brazil --away Argentina \
    --home-odds 2.4 --draw-odds 3.2 --away-odds 2.9
```

```text
  Brazil  vs  Argentina   (neutral venue)

  Ensemble :  Brazil       35.2%    Draw  28.1%    Argentina    36.7%
    Dixon-Coles:   34.0% / 25.6% /  40.3%      Elo:   27.8% / 30.1% /  42.1%
    Market     :   38.8% / 29.1% /  32.1%   (vig removed; anchored into the ensemble)

  Expected goals : 1.34 – 1.48
  Most likely    : 1–1
```

```bash
# 4. Watch a tournament unfold live in your terminal:
cargo run --release -p oracle-cli -- watch          # press q to quit

# 5. Backtest and benchmark against the bookmaker (synthetic data, or --data a real CSV):
cargo run --release -p oracle-cli -- backtest
cargo run --release -p oracle-cli -- backtest --data path/to/football-data.csv
```

```text
  Model                   Brier   LogLoss      Acc
  ------------------------------------------------
  Uniform baseline       0.6667    1.0986    33.3%
  Dixon-Coles (goals)    0.6283    1.0433    46.2%
  Dixon-Coles (xG)       0.6227    1.0360    48.2%
  Elo                    0.6719    1.1273    46.4%
  Ensemble (+Market)     0.6272    1.0427    47.5%
  Market (bookmaker)     0.6197    1.0318    48.8%

  learned weights: DC 0.37 / Elo 0.22 / Market 0.41   temperature 0.77

  Ensemble calibration (ECE 0.017):
          bucket   predicted   empirical        n
       0-20 %         18.3%      18.5%       54
      20-40 %         29.6%      28.5%     1801
      40-60 %         46.5%      50.2%      524
```

Three things are visible here. Fitting on **xG** beats fitting on goals (a lower-noise
signal). Stacking learns to lean on the **market** (the heaviest weight), since the
bookmaker's vig-free implied odds are the hard bar to beat: the engine approaches the
market but does not clear it. And the **reliability table + ECE** confirm the ensemble is
well-calibrated (predicted ≈ empirical in every bucket). `--data` runs the same split on a real
[football-data.co.uk](https://www.football-data.co.uk) CSV (with closing odds, and xG
columns if present).

> **Validated on real data.** On 1,520 real Premier League matches with real Bet365 closing
> odds, the stacked ensemble (Brier **0.5416**) matches the bookmaker's closing line (0.5421)
> out-of-sample and stays well-calibrated (ECE 0.018). Numbers and a one-command reproducer
> are in [`docs/VALIDATION.md`](docs/VALIDATION.md) (`bash scripts/fetch-results.sh`).

### Run the server and live dashboard

```bash
cargo run --release -p oracle-cli -- serve         # or: cargo run -p oracle-api --bin oracle-server
# record every event to a durable log and recover from it on restart:
cargo run --release -p oracle-cli -- serve --event-log oracle.jsonl
# open the live dashboard:
open http://localhost:8080/
# or hit the API directly:
curl localhost:8080/predict/tournament | jq '.teams[:5]'
curl localhost:8080/predict/match/1
```

Visiting `/` serves a self-contained dashboard (no build step, no CDN) that subscribes to
the `/live` WebSocket and renders live match win bars, a championship-odds leaderboard, a
probability-over-time chart, and a feed-health indicator, all updating in real time. With
`--event-log`, every event is appended as JSON and replayed on the next start, so a restart
mid-tournament recovers its state instead of starting cold.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | **live web dashboard** |
| `GET` | `/api` | service info + endpoint list (JSON) |
| `GET` | `/health` | liveness probe |
| `GET` | `/teams` | current Elo ratings |
| `GET` | `/matches` | all match predictions (compact) |
| `GET` | `/predict/match/{id}` | one match: live odds + exact-score grid |
| `GET` | `/predict/tournament` | champion-odds table |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/live` | **WebSocket**: pushes a compact live view on every update |

### Go live (optional)

Drop a free API key in `.env` (`cp .env.example .env`) and the engine switches to the
real 2026 World Cup feed automatically:

```bash
FOOTBALL_DATA_API_KEY=your_key   # from football-data.org
```

No key? It runs the deterministic simulation, and every command above works unchanged.

### Docker

```bash
docker compose up --build         # serves on :8080
```

## 🛠️ What this project demonstrates

- **Workspace architecture & dependency inversion** -> a pure domain core with a
  trait-based data seam (`DataProvider`) and transport layers as thin shells.
- **Async, event-driven concurrency** -> `tokio` mpsc ingestion with back-pressure,
  `broadcast` fan-out pub/sub, **single-writer state with lock-free `arc-swap` reads**,
  graceful cancellation.
- **Data-parallelism** -> `rayon`-parallel Monte-Carlo with deterministic per-iteration
  seeding; ~50k full tournament simulations/second.
- **Applied statistics** -> Dixon-Coles MLE with time decay (**fit on xG** when present)
  and **ridge regularization** that shrinks sparse-data teams, **online updating from each
  finished match** so the model learns in-tournament, Elo, Bayesian conditioning, lineup-,
  suspension-, and venue-aware adjustments, a **stacked** `[Dixon-Coles, Elo, Market]` ensemble, and
  honest evaluation: **proper scoring rules** vs the **bookmaker's implied odds**, a
  **reliability curve + ECE**, and **Monte-Carlo standard error** on the forecast.
- **Resilient ingestion** -> an authoritative `ScoreSync` reconciliation (so a dropped or
  duplicated poll can't corrupt the score), a feed-health signal with exponential backoff,
  a **durable append-only event log** that is replayed on boot for crash recovery, a
  hand-rolled token-bucket rate limiter + TTL cache, structured `tracing`, Prometheus
  metrics, `#![forbid(unsafe_code)]`, unit + property + integration tests, Criterion
  benchmarks, CI, and Docker.
- **Full-stack delivery** -> a dependency-free live web dashboard (vanilla JS + canvas)
  served by the API and driven entirely off the `/live` WebSocket.

## 🧪 Tests & benchmarks

```bash
cargo test --workspace          # unit + integration (incl. calibration guard)
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p oracle-sim       # Monte-Carlo throughput
```

## 📌 Scope & honest limitations

- The bundled roster/draw is a **representative sample** for offline use, not FIFA's
  official draw; the live adapter pulls the real teams, fixtures, and results.
- The default offline data (training history, the "bookmaker" line) is **synthetic but
  reproducible**, so everything runs without a network. That validates the *machinery*;
  the model's *real* skill is measured separately on real matches with real odds, see
  [`docs/VALIDATION.md`](docs/VALIDATION.md). World-Cup-specific real validation needs
  international results + odds (the same `--data` path accepts them).
- Squads and venue assignments are **synthetic** for offline use, so the lineup and venue
  features are fully demonstrable via the simulation feed. The live football-data.org
  adapter ingests results (and **line-ups** on tiers that expose them); odds and xG are not
  offered by that provider, so they come from the CSV path or a dedicated source. Rest days
  are derived from the real fixture schedule.
- In-progress **group** matches are conditioned on their live score, but the knockout
  simulator still builds a fresh bracket each run (a standard seeded single-elimination
  template, not FIFA's exact slotting); conditioning live *knockout* matches is future
  work. All documented in the code.
- The goal model now **learns in-tournament** from each finished match (a one-step online
  Poisson update); the deeper version, a full **dynamic / state-space (Kalman) rating** with
  process noise, is still future work. Also open: full **posterior intervals** (we report
  Monte-Carlo standard error, not parameter uncertainty) and richer **knockout realism**
  (extra time, a less coin-flip shootout). Deliberately deferred: **squad market value**
  (largely redundant with the strength ratings offline) and **stakes / dead-rubber
  rotation** (speculative).

## 📄 License

MIT © Ruhan Sahasi. See [LICENSE](LICENSE).
