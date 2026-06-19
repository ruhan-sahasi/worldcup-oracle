# Validation on real data

The synthetic backtest proves the *machinery*; this proves the *model has real skill*. The
engine is validated here on **real match results with real bookmaker closing odds**, against
the bookmaker itself, with a proper out-of-sample temporal split and a calibration check.

## How to reproduce

```bash
# Download real results + Bet365 closing odds (public, no key) and run the backtest:
bash scripts/fetch-results.sh /tmp/real_results.csv          # football-data.co.uk, English top flight
cargo run --release -p oracle-cli -- backtest --data /tmp/real_results.csv
```

The data (football-data.co.uk) is not committed; the script downloads it. `--data` uses a
three-way temporal split (fit on the oldest 60%, learn the ensemble weights on the next 20%,
report on the most recent 20%) and, for real club data, applies home advantage.

## A representative run

1,520 English Premier League matches, oldest-first; reported on the 304 most recent
(held-out) matches. Lower Brier / log-loss is better; the bookmaker is the bar to beat.

| Model | Brier | Log-loss | Accuracy |
|-------|------:|---------:|---------:|
| Uniform baseline | 0.6667 | 1.0986 | 33.3% |
| Dixon-Coles | 0.5652 | 0.9575 | 55.3% |
| Elo | 0.5726 | 0.9875 | 55.9% |
| **Ensemble (+Market)** | **0.5416** | **0.9194** | 58.2% |
| Market (bookmaker) | 0.5421 | 0.9204 | 58.9% |

Learned ensemble weights: Dixon-Coles 0.21 / Elo 0.15 / Market 0.63, temperature 1.07.

Calibration of the ensemble (expected calibration error **0.018**):

| Predicted bucket | Mean predicted | Empirical | n |
|------------------|---------------:|----------:|--:|
| 0-20% | 13.6% | 14.8% | 229 |
| 20-40% | 27.7% | 25.8% | 418 |
| 40-60% | 49.0% | 51.4% | 148 |
| 60-80% | 68.3% | 67.8% | 90 |
| 80-100% | 85.4% | 92.6% | 27 |

## What this shows

- Both component models beat the uniform baseline by a wide margin on real matches (Brier
  ~0.57 vs 0.667, ~55% accuracy), so the fit captures real signal, not just synthetic noise.
- Stacking learns to lean on the market (weight 0.63) and the resulting ensemble **matches
  the bookmaker's closing line** (Brier 0.5416 vs 0.5421, log-loss 0.9194 vs 0.9204). The
  market is famously hard to beat; landing on it out-of-sample is a strong, honest result.
- The ensemble is **well-calibrated** (ECE 0.018): predicted probability tracks empirical
  frequency across every bucket.

## Honest scope

- This validates the model on real **club** football (Premier League), the realistic
  offline proxy with public odds. World-Cup-specific real validation needs international
  results with odds; the same `--data` path accepts them.
- Exact numbers shift with which seasons football-data.co.uk currently serves (the script
  only concatenates seasons sharing the newest schema), but the picture is stable.
- Expected goals are not in these CSVs, so this run fits on goals; supply a source with xG
  columns to exercise that path on real data too.
