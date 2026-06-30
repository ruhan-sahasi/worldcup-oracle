# Methodology

This is the *why* behind the model: the forecasting philosophy and the reasoning behind each
modelling choice. For the *how* (crate-by-crate mechanics) see [`ARCHITECTURE.md`](ARCHITECTURE.md);
for measured skill on real data see [`VALIDATION.md`](VALIDATION.md).

## Thesis

A World Cup forecaster is easy to start and hard to make honest. Three commitments shape this one:

1. **Model the things a strength rating cannot.** A single number per team (Elo, market value, a
   power ranking) captures *level*. The interesting, hard-to-fake signal lives in the *interactions*
   and *context* a scalar misses: a style that troubles a specific opponent, a crowd that is hostile
   in a nominally neutral game, a side drained by extra time the round before. Most of the
   unconventional signals below were chosen precisely because additive strength is structurally
   incapable of representing them.
2. **Quantify uncertainty, then propagate it.** A point forecast that says "Brazil 31%" without a
   sense of how sure it is is half a forecast. The model carries parameter uncertainty from the fit
   all the way into the champion odds, and can report a full posterior credible interval on any
   matchup.
3. **Evaluate honestly, against the hardest baseline.** Skill is measured with proper scoring rules,
   out-of-sample, with confidence intervals, against the bookmaker's vig-free line. The repository
   states plainly where it matches the market and where it does not.

## The forecasting stack

```
team histories ─▶ goal model (Dixon-Coles) ┐
results        ─▶ Elo                        ├─▶ log-opinion-pool ensemble ─▶ per-match odds
results        ─▶ state-space (Kalman)       ┘                                      │
bookmaker odds ─▶ market member ─────────────┘                                      ▼
                                                          Monte-Carlo tournament simulator ─▶ champion odds
```

The **Dixon-Coles bivariate-Poisson** goal model is the core: it turns team histories into a full
exact-score distribution. **Elo** and a **state-space (Kalman)** rating are complementary strength
signals; the **market** (vig-free bookmaker odds) is the fourth member. A **log-opinion-pool
ensemble** blends them with weights learned by out-of-fold stacking. For tournament odds, a parallel
**Monte-Carlo** plays the rest of the competition tens of thousands of times.

## The unconventional signals

Nine signals beyond raw strength. Each is grouped by what it captures, with its rationale and an
honest note on how it is sourced. Offline, several are *reasoned-synthetic* (a defensible model, not
measured data) so the engine runs without a network; on a live feed they would come from real data.

### Context (what surrounds the match)
- **Crowd composition** - a continuous partisanship signal, not the binary "is it a host". Mexico
  draws a near-home crowd across US venues; a diaspora-heavy stand tilts a "neutral" game. Captures
  what a host flag misses.
- **Travel & circadian load** - distance and (eastward-weighted) time-zone shifts since a side's
  last match sap it. A continent-spanning 2026 makes the home/away differential a real edge.
- **Heat** - afternoon kickoffs in a North American summer suppress tempo for both sides; fewer
  goals also *flatten the favourite's edge*, which falls out of the Poisson variance for free.

### Structural (what additive strength cannot express)
- **Style matchup** - a low-rank style embedding per team scored by a *bilinear* form with an
  antisymmetric kernel: a non-transitive, rock-paper-scissors edge (style A troubles B troubles C).
  An additive model cannot represent a cycle.
- **Overdispersion** - real goals are mildly overdispersed (blowouts and goalless games beat
  Poisson). The margins are **negative-binomial** (a Gamma-Poisson mixture), fit from the data; a
  fixed-rate Poisson cannot have fat tails.
- **Confederation pooling** - confederations rarely play each other, so cross-confederation strength
  is poorly pinned down. The fit is **hierarchical**: each team is pooled toward its confederation
  level rather than the global mean, so sparse teams borrow strength sensibly.

### Knockout dynamics (effects that exist only in single-elimination)
- **Shootout skill** - shootout outcomes have a persistent component (technique, the keeper,
  composure) independent of open play, so a flat 50/50 leaves signal on the table.
- **Knockout pedigree** - a knockout-*only* tilt for single-elimination temperament/experience;
  2026's 48-team field sends more debutants deep than ever.
- **Extra-time fatigue** - a side whose tie went to extra time is dampened the *next* round. This is
  a genuinely *dynamic within-tournament state* - only expressible because the simulator plays the
  bracket round by round, not as independent ties.

## Fitting and inference

The goal-model parameters progressed deliberately:

1. **Time-weighted maximum likelihood.** Matches are down-weighted by `exp(-ξ·age)` so recent form
   counts more; an **L2 (ridge)** penalty shrinks sparse-data teams (the regularization a sparse,
   unbalanced international schedule needs).
2. **L-BFGS** replaced a hand-rolled gradient ascent: quasi-Newton with a line search, converging in
   tens of iterations instead of hundreds and (verified by cross-validation) finding a slightly
   better optimum.
3. **Laplace (Fisher-information) posterior.** Treating the ridge as a Gaussian prior, a team's
   strength posterior precision is `prior + Fisher information`, giving a principled per-team
   standard deviation. The Monte-Carlo resamples each team's strength from it per iteration, so
   champion odds carry parameter uncertainty rather than treating a point estimate as certain.
4. **Full HMC posterior.** Beyond the Gaussian approximation, Hamiltonian Monte Carlo draws the true
   posterior of a matchup's win/draw/win probabilities (`predict --posterior` reports a 90% credible
   interval). Leapfrog integration on the exact log-posterior gradient, Laplace-variance
   preconditioning, trajectory-length jitter to avoid resonances.

Hyperparameters (`ξ`, ridge, score model) are chosen by **Bayesian optimization** (a Gaussian-process
surrogate + Expected Improvement) over a continuous space, not a hand-specified grid.

## Evaluation philosophy

Skill is the point, so it is measured carefully:

- **Proper scoring rules** (Brier, log-loss) plus a **reliability curve + ECE**, never just accuracy.
- The **bookmaker's vig-free implied odds** are the baseline, because beating the market is the
  honest bar - and a hard one.
- **Rolling-origin cross-validation** with **bootstrap confidence intervals** (`backtest --cv N`):
  expanding-window folds with no look-ahead, scored with a 95% CI per metric. A single split is one
  noisy draw; non-overlapping intervals are the test of whether a change is real or within noise.
- The honest finding: on held-out data the ensemble's interval **overlaps the bookmaker's** - it
  matches the market within noise rather than beating it. That is a strong, truthful result, and the
  CV harness is what lets it be stated with confidence.
- **Signal ablation** (`wc-oracle sensitivity`): the natural skeptic's question about nine
  unconventional signals is "do they actually matter?". The analysis disables each signal in turn
  and re-simulates the whole tournament on a shared RNG seed (so the delta reflects the signal, not
  Monte-Carlo noise), reporting how far each one moves the championship distribution (total
  variation distance) and which teams move most. A signal that barely shifts the title picture is
  reported honestly as such; the point is to *measure* each contribution rather than assert it.

## Honest limitations

- The bundled roster/draw is a representative sample, not the official FIFA draw; the live adapter
  pulls real teams and fixtures.
- Offline, the synthetic history is overdispersed-Poisson-generated; several signals (crowd, style,
  heat, shootout skill, knockout pedigree) are **reasoned-synthetic** - defensible models, not
  measured data. Travel, time zones, and rest come from the real schedule and venue coordinates. The
  feature *machinery* is real and tested; the offline *inputs* are illustrative. World-Cup-specific
  real validation needs international results with odds (the `--data` path accepts them).
- Deliberately deferred (documented in the code): the FIFA best-thirds 495-row lookup table (a
  deterministic rule is used instead), separate attack/defense state-space states, squad market
  value, dead-rubber rotation, and referee effects.
