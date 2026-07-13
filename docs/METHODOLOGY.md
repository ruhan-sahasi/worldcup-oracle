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
ensemble** blends them with weights learned by out-of-fold stacking, and leans harder on the market
for **knockout ties**, where the heavily-traded single-match closing line is the sharpest signal
and the stacked weights (learned mostly on group/league play) under-weight it. For tournament odds, a parallel
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

## A second model: Bradley-Terry-Davidson

The Dixon-Coles model above reasons about *goals*. As a tournament fills up with real results, a
second, deliberately different forecaster earns its place: one that reasons about *outcomes*
directly. **Bradley-Terry-Davidson** gives each team a single latent strength and reads a match as a
paired comparison - `P(i beats j) = πᵢ / (πᵢ + πⱼ + ν√(πᵢπⱼ))`, with a tie parameter `ν` (Davidson
1970) for the draw. It is fit by time-weighted maximum likelihood on who-beat-whom, so **deep in the
tournament the accumulated results dominate the priors** - exactly when a second opinion is most
useful. Dropping the tie term recovers plain Bradley-Terry, `πᵢ/(πᵢ+πⱼ)`, the natural "who advances"
probability for a knockout tie.

Its **winner prediction is exact, not Monte-Carlo**: given a knockout bracket, champion odds fall
out of a bottom-up **dynamic program** - a team wins a node by winning its own sub-bracket and then
beating the probability-weighted field from the other side, and the champion probabilities provably
sum to one. This is a genuinely distinct model (a different family, a different fit target, and a
different inference) offered *alongside* the goal-model ensemble, not folded into it - two
independent reads on the same tournament.

Deep in the tournament that DP runs over the **real, current bracket**. The live engine
**materializes each knockout round as it is decided** - the Round of 32 the moment the group stage
finishes, then the Round of 16, quarters, semis and final each time the round beneath it is fully
played, pairing the adjacent winners up the bracket (a level tie resolves the way the simulator's
shootout does). The champion DP then starts from the **deepest round that exists**: a decided tie
enters as a point mass on its winner, an undecided one as the pairwise advance split. So the second
model's live champion odds condition on the knockout results already played and project only what
remains, and eliminated teams correctly carry zero title probability.

With two independent title forecasts live, the engine also publishes a **consensus**: a plain 50/50
average of the two champion distributions (model averaging tends to be better calibrated than either
input), alongside a single measure of how far the two disagree right now, the **Jensen-Shannon
divergence** between them (in bits, bounded `[0, 1]`; zero when the two models are identical). The
per-team gap `bradley_terry - ensemble` then shows exactly *which* contenders the two models see
differently. This is deliberately honest: rather than hide the second model inside a blend, it keeps
both reads visible and quantifies their disagreement as a live uncertainty signal.

## A within-tournament power ranking: Massey least squares

Elo, the state-space filter, and Bradley-Terry all update a belief game by game and all lean on a
pre-tournament prior. **Massey's method** takes the opposite approach: it treats the whole set of
results as one **linear system** and solves it in closed form. With `T` the diagonal matrix of games
played and `P` the matrix of pairwise game counts, the Massey matrix is `M = T - P`, and the ratings
`r` solve `M r = p` where `p` is each team's cumulative goal margin. The least-squares fit that best
explains every margin at once is, by construction, **strength-of-schedule adjusted**: the same margin
counts for more against a higher-rated opponent. Because `M` is singular (its rows sum to zero), a
small **ridge** is added to the diagonal, which both makes the fit well-posed when the results graph
is thin or disconnected early on and shrinks sparsely-observed teams toward the mean.

The rating also splits into **offense** and **defense** (`r = offense + defense`): the defensive
ratings solve `(T + P) d = T r - f` for the goals-scored vector `f`, and offense is the remainder, so
the same fit says both how strong a team is and whether that strength comes from scoring or from
stopping goals. Fed only the matches actually played, this is a deliberately *prior-free,
within-tournament* read on who has been strongest here, published live and offered alongside the
prior-anchored models rather than folded into them.

Because that ranking is prior-free, comparing it to the pre-tournament strength prior gives a clean
read on **who has over- or under-performed**: rank the teams both by the prior and by the live Massey
fit over the same set, and the gap `pre_rank - power_rank` is how many places a team has climbed or
slid. The biggest movers each way are the tournament's risers and fallers, straight from the
difference between what was expected and what the results alone now say.

The same bracket dynamic program that produces the champion odds also yields each contender's **road
to the final**. A team's opponent in the round that is `k` steps ahead is the winner of the sibling
sub-bracket of `2^(k-1)` current ties, and that sub-bracket's win distribution is exactly the DP run
over that slice of the bracket. Weighting each possible opponent's Elo by its chance of getting there
gives the expected strength of the opponent each remaining round, and their mean is a single
`difficulty` for the whole path. Because every surviving team faces the same number of rounds, those
difficulties are directly comparable: an exact, not sampled, read on who has the easier road left.

Resolving every remaining tie to its favourite instead gives the **predicted bracket**: from the
current round forward, each tie goes to its Bradley-Terry favourite, the favourites are paired up the
bracket, and the projection runs to a single champion. Multiplying the favourites' win probabilities
gives the probability that this *exact* bracket occurs, which is deliberately reported: it is
typically a fraction of a percent, an honest reminder that a single most-likely bracket is still an
unlikely one, and that the champion-odds distribution, not the modal bracket, is the real forecast.

## Evaluation philosophy

Skill is the point, so it is measured carefully:

- **Proper scoring rules** (Brier, log-loss) plus a **reliability curve + ECE**, never just accuracy.
  Once a tournament is under way the engine publishes that reliability curve **live** (`/calibration`):
  it bins the model's own leak-free pre-match calls by predicted probability and compares each bin to
  the frequency the outcome actually occurred, so a viewer can watch whether a 70% call really lands
  about 70% of the time, with the expected calibration error as the one-number summary.
- The **bookmaker's vig-free implied odds** are the baseline, because beating the market is the
  honest bar - and a hard one.
- **Rolling-origin cross-validation** with **bootstrap confidence intervals** (`backtest --cv N`):
  expanding-window folds with no look-ahead, scored with a 95% CI per metric. A single split is one
  noisy draw; non-overlapping intervals are the test of whether a change is real or within noise.
- The honest finding: on held-out data the ensemble's interval **overlaps the bookmaker's** - it
  matches the market within noise rather than beating it. That is a strong, truthful result, and the
  CV harness is what lets it be stated with confidence.
- **Live recalibration** (in the engine, once a tournament is under way): the finished matches are
  a calibration set. Each match's leak-free pre-match forecast (recorded before its result updates
  the model) is paired with the realized outcome, and once enough have accumulated a single
  **temperature** is refit by minimizing log-loss and applied to the remaining forecasts. This is
  post-hoc temperature scaling, correcting any systematic over/under-confidence the offline fit
  could not anticipate. It rides on top of the model's other in-tournament learning (online
  goal-model updates and the state-space rating, which injects a per-match process-noise bump for
  tournament games so it keeps tracking form fast instead of growing overconfident) - those move
  the point estimates; recalibration fixes the *sharpness*.
- **Context recalibration** (in the engine): the reasoned-synthetic context effects (host, crowd,
  travel, heat) carry an *aggregate* strength gain, refit in-tournament against the results. Each
  finished match contributes the context's predicted margin contribution and the residual it should
  explain; a strongly-shrunk 1-D ridge (prior mean 1) nudges the gain toward what the tournament
  actually shows, so if context is playing out stronger or weaker than the priors the remaining
  forecasts follow. Deliberately *one* gain, not per-signal: a single tournament cannot reliably
  separate the correlated host/crowd/heat/travel effects, so the honest statistic is their combined
  strength, shrunk hard toward the prior until real evidence accumulates.
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
