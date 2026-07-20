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

The same DP measures **match leverage**: how much a single tie's result reshapes the whole title
race. For each undecided tie in the current round the champion DP is re-run twice, forcing each side
through, and the leverage is the total-variation distance between the two resulting champion
distributions. This captures something a scoreline-closeness heuristic cannot: a coin-flip between
two long shots barely moves the title picture, while a tie between two contenders deep in the bracket
moves it a great deal, so the ranking reflects who sits where in the bracket, not just how even the
tie is. By construction leverage grows as the final nears, since each result is diluted by fewer
remaining rounds.

Stepping back from any single team or tie, the shape of the champion-odds distribution itself says
how decided the tournament is. Its **Shannon entropy** `H = -Sigma p log2 p` (in bits) is summarized
as an **effective number of contenders** `2^H`: the race is as open as if that many teams were
equally likely, so a lone runaway favourite reads near one while a wide-open field reads high.
Dividing by the ceiling `log2(k)` over the `k` teams with any chance gives a normalized openness in
`[0, 1]`. Unlike the bracket views this is meaningful from the group stage onward, and it falls
monotonically in spirit as results eliminate teams and concentrate the mass on the survivors.

The engine also keeps the title race's **trajectory**, not only its current state: one sample of the
champion odds per forecast recompute, in a bounded ring buffer that covers a whole tournament. Two
readings fall out of that series. **Momentum** compares the latest odds with a sample a fixed window
of recomputes back and reports the biggest risers and fallers, so a team surging or fading reads
immediately (a fresh contender counts up from zero, a just-eliminated one down to it). **Lead
changes** walk the series and record each time the favourite changed hands, with a small hysteresis
margin so Monte-Carlo jitter between two near-tied favourites never manufactures a spurious flip. It
is persisted server-side, so the story survives a browser reload rather than living only in the
dashboard's in-memory chart.

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

## Betting the model against the market

Matching the market on a proper scoring rule is one thing; **beating the price** is another, and the
`oracle-market` crate asks it directly. It is a small, pure library, built in layers that are each
unit-tested in isolation: decimal odds and the **overround**; **de-vigging** (proportional, and
**Shin's method**, which strips more margin from longshots to correct the favourite/longshot bias);
per-outcome **edge** and **expected value**; **Kelly** and fractional-Kelly staking; a bet-selection
**policy** (back the top-EV side only past an edge threshold, stake fractional Kelly, cap the risk);
and a compounding **paper-trading bankroll** that settles each selected bet against the real result
and reports ROI, hit rate, turnover, yield, and max drawdown, next to a Brier/log-loss comparison of
the model versus the market on the same games.

The backtest is deliberately **out of sample**: the model is fit on one season's synthetic history
and bets a *different* season, whose noisy fair lines are priced with a vig to give the odds a bettor
would face. It bets the goal model's own probabilities, never the market-anchored ensemble, which
would be circular. The honest result reinforces the thesis above: with a tight edge filter the model
can post a positive ROI over a handful of bets, but that is variance (a dozen bets, an absurd yield);
loosen the filter to a real sample and it bleeds roughly the margin, because its Brier score is not
actually better than the market's. A calibrated model is not the same as an edge, and the harness is
built to show that rather than hide it.

## Who scores: goalscorer markets and the Golden Boot

Every layer above reasons about teams; `oracle-players` reasons about the individuals who score. It
takes a team's expected goals in a match and **shares them across the on-pitch players in proportion
to their attacking weight** (the squad model's own attack scores, talisman boost included, so nothing
new is invented), giving each player an expected-goals rate. From a Poisson on that rate fall the
markets a fan actually bets: **anytime scorer** (`1 - e^-xg`), **brace** and **hat-trick**
(at-least-two and at-least-three), and, treating each player's goals as a competing Poisson process,
the **first goalscorer** (`(rate / total)(1 - e^-total)`, with the leftover a goalless match).

The **Golden Boot** race lifts this to the whole tournament. Each team's expected tournament goals
are its expected number of matches (three group games plus the round-reach probabilities from a quick
simulation) times its representative per-match expected goals, shared across its outfielders, and a
Monte-Carlo (a seeded SplitMix64 drawing each player's goals by Knuth's method) counts how often each
finishes top scorer or on the podium. The expected-matches and per-match-goals steps are deliberate
approximations, labelled as such; the allocation, the Poisson markets, and the race simulation on top
are exact and independently unit-tested.

## In-play: trading the live position

`oracle-live` follows a match after kickoff. Its in-play win probability adds the current score to the
remaining goals, which are Poisson with a rate **prorated by the time left**: at kickoff it is the
pre-match forecast, by the whistle the remaining rate is zero so it settles to the result on the
board, and in between a lead is worth steadily more as the clock runs down. Simulating a goal
timeline and sampling that model minute by minute gives the live win-probability path, the drama
graph a trader watches.

On top sit the exchange tools. **Hedging** a back bet of stake `S` taken at odds `B` means laying
`S B / L` at the current lay odds `L`, which locks in `S (B - L) / L` whichever way the match goes;
**cash-out** value is exactly that locked figure at the current price. A trading backtest then backs
the pre-match favourite at fair odds and either cashes out at a profit target or stop or holds to
settlement, over thousands of simulated matches, against a hold-to-settlement baseline. Because the
odds are fair, the result is the honest one: both average essentially zero, so cash-out reshapes the
**distribution** of P&L (lower variance, different drawdown) but does not manufacture an edge. It is
the in-play echo of the market backtest's finding: the machinery is real, and it is honest about not
printing money.

## The full board: derivative markets

The goal model does not just give win/draw/win, it gives the whole **joint distribution over
scorelines** (the score grid). `oracle-derivatives` re-expresses that one object as every market a
book quotes, each a closed-form sum over the grid with no Monte-Carlo: **totals** (the goal
distribution and the over/under ladder), **both teams to score**, **clean sheets** and
**win-to-nil**, **double chance** and **draw-no-bet**, and the **correct-score** board.

The centrepiece is the **Asian handicap**, which is more intricate than it looks. From the winning-
margin distribution, a handicap line settles win/push/lose on the margin adjusted by the line, so a
whole line can push (the stake is refunded) while a half line cannot, and the fair odds are
`(1 - push) / win` to remove the refunded mass. A **quarter** line splits the stake evenly across the
two adjacent lines, so its effective settlement is the average of theirs, capturing the half-win and
half-loss outcomes exactly. All of it is priced off the same grid the headline forecast comes from,
so the board is internally consistent by construction (the handicap at the level line reproduces the
1x2, over-plus-under is one, the correct-score board is exhaustive).

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
