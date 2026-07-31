//! Fit the models on a dataset and score them out of sample.
//!
//! This is the evaluation the `backtest` command has always performed. It lived inside that
//! command, interleaved with its printing, which meant nothing else could run it - so the skill
//! regression gate would have had to reimplement the split, the fit and the scoring, and the two
//! would have been free to drift apart while both looked right.

use oracle_domain::{Outcome, Probabilities};
use oracle_ingest::data::{self, MatchRecord};
use oracle_model::{
    bootstrap_score_ci, score, CalibrationReport, DixonColesConfig, Ensemble, GoalModel, MetricCi,
    Observation, ReliabilityReport,
};
use oracle_ratings::RatingStore;
use serde::{Deserialize, Serialize};

/// The minimum dataset an evaluation is willing to run on.
///
/// Below this the three-way split leaves a test set too small for its metrics to mean anything, and
/// a gate built on noise is worse than no gate.
pub const MIN_MATCHES: usize = 50;

/// Which forecaster a scored row belongs to.
///
/// An enum rather than a string because the gate matches baseline rows to fresh ones by model, and a
/// renamed or mistyped label would silently drop a comparison rather than fail it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// The naive uniform forecast, the bar any model must clear.
    UniformBaseline,
    /// Dixon-Coles fit on goals only, present only when the dataset also carries xG.
    DixonColesGoals,
    /// Dixon-Coles fit on xG when the dataset has it, otherwise on goals.
    DixonColes,
    Elo,
    /// The learned log-opinion-pool blend.
    Ensemble,
    /// The bookmaker's de-vigged closing line, when the dataset carries odds.
    Market,
}

impl Model {
    /// A short, stable label for display and for baseline files.
    pub fn label(self) -> &'static str {
        match self {
            Model::UniformBaseline => "Uniform baseline",
            Model::DixonColesGoals => "Dixon-Coles (goals)",
            Model::DixonColes => "Dixon-Coles",
            Model::Elo => "Elo",
            Model::Ensemble => "Ensemble",
            Model::Market => "Market (bookmaker)",
        }
    }
}

/// One forecaster's out-of-sample scores.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelSkill {
    pub model: Model,
    pub scored: usize,
    pub brier: f64,
    pub log_loss: f64,
    pub accuracy: f64,
    /// Bootstrap 95% interval on the Brier score, when [`EvalConfig::bootstrap`] asked for one.
    ///
    /// This is **not** what the gate decides on, and the distinction is the whole reason it is worth
    /// reporting. The evaluation is deterministic over a fixed fixture, so a moved metric is a real
    /// change to the model with no sampling noise to explain it - the gate compares the point values
    /// and is right to. What the interval answers is the *next* question a reader has: would this
    /// change survive on different matches, or is it smaller than the spread of the sample it was
    /// measured on? A gate cannot answer that, and quietly conflating the two would either hide real
    /// regressions or fail on ones that mean nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brier_ci: Option<MetricCi>,
    /// Bootstrap 95% interval on the log loss. See [`brier_ci`](Self::brier_ci).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_loss_ci: Option<MetricCi>,
}

/// A complete evaluation of a dataset: how each forecaster scored on the held-out split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillReport {
    /// Matches in each split.
    pub train: usize,
    pub validation: usize,
    pub test: usize,
    /// Whether the dataset carried xG, and whether it carried bookmaker odds. Both change which
    /// rows are present, so a comparison across datasets that differ in these is meaningless.
    pub xg_present: bool,
    pub market_present: bool,
    pub models: Vec<ModelSkill>,
    /// The learned ensemble weights, normalized, and its temperature.
    pub ensemble_weights: Vec<f64>,
    pub ensemble_temperature: f64,
    /// The ensemble's reliability curve over the test split.
    pub reliability: ReliabilityReport,
}

impl SkillReport {
    /// One forecaster's row, if it was scored.
    pub fn get(&self, model: Model) -> Option<&ModelSkill> {
        self.models.iter().find(|m| m.model == model)
    }
}

/// How a dataset is split and evaluated.
#[derive(Debug, Clone, Copy)]
pub struct EvalConfig {
    /// Whether every fixture is at a neutral venue. Real club data has genuine home advantage; the
    /// synthetic World Cup does not.
    pub neutral: bool,
    /// Whether to seed Elo from the known synthetic strengths. Only meaningful for synthetic data,
    /// where those strengths exist.
    pub seed_elo_from_strengths: bool,
    /// Bins in the reported reliability curve.
    pub reliability_bins: usize,
    /// Bootstrap resamples and seed for the confidence intervals, or `None` to skip them.
    ///
    /// Skipped by default because they cost a few thousand rescorings per model and most callers -
    /// including the gate's pass/fail decision - do not need them.
    pub bootstrap: Option<Bootstrap>,
}

/// How to compute the bootstrap intervals.
#[derive(Debug, Clone, Copy)]
pub struct Bootstrap {
    pub resamples: usize,
    /// Seed for the resampling, so the intervals are reproducible. A gate that reported a different
    /// interval on every run would look flaky even though its verdict was stable.
    pub seed: u64,
}

impl Default for Bootstrap {
    fn default() -> Self {
        Self {
            resamples: 2000,
            seed: 20_260_611,
        }
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            neutral: true,
            seed_elo_from_strengths: true,
            reliability_bins: 5,
            bootstrap: None,
        }
    }
}

/// Why an evaluation could not be run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// Fewer than [`MIN_MATCHES`] rows.
    TooFewMatches { got: usize, need: usize },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::TooFewMatches { got, need } => {
                write!(f, "need at least {need} matches to backtest (got {got})")
            }
        }
    }
}

impl std::error::Error for EvalError {}

/// Fit on the oldest 60% of `records`, learn the ensemble on the next 20%, and score every
/// forecaster on the most recent 20%.
///
/// The temporal split is the point: the test set is strictly newer than everything the models saw,
/// so nothing here is contaminated by look-ahead. Ordering is by match age, which the caller does
/// not have to pre-sort.
///
/// Deterministic given the same records and config - no randomness is involved anywhere in the fit
/// or the scoring - which is what lets a regression gate compare two runs and attribute any
/// difference to the model rather than to chance.
pub fn evaluate(records: &[MatchRecord], config: EvalConfig) -> Result<SkillReport, EvalError> {
    if records.len() < MIN_MATCHES {
        return Err(EvalError::TooFewMatches {
            got: records.len(),
            need: MIN_MATCHES,
        });
    }
    // Oldest first. `age_days` counts backwards from now, so descending age is chronological.
    //
    // The tie-break matters more than it looks. Datasets routinely hold several matches on the same
    // day, and a sort on age alone is only *stable*, so tied rows keep whatever order the caller
    // passed them in - which means the file's row order decides which of them land in train, in
    // validation, and in test. A regression gate reading a committed fixture cannot depend on that:
    // reordering the rows would move the metrics with no model change. Breaking ties on the match's
    // own content makes the ordering total, so the split is a function of the data alone.
    let mut records = records.to_vec();
    records.sort_by(|a, b| {
        b.obs
            .age_days
            .partial_cmp(&a.obs.age_days)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.obs.home.cmp(&b.obs.home))
            .then_with(|| a.obs.away.cmp(&b.obs.away))
            .then_with(|| a.obs.score.home.cmp(&b.obs.score.home))
            .then_with(|| a.obs.score.away.cmp(&b.obs.score.away))
    });

    let n = records.len();
    let train_end = n * 6 / 10;
    let val_end = n * 8 / 10;
    let train_obs: Vec<Observation> = records[..train_end].iter().map(|r| r.obs).collect();
    let validation = &records[train_end..val_end];
    let test = &records[val_end..];

    let xg_present = train_obs.iter().any(|o| o.home_xg.is_some());
    let market_present = records.iter().any(|r| r.market.is_some());
    let neutral = config.neutral;

    // The headline goal model, fit on xG when the dataset has it (a lower-noise target).
    let model = GoalModel::fit(&train_obs, DixonColesConfig::default());
    // A goals-only refit, so the xG lever is visible rather than assumed.
    let model_goals = xg_present.then(|| {
        let stripped: Vec<Observation> = train_obs
            .iter()
            .map(|o| Observation::new(o.home, o.away, o.score, o.age_days))
            .collect();
        GoalModel::fit(&stripped, DixonColesConfig::default())
    });

    let mut ratings = RatingStore::with_defaults();
    if config.seed_elo_from_strengths {
        for (team, rating) in data::team_strengths() {
            ratings.seed(team, rating);
        }
    }
    for r in &records[..train_end] {
        ratings.record(r.obs.home, r.obs.away, r.obs.score, neutral);
    }

    // Learn the ensemble on the validation split, with the market as a third member when present.
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

    // Score everything on the held-out test split.
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

    let row = |model: Model, preds: &[(Probabilities, Outcome)]| -> Option<ModelSkill> {
        if preds.is_empty() {
            return None;
        }
        let r = score(preds);
        let (brier_ci, log_loss_ci) = match config.bootstrap {
            Some(b) => {
                let (brier, log_loss, _accuracy) = bootstrap_score_ci(preds, b.resamples, b.seed);
                (Some(brier), Some(log_loss))
            }
            None => (None, None),
        };
        Some(ModelSkill {
            model,
            scored: preds.len(),
            brier: r.brier,
            log_loss: r.log_loss,
            accuracy: r.accuracy,
            brier_ci,
            log_loss_ci,
        })
    };
    let baseline = CalibrationReport::uniform_baseline(test.len());
    let mut models = vec![ModelSkill {
        model: Model::UniformBaseline,
        scored: test.len(),
        brier: baseline.brier,
        log_loss: baseline.log_loss,
        accuracy: baseline.accuracy,
        // The uniform baseline is a constant, not an estimate: it makes the same prediction on every
        // match, so resampling would return the same number every time. An interval of zero width
        // would be technically correct and read as suspicious, so there is none.
        brier_ci: None,
        log_loss_ci: None,
    }];
    models.extend(
        [
            row(Model::DixonColesGoals, &dc_goals_preds),
            row(Model::DixonColes, &dc_preds),
            row(Model::Elo, &elo_preds),
            row(Model::Ensemble, &ens_preds),
            row(Model::Market, &market_preds),
        ]
        .into_iter()
        .flatten(),
    );

    let wsum: f64 = ensemble.weights.iter().sum::<f64>().max(1e-9);
    Ok(SkillReport {
        train: train_obs.len(),
        validation: validation.len(),
        test: test.len(),
        xg_present,
        market_present,
        models,
        ensemble_weights: ensemble.weights.iter().map(|w| w / wsum).collect(),
        ensemble_temperature: ensemble.temperature,
        reliability: oracle_model::reliability(&ens_preds, config.reliability_bins),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(n: usize) -> Vec<MatchRecord> {
        data::synthetic_history_with_market(n, 7)
    }

    #[test]
    fn an_evaluation_splits_the_data_sixty_twenty_twenty() {
        let r = evaluate(&synthetic(1000), EvalConfig::default()).unwrap();
        assert_eq!(r.train, 600);
        assert_eq!(r.validation, 200);
        assert_eq!(r.test, 200);
    }

    #[test]
    fn every_model_beats_the_uniform_baseline() {
        let r = evaluate(&synthetic(4000), EvalConfig::default()).unwrap();
        let baseline = r.get(Model::UniformBaseline).unwrap().brier;
        for m in &r.models {
            if m.model == Model::UniformBaseline {
                continue;
            }
            assert!(
                m.brier < baseline,
                "{} scored {} against a baseline of {baseline}",
                m.model.label(),
                m.brier
            );
        }
    }

    #[test]
    fn a_dataset_with_odds_scores_the_market_and_a_three_member_ensemble() {
        let r = evaluate(&synthetic(1000), EvalConfig::default()).unwrap();
        assert!(r.market_present);
        assert!(r.get(Model::Market).is_some());
        assert_eq!(r.ensemble_weights.len(), 3, "DC, Elo, market");
        assert!((r.ensemble_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_dataset_without_odds_omits_the_market_row() {
        let stripped: Vec<MatchRecord> = synthetic(1000)
            .into_iter()
            .map(|mut r| {
                r.market = None;
                r
            })
            .collect();
        let r = evaluate(&stripped, EvalConfig::default()).unwrap();
        assert!(!r.market_present);
        assert!(r.get(Model::Market).is_none());
        assert_eq!(r.ensemble_weights.len(), 2, "DC and Elo only");
    }

    #[test]
    fn the_same_input_twice_gives_bit_identical_metrics() {
        // The property the regression gate rests on. Nothing in the fit or the scoring is random, so
        // two runs over the same data must agree to the last bit - otherwise the gate could fail on
        // noise, and a gate that cries wolf gets switched off.
        //
        // This did not hold until the goal-model fit stopped summing over a HashMap's own iteration
        // order; see the comment in dixon_coles.rs.
        let records = synthetic(1000);
        let a = evaluate(&records, EvalConfig::default()).unwrap();
        let b = evaluate(&records, EvalConfig::default()).unwrap();
        assert_eq!(a.models, b.models, "metrics are not reproducible");
        assert_eq!(a.ensemble_weights, b.ensemble_weights);
        assert_eq!(a.ensemble_temperature, b.ensemble_temperature);
        assert_eq!(a.reliability.ece, b.reliability.ece);
    }

    #[test]
    fn repeated_evaluation_is_stable_over_many_runs() {
        // Once is luck; the hash seed differs per HashMap instance, so a residual order dependency
        // would show up intermittently rather than every time.
        let records = synthetic(400);
        let first = evaluate(&records, EvalConfig::default()).unwrap();
        for i in 0..12 {
            let again = evaluate(&records, EvalConfig::default()).unwrap();
            assert_eq!(first.models, again.models, "run {i} disagreed");
        }
    }

    #[test]
    fn reordering_the_input_does_not_move_the_metrics() {
        // A weaker but still necessary property: a committed fixture's row order must not decide the
        // result. Before the total tie-break in `evaluate` this failed outright, because a stable
        // sort on age alone left same-day matches in caller order and so moved them between splits.
        //
        // What remains is floating-point summation order inside the ensemble's iterative fit, worth
        // about 3e-14 on the Brier score. Forcing that to zero would mean imposing a canonical
        // summation order through the whole optimizer, for a difference eleven orders of magnitude
        // below the smallest regression anyone would care about.
        let forward = synthetic(1000);
        let mut reversed = forward.clone();
        reversed.reverse();
        let a = evaluate(&forward, EvalConfig::default()).unwrap();
        let b = evaluate(&reversed, EvalConfig::default()).unwrap();

        assert_eq!(a.test, b.test, "the split itself must be identical");
        for (x, y) in a.models.iter().zip(&b.models) {
            assert_eq!(x.model, y.model, "same models in the same order");
            assert_eq!(x.scored, y.scored);
            for (label, p, q) in [
                ("brier", x.brier, y.brier),
                ("log_loss", x.log_loss, y.log_loss),
                ("accuracy", x.accuracy, y.accuracy),
            ] {
                assert!(
                    (p - q).abs() < 1e-9,
                    "{} {label} moved by {:.3e} on reordering",
                    x.model.label(),
                    (p - q).abs()
                );
            }
        }
    }

    #[test]
    fn too_few_matches_is_refused_rather_than_scored() {
        let err = evaluate(&synthetic(1000)[..49], EvalConfig::default()).unwrap_err();
        assert_eq!(
            err,
            EvalError::TooFewMatches {
                got: 49,
                need: MIN_MATCHES
            }
        );
        assert!(evaluate(&synthetic(1000)[..50], EvalConfig::default()).is_ok());
    }

    #[test]
    fn intervals_are_absent_unless_asked_for() {
        let r = evaluate(&synthetic(400), EvalConfig::default()).unwrap();
        assert!(r.models.iter().all(|m| m.brier_ci.is_none()));
        assert!(r.models.iter().all(|m| m.log_loss_ci.is_none()));
    }

    #[test]
    fn intervals_bracket_their_point_estimate() {
        let cfg = EvalConfig {
            bootstrap: Some(Bootstrap {
                resamples: 400,
                seed: 11,
            }),
            ..Default::default()
        };
        let r = evaluate(&synthetic(1000), cfg).unwrap();
        for m in r
            .models
            .iter()
            .filter(|m| m.model != Model::UniformBaseline)
        {
            let ci = m.brier_ci.expect("an interval was requested");
            assert!(
                ci.lo <= m.brier && m.brier <= ci.hi,
                "{}: {} outside [{}, {}]",
                m.model.label(),
                m.brier,
                ci.lo,
                ci.hi
            );
            assert!(ci.lo < ci.hi, "a degenerate interval is not informative");
            assert!(
                (ci.point - m.brier).abs() < 1e-12,
                "point matches the score"
            );
            let ll = m.log_loss_ci.expect("log-loss interval");
            assert!(ll.lo <= m.log_loss && m.log_loss <= ll.hi);
        }
    }

    #[test]
    fn the_uniform_baseline_has_no_interval() {
        // It predicts the same thing on every match, so resampling returns the same number and a
        // zero-width interval would read as suspicious precision rather than as a constant.
        let cfg = EvalConfig {
            bootstrap: Some(Bootstrap::default()),
            ..Default::default()
        };
        let r = evaluate(&synthetic(400), cfg).unwrap();
        let base = r.get(Model::UniformBaseline).unwrap();
        assert!(base.brier_ci.is_none());
    }

    #[test]
    fn intervals_are_reproducible_for_a_seed() {
        // A gate reporting a different interval on every run would look flaky even with a stable
        // verdict.
        let cfg = EvalConfig {
            bootstrap: Some(Bootstrap {
                resamples: 300,
                seed: 99,
            }),
            ..Default::default()
        };
        let records = synthetic(400);
        let a = evaluate(&records, cfg).unwrap();
        let b = evaluate(&records, cfg).unwrap();
        assert_eq!(a.models, b.models);
    }

    #[test]
    fn asking_for_intervals_does_not_move_the_point_estimates() {
        // The intervals are extra reporting, not a different measurement. If requesting them shifted
        // a metric, a baseline recorded with them would not compare against a run without them.
        let records = synthetic(1000);
        let plain = evaluate(&records, EvalConfig::default()).unwrap();
        let with_ci = evaluate(
            &records,
            EvalConfig {
                bootstrap: Some(Bootstrap::default()),
                ..Default::default()
            },
        )
        .unwrap();
        for (a, b) in plain.models.iter().zip(&with_ci.models) {
            assert_eq!(a.model, b.model);
            assert_eq!(a.brier, b.brier, "{} Brier moved", a.model.label());
            assert_eq!(a.log_loss, b.log_loss);
            assert_eq!(a.accuracy, b.accuracy);
        }
    }

    #[test]
    fn model_labels_are_distinct() {
        // The gate matches rows by model, and labels appear in baseline files; a collision would
        // silently merge two forecasters.
        let all = [
            Model::UniformBaseline,
            Model::DixonColesGoals,
            Model::DixonColes,
            Model::Elo,
            Model::Ensemble,
            Model::Market,
        ];
        let mut labels: Vec<&str> = all.iter().map(|m| m.label()).collect();
        labels.sort_unstable();
        let before = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), before, "two models share a label");
    }
}
