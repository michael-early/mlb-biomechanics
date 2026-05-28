# Rigor Roadmap: MLB Biomechanics Velocity Study

This document tracks the implemented rigor upgrade and the next improvements needed to make the study more credible as a baseball data science project. The priority is still rigor: validation, uncertainty, model comparison, diagnostics, and a defensible baseball narrative.

## Current Limits

- Engineered metrics are interpretable z-score composites, but raw biomechanical inputs currently outperform engineered composites in repeated grouped CV.
- The model still underpredicts the 90+ mph band, which is the most baseball-relevant failure mode.
- The local Statcast bridge uses a larger sample, but it is still capped by Baseball Savant range export behavior.
- The public data cannot identify OpenBiomechanics athletes as MLB players, so the project must avoid player-level translation claims.

## Implemented Priority 1: Leakage-Safe Grouped Validation

- Use leakage-safe train-fit/test-transform biomechanical metric scaling.
- Clip fold-fit z-scores to reduce small-sample outlier extrapolation.
- Use repeated grouped K-fold validation by `session`.
- Ensure the same session never appears in both train and test within a fold.
- Report fold-level RMSE, MAE, and R2.
- Report mean, standard deviation, and approximate confidence intervals across folds.

Discussion point:

> I treated session leakage as the first modeling risk. The validation design groups by session so a pitcher's repeated throws do not leak across train/test folds.

## Implemented Priority 2: Model Comparison

- Compare a mean baseline, OLS, ridge with grouped inner-CV alpha selection, and dependency-free kNN.
- Compare engineered metrics against raw metric inputs, raw plus engineered metrics, and simple engineered interactions.
- Keep ridge as the primary interpretable model unless a more complex model clearly wins and remains explainable.
- Write model comparison outputs to:
  - `reports/tables/cv_model_comparison.csv`
  - `reports/tables/feature_set_comparison.csv`
  - `reports/tables/metric_ablation.csv`
  - `reports/tables/metric_correlations.csv`

Discussion point:

> I did not jump straight to a black-box model. I benchmarked simple, interpretable models and a nonlinear baseline so the performance claim is anchored against reasonable alternatives.

## Implemented Priority 3: Uncertainty and Diagnostics

- Bootstrap holdout RMSE, MAE, and R2 intervals by session.
- Repeat permutation importance shuffles to measure feature-importance stability.
- Summarize residuals by velocity band and session.
- Add an explicit high-velocity error table for the 90+ mph band.
- Write diagnostics to:
  - `reports/tables/bootstrap_intervals.csv`
  - `reports/tables/residual_diagnostics.csv`
  - `reports/tables/velocity_permutation_importance.csv`
  - `reports/tables/high_velocity_error_analysis.csv`

Discussion point:

> The takeaway is not just a single RMSE. I added uncertainty and residual diagnostics to show where the model is reliable and where it needs more data.

## Implemented Priority 4: Better Statcast Bridge

- The code supports Baseball Savant CSV ingestion and pitcher-pitch-type aggregation.
- The current local sample is a capped Baseball Savant range export with 25,000 pitch rows across April 24-30, 2025.
- Whiff rate now uses swings as the denominator.
- Chase rate now uses out-of-zone pitches as the denominator.
- Hard-hit rate now uses batted balls as the denominator.
- A sample manifest records rows, date range, denominator counts, and pitcher-pitch-type rows.
- The next data upgrade should use chunked daily/weekly Baseball Savant exports to build a 4-8 week or full-season Statcast dataset without range caps.
- Keep this as a bridge, not a direct player-level join, unless identity-linked private data becomes available.

Discussion point:

> Public biomechanics data and public Statcast data answer adjacent questions. The bridge shows how I would structure the team version once private identity-linked data exists.

## Implemented Priority 5: Public Report Narrative

- Center the report on one defensible claim: kinetic-chain transfer is the strongest engineered signal for fastball velocity in the public OpenBiomechanics sample.
- State the stricter caveat: raw metric inputs currently outperform engineered composites in repeated CV.
- State the high-velocity caveat: the holdout model underpredicts 90+ mph pitches.
- Separate observed evidence from future private-data extensions.
- Keep limitations visible near the top of the report.

Discussion point:

> I designed the project to be technically honest. It shows what can be learned from public data and exactly what additional private data would unlock for a club.

## Next Best Work

1. Add chunked Statcast downloading so a full 4-8 week or season sample can be reproduced without CSV range caps.
2. Investigate why engineered composites trail raw inputs; revise formulas, weights, or feature groups only if ablations justify it.
3. Improve the 90+ mph region with stratified modeling, interaction terms, robust loss, or more high-velocity training examples.
4. Add visual plots for fold performance, residuals, and permutation-importance intervals.
5. Consider an optional scikit-learn path for random forest or gradient boosting while preserving the dependency-free fallback.
