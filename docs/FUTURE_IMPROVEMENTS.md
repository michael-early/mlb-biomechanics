# Future Improvements: MLB Quant Rigor Upgrade

This document is the implementation roadmap for turning the working MVP into a stronger MLB quantitative-analysis talking point. The priority is rigor: validation, uncertainty, model comparison, and a defensible baseball narrative.

## Current MVP Limits

- The original MVP used one grouped holdout split, which was useful for a demo but too fragile for a serious modeling claim.
- The first Statcast bridge used a small one-day sample; the upgraded local sample is larger but still capped by Baseball Savant range export behavior.
- The report showed point estimates without uncertainty intervals or residual diagnostics.
- The public data cannot identify OpenBiomechanics athletes as MLB players, so the project must avoid player-level translation claims.

## Priority 1: Grouped Validation

Implemented direction:

- Use grouped K-fold validation by `session`.
- Ensure the same session never appears in both train and test within a fold.
- Report fold-level RMSE, MAE, and R2.
- Report mean, standard deviation, and approximate confidence intervals across folds.

Interview angle:

> I treated session leakage as the first modeling risk. The validation design groups by session so a pitcher's repeated throws do not leak across train/test folds.

## Priority 2: Model Comparison

Implemented direction:

- Compare a mean baseline, OLS, ridge with grouped inner-CV alpha selection, and dependency-free kNN.
- Keep ridge as the primary interpretable model unless a more complex model clearly wins and remains explainable.
- Write model comparison outputs to `reports/tables/cv_model_comparison.csv`.

Interview angle:

> I did not jump straight to a black-box model. I benchmarked simple, interpretable models and a nonlinear baseline so the performance claim is anchored against reasonable alternatives.

## Priority 3: Uncertainty and Diagnostics

Implemented direction:

- Bootstrap holdout RMSE, MAE, and R2 intervals.
- Repeat permutation importance shuffles to measure feature-importance stability.
- Summarize residuals by velocity band and session.
- Write diagnostics to:
  - `reports/tables/bootstrap_intervals.csv`
  - `reports/tables/residual_diagnostics.csv`
  - `reports/tables/velocity_permutation_importance.csv`

Interview angle:

> The takeaway is not just a single RMSE. I added uncertainty and residual diagnostics to show where the model is reliable and where it needs more data.

## Priority 4: Larger Statcast Bridge

Current direction:

- The code supports Baseball Savant CSV ingestion and pitcher-pitch-type aggregation.
- The current local sample is a capped Baseball Savant range export with 25,000 pitch rows across April 24-30, 2025.
- The next data upgrade should use chunked daily/weekly Baseball Savant exports to build a 4-8 week or full-season Statcast dataset without range caps.
- Keep this as a bridge, not a direct player-level join, unless identity-linked private data becomes available.

Interview angle:

> Public biomechanics data and public Statcast data answer adjacent questions. The bridge shows how I would structure the team version once private identity-linked data exists.

## Priority 5: Public Report Narrative

Implemented direction:

- Center the report on one defensible claim: kinetic-chain transfer is the strongest engineered signal for fastball velocity in the public OpenBiomechanics sample.
- Separate observed evidence from future private-data extensions.
- Keep limitations visible near the top of the report.

Interview angle:

> I designed the project to be technically honest. It shows what can be learned from public data and exactly what additional private data would unlock for a club.

## Next Best Work

1. Add chunked Statcast downloading so a full 4-8 week or season sample can be reproduced without CSV range caps.
2. Add a final model-card section with data, target, validation design, model comparison, limitations, and intended use.
3. Add visual plots for fold performance, residuals, and permutation-importance intervals.
4. Consider an optional scikit-learn path for random forest or gradient boosting while preserving the dependency-free fallback.
