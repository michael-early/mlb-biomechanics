# MLB Biomechanics MVP

Portfolio project for MLB quantitative roles: build reproducible machine-learning metrics from public baseball biomechanics data and connect those metrics to pitch-performance outcomes.

## MVP Thesis

Public motion-capture data can be turned into interpretable biomechanical performance metrics that explain pitch velocity and create a bridge to MLB Statcast outcomes such as whiff rate, run value, and hard-contact suppression.

The MVP focuses on pitching. It uses OpenBiomechanics pitching data for biomechanical feature engineering and velocity modeling, then uses public Statcast data to show how velocity and pitch-quality proxies relate to MLB on-field outcomes. Because OpenBiomechanics athletes are anonymized, the project will not claim direct player-level matching to MLB results.

## Planned Resume Artifact

- A clean Python repository with reproducible data ingestion, feature engineering, modeling, evaluation, and reporting.
- A Streamlit dashboard for exploring pitcher biomechanics metrics, model explanations, and performance relationships.
- A polished analysis write-up with baseball interpretation, model validation, and limitations.
- A concise resume bullet, for example:

  Built an end-to-end Python ML pipeline using public baseball motion-capture and Statcast data to engineer biomechanical pitching metrics, predict pitch velocity, and quantify how metric-derived pitch-quality proxies relate to MLB whiff and run-value outcomes.

## Primary Public Sources

- OpenBiomechanics Project: https://www.openbiomechanics.org/
- OpenBiomechanics GitHub: https://github.com/drivelineresearch/openbiomechanics
- Baseball Savant Statcast CSV docs: https://baseballsavant.mlb.com/csv-docs
- pybaseball: https://pypi.org/project/pybaseball/

## MVP Stack

- Python 3.11+
- pandas or Polars, NumPy, scikit-learn
- XGBoost or LightGBM if dependency setup is straightforward
- SHAP or permutation importance for model explanation
- DuckDB or Parquet for local data storage
- Streamlit for the portfolio dashboard
- pytest for core feature/model tests

## Initial Scope

1. Ingest OpenBiomechanics pitching processed metrics and metadata.
2. Engineer interpretable biomechanical metrics such as sequencing efficiency, pelvis-to-torso transfer, arm-slot stability, stride/lead-leg indicators, and release consistency.
3. Train velocity prediction models with careful validation and model explanation.
4. Ingest public Statcast pitch data for MLB pitchers and model relationships among velocity, movement, release characteristics, and outcomes.
5. Create a dashboard and written report that clearly separate observed results, inferred relationships, and limitations.

