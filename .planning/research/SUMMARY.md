# Research Summary: MLB Biomechanics MVP

**Created:** 2026-05-27

## Key Findings

OpenBiomechanics is the right public biomechanics foundation for the MVP. Its site describes free high-fidelity elite-level motion-capture data, including processed biomechanics data, point-of-interest metrics, C3D files, synchronized video samples, and tutorial scripts. It states that the repository currently contains data from 100 pitchers and 98 hitters.

Baseball Savant/Statcast is the right public on-field performance foundation. Baseball Savant provides CSV documentation for Statcast Search data downloads, and pybaseball provides a Python package that can query Statcast data while pointing users back to Baseball Savant's CSV field definitions.

The main project risk is overclaiming linkage. OpenBiomechanics data is anonymized and is not pitch-by-pitch MLB identity-matched performance data. The MVP should therefore make two defensible claims:

1. Biomechanical metrics can explain/predict pitch velocity within the public biomechanics dataset.
2. Velocity and related public pitch traits are associated with MLB pitch outcomes in Statcast.

The project should not claim that a specific anonymized OpenBiomechanics athlete produced specific MLB outcomes.

## Recommended MVP Scope

Pitching only. The first version should avoid hitting and avoid raw marker-level modeling except as optional exploration. Pitching has clearer performance outcomes, clearer biomechanics-to-velocity targets, and easier Statcast linkage.

## Recommended Stack

- Python 3.11+
- pandas or Polars for tabular data
- DuckDB and Parquet for local analytical storage
- scikit-learn for baseline and interpretable models
- XGBoost or LightGBM if installation is straightforward
- SHAP, permutation importance, or partial dependence for model explanation
- Streamlit for dashboard packaging
- pytest for metric and transformation tests

## Recommended Feature Categories

### Table Stakes

- Reproducible data ingestion for OpenBiomechanics and Statcast.
- Data dictionary and variable documentation.
- Biomechanical metric engineering with formulas and baseball interpretations.
- Velocity prediction model with baseline comparison.
- Statcast performance bridge using public pitch outcomes.
- Dashboard/report and polished README.

### Differentiators

- Honest, explicit public-data limitation handling.
- Metric names and explanations that sound like baseball operations work rather than generic ML features.
- Model-card style validation and leakage discussion.
- Dashboard that compares biomechanical metrics, velocity predictions, and Statcast outcome relationships.

### Anti-Features

- Deanonymization attempts.
- Medical injury diagnosis.
- Causal claims from observational public data.
- Premature deep learning that obscures baseball interpretation.

## Source Links

- OpenBiomechanics: https://www.openbiomechanics.org/
- OpenBiomechanics GitHub: https://github.com/drivelineresearch/openbiomechanics
- Driveline OpenBiomechanics announcement: https://www.drivelinebaseball.com/2022/12/openbiomechanics-project/
- Baseball Savant CSV docs: https://baseballsavant.mlb.com/csv-docs
- pybaseball package: https://pypi.org/project/pybaseball/

