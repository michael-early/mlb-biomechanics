# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Purpose

`mlb-biomechanics` is a portfolio MVP for MLB quantitative roles. It uses public pitching biomechanics data from OpenBiomechanics and public Baseball Savant/Statcast data to engineer interpretable biomechanical metrics, model fastball velocity, and bridge pitch traits to on-field performance outcomes.

The core project standard is credibility: improve the baseball and modeling rigor without overstating what public anonymized data can prove.

## Current State

Implemented MVP:

- Python package under `src/mlb_biomechanics/`
- CLI entry point: `PYTHONPATH=src python3 -m mlb_biomechanics`
- OpenBiomechanics ingestion from public CSVs
- Baseball Savant/Statcast CSV ingestion
- Engineered biomechanics metrics
- Ridge-regression velocity model
- Statcast bridge for whiff, chase, hard-hit, and pitcher run value
- Static HTML report at `reports/mvp_report.html`
- Optional Streamlit app at `app.py`
- Unit tests under `tests/`

Current local model result on OpenBiomechanics fastball data:

- 411 biomechanics rows
- 100 sessions
- train: 292 rows / 75 sessions
- test: 119 rows / 25 sessions
- baseline RMSE: 4.33 mph
- ridge RMSE: 3.07 mph
- ridge R2: 0.48
- strongest current signal: `kinetic_chain_transfer_score`

Current Statcast bridge uses a one-day Baseball Savant sample from April 1, 2025. Treat this as proof of pipeline, not as a final baseball conclusion.

## Commands

Run from repo root:

```bash
PYTHONPATH=src python3 -m mlb_biomechanics download-data
PYTHONPATH=src python3 -m mlb_biomechanics build
PYTHONPATH=src python3 -m unittest discover -s tests
```

Optional dashboard:

```bash
streamlit run app.py
```

If Streamlit is unavailable, use `reports/mvp_report.html`.

## Important Files

- `README.md` - user-facing project description and results
- `.planning/PROJECT.md` - project context and boundaries
- `.planning/REQUIREMENTS.md` - MVP requirements
- `.planning/ROADMAP.md` - planned phases
- `.planning/STATE.md` - current project state
- `src/mlb_biomechanics/metrics.py` - metric definitions
- `src/mlb_biomechanics/modeling.py` - split, ridge model, evaluation, Statcast bridge
- `src/mlb_biomechanics/report.py` - report generation
- `reports/mvp_report.html` - portfolio report

## Data Boundaries

OpenBiomechanics athletes are anonymized. Do not:

- attempt to deanonymize athletes
- claim direct player-level matching to MLB outcomes
- claim causal effects from the current observational pipeline
- frame this as injury diagnosis or medical risk prediction

The defensible framing is:

1. Public biomechanics metrics can predict fastball velocity within OpenBiomechanics.
2. Public Statcast traits can be summarized against pitch outcomes.
3. A team with identity-linked private data could test whether biomechanical changes translate to MLB outcomes.

## Next High-Value Improvements

Prioritize these before adding visual polish:

1. Replace the single holdout split with grouped K-fold cross-validation by `session`.
2. Compare ridge against at least one nonlinear model that can run with available dependencies.
3. Add bootstrap confidence intervals for RMSE, MAE, R2, and permutation importance.
4. Add residual diagnostics by velocity band and session.
5. Use a larger Statcast sample and aggregate by pitch type / pitcher / game context.
6. Improve the report narrative so it clearly distinguishes MVP result, sample limitation, and next private-data extension.

## Coding Constraints

- Keep the project runnable with standard library + NumPy + pandas.
- Do not require scikit-learn unless adding it as an optional path with graceful fallback.
- Keep tests based on `unittest` unless the dependency plan changes.
- Generated `data/` artifacts are gitignored; reports and small summary tables may be committed.
- After changes, run:

```bash
PYTHONPATH=src python3 -m mlb_biomechanics build
PYTHONPATH=src python3 -m unittest discover -s tests
```

