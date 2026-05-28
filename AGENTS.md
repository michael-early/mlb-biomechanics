# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Purpose

`mlb-biomechanics` is a standalone MLB quantitative analysis project. It uses public pitching biomechanics data from OpenBiomechanics to engineer interpretable metrics and model fastball velocity. It also uses public Baseball Savant/Statcast data for a separate pitch-trait/outcome bridge.

The core project standard is credibility: improve the baseball and modeling rigor without overstating what public anonymized data can prove.

## Current State

Implemented study:

- Python package under `src/mlb_biomechanics/`
- CLI entry point: `PYTHONPATH=src python3 -m mlb_biomechanics`
- OpenBiomechanics ingestion from public CSVs
- Baseball Savant/Statcast CSV ingestion
- Engineered biomechanics metrics
- Leakage-safe train-fit/test-transform metric construction
- Ridge-regression velocity model with repeated grouped validation
- Feature-set comparisons, metric ablations, session bootstrap intervals, and high-velocity diagnostics
- Statcast bridge for whiff, chase, hard-hit, and pitcher run value with baseball-valid rate denominators
- Static HTML report at `reports/mvp_report.html`
- Optional Streamlit app at `app.py`
- Unit tests under `tests/`

Current local model result on OpenBiomechanics fastball data:

- 411 biomechanics rows
- 100 sessions
- train: 292 rows / 75 sessions
- test: 119 rows / 25 sessions
- baseline RMSE: 4.33 mph
- ridge engineered-metrics holdout RMSE: 2.98 mph
- ridge engineered-metrics holdout R2: 0.51
- repeated grouped-CV ridge engineered-metrics RMSE: 3.51 mph
- repeated grouped-CV raw metric-input RMSE: 3.18 mph
- strongest current signal: `kinetic_chain_transfer_score`
- key current weakness: engineered composites are interpretable but trail raw inputs in repeated CV
- high-velocity weakness: 90+ mph pitches are underpredicted by about 3.08 mph on the current holdout

Current Statcast bridge uses a capped Baseball Savant range sample: 25,000 pitch rows across April 24-30, 2025 in the current local data. Treat this as a stronger pipeline sample, not as a final full-season baseball conclusion.

The source of truth for implemented rigor work and remaining upgrades is `docs/FUTURE_IMPROVEMENTS.md`.

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
- `docs/FUTURE_IMPROVEMENTS.md` - documented next-step roadmap and discussion points
- `reports/mvp_report.html` - generated project report

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

Prioritize these before adding visual polish. See `docs/FUTURE_IMPROVEMENTS.md` for the full documented plan:

1. Add chunked Baseball Savant ingestion for a reproducible 4-8 week or full-season Statcast sample.
2. Investigate why raw biomechanical inputs outperform engineered composites; revise metric formulas only if ablations support it.
3. Improve high-velocity performance for the 90+ mph band.
4. Add visual plots for CV performance, residuals, permutation importance, and Statcast outcomes.
5. Keep the report narrative honest: velocity modeling in OpenBiomechanics, separate public Statcast bridge, no direct MLB player linkage.

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
