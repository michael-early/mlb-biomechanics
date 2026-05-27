# Roadmap: MLB Biomechanics MVP

**Created:** 2026-05-27
**Goal:** Build a resume-ready MVP that creates interpretable biomechanical pitching metrics and relates them to pitch performance using public data.

## Milestone 1: Resume-Ready MVP

### Phase 1: Data Foundation

**Objective:** Establish reproducible access to public biomechanics and Statcast data.

**Deliverables:**
- Python project scaffold with dependency management.
- Data directories and gitignore rules for raw/intermediate/processed data.
- OpenBiomechanics processed pitching data ingestion.
- Statcast ingestion through Baseball Savant CSV or pybaseball.
- Schema summaries for both data sources.
- Written limitations note on anonymized biomechanics data and MLB linkage.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05

**Acceptance Criteria:**
- A fresh checkout can run a documented command to build local sample datasets.
- Data-source URLs and licensing/usage notes are visible in the README or docs.
- The project explicitly prevents claims of direct athlete-to-MLB player matching.

### Phase 2: Biomechanical Metric Engineering

**Objective:** Convert raw/processed biomechanical fields into interpretable baseball metrics.

**Deliverables:**
- Metric calculation module.
- Metric dictionary with formulas, inputs, baseball interpretation, expected relationship to velocity/performance, and caveats.
- Exploratory notebook or report showing metric distributions and correlations.
- Unit tests for metric calculations on fixture data.

**Candidate Metrics:**
- Kinematic sequencing efficiency.
- Pelvis-to-torso transfer index.
- Arm-slot stability score.
- Release consistency score.
- Lead-leg block or lower-body contribution score.
- Stride/release timing score.

**Requirements:** METR-01, METR-02, METR-03, METR-04

**Acceptance Criteria:**
- Each MVP metric is computed by a reproducible script.
- Each metric can be explained in baseball language without relying on black-box terminology.
- Tests cover representative normal and missing-data cases.

### Phase 3: Biomechanics-to-Velocity Modeling

**Objective:** Evaluate whether biomechanical metrics explain or predict pitch velocity.

**Deliverables:**
- Baseline model, such as mean prediction or linear regression.
- Stronger supervised model, such as regularized regression, random forest, gradient boosting, XGBoost, or LightGBM.
- Validation report with MAE, RMSE, R2, residual diagnostics, and feature importance.
- Model-card style write-up covering limits, leakage risks, and interpretation.

**Requirements:** MODL-01, MODL-02, MODL-03, MODL-04

**Acceptance Criteria:**
- The stronger model is compared against the baseline.
- Validation split avoids obvious leakage and is documented.
- The analysis identifies which biomechanical metrics are most associated with velocity.

### Phase 4: Statcast Performance Bridge

**Objective:** Relate pitch traits and performance outcomes in public MLB data, then connect those findings back to the biomechanics-derived metric story.

**Deliverables:**
- Statcast modeling dataset aggregated at pitch-type, pitcher-season, or pitcher-pitch-type level.
- Outcome models for whiff rate, run value, chase rate, hard contact, or expected wOBA.
- Comparison between velocity/release/movement traits and outcomes.
- Clear narrative explaining what is directly observed versus inferred from biomechanics.

**Requirements:** MODL-05

**Acceptance Criteria:**
- At least one on-field outcome is modeled and visualized.
- The write-up explains why public data supports a performance bridge, not a direct private-data player linkage.
- The analysis includes limitations and what private team data would unlock.

### Phase 5: Portfolio Packaging

**Objective:** Package the project so an MLB quant reviewer can understand, run, and evaluate it quickly.

**Deliverables:**
- Polished README with thesis, screenshots, setup, commands, data sources, model results, and resume bullet.
- Streamlit dashboard or static report with metric explorer, model results, and Statcast bridge views.
- Test suite for core transformations.
- Final written analysis under `reports/`.
- Optional architecture diagram and project poster image.

**Requirements:** PROD-01, PROD-02, PROD-03, PROD-04, PROD-05

**Acceptance Criteria:**
- Reviewer can run setup and reproduce the main outputs from documented commands.
- README makes the baseball value clear in under two minutes.
- Dashboard/report presents model results without overstating claims.

## Suggested Build Order

1. Create Python scaffold and data contracts.
2. Ingest a small sample of OpenBiomechanics and Statcast data.
3. Implement metrics on the small sample with tests.
4. Scale ingestion to full local datasets.
5. Train baseline and stronger velocity models.
6. Add Statcast outcome modeling.
7. Build dashboard/report.
8. Polish README and resume framing.

## Open Questions

- Should the first implementation use pandas for familiarity or Polars for speed and modern data-engineering polish?
- Should the dashboard be Streamlit for speed or a static Quarto/Jupyter Book report for a more research-paper feel?
- Which Statcast outcome should be primary: whiff rate, run value, chase rate, or hard-contact suppression?
- Should v1 include only processed OpenBiomechanics point-of-interest metrics, or also a small raw C3D parsing demo?

