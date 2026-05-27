# Requirements: MLB Biomechanics

**Defined:** 2026-05-27
**Core Value:** Create credible, interpretable biomechanical pitching metrics and show how they relate to measurable pitch performance without overstating public-data limitations.

## v1 Requirements

Requirements for the first public resume-ready MVP.

### Data

- [ ] **DATA-01**: Project can download or load OpenBiomechanics pitching processed metrics and metadata from documented public sources.
- [ ] **DATA-02**: Project stores raw and processed datasets in a reproducible local layout with generated data excluded from git where appropriate.
- [ ] **DATA-03**: Project creates a data dictionary or schema summary for biomechanical variables used in modeling.
- [ ] **DATA-04**: Project can download or load Statcast pitch-level data for a documented season/sample using Baseball Savant or pybaseball.
- [ ] **DATA-05**: Project documents the identity-linkage limitation between anonymized biomechanics athletes and MLB players.

### Metrics

- [ ] **METR-01**: Project computes a small, named set of biomechanical pitching metrics from OpenBiomechanics variables.
- [ ] **METR-02**: Each metric has a baseball interpretation, formula/source variables, expected directionality, and caveats.
- [ ] **METR-03**: Metrics include at least one sequencing/transfer metric, one release-stability metric, and one lower-body/lead-leg metric.
- [ ] **METR-04**: Metric outputs are reproducible from scripts and covered by basic validation tests.

### Modeling

- [ ] **MODL-01**: Project trains a baseline model predicting pitch velocity from biomechanical metrics.
- [ ] **MODL-02**: Project trains at least one stronger supervised model, such as regularized regression, random forest, gradient boosting, XGBoost, or LightGBM.
- [ ] **MODL-03**: Model evaluation includes train/test separation, cross-validation or grouped validation where applicable, and error metrics such as MAE/RMSE/R2.
- [ ] **MODL-04**: Project explains model behavior using interpretable coefficients, permutation importance, SHAP, or partial dependence.
- [ ] **MODL-05**: Project performs a separate Statcast analysis connecting pitch traits to outcomes such as whiff rate, run value, chase rate, or hard-contact suppression.

### Product

- [ ] **PROD-01**: Repository includes a polished README with project thesis, data sources, setup instructions, limitations, and resume framing.
- [ ] **PROD-02**: Repository includes a dashboard or report that lets a reviewer inspect metrics, model performance, and baseball interpretation.
- [ ] **PROD-03**: Project includes automated tests for core data transformations and metric calculations.
- [ ] **PROD-04**: Project includes a final analysis narrative that distinguishes observed evidence, inferred relationships, and future private-data extensions.
- [ ] **PROD-05**: Project can be run from a documented command sequence on a fresh local checkout.

## v2 Requirements

Deferred until after the first resume-ready MVP.

### Advanced Biomechanics

- **ADV-01**: Parse raw C3D marker trajectories directly and reproduce selected processed metrics.
- **ADV-02**: Add time-series sequence modeling over full motion traces.
- **ADV-03**: Add uncertainty intervals for athlete-level or pitch-level metric estimates.

### Video and Scouting

- **VID-01**: Estimate pitching pose from public or user-provided video.
- **VID-02**: Compare video-derived mechanics against motion-capture-derived metrics.
- **SCOUT-01**: Generate pitcher development or scouting notes from model outputs.

### Broader Baseball Scope

- **HIT-01**: Extend the project to OpenBiomechanics hitting data and batted-ball performance.
- **TEAM-01**: Add private-data integration interfaces for a hypothetical team environment.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Deanonymizing OpenBiomechanics athletes | Violates the spirit of the dataset and cannot be defended from public evidence. |
| Medical injury diagnosis | Requires clinical validation and is not needed for an MLB quant portfolio MVP. |
| Production web authentication or multi-user storage | The MVP is a public analytical artifact, not a SaaS product. |
| Full raw C3D processing in v1 | Valuable but too large for the first resume-ready release; use processed metrics first. |
| Claiming direct causal impact on MLB outcomes | Public data supports association and modeling, not causal claims. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| METR-01 | Phase 2 | Pending |
| METR-02 | Phase 2 | Pending |
| METR-03 | Phase 2 | Pending |
| METR-04 | Phase 2 | Pending |
| MODL-01 | Phase 3 | Pending |
| MODL-02 | Phase 3 | Pending |
| MODL-03 | Phase 3 | Pending |
| MODL-04 | Phase 3 | Pending |
| MODL-05 | Phase 4 | Pending |
| PROD-01 | Phase 5 | Pending |
| PROD-02 | Phase 5 | Pending |
| PROD-03 | Phase 5 | Pending |
| PROD-04 | Phase 5 | Pending |
| PROD-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-05-27*
*Last updated: 2026-05-27 after initial definition*

