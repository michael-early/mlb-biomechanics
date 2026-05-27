# MLB Biomechanics

## What This Is

MLB Biomechanics is a portfolio-grade quantitative baseball project that uses public baseball biomechanics and Statcast data to create interpretable high-performance metrics for pitchers. The MVP will turn motion-capture variables into biomechanical performance scores, model pitch velocity, and connect those scores to on-field performance through public MLB pitch-outcome data.

The project is designed for MLB quantitative analyst applications. It should demonstrate baseball domain judgment, reproducible data engineering, machine-learning rigor, and clear communication about what public data can and cannot prove.

## Core Value

Create credible, interpretable biomechanical pitching metrics and show how they relate to measurable pitch performance without overstating public-data limitations.

## Requirements

### Validated

(None yet - ship to validate)

### Active

- [ ] Ingest public OpenBiomechanics pitching data and document the available variables, athlete anonymity constraints, and licensing/usage assumptions.
- [ ] Engineer a compact set of interpretable pitcher biomechanics metrics that map to baseball concepts such as kinetic sequencing, transfer efficiency, release stability, stride/lead-leg behavior, and arm-slot consistency.
- [ ] Train and evaluate models that predict pitch velocity from biomechanical inputs using proper train/test separation, cross-validation, and interpretable feature analysis.
- [ ] Ingest public Statcast data and model relationships among pitch velocity, release traits, movement, pitch characteristics, and on-field outcomes such as whiff rate, run value, chase rate, or hard-contact suppression.
- [ ] Build a portfolio dashboard and written analysis that make the results understandable to baseball analysts and quantitative hiring managers.
- [ ] Provide a reproducible local workflow with clear setup, data-download instructions, tests for core transformations, and a resume-ready README.

### Out of Scope

- Directly matching anonymized OpenBiomechanics athletes to MLB player identities - the public dataset is anonymized, so this would be speculative without private data.
- Injury diagnosis or medical risk prediction - this would require clinical framing, stronger validation, and appropriate medical caution.
- Full marker-level biomechanical model development from raw C3D as the first milestone - the MVP will start with processed point-of-interest metrics to keep scope resume-polishable.
- Real-time computer vision from broadcast video - interesting future work, but too large for the MVP and not necessary to demonstrate quantitative baseball skill.
- Team-internal decision tooling - the MVP should be a public portfolio artifact, not a private front-office system clone.

## Context

OpenBiomechanics provides free high-fidelity, elite-level athletic motion-capture data, including baseball pitching data, processed biomechanics data, point-of-interest metrics, and raw C3D files. Its public documentation says the repository contains data from 100 pitchers and 98 hitters, and the project is explicitly intended to support public research and education.

Baseball Savant provides public Statcast CSV documentation and query access for per-pitch MLB data. pybaseball can be used as a Python convenience layer for pulling Statcast data, but the project should document the underlying Baseball Savant source and cache downloaded data locally.

The central analytical limitation is identity linkage: OpenBiomechanics athletes are anonymized and should not be represented as known MLB players. The MVP should handle this by building a biomechanics-to-velocity model on OpenBiomechanics, then separately modeling how velocity and comparable pitch traits relate to MLB outcomes in Statcast. Any synthetic or transfer-linking analysis must be clearly labeled.

## Constraints

- **Data ethics**: Do not attempt to deanonymize OpenBiomechanics athletes - the public dataset is anonymized and should remain that way.
- **Credibility**: Do not claim causal effects or player-level MLB translation without evidence - the project is strongest when limitations are explicit.
- **Scope**: Focus the MVP on pitching, not pitching and hitting - pitching has clearer public outcome links through velocity and pitch-level Statcast outcomes.
- **Technical stack**: Prefer Python-first tooling - this is the most legible stack for MLB quant/ML portfolio review.
- **Reproducibility**: Cache raw downloads and generated datasets behind documented scripts - reviewers should be able to rerun the analysis or inspect cached schemas.
- **Portfolio polish**: The final artifact must include a concise README, clear figures, dashboard, model evaluation, and baseball interpretation - code alone is not enough for the resume goal.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Scope MVP to pitching biomechanics first | Pitching has direct biomechanical targets such as velocity and public pitch-level outcomes through Statcast. | - Pending |
| Use OpenBiomechanics for biomechanics data | It is the strongest public baseball motion-capture source and includes processed metrics suitable for an MVP. | - Pending |
| Use Baseball Savant/Statcast for on-field outcomes | It provides public pitch-level MLB performance data with documented CSV fields. | - Pending |
| Avoid player-level identity matching | OpenBiomechanics athletes are anonymized; claiming direct MLB linkage would be indefensible. | - Pending |
| Build interpretable ML before deep learning | Hiring value comes from defensible baseball insight, validation, and communication more than model complexity. | - Pending |
| Deliver dashboard plus written analysis | MLB quant reviewers need to see both technical rigor and baseball-facing explanation. | - Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-27 after initialization*

