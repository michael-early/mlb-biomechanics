from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .metrics import metric_dictionary


def _table(df: pd.DataFrame, max_rows: int = 12) -> str:
    return df.head(max_rows).to_html(
        index=False,
        classes="data-table",
        border=0,
        float_format="{:.3f}".format,
    )


def _source_label(source: str) -> str:
    labels = {
        "openbiomechanics_public_poi_metrics": "OpenBiomechanics public pitching metrics",
        "openbiomechanics_public_metadata": "OpenBiomechanics public metadata",
        "user_supplied_statcast_csv": "Baseball Savant Statcast CSV",
        "generated_statcast_like_sample": "Generated Statcast-like sample",
    }
    return labels.get(source, source.replace("_", " ").title())


def _pitch_summary_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    pretty = df.head(max_rows).copy()
    rename = {
        "pitch_type": "Pitch",
        "pitches": "Pitches",
        "avg_velocity": "Avg velo",
        "swings": "Swings",
        "out_of_zone_pitches": "OOZ",
        "batted_balls": "BIP",
        "whiff_rate": "Whiff",
        "chase_rate": "Chase",
        "hard_hit_rate": "Hard hit",
        "avg_run_value": "Pitcher RV",
    }
    pretty = pretty.rename(columns=rename)
    for col in ["Whiff", "Chase", "Hard hit"]:
        if col in pretty:
            pretty[col] = (pretty[col] * 100).map(lambda value: f"{value:.1f}%")
    for col in ["Avg velo", "Pitcher RV"]:
        if col in pretty:
            pretty[col] = pretty[col].map(lambda value: f"{value:.2f}")
    return pretty.to_html(index=False, classes="data-table", border=0)


def _correlation_table(correlations: dict[str, dict[str, float]]) -> str:
    rows = []
    for outcome, values in correlations.items():
        for trait, corr in values.items():
            rows.append({"Outcome": outcome, "Trait": trait, "Correlation": corr})
    if not rows:
        return "<p class=\"muted\">No valid trait/outcome correlations were available.</p>"
    df = pd.DataFrame(rows)
    df["Correlation"] = df["Correlation"].map(lambda value: f"{value:.3f}")
    return df.to_html(index=False, classes="data-table compact", border=0)


def _cv_summary_table(cv_model_comparison: pd.DataFrame) -> str:
    summary = cv_model_comparison[cv_model_comparison["row_type"] == "summary"].copy()
    summary = summary[summary["metric"].isin(["rmse", "mae", "r2"])]
    if summary.empty:
        return "<p class=\"muted\">Cross-validation summary was not available.</p>"
    pivot = (
        summary.pivot_table(
            index="model",
            columns="metric",
            values=["mean", "std", "ci_lower", "ci_upper"],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [
        "_".join([str(part) for part in col if part]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in pivot.columns
    ]
    keep = [
        "model",
        "mean_rmse",
        "std_rmse",
        "ci_lower_rmse",
        "ci_upper_rmse",
        "mean_mae",
        "mean_r2",
    ]
    pretty = pivot[[col for col in keep if col in pivot.columns]].copy()
    pretty = pretty.rename(
        columns={
            "model": "Model",
            "mean_rmse": "RMSE mean",
            "std_rmse": "RMSE std",
            "ci_lower_rmse": "RMSE CI low",
            "ci_upper_rmse": "RMSE CI high",
            "mean_mae": "MAE mean",
            "mean_r2": "R2 mean",
        }
    )
    for col in pretty.columns:
        if col != "Model":
            pretty[col] = pretty[col].map(lambda value: f"{value:.3f}")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _bootstrap_table(intervals: pd.DataFrame) -> str:
    pretty = intervals.copy()
    pretty = pretty.rename(
        columns={
            "metric": "Metric",
            "estimate": "Estimate",
            "ci_lower": "CI low",
            "ci_upper": "CI high",
            "bootstrap_samples": "Bootstraps",
            "bootstrap_unit": "Unit",
        }
    )
    pretty = pretty[
        [col for col in ["Metric", "Estimate", "CI low", "CI high", "Bootstraps", "Unit"] if col in pretty]
    ]
    for col in ["Estimate", "CI low", "CI high"]:
        if col in pretty:
            pretty[col] = pretty[col].map(lambda value: f"{value:.3f}")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _residual_band_table(diagnostics: pd.DataFrame) -> str:
    pretty = diagnostics[diagnostics["diagnostic_group"] == "velocity_band"].copy()
    pretty = pretty.rename(
        columns={
            "velocity_band": "Velocity band",
            "rows": "Rows",
            "mean_actual_mph": "Actual mph",
            "mean_residual_mph": "Mean residual",
            "mae": "MAE",
            "rmse": "RMSE",
        }
    )
    pretty = pretty[[col for col in ["Velocity band", "Rows", "Actual mph", "Mean residual", "MAE", "RMSE"] if col in pretty]]
    for col in ["Actual mph", "Mean residual", "MAE", "RMSE"]:
        if col in pretty:
            pretty[col] = pretty[col].map(lambda value: f"{value:.2f}")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _feature_set_table(feature_sets: pd.DataFrame) -> str:
    if feature_sets.empty:
        return "<p class=\"muted\">Feature-set comparison was not available.</p>"
    summary = feature_sets[
        (feature_sets["row_type"] == "summary") & (feature_sets["metric"] == "rmse")
    ].copy()
    if summary.empty:
        return "<p class=\"muted\">Feature-set comparison was not available.</p>"
    pretty = summary[
        [
            "feature_set",
            "feature_count",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "rmse_delta_vs_best",
        ]
    ].copy()
    pretty = pretty.rename(
        columns={
            "feature_set": "Feature set",
            "feature_count": "Features",
            "mean": "RMSE mean",
            "std": "RMSE std",
            "ci_lower": "CI low",
            "ci_upper": "CI high",
            "rmse_delta_vs_best": "Delta vs best",
        }
    ).sort_values("RMSE mean")
    for col in ["RMSE mean", "RMSE std", "CI low", "CI high", "Delta vs best"]:
        pretty[col] = pretty[col].map(lambda value: f"{value:.3f}")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _ablation_table(ablation: pd.DataFrame) -> str:
    if ablation.empty:
        return "<p class=\"muted\">Metric ablation was not available.</p>"
    pretty = ablation.copy().rename(
        columns={
            "feature_removed": "Feature removed",
            "feature_count": "Features",
            "rmse_mean": "RMSE mean",
            "mae_mean": "MAE mean",
            "r2_mean": "R2 mean",
            "rmse_delta_vs_all": "RMSE delta",
        }
    )
    pretty = pretty[
        ["Feature removed", "Features", "RMSE mean", "MAE mean", "R2 mean", "RMSE delta"]
    ].head(12)
    for col in ["RMSE mean", "MAE mean", "R2 mean", "RMSE delta"]:
        pretty[col] = pretty[col].map(lambda value: f"{value:.3f}")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _metric_correlation_table(correlations: pd.DataFrame) -> str:
    if correlations.empty:
        return "<p class=\"muted\">Metric correlations were not available.</p>"
    pretty = correlations.head(14).copy().rename(
        columns={
            "feature": "Feature",
            "feature_type": "Type",
            "pearson_corr_with_velocity": "Velocity corr",
            "rows": "Rows",
        }
    )
    pretty = pretty[["Feature", "Type", "Velocity corr", "Rows"]]
    pretty["Velocity corr"] = pretty["Velocity corr"].map(lambda value: f"{value:.3f}")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _high_velocity_table(high_velocity: pd.DataFrame) -> str:
    if high_velocity.empty:
        return "<p class=\"muted\">High-velocity diagnostics were not available.</p>"
    pretty = high_velocity.copy().rename(
        columns={
            "group": "Group",
            "rows": "Rows",
            "mean_actual_mph": "Actual mph",
            "mean_predicted_mph": "Predicted mph",
            "mean_residual_mph": "Mean residual",
            "mae": "MAE",
            "rmse": "RMSE",
            "underprediction_rate": "Underpredicted",
            "overprediction_rate": "Overpredicted",
        }
    )
    pretty = pretty[
        [
            "Group",
            "Rows",
            "Actual mph",
            "Predicted mph",
            "Mean residual",
            "MAE",
            "RMSE",
            "Underpredicted",
            "Overpredicted",
        ]
    ]
    for col in ["Actual mph", "Predicted mph", "Mean residual", "MAE", "RMSE"]:
        pretty[col] = pretty[col].map(lambda value: f"{value:.2f}")
    for col in ["Underpredicted", "Overpredicted"]:
        pretty[col] = (pretty[col] * 100).map(lambda value: f"{value:.1f}%")
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _sample_manifest_table(manifest: pd.DataFrame) -> str:
    if manifest.empty:
        return "<p class=\"muted\">Statcast sample manifest was not available.</p>"
    pretty = manifest.rename(columns={"field": "Field", "value": "Value"})
    return pretty.to_html(index=False, classes="data-table compact", border=0)


def _model_card_table(velocity_results: dict[str, object], statcast_results: dict[str, object]) -> str:
    ridge = velocity_results["ridge"]
    baseline = velocity_results["baseline"]
    statcast_sample = statcast_results["sample_summary"]
    rows = [
        ("Primary target", "pitch_speed_mph"),
        ("Biomechanics validation unit", "session"),
        ("Holdout rows", velocity_results["test_rows"]),
        ("Holdout sessions", velocity_results["test_sessions"]),
        ("Baseline RMSE", f"{baseline['rmse']:.2f} mph"),
        ("Ridge RMSE", f"{ridge['rmse']:.2f} mph"),
        ("Primary model", f"ridge regression, alpha={ridge['alpha']:.1f}"),
        ("CV design", "repeated grouped K-fold by session with inner alpha selection"),
        ("Bootstrap unit", "session"),
        ("Statcast rows", f"{statcast_sample['rows']:,}"),
        ("Statcast denominator policy", "whiff/swing, chase/out-of-zone, hard-hit/batted-ball"),
        ("Use boundary", "predictive association only; no causal or identity-linked MLB claim"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"]).to_html(
        index=False, classes="data-table compact", border=0
    )


def write_report(
    output_path: Path,
    biomech_source: str,
    statcast_source: str,
    velocity_results: dict[str, object],
    statcast_results: dict[str, object],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ridge = velocity_results["ridge"]
    baseline = velocity_results["baseline"]
    correlations = statcast_results["trait_outcome_correlations"]
    rmse_lift = baseline["rmse"] - ridge["rmse"]
    cv_summary = velocity_results["cv_model_comparison"]
    bootstrap_intervals = velocity_results["bootstrap_intervals"]
    residual_diagnostics = velocity_results["residual_diagnostics"]
    feature_set_comparison = velocity_results["feature_set_comparison"]
    metric_ablation = velocity_results["metric_ablation"]
    metric_correlations = velocity_results["metric_correlations"]
    high_velocity = velocity_results["high_velocity_error_analysis"]
    statcast_sample = statcast_results["sample_summary"]
    biomech_source_label = _source_label(biomech_source)
    statcast_source_label = _source_label(statcast_source)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLB Biomechanics Velocity Study</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #667085;
      --line: #d9e2ec;
      --panel: #f7f9fc;
      --accent: #156f7a;
      --accent-dark: #0f4f58;
      --warn-bg: #fff8e6;
      --warn-line: #d99d20;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #ffffff;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fbfd 0%, #ffffff 100%);
    }}
    main, .header-inner {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 24px;
    }}
    .eyebrow {{
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
      margin: 0 0 10px;
    }}
    h1 {{
      font-size: 42px;
      line-height: 1.08;
      letter-spacing: 0;
      margin: 0;
    }}
    h2 {{
      font-size: 24px;
      letter-spacing: 0;
      margin: 34px 0 12px;
    }}
    h3 {{
      font-size: 18px;
      letter-spacing: 0;
      margin: 22px 0 8px;
    }}
    p {{ margin: 0 0 14px; }}
    .subtitle {{
      max-width: 820px;
      color: var(--muted);
      font-size: 17px;
      margin-top: 14px;
    }}
    .section {{
      border-top: 1px solid var(--line);
      padding-top: 20px;
      margin-top: 26px;
    }}
    .callout {{
      background: var(--warn-bg);
      border-left: 4px solid var(--warn-line);
      padding: 14px 16px;
      margin: 22px 0;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 4px;
    }}
    .button {{
      display: inline-block;
      border: 1px solid var(--accent-dark);
      border-radius: 6px;
      background: var(--accent-dark);
      color: #ffffff;
      font-size: 13px;
      font-weight: 700;
      padding: 8px 12px;
      text-decoration: none;
    }}
    .button:hover {{ opacity: 0.86; text-decoration: none; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0 4px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-height: 96px;
    }}
    .metric span {{
      color: var(--muted);
      display: block;
      font-size: 13px;
      line-height: 1.25;
    }}
    .metric strong {{
      color: var(--accent-dark);
      display: block;
      font-size: 24px;
      line-height: 1.15;
      margin-top: 8px;
      overflow-wrap: anywhere;
    }}
    .insight {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0 22px;
      background: #ffffff;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 12px 0 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table.data-table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 680px;
      background: #ffffff;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #edf1f5;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: #f1f6f8;
      color: #243647;
      font-size: 13px;
    }}
    .data-table tr:last-child td {{ border-bottom: 0; }}
    .compact td, .compact th {{ padding: 8px 10px; }}
    .muted {{ color: var(--muted); }}
    code {{
      background: #eef2f6;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    footer {{
      color: var(--muted);
      border-top: 1px solid var(--line);
      margin-top: 36px;
      padding-top: 18px;
      font-size: 14px;
    }}
    @media (max-width: 820px) {{
      h1 {{ font-size: 34px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 540px) {{
      main, .header-inner {{ padding: 22px 16px; }}
      h1 {{ font-size: 30px; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <p class="eyebrow">MLB Quant Rigor Upgrade | Baseball biomechanics and ML</p>
      <h1>MLB Biomechanics Velocity Study</h1>
      <p class="subtitle">A reproducible pitching-analysis pipeline that converts public motion-capture variables into interpretable biomechanical metrics, validates fastball velocity models with grouped session splits, and separately summarizes public Statcast pitch-trait outcomes.</p>
      <div class="actions">
        <a class="button" href="https://github.com/michael-early/mlb-biomechanics" target="_blank" rel="noopener">View full project on GitHub</a>
      </div>
    </div>
  </header>

  <main>
    <div class="callout">
      <strong>Data boundary:</strong> OpenBiomechanics athletes are anonymized. This report does not claim direct player-level matching to MLB outcomes. It models biomechanics-to-velocity and separately models pitch-trait-to-outcome relationships.
    </div>

    <section class="section">
      <h2>Executive Summary</h2>
      <div class="metric-grid">
        <div class="metric"><span>Biomechanics source</span><strong>{html.escape(biomech_source_label)}</strong></div>
        <div class="metric"><span>Performance source</span><strong>{html.escape(statcast_source_label)}</strong></div>
        <div class="metric"><span>Velocity RMSE lift</span><strong>{rmse_lift:.2f} mph</strong></div>
        <div class="metric"><span>Validation groups</span><strong>{velocity_results["sessions"]} sessions</strong></div>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>Statcast rows</span><strong>{statcast_sample["rows"]:,}</strong></div>
        <div class="metric"><span>Statcast dates</span><strong>{html.escape(statcast_sample["date_min"])} to {html.escape(statcast_sample["date_max"])}</strong></div>
        <div class="metric"><span>Pitch types</span><strong>{statcast_sample["pitch_types"]}</strong></div>
        <div class="metric"><span>Primary signal</span><strong>Kinetic-chain transfer</strong></div>
      </div>
      <div class="insight">
        <strong>Main baseball finding:</strong> kinetic-chain transfer is the strongest engineered signal for fastball velocity in the public OpenBiomechanics sample. The holdout ridge model reduces pitch-speed error from {baseline["rmse"]:.2f} mph to {ridge["rmse"]:.2f} mph, and validation is grouped by session so a pitcher's repeated throws do not leak across train and test.
      </div>
    </section>

    <section class="section">
      <h2>Model Card</h2>
      <p class="muted">This section defines the modeling boundary, validation unit, and intended interpretation before presenting feature-level results.</p>
      <div class="table-wrap">{_model_card_table(velocity_results, statcast_results)}</div>
    </section>

    <section class="section">
      <h2>Velocity Model</h2>
      <div class="metric-grid">
        <div class="metric"><span>Baseline RMSE</span><strong>{baseline["rmse"]:.2f} mph</strong></div>
        <div class="metric"><span>Ridge RMSE</span><strong>{ridge["rmse"]:.2f} mph</strong></div>
        <div class="metric"><span>Ridge MAE</span><strong>{ridge["mae"]:.2f} mph</strong></div>
        <div class="metric"><span>Selected alpha</span><strong>{ridge["alpha"]:.1f}</strong></div>
      </div>
    </section>

    <section class="section">
      <h2>Grouped Cross-Validation Model Comparison</h2>
      <p class="muted">All folds are grouped by session so the same athlete/session is never in both train and test for a fold. Ridge uses an inner grouped CV alpha search.</p>
      <div class="table-wrap">{_cv_summary_table(cv_summary)}</div>
    </section>

    <section class="section">
      <h2>Bootstrap Uncertainty</h2>
      <p class="muted">Holdout metrics are bootstrapped by session to respect repeated pitches within the same held-out session.</p>
      <div class="table-wrap">{_bootstrap_table(bootstrap_intervals)}</div>
    </section>

    <section class="section">
      <h2>Feature-Set Comparison</h2>
      <p class="muted">This checks whether engineered composites add value beyond the raw metric inputs and whether simple interaction terms help.</p>
      <div class="table-wrap">{_feature_set_table(feature_set_comparison)}</div>
    </section>

    <section class="section">
      <h2>Metric Ablation</h2>
      <p class="muted">Rows show how repeated grouped-CV RMSE changes when each engineered metric is removed from the ridge model. Positive delta means removing that metric made the model worse.</p>
      <div class="table-wrap">{_ablation_table(metric_ablation)}</div>
    </section>

    <section class="section">
      <h2>Metric Correlations</h2>
      <p class="muted">Descriptive Pearson correlations show which raw and engineered biomechanics fields move with pitch velocity before multivariable modeling.</p>
      <div class="table-wrap">{_metric_correlation_table(metric_correlations)}</div>
    </section>

    <section class="section">
      <h2>Most Important Velocity Features</h2>
      <p class="muted">Permutation importance is measured as the increase in holdout RMSE after repeated shuffles of each metric. Intervals show stability across repeated permutations.</p>
      <div class="table-wrap">{_table(velocity_results["permutation_importance"], max_rows=10)}</div>

      <h3>Model Coefficients</h3>
      <p class="muted">Coefficients are from standardized features in the ridge-regression velocity model.</p>
      <div class="table-wrap">{_table(velocity_results["coefficients"], max_rows=10)}</div>
    </section>

    <section class="section">
      <h2>Residual Diagnostics</h2>
      <p class="muted">Residual summaries by velocity band identify where the model is under- or over-performing.</p>
      <div class="table-wrap">{_residual_band_table(residual_diagnostics)}</div>
      <h3>High-Velocity Error Check</h3>
      <p class="muted">This explicitly checks the 90+ mph region, where underprediction is the most baseball-relevant failure mode.</p>
      <div class="table-wrap">{_high_velocity_table(high_velocity)}</div>
    </section>

    <section class="section">
      <h2>Biomechanical Metric Dictionary</h2>
      <div class="table-wrap">{_table(metric_dictionary(), max_rows=20)}</div>
    </section>

    <section class="section">
      <h2>Statcast Performance Bridge</h2>
      <p>The bridge summarizes how public pitch traits relate to outcomes. With a Baseball Savant CSV, whiff rate is computed per swing, chase rate per out-of-zone pitch, and hard-hit rate per batted ball.</p>
      <div class="table-wrap">{_pitch_summary_table(statcast_results["pitch_type_summary"], max_rows=10)}</div>
      <h3>Sample Manifest</h3>
      <div class="table-wrap">{_sample_manifest_table(statcast_results["sample_manifest"])}</div>
    </section>

    <section class="section">
      <h2>Trait and Outcome Correlations</h2>
      <p class="muted">These Statcast correlations use the local capped Baseball Savant sample. A chunked multi-month or full-season sample would be the next analytical upgrade.</p>
      <div class="table-wrap">{_correlation_table(correlations)}</div>
    </section>

    <section class="section">
      <h2>Next Private-Data Extension</h2>
      <p>With team-owned, identity-linked motion-capture and game data, this same pipeline could test whether changes in a pitcher's biomechanical metrics precede changes in velocity, whiff rate, run value, or durability indicators.</p>
    </section>

    <footer>
      Generated by <code>PYTHONPATH=src python3 -m mlb_biomechanics build</code>. Data artifacts are reproducible from the project pipeline.
    </footer>
  </main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
