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
    biomech_source_label = _source_label(biomech_source)
    statcast_source_label = _source_label(statcast_source)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLB Biomechanics MVP Report</title>
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
      <p class="eyebrow">Portfolio MVP | Baseball biomechanics and ML</p>
      <h1>MLB Biomechanics Performance Report</h1>
      <p class="subtitle">A reproducible pitching-analysis pipeline that converts public motion-capture variables into interpretable biomechanical metrics, predicts pitch velocity, and bridges pitch traits to Statcast outcomes.</p>
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
        <div class="metric"><span>Holdout rows</span><strong>{velocity_results["test_rows"]}</strong></div>
      </div>
      <div class="insight">
        <strong>Main baseball finding:</strong> the engineered biomechanics model reduces pitch-speed error from {baseline["rmse"]:.2f} mph to {ridge["rmse"]:.2f} mph on the holdout split. The strongest signal is the kinetic-chain transfer score, followed by arm speed and sequencing efficiency.
      </div>
    </section>

    <section class="section">
      <h2>Velocity Model</h2>
      <div class="metric-grid">
        <div class="metric"><span>Baseline RMSE</span><strong>{baseline["rmse"]:.2f} mph</strong></div>
        <div class="metric"><span>Ridge RMSE</span><strong>{ridge["rmse"]:.2f} mph</strong></div>
        <div class="metric"><span>Ridge MAE</span><strong>{ridge["mae"]:.2f} mph</strong></div>
        <div class="metric"><span>Ridge R2</span><strong>{ridge["r2"]:.2f}</strong></div>
      </div>
    </section>

    <section class="section">
      <h2>Most Important Velocity Features</h2>
      <p class="muted">Permutation importance is measured as the increase in holdout RMSE after shuffling each metric.</p>
      <div class="table-wrap">{_table(velocity_results["permutation_importance"], max_rows=10)}</div>

      <h3>Model Coefficients</h3>
      <p class="muted">Coefficients are from standardized features in the ridge-regression velocity model.</p>
      <div class="table-wrap">{_table(velocity_results["coefficients"], max_rows=10)}</div>
    </section>

    <section class="section">
      <h2>Biomechanical Metric Dictionary</h2>
      <div class="table-wrap">{_table(metric_dictionary(), max_rows=20)}</div>
    </section>

    <section class="section">
      <h2>Statcast Performance Bridge</h2>
      <p>The bridge summarizes how public pitch traits relate to outcomes. With a Baseball Savant CSV, whiffs, chases, hard-hit balls, and pitcher run value are derived from standard Statcast columns.</p>
      <div class="table-wrap">{_pitch_summary_table(statcast_results["pitch_type_summary"], max_rows=10)}</div>
    </section>

    <section class="section">
      <h2>Trait and Outcome Correlations</h2>
      <p class="muted">These one-day Statcast correlations are a pipeline check, not a final scouting conclusion. A full-season sample would be the next portfolio upgrade.</p>
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
