from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .metrics import metric_dictionary


def _table(df: pd.DataFrame, max_rows: int = 12) -> str:
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0, float_format="{:.3f}".format)


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

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MLB Biomechanics MVP Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1, h2 {{ color: #0b1f33; }}
    .callout {{ background: #eef6ff; border-left: 4px solid #2176c7; padding: 12px 16px; margin: 18px 0; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
    table.data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e4e7ec; padding: 8px; text-align: left; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>MLB Biomechanics MVP Report</h1>
  <p>This report turns pitching biomechanics variables into interpretable metrics, predicts pitch velocity, and bridges comparable pitch traits to Statcast-style outcomes.</p>

  <div class="callout">
    <strong>Data limitation:</strong> OpenBiomechanics athletes are anonymized. This project does not claim direct player-level matching to MLB outcomes. It models biomechanics-to-velocity and separately models pitch-trait-to-outcome relationships.
  </div>

  <h2>Run Summary</h2>
  <div class="metric-grid">
    <div class="metric">Biomechanics source<strong>{html.escape(biomech_source)}</strong></div>
    <div class="metric">Performance source<strong>{html.escape(statcast_source)}</strong></div>
    <div class="metric">Train rows<strong>{velocity_results["train_rows"]}</strong></div>
    <div class="metric">Test rows<strong>{velocity_results["test_rows"]}</strong></div>
  </div>

  <h2>Velocity Model</h2>
  <div class="metric-grid">
    <div class="metric">Baseline RMSE<strong>{baseline["rmse"]:.2f} mph</strong></div>
    <div class="metric">Ridge RMSE<strong>{ridge["rmse"]:.2f} mph</strong></div>
    <div class="metric">Ridge MAE<strong>{ridge["mae"]:.2f} mph</strong></div>
    <div class="metric">Ridge R2<strong>{ridge["r2"]:.2f}</strong></div>
  </div>

  <h2>Biomechanical Metrics</h2>
  {_table(metric_dictionary(), max_rows=20)}

  <h2>Most Important Velocity Features</h2>
  {_table(velocity_results["permutation_importance"], max_rows=10)}

  <h2>Model Coefficients</h2>
  {_table(velocity_results["coefficients"], max_rows=10)}

  <h2>Statcast Performance Bridge</h2>
  <p>The bridge summarizes how public pitch traits relate to outcomes. If no user-supplied Statcast CSV is present, this section uses a generated Statcast-like sample and labels it accordingly.</p>
  {_table(statcast_results["pitch_type_summary"], max_rows=10)}

  <h2>Trait/Outcome Correlations</h2>
  <pre>{html.escape(json.dumps(correlations, indent=2))}</pre>

  <h2>Next Private-Data Extension</h2>
  <p>With team-owned, identity-linked motion-capture and game data, this same pipeline could evaluate whether a pitcher's biomechanical metric changes precede changes in MLB velocity, whiff rate, run value, or durability indicators.</p>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path

