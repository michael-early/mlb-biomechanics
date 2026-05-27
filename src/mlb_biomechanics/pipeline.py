from __future__ import annotations

import json

from .data import load_biomechanics, load_metadata, load_statcast
from .metrics import add_biomechanical_metrics, metric_dictionary
from .modeling import evaluate_velocity_model, statcast_performance_bridge
from .paths import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs
from .report import write_report


def build_mvp() -> dict[str, str]:
    ensure_project_dirs()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    biomechanics, biomechanics_source = load_biomechanics()
    metadata, metadata_source = load_metadata()
    statcast, statcast_source = load_statcast()

    metrics = add_biomechanical_metrics(biomechanics)
    velocity_results = evaluate_velocity_model(metrics)
    statcast_results = statcast_performance_bridge(statcast)

    metrics_path = PROCESSED_DIR / "biomechanics_metrics.csv"
    predictions_path = PROCESSED_DIR / "velocity_predictions.csv"
    coeffs_path = REPORTS_DIR / "tables" / "velocity_model_coefficients.csv"
    importance_path = REPORTS_DIR / "tables" / "velocity_permutation_importance.csv"
    metric_dict_path = REPORTS_DIR / "tables" / "biomechanical_metric_dictionary.csv"
    statcast_summary_path = REPORTS_DIR / "tables" / "statcast_pitch_type_summary.csv"
    pitcher_pitch_type_path = REPORTS_DIR / "tables" / "statcast_pitcher_pitch_type_summary.csv"
    cv_model_comparison_path = REPORTS_DIR / "tables" / "cv_model_comparison.csv"
    bootstrap_intervals_path = REPORTS_DIR / "tables" / "bootstrap_intervals.csv"
    residual_diagnostics_path = REPORTS_DIR / "tables" / "residual_diagnostics.csv"
    alpha_selection_path = REPORTS_DIR / "tables" / "ridge_alpha_selection.csv"
    model_metrics_path = PROCESSED_DIR / "velocity_model_metrics.json"
    data_sources_path = PROCESSED_DIR / "data_sources.json"
    report_path = REPORTS_DIR / "mvp_report.html"

    metrics.to_csv(metrics_path, index=False)
    velocity_results["predictions"].to_csv(predictions_path, index=False)
    velocity_results["coefficients"].to_csv(coeffs_path, index=False)
    velocity_results["permutation_importance"].to_csv(importance_path, index=False)
    metric_dictionary().to_csv(metric_dict_path, index=False)
    statcast_results["pitch_type_summary"].to_csv(statcast_summary_path, index=False)
    statcast_results["pitcher_pitch_type_summary"].to_csv(pitcher_pitch_type_path, index=False)
    velocity_results["cv_model_comparison"].to_csv(cv_model_comparison_path, index=False)
    velocity_results["bootstrap_intervals"].to_csv(bootstrap_intervals_path, index=False)
    velocity_results["residual_diagnostics"].to_csv(residual_diagnostics_path, index=False)
    velocity_results["alpha_selection"].to_csv(alpha_selection_path, index=False)

    model_metrics = {
        "train_rows": velocity_results["train_rows"],
        "test_rows": velocity_results["test_rows"],
        "baseline": velocity_results["baseline"],
        "ridge": velocity_results["ridge"],
        "sessions": velocity_results["sessions"],
    }
    model_metrics_path.write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")
    data_sources_path.write_text(
        json.dumps(
            {
                "biomechanics": biomechanics_source,
                "metadata": metadata_source,
                "metadata_rows": int(len(metadata)),
                "statcast": statcast_source,
                "statcast_rows": statcast_results["sample_summary"]["rows"],
                "statcast_date_min": statcast_results["sample_summary"]["date_min"],
                "statcast_date_max": statcast_results["sample_summary"]["date_max"],
                "statcast_pitch_types": statcast_results["sample_summary"]["pitch_types"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(report_path, biomechanics_source, statcast_source, velocity_results, statcast_results)

    return {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "model_metrics": str(model_metrics_path),
        "data_sources": str(data_sources_path),
        "report": str(report_path),
        "metric_dictionary": str(metric_dict_path),
        "statcast_summary": str(statcast_summary_path),
        "pitcher_pitch_type_summary": str(pitcher_pitch_type_path),
        "cv_model_comparison": str(cv_model_comparison_path),
        "bootstrap_intervals": str(bootstrap_intervals_path),
        "residual_diagnostics": str(residual_diagnostics_path),
        "alpha_selection": str(alpha_selection_path),
    }
