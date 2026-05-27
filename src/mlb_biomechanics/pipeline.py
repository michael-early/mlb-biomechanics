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
    model_metrics_path = PROCESSED_DIR / "velocity_model_metrics.json"
    data_sources_path = PROCESSED_DIR / "data_sources.json"
    report_path = REPORTS_DIR / "mvp_report.html"

    metrics.to_csv(metrics_path, index=False)
    velocity_results["predictions"].to_csv(predictions_path, index=False)
    velocity_results["coefficients"].to_csv(coeffs_path, index=False)
    velocity_results["permutation_importance"].to_csv(importance_path, index=False)
    metric_dictionary().to_csv(metric_dict_path, index=False)
    statcast_results["pitch_type_summary"].to_csv(statcast_summary_path, index=False)

    model_metrics = {
        "train_rows": velocity_results["train_rows"],
        "test_rows": velocity_results["test_rows"],
        "baseline": velocity_results["baseline"],
        "ridge": velocity_results["ridge"],
    }
    model_metrics_path.write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")
    data_sources_path.write_text(
        json.dumps(
            {
                "biomechanics": biomechanics_source,
                "metadata": metadata_source,
                "metadata_rows": int(len(metadata)),
                "statcast": statcast_source,
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
    }
