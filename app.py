"""Optional Streamlit entry point.

Run after building outputs:

    PYTHONPATH=src python -m mlb_biomechanics build
    streamlit run app.py

If Streamlit is not installed, open `reports/mvp_report.html` instead.
"""

from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - informational path
    raise SystemExit("Streamlit is optional. Open reports/mvp_report.html or install streamlit.") from exc


ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="MLB Biomechanics MVP", layout="wide")
st.title("MLB Biomechanics MVP")
st.caption("Biomechanical pitching metrics, velocity modeling, and Statcast performance bridge.")

metrics_path = ROOT / "data" / "processed" / "biomechanics_metrics.csv"
model_path = ROOT / "data" / "processed" / "velocity_model_metrics.json"
importance_path = ROOT / "reports" / "tables" / "velocity_permutation_importance.csv"
statcast_path = ROOT / "reports" / "tables" / "statcast_pitch_type_summary.csv"

if not metrics_path.exists():
    st.warning("Run `PYTHONPATH=src python -m mlb_biomechanics build` first.")
    st.stop()

metrics = pd.read_csv(metrics_path)
importance = pd.read_csv(importance_path)
statcast = pd.read_csv(statcast_path)

st.subheader("Biomechanical Metrics")
st.dataframe(metrics.head(100), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Velocity Feature Importance")
    st.bar_chart(importance.set_index("feature")["rmse_increase"])
with right:
    st.subheader("Statcast Pitch-Type Bridge")
    st.dataframe(statcast, use_container_width=True)

st.info(
    "OpenBiomechanics athletes are anonymized. The MVP models biomechanics-to-velocity and "
    "separately relates public pitch traits to outcomes; it does not claim direct MLB identity linkage."
)

