from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from .paths import OBP_METADATA_URL, OBP_POI_URL, RAW_DIR, ensure_project_dirs
from .sample_data import make_sample_biomechanics, make_sample_statcast


def download_public_sources(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download public CSVs used by the MVP.

    Statcast downloads are intentionally not automated here because Baseball Savant queries are
    user-selected. Place any exported Statcast CSV at `data/raw/statcast/statcast_sample.csv`.
    """

    ensure_project_dirs()
    obp_dir = raw_dir / "openbiomechanics"
    obp_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "openbiomechanics_poi": obp_dir / "poi_metrics.csv",
        "openbiomechanics_metadata": obp_dir / "metadata.csv",
    }
    for url, path in [
        (OBP_POI_URL, outputs["openbiomechanics_poi"]),
        (OBP_METADATA_URL, outputs["openbiomechanics_metadata"]),
    ]:
        urllib.request.urlretrieve(url, path)
    return outputs


def load_biomechanics(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, str]:
    path = raw_dir / "openbiomechanics" / "poi_metrics.csv"
    if path.exists():
        return pd.read_csv(path), "openbiomechanics_public_poi_metrics"
    return make_sample_biomechanics(), "generated_biomechanics_sample"


def load_metadata(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, str]:
    path = raw_dir / "openbiomechanics" / "metadata.csv"
    if path.exists():
        return pd.read_csv(path), "openbiomechanics_public_metadata"
    return pd.DataFrame(), "metadata_unavailable"


def load_statcast(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, str]:
    candidates = [
        raw_dir / "statcast" / "statcast_sample.csv",
        raw_dir / "statcast" / "statcast.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), "user_supplied_statcast_csv"
    return make_sample_statcast(), "generated_statcast_like_sample"

