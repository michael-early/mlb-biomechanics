from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

OBP_POI_URL = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
    "refs/heads/main/baseball_pitching/data/poi/poi_metrics.csv"
)
OBP_METADATA_URL = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
    "refs/heads/main/baseball_pitching/data/metadata.csv"
)


def ensure_project_dirs() -> None:
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, REPORTS_DIR / "tables"]:
        path.mkdir(parents=True, exist_ok=True)

