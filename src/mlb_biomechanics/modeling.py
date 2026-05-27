from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import METRIC_FEATURES


@dataclass
class RidgeModel:
    features: list[str]
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=float)
        x_scaled = (x - self.means) / self.scales
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        return x_design @ self.coef

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.features, "coefficient": self.coef[1:]})


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def train_test_split_by_session(
    df: pd.DataFrame, test_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "session" not in df.columns:
        mask = np.arange(len(df)) % round(1 / test_fraction) == 0
        return df.loc[~mask].copy(), df.loc[mask].copy()

    sessions = pd.Series(df["session"].dropna().unique()).sort_values().to_numpy()
    test_count = max(1, int(round(len(sessions) * test_fraction)))
    test_sessions = set(sessions[-test_count:])
    test_mask = df["session"].isin(test_sessions)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def fit_ridge(
    train: pd.DataFrame,
    target: str = "pitch_speed_mph",
    features: list[str] | None = None,
    alpha: float = 1.0,
) -> RidgeModel:
    features = features or METRIC_FEATURES
    x = train[features].to_numpy(dtype=float)
    y = train[target].to_numpy(dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    x_scaled = (x - means) / scales
    x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
    penalty = np.eye(x_design.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
    return RidgeModel(features=features, means=means, scales=scales, coef=coef)


def evaluate_velocity_model(df: pd.DataFrame) -> dict[str, object]:
    clean = df.dropna(subset=["pitch_speed_mph", *METRIC_FEATURES]).copy()
    train, test = train_test_split_by_session(clean)
    model = fit_ridge(train)
    y_train = train["pitch_speed_mph"].to_numpy(dtype=float)
    y_test = test["pitch_speed_mph"].to_numpy(dtype=float)
    baseline_pred = np.repeat(y_train.mean(), len(test))
    model_pred = model.predict(test)
    baseline_rmse = rmse(y_test, baseline_pred)
    model_rmse = rmse(y_test, model_pred)

    predictions = test[
        ["session_pitch", "session", "pitch_type", "pitch_speed_mph", *METRIC_FEATURES]
    ].copy()
    predictions["predicted_pitch_speed_mph"] = model_pred
    predictions["residual_mph"] = predictions["pitch_speed_mph"] - predictions[
        "predicted_pitch_speed_mph"
    ]

    return {
        "model": model,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline": {
            "rmse": baseline_rmse,
            "mae": mae(y_test, baseline_pred),
            "r2": r2(y_test, baseline_pred),
        },
        "ridge": {
            "rmse": model_rmse,
            "mae": mae(y_test, model_pred),
            "r2": r2(y_test, model_pred),
            "rmse_lift_vs_baseline": baseline_rmse - model_rmse,
        },
        "predictions": predictions,
        "coefficients": model.coefficients().sort_values("coefficient", key=np.abs, ascending=False),
        "permutation_importance": permutation_importance(model, test, "pitch_speed_mph"),
    }


def permutation_importance(
    model: RidgeModel, test: pd.DataFrame, target: str, seed: int = 17
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = test[target].to_numpy(dtype=float)
    base_rmse = rmse(y, model.predict(test))
    rows = []
    for feature in model.features:
        shuffled = test.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        rows.append({"feature": feature, "rmse_increase": rmse(y, model.predict(shuffled)) - base_rmse})
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)


def statcast_performance_bridge(statcast: pd.DataFrame) -> dict[str, object]:
    df = statcast.copy()
    if "description" in df.columns:
        description = df["description"].astype(str)
        swing_descriptions = {
            "swinging_strike",
            "swinging_strike_blocked",
            "foul",
            "foul_tip",
            "foul_bunt",
            "hit_into_play",
            "hit_into_play_no_out",
            "hit_into_play_score",
        }
        whiff_descriptions = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
        swing = description.isin(swing_descriptions)
        if "whiff" not in df.columns:
            df["whiff"] = description.isin(whiff_descriptions).astype(int)
        if "chase" not in df.columns and "zone" in df.columns:
            zone = pd.to_numeric(df["zone"], errors="coerce")
            df["chase"] = (swing & ~zone.between(1, 9)).astype(int)

    if "hard_hit" not in df.columns and "launch_speed" in df.columns:
        df["hard_hit"] = (pd.to_numeric(df["launch_speed"], errors="coerce") >= 95).astype(int)

    if "delta_run_exp" in df.columns and "run_value" not in df.columns:
        # Baseball Savant's delta_run_exp is from the batting/offense perspective.
        # Multiply by -1 so positive values are better for the pitcher.
        df["run_value"] = -pd.to_numeric(df["delta_run_exp"], errors="coerce")

    if "estimated_woba_using_speedangle" in df.columns and "run_value" not in df.columns:
        df["run_value"] = -pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce")

    for binary in ["whiff", "chase", "hard_hit"]:
        if binary not in df.columns:
            df[binary] = 0

    trait_columns = [c for c in ["release_speed", "release_extension", "pfx_x", "pfx_z", "plate_x"] if c in df]
    grouped = (
        df.dropna(subset=trait_columns)
        .groupby("pitch_type", dropna=False)
        .agg(
            pitches=("pitch_type", "size"),
            avg_velocity=("release_speed", "mean"),
            whiff_rate=("whiff", "mean"),
            chase_rate=("chase", "mean"),
            hard_hit_rate=("hard_hit", "mean"),
            avg_run_value=("run_value", "mean"),
        )
        .reset_index()
        .sort_values("pitches", ascending=False)
    )

    correlations = {}
    for outcome in ["whiff", "chase", "hard_hit", "run_value"]:
        if outcome in df.columns:
            correlations[outcome] = {
                col: float(df[[col, outcome]].corr(numeric_only=True).iloc[0, 1])
                for col in trait_columns
                if df[col].std(ddof=0) > 0 and df[outcome].std(ddof=0) > 0
            }

    return {"pitch_type_summary": grouped, "trait_outcome_correlations": correlations}
