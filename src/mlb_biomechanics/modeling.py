from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import METRIC_FEATURES


TARGET = "pitch_speed_mph"
ALPHA_GRID = [0.0, 0.1, 1.0, 10.0, 100.0]


class PredictiveModel(Protocol):
    features: list[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ConstantModel:
    features: list[str]
    value: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.value, len(df))


@dataclass
class RidgeModel:
    features: list[str]
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray
    alpha: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=float)
        x_scaled = (x - self.means) / self.scales
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        return x_design @ self.coef

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.features,
                "coefficient": self.coef[1:],
                "alpha": self.alpha,
            }
        )


@dataclass
class KNNRegressor:
    features: list[str]
    means: np.ndarray
    scales: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    k: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=float)
        x_scaled = (x - self.means) / self.scales
        predictions = []
        for row in x_scaled:
            distances = np.sqrt(np.sum((self.train_x - row) ** 2, axis=1))
            neighbor_idx = np.argsort(distances)[: self.k]
            predictions.append(float(np.mean(self.train_y[neighbor_idx])))
        return np.asarray(predictions)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


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


def grouped_kfold_splits(
    df: pd.DataFrame, group_col: str = "session", n_splits: int = 5
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create deterministic grouped folds with no group leakage between train and test."""

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    groups = pd.Series(df[group_col].dropna().unique()).sort_values().to_numpy()
    if len(groups) < 2:
        raise ValueError("Grouped K-fold requires at least two groups")
    n_splits = min(n_splits, len(groups))
    folds = np.array_split(groups, n_splits)
    splits = []
    for fold_groups in folds:
        test_groups = set(fold_groups.tolist())
        test_mask = df[group_col].isin(test_groups)
        splits.append((df.loc[~test_mask].copy(), df.loc[test_mask].copy()))
    return splits


def _standardize_train_x(
    train: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train[features].to_numpy(dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    return (x - means) / scales, means, scales


def fit_constant(
    train: pd.DataFrame, target: str = TARGET, features: list[str] | None = None
) -> ConstantModel:
    return ConstantModel(features=features or METRIC_FEATURES, value=float(train[target].mean()))


def fit_ridge(
    train: pd.DataFrame,
    target: str = TARGET,
    features: list[str] | None = None,
    alpha: float = 1.0,
) -> RidgeModel:
    features = features or METRIC_FEATURES
    x_scaled, means, scales = _standardize_train_x(train, features)
    y = train[target].to_numpy(dtype=float)
    x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
    penalty = np.eye(x_design.shape[1]) * alpha
    penalty[0, 0] = 0
    normal_matrix = x_design.T @ x_design + penalty
    rhs = x_design.T @ y
    try:
        coef = np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(normal_matrix) @ rhs
    return RidgeModel(features=features, means=means, scales=scales, coef=coef, alpha=alpha)


def fit_knn(
    train: pd.DataFrame,
    target: str = TARGET,
    features: list[str] | None = None,
    k: int = 7,
) -> KNNRegressor:
    features = features or METRIC_FEATURES
    x_scaled, means, scales = _standardize_train_x(train, features)
    return KNNRegressor(
        features=features,
        means=means,
        scales=scales,
        train_x=x_scaled,
        train_y=train[target].to_numpy(dtype=float),
        k=min(k, len(train)),
    )


def evaluate_predictions(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: PredictiveModel,
    fold: int | str,
    target: str = TARGET,
) -> dict[str, float | int | str]:
    y_test = test[target].to_numpy(dtype=float)
    y_pred = model.predict(test)
    scores = metrics_dict(y_test, y_pred)
    return {
        "row_type": "fold",
        "model": model_name,
        "fold": fold,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **scores,
    }


def summarize_cv_results(cv_rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, float | str]] = []
    for model, group in cv_rows.groupby("model", sort=False):
        for metric in ["rmse", "mae", "r2"]:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_half_width = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            summary_rows.append(
                {
                    "row_type": "summary",
                    "model": model,
                    "fold": f"{metric}_mean",
                    "train_rows": int(group["train_rows"].mean()),
                    "test_rows": int(group["test_rows"].mean()),
                    "rmse": mean if metric == "rmse" else np.nan,
                    "mae": mean if metric == "mae" else np.nan,
                    "r2": mean if metric == "r2" else np.nan,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci_lower": mean - ci_half_width,
                    "ci_upper": mean + ci_half_width,
                }
            )
    return pd.DataFrame(summary_rows)


def select_ridge_alpha(
    train: pd.DataFrame,
    alphas: list[float] | None = None,
    features: list[str] | None = None,
    target: str = TARGET,
    n_splits: int = 4,
) -> tuple[float, pd.DataFrame]:
    alphas = alphas or ALPHA_GRID
    features = features or METRIC_FEATURES
    rows = []
    for fold_idx, (inner_train, inner_test) in enumerate(
        grouped_kfold_splits(train, n_splits=n_splits), start=1
    ):
        for alpha in alphas:
            model = fit_ridge(inner_train, target=target, features=features, alpha=alpha)
            scores = metrics_dict(
                inner_test[target].to_numpy(dtype=float),
                model.predict(inner_test),
            )
            rows.append(
                {
                    "alpha": alpha,
                    "fold": fold_idx,
                    "rmse": scores["rmse"],
                    "mae": scores["mae"],
                    "r2": scores["r2"],
                }
            )
    table = pd.DataFrame(rows)
    alpha_summary = table.groupby("alpha", as_index=False)["rmse"].mean()
    best_alpha = float(alpha_summary.sort_values(["rmse", "alpha"]).iloc[0]["alpha"])
    table["selected_alpha"] = best_alpha
    return best_alpha, table


def cross_validate_model_suite(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = TARGET,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features or METRIC_FEATURES
    fold_rows: list[dict[str, float | int | str]] = []
    alpha_tables = []
    for fold_idx, (train, test) in enumerate(grouped_kfold_splits(df, n_splits=n_splits), start=1):
        best_alpha, alpha_table = select_ridge_alpha(train, features=features, target=target)
        alpha_table["outer_fold"] = fold_idx
        alpha_tables.append(alpha_table)
        models: list[tuple[str, PredictiveModel]] = [
            ("mean_baseline", fit_constant(train, target=target, features=features)),
            ("ols", fit_ridge(train, target=target, features=features, alpha=0.0)),
            ("ridge_alpha_grid", fit_ridge(train, target=target, features=features, alpha=best_alpha)),
            ("knn_k7", fit_knn(train, target=target, features=features, k=7)),
        ]
        for model_name, model in models:
            row = evaluate_predictions(model_name, train, test, model, fold_idx, target=target)
            row["selected_alpha"] = best_alpha if model_name == "ridge_alpha_grid" else np.nan
            fold_rows.append(row)

    fold_table = pd.DataFrame(fold_rows)
    summary = summarize_cv_results(fold_table)
    alpha_selection = pd.concat(alpha_tables, ignore_index=True)
    return pd.concat([fold_table, summary], ignore_index=True, sort=False), alpha_selection


def permutation_importance(
    model: PredictiveModel,
    test: pd.DataFrame,
    target: str,
    seed: int = 17,
    n_repeats: int = 50,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = test[target].to_numpy(dtype=float)
    base_rmse = rmse(y, model.predict(test))
    rows = []
    for feature in model.features:
        increases = []
        for _ in range(n_repeats):
            shuffled = test.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            increases.append(rmse(y, model.predict(shuffled)) - base_rmse)
        values = np.asarray(increases)
        rows.append(
            {
                "feature": feature,
                "rmse_increase": float(values.mean()),
                "rmse_increase_std": float(values.std(ddof=1)),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
                "repeats": n_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)


def bootstrap_metric_intervals(
    predictions: pd.DataFrame,
    target_col: str = TARGET,
    prediction_col: str = "predicted_pitch_speed_mph",
    model_name: str = "ridge_holdout",
    n_bootstrap: int = 500,
    seed: int = 23,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = predictions[target_col].to_numpy(dtype=float)
    y_pred = predictions[prediction_col].to_numpy(dtype=float)
    estimates = metrics_dict(y_true, y_pred)
    rows = []
    for metric_name, estimate in estimates.items():
        boot_values = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true), len(y_true))
            boot_values.append(metrics_dict(y_true[idx], y_pred[idx])[metric_name])
        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "estimate": estimate,
                "ci_lower": float(np.percentile(boot_values, 2.5)),
                "ci_upper": float(np.percentile(boot_values, 97.5)),
                "bootstrap_samples": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def residual_diagnostics(
    predictions: pd.DataFrame,
    target_col: str = TARGET,
    residual_col: str = "residual_mph",
) -> pd.DataFrame:
    out = predictions.copy()
    out["abs_residual_mph"] = out[residual_col].abs()
    out["velocity_band"] = pd.cut(
        out[target_col],
        bins=[-np.inf, 80, 85, 90, 95, np.inf],
        labels=["<80", "80-85", "85-90", "90-95", "95+"],
        right=False,
    ).astype(str)
    band_summary = (
        out.groupby("velocity_band", dropna=False)
        .agg(
            rows=("velocity_band", "size"),
            mean_actual_mph=(target_col, "mean"),
            mean_residual_mph=(residual_col, "mean"),
            mae=("abs_residual_mph", "mean"),
            rmse=(residual_col, lambda value: float(np.sqrt(np.mean(value**2)))),
        )
        .reset_index()
    )
    session_summary = (
        out.groupby("session", dropna=False)
        .agg(
            rows=("session", "size"),
            mean_actual_mph=(target_col, "mean"),
            mean_residual_mph=(residual_col, "mean"),
            mae=("abs_residual_mph", "mean"),
            rmse=(residual_col, lambda value: float(np.sqrt(np.mean(value**2)))),
        )
        .reset_index()
    )
    band_summary.insert(0, "diagnostic_group", "velocity_band")
    session_summary.insert(0, "diagnostic_group", "session")
    session_summary = session_summary.rename(columns={"session": "velocity_band"})
    return pd.concat([band_summary, session_summary], ignore_index=True, sort=False)


def evaluate_velocity_model(df: pd.DataFrame) -> dict[str, object]:
    clean = df.dropna(subset=[TARGET, *METRIC_FEATURES]).copy()
    train, test = train_test_split_by_session(clean)
    best_alpha, alpha_selection = select_ridge_alpha(train)
    model = fit_ridge(train, alpha=best_alpha)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)
    baseline_model = fit_constant(train)
    baseline_pred = baseline_model.predict(test)
    model_pred = model.predict(test)
    baseline_scores = metrics_dict(y_test, baseline_pred)
    model_scores = metrics_dict(y_test, model_pred)

    predictions = test[["session_pitch", "session", "pitch_type", TARGET, *METRIC_FEATURES]].copy()
    predictions["predicted_pitch_speed_mph"] = model_pred
    predictions["residual_mph"] = predictions[TARGET] - predictions["predicted_pitch_speed_mph"]
    predictions["velocity_band"] = pd.cut(
        predictions[TARGET],
        bins=[-np.inf, 80, 85, 90, 95, np.inf],
        labels=["<80", "80-85", "85-90", "90-95", "95+"],
        right=False,
    ).astype(str)

    cv_model_comparison, cv_alpha_selection = cross_validate_model_suite(clean)

    return {
        "model": model,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sessions": int(clean["session"].nunique()) if "session" in clean.columns else None,
        "baseline": baseline_scores,
        "ridge": {
            **model_scores,
            "alpha": best_alpha,
            "rmse_lift_vs_baseline": baseline_scores["rmse"] - model_scores["rmse"],
        },
        "predictions": predictions,
        "coefficients": model.coefficients().sort_values("coefficient", key=np.abs, ascending=False),
        "permutation_importance": permutation_importance(model, test, TARGET),
        "bootstrap_intervals": bootstrap_metric_intervals(predictions),
        "residual_diagnostics": residual_diagnostics(predictions),
        "cv_model_comparison": cv_model_comparison,
        "alpha_selection": alpha_selection,
        "cv_alpha_selection": cv_alpha_selection,
    }


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

    pitcher_pitch_type = pd.DataFrame()
    if "pitcher" in df.columns:
        pitcher_pitch_type = (
            df.dropna(subset=trait_columns)
            .groupby(["pitcher", "pitch_type"], dropna=False)
            .agg(
                pitches=("pitch_type", "size"),
                avg_velocity=("release_speed", "mean"),
                whiff_rate=("whiff", "mean"),
                chase_rate=("chase", "mean"),
                hard_hit_rate=("hard_hit", "mean"),
                avg_run_value=("run_value", "mean"),
            )
            .reset_index()
            .sort_values(["pitches", "avg_run_value"], ascending=[False, False])
        )

    correlations = {}
    for outcome in ["whiff", "chase", "hard_hit", "run_value"]:
        if outcome in df.columns:
            correlations[outcome] = {
                col: float(df[[col, outcome]].corr(numeric_only=True).iloc[0, 1])
                for col in trait_columns
                if df[col].std(ddof=0) > 0 and df[outcome].std(ddof=0) > 0
            }

    sample_summary = {
        "rows": int(len(df)),
        "pitch_types": int(df["pitch_type"].nunique(dropna=True)) if "pitch_type" in df.columns else 0,
        "date_min": str(df["game_date"].min()) if "game_date" in df.columns else "unknown",
        "date_max": str(df["game_date"].max()) if "game_date" in df.columns else "unknown",
    }

    return {
        "pitch_type_summary": grouped,
        "pitcher_pitch_type_summary": pitcher_pitch_type,
        "trait_outcome_correlations": correlations,
        "sample_summary": sample_summary,
    }
