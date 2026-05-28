from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import (
    BiomechanicalMetricTransformer,
    METRIC_FEATURES,
    RAW_METRIC_INPUTS,
    add_biomechanical_metrics,
)


TARGET = "pitch_speed_mph"
ALPHA_GRID = [0.0, 0.1, 1.0, 10.0, 100.0]
INTERACTION_FEATURES = [
    "kinetic_chain_x_arm_speed",
    "kinetic_chain_x_lower_body",
    "arm_speed_x_sequencing",
    "kinetic_chain_transfer_score_squared",
    "arm_speed_score_squared",
]


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
        x = np.where(np.isnan(x), self.means, x)
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
        x = np.where(np.isnan(x), self.means, x)
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


def _numeric_existing_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [feature for feature in candidates if feature in df.columns]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"kinetic_chain_transfer_score", "arm_speed_score"}.issubset(out.columns):
        out["kinetic_chain_x_arm_speed"] = (
            out["kinetic_chain_transfer_score"] * out["arm_speed_score"]
        )
        out["kinetic_chain_transfer_score_squared"] = out["kinetic_chain_transfer_score"] ** 2
        out["arm_speed_score_squared"] = out["arm_speed_score"] ** 2
    if {"kinetic_chain_transfer_score", "lower_body_force_score"}.issubset(out.columns):
        out["kinetic_chain_x_lower_body"] = (
            out["kinetic_chain_transfer_score"] * out["lower_body_force_score"]
        )
    if {"arm_speed_score", "sequencing_efficiency"}.issubset(out.columns):
        out["arm_speed_x_sequencing"] = out["arm_speed_score"] * out["sequencing_efficiency"]
    for feature in INTERACTION_FEATURES:
        if feature in out.columns:
            out[feature] = out[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def prepare_metric_train_test(
    train: pd.DataFrame, test: pd.DataFrame, include_interactions: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformer = BiomechanicalMetricTransformer().fit(train)
    train_prepared = transformer.transform(train)
    test_prepared = transformer.transform(test)
    if include_interactions:
        train_prepared = add_interaction_features(train_prepared)
        test_prepared = add_interaction_features(test_prepared)
    return train_prepared, test_prepared


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


def repeated_grouped_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "session",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 17,
) -> list[tuple[int, int, pd.DataFrame, pd.DataFrame]]:
    """Create repeated grouped folds with randomized group order and no group leakage."""

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    groups = pd.Series(df[group_col].dropna().unique()).to_numpy()
    if len(groups) < 2:
        raise ValueError("Repeated grouped K-fold requires at least two groups")
    n_splits = min(n_splits, len(groups))
    rng = np.random.default_rng(seed)
    splits = []
    for repeat in range(1, n_repeats + 1):
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, n_splits)
        for fold_idx, fold_groups in enumerate(folds, start=1):
            test_groups = set(fold_groups.tolist())
            test_mask = df[group_col].isin(test_groups)
            splits.append((repeat, fold_idx, df.loc[~test_mask].copy(), df.loc[test_mask].copy()))
    return splits


def _standardize_train_x(
    train: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train[features].to_numpy(dtype=float)
    means = np.nanmean(x, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    x = np.where(np.isnan(x), means, x)
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
    fit_metric_transformer: bool = True,
    include_interactions: bool = False,
) -> tuple[float, pd.DataFrame]:
    alphas = alphas or ALPHA_GRID
    features = features or METRIC_FEATURES
    rows = []
    for fold_idx, (inner_train, inner_test) in enumerate(
        grouped_kfold_splits(train, n_splits=n_splits), start=1
    ):
        if fit_metric_transformer:
            inner_train, inner_test = prepare_metric_train_test(
                inner_train, inner_test, include_interactions=include_interactions
            )
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
    n_repeats: int = 3,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features or METRIC_FEATURES
    fold_rows: list[dict[str, float | int | str]] = []
    alpha_tables = []
    for repeat, fold_idx, train_raw, test_raw in repeated_grouped_kfold_splits(
        df, n_splits=n_splits, n_repeats=n_repeats, seed=seed
    ):
        fold_label = f"r{repeat}_f{fold_idx}"
        best_alpha, alpha_table = select_ridge_alpha(
            train_raw,
            features=features,
            target=target,
            fit_metric_transformer=True,
        )
        alpha_table["outer_repeat"] = repeat
        alpha_table["outer_fold"] = fold_idx
        train, test = prepare_metric_train_test(train_raw, test_raw)
        alpha_tables.append(alpha_table)
        models: list[tuple[str, PredictiveModel]] = [
            ("mean_baseline", fit_constant(train, target=target, features=features)),
            ("ols", fit_ridge(train, target=target, features=features, alpha=0.0)),
            ("ridge_alpha_grid", fit_ridge(train, target=target, features=features, alpha=best_alpha)),
            ("knn_k7", fit_knn(train, target=target, features=features, k=7)),
        ]
        for model_name, model in models:
            row = evaluate_predictions(model_name, train, test, model, fold_label, target=target)
            row["repeat"] = repeat
            row["selected_alpha"] = best_alpha if model_name == "ridge_alpha_grid" else np.nan
            fold_rows.append(row)

    fold_table = pd.DataFrame(fold_rows)
    summary = summarize_cv_results(fold_table)
    alpha_selection = pd.concat(alpha_tables, ignore_index=True)
    return pd.concat([fold_table, summary], ignore_index=True, sort=False), alpha_selection


def _ridge_cv_table(
    df: pd.DataFrame,
    model_name: str,
    features: list[str],
    target: str = TARGET,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 17,
    fit_metric_transformer: bool = True,
    include_interactions: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for repeat, fold_idx, train_raw, test_raw in repeated_grouped_kfold_splits(
        df, n_splits=n_splits, n_repeats=n_repeats, seed=seed
    ):
        if fit_metric_transformer:
            train, test = prepare_metric_train_test(
                train_raw, test_raw, include_interactions=include_interactions
            )
        else:
            train, test = train_raw.copy(), test_raw.copy()

        available_features = [feature for feature in features if feature in train.columns]
        if not available_features:
            continue

        best_alpha, _ = select_ridge_alpha(
            train_raw,
            features=available_features,
            target=target,
            fit_metric_transformer=fit_metric_transformer,
            include_interactions=include_interactions,
        )
        model = fit_ridge(train, target=target, features=available_features, alpha=best_alpha)
        row = evaluate_predictions(
            model_name,
            train,
            test,
            model,
            f"r{repeat}_f{fold_idx}",
            target=target,
        )
        row["repeat"] = repeat
        row["selected_alpha"] = best_alpha
        row["feature_count"] = len(available_features)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_metric(summary: pd.DataFrame, model_name: str, metric: str) -> float:
    row = summary[
        (summary["row_type"] == "summary")
        & (summary["model"] == model_name)
        & (summary["metric"] == metric)
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["mean"])


def compare_feature_sets(
    df: pd.DataFrame,
    target: str = TARGET,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 17,
) -> pd.DataFrame:
    raw_features = _numeric_existing_features(df, RAW_METRIC_INPUTS)
    specs = [
        {
            "feature_set": "engineered_metrics",
            "model": "ridge_engineered_metrics",
            "features": METRIC_FEATURES,
            "fit_metric_transformer": True,
            "include_interactions": False,
        },
        {
            "feature_set": "raw_metric_inputs",
            "model": "ridge_raw_metric_inputs",
            "features": raw_features,
            "fit_metric_transformer": False,
            "include_interactions": False,
        },
        {
            "feature_set": "raw_plus_engineered",
            "model": "ridge_raw_plus_engineered",
            "features": raw_features + METRIC_FEATURES,
            "fit_metric_transformer": True,
            "include_interactions": False,
        },
        {
            "feature_set": "engineered_interactions",
            "model": "ridge_engineered_interactions",
            "features": METRIC_FEATURES + INTERACTION_FEATURES,
            "fit_metric_transformer": True,
            "include_interactions": True,
        },
    ]

    fold_tables = []
    for spec in specs:
        if not spec["features"]:
            continue
        table = _ridge_cv_table(
            df,
            model_name=str(spec["model"]),
            features=list(spec["features"]),
            target=target,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_metric_transformer=bool(spec["fit_metric_transformer"]),
            include_interactions=bool(spec["include_interactions"]),
        )
        if table.empty:
            continue
        table["feature_set"] = spec["feature_set"]
        fold_tables.append(table)

    if not fold_tables:
        return pd.DataFrame()

    fold_table = pd.concat(fold_tables, ignore_index=True, sort=False)
    summary = summarize_cv_results(fold_table)
    feature_meta = (
        fold_table.groupby("model", as_index=False)
        .agg(feature_set=("feature_set", "first"), feature_count=("feature_count", "first"))
    )
    summary = summary.merge(feature_meta, on="model", how="left")
    rmse_by_model = summary[summary["metric"] == "rmse"][["model", "mean"]].rename(
        columns={"mean": "rmse_mean"}
    )
    best_rmse = float(rmse_by_model["rmse_mean"].min())
    summary = summary.merge(rmse_by_model, on="model", how="left")
    summary["rmse_delta_vs_best"] = summary["rmse_mean"] - best_rmse
    return pd.concat([fold_table, summary], ignore_index=True, sort=False)


def metric_ablation(
    df: pd.DataFrame,
    target: str = TARGET,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 17,
) -> pd.DataFrame:
    rows = []
    base_model = "all_engineered_metrics"
    base_table = _ridge_cv_table(
        df,
        model_name=base_model,
        features=METRIC_FEATURES,
        target=target,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_metric_transformer=True,
    )
    base_summary = summarize_cv_results(base_table)
    base_rmse = _summary_metric(base_summary, base_model, "rmse")
    base_mae = _summary_metric(base_summary, base_model, "mae")
    base_r2 = _summary_metric(base_summary, base_model, "r2")
    rows.append(
        {
            "feature_removed": "none",
            "feature_count": len(METRIC_FEATURES),
            "rmse_mean": base_rmse,
            "mae_mean": base_mae,
            "r2_mean": base_r2,
            "rmse_delta_vs_all": 0.0,
        }
    )

    for feature in METRIC_FEATURES:
        keep = [candidate for candidate in METRIC_FEATURES if candidate != feature]
        model_name = f"without_{feature}"
        table = _ridge_cv_table(
            df,
            model_name=model_name,
            features=keep,
            target=target,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_metric_transformer=True,
        )
        summary = summarize_cv_results(table)
        rmse_mean = _summary_metric(summary, model_name, "rmse")
        rows.append(
            {
                "feature_removed": feature,
                "feature_count": len(keep),
                "rmse_mean": rmse_mean,
                "mae_mean": _summary_metric(summary, model_name, "mae"),
                "r2_mean": _summary_metric(summary, model_name, "r2"),
                "rmse_delta_vs_all": rmse_mean - base_rmse,
            }
        )

    return pd.DataFrame(rows).sort_values("rmse_delta_vs_all", ascending=False)


def metric_correlations(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    metrics_df = add_biomechanical_metrics(df.dropna(subset=[target]).copy())
    rows = []
    for feature_type, features in [
        ("engineered_metric", METRIC_FEATURES),
        ("raw_metric_input", _numeric_existing_features(metrics_df, RAW_METRIC_INPUTS)),
    ]:
        for feature in features:
            if feature not in metrics_df.columns:
                continue
            pair = metrics_df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair[feature].std(ddof=0) == 0 or pair[target].std(ddof=0) == 0:
                corr = np.nan
            else:
                corr = float(pair[feature].corr(pair[target]))
            rows.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "pearson_corr_with_velocity": corr,
                    "abs_corr": abs(corr) if not pd.isna(corr) else np.nan,
                    "rows": int(len(pair)),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


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
    group_col: str | None = "session",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = predictions[target_col].to_numpy(dtype=float)
    y_pred = predictions[prediction_col].to_numpy(dtype=float)
    estimates = metrics_dict(y_true, y_pred)
    if group_col and group_col in predictions.columns:
        bootstrap_unit = group_col
        positions = np.arange(len(predictions))
        group_values = predictions[group_col].to_numpy()
        group_to_indices = {
            group: positions[group_values == group]
            for group in predictions[group_col].dropna().unique()
        }
        groups = np.asarray(list(group_to_indices))
    else:
        bootstrap_unit = "row"
        groups = np.asarray([])

    rows = []
    for metric_name, estimate in estimates.items():
        boot_values = []
        for _ in range(n_bootstrap):
            if len(groups) > 0:
                sampled_groups = rng.choice(groups, size=len(groups), replace=True)
                idx = np.concatenate([group_to_indices[group] for group in sampled_groups])
            else:
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
                "bootstrap_unit": bootstrap_unit,
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


def high_velocity_error_analysis(
    predictions: pd.DataFrame,
    target_col: str = TARGET,
    prediction_col: str = "predicted_pitch_speed_mph",
    threshold_mph: float = 90.0,
) -> pd.DataFrame:
    rows = []
    out = predictions.copy()
    out["residual_mph"] = out[target_col] - out[prediction_col]
    out["abs_residual_mph"] = out["residual_mph"].abs()
    groups = {
        "all_pitches": out,
        f"<{threshold_mph:.0f}_mph": out[out[target_col] < threshold_mph],
        f"{threshold_mph:.0f}+_mph": out[out[target_col] >= threshold_mph],
    }
    for group_name, group in groups.items():
        if group.empty:
            rows.append(
                {
                    "group": group_name,
                    "rows": 0,
                    "mean_actual_mph": np.nan,
                    "mean_predicted_mph": np.nan,
                    "mean_residual_mph": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "underprediction_rate": np.nan,
                    "overprediction_rate": np.nan,
                }
            )
            continue
        residual = group["residual_mph"].to_numpy(dtype=float)
        rows.append(
            {
                "group": group_name,
                "rows": int(len(group)),
                "mean_actual_mph": float(group[target_col].mean()),
                "mean_predicted_mph": float(group[prediction_col].mean()),
                "mean_residual_mph": float(group["residual_mph"].mean()),
                "mae": float(group["abs_residual_mph"].mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "underprediction_rate": float((group["residual_mph"] > 0).mean()),
                "overprediction_rate": float((group["residual_mph"] < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def velocity_band_performance(diagnostics: pd.DataFrame) -> pd.DataFrame:
    bands = diagnostics[diagnostics["diagnostic_group"] == "velocity_band"].copy()
    if bands.empty:
        return bands
    bands["abs_mean_residual_mph"] = bands["mean_residual_mph"].abs()
    bands["bias_direction"] = np.where(
        bands["mean_residual_mph"] > 0,
        "underpredicts",
        np.where(bands["mean_residual_mph"] < 0, "overpredicts", "neutral"),
    )
    return bands.sort_values("velocity_band")


def evaluate_velocity_model(df: pd.DataFrame, include_extended: bool = True) -> dict[str, object]:
    clean = df.dropna(subset=[TARGET]).copy()
    train_raw, test_raw = train_test_split_by_session(clean)
    best_alpha, alpha_selection = select_ridge_alpha(train_raw)
    train, test = prepare_metric_train_test(train_raw, test_raw)
    model = fit_ridge(train, alpha=best_alpha)
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

    if include_extended:
        cv_model_comparison, cv_alpha_selection = cross_validate_model_suite(clean)
        feature_set_comparison = compare_feature_sets(clean)
        ablation = metric_ablation(clean)
        correlations = metric_correlations(clean)
    else:
        cv_model_comparison = pd.DataFrame()
        cv_alpha_selection = pd.DataFrame()
        feature_set_comparison = pd.DataFrame()
        ablation = pd.DataFrame()
        correlations = pd.DataFrame()
    diagnostics = residual_diagnostics(predictions)

    return {
        "model": model,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sessions": int(train["session"].nunique()) if "session" in train.columns else None,
        "test_sessions": int(test["session"].nunique()) if "session" in test.columns else None,
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
        "residual_diagnostics": diagnostics,
        "velocity_band_performance": velocity_band_performance(diagnostics),
        "high_velocity_error_analysis": high_velocity_error_analysis(predictions),
        "cv_model_comparison": cv_model_comparison,
        "alpha_selection": alpha_selection,
        "cv_alpha_selection": cv_alpha_selection,
        "feature_set_comparison": feature_set_comparison,
        "metric_ablation": ablation,
        "metric_correlations": correlations,
    }


def statcast_performance_bridge(statcast: pd.DataFrame) -> dict[str, object]:
    df = statcast.copy()
    if "pitch_type" not in df.columns:
        df["pitch_type"] = "unknown"

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
        df["swing"] = description.isin(swing_descriptions).astype(int)
        if "whiff" not in df.columns:
            df["whiff"] = description.isin(whiff_descriptions).astype(int)
        if "chase" not in df.columns and "zone" in df.columns:
            zone = pd.to_numeric(df["zone"], errors="coerce")
            df["out_of_zone"] = (~zone.between(1, 9)).astype(int)
            df["chase"] = ((df["swing"] == 1) & (df["out_of_zone"] == 1)).astype(int)

    if "zone" in df.columns and "out_of_zone" not in df.columns:
        zone = pd.to_numeric(df["zone"], errors="coerce")
        df["out_of_zone"] = (~zone.between(1, 9)).astype(int)

    if "launch_speed" in df.columns:
        launch_speed = pd.to_numeric(df["launch_speed"], errors="coerce")
        df["batted_ball"] = launch_speed.notna().astype(int)
        if "hard_hit" not in df.columns:
            df["hard_hit"] = ((launch_speed >= 95) & launch_speed.notna()).astype(int)

    if "delta_run_exp" in df.columns and "run_value" not in df.columns:
        # Baseball Savant's delta_run_exp is from the batting/offense perspective.
        # Multiply by -1 so positive values are better for the pitcher.
        df["run_value"] = -pd.to_numeric(df["delta_run_exp"], errors="coerce")

    if "estimated_woba_using_speedangle" in df.columns and "run_value" not in df.columns:
        df["run_value"] = -pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce")

    for binary in ["whiff", "chase", "hard_hit"]:
        if binary not in df.columns:
            df[binary] = 0
        df[binary] = pd.to_numeric(df[binary], errors="coerce").fillna(0).astype(int)

    if "swing" not in df.columns:
        df["swing"] = 1 if "whiff" in df.columns else 0
    if "out_of_zone" not in df.columns:
        df["out_of_zone"] = 1 if "chase" in df.columns else 0
    if "batted_ball" not in df.columns:
        df["batted_ball"] = 1 if "hard_hit" in df.columns else 0
    if "run_value" not in df.columns:
        df["run_value"] = np.nan
    for denominator in ["swing", "out_of_zone", "batted_ball"]:
        df[denominator] = pd.to_numeric(df[denominator], errors="coerce").fillna(0).astype(int)

    trait_columns = [c for c in ["release_speed", "release_extension", "pfx_x", "pfx_z", "plate_x"] if c in df]
    for optional_numeric in ["release_speed", "release_extension", "pfx_x", "pfx_z", "plate_x"]:
        if optional_numeric not in df.columns:
            df[optional_numeric] = np.nan

    def _summarize(group_cols: list[str]) -> pd.DataFrame:
        base = df.dropna(subset=trait_columns) if trait_columns else df
        summary = (
            base.groupby(group_cols, dropna=False)
            .agg(
                pitches=("pitch_type", "size"),
                avg_velocity=("release_speed", "mean"),
                avg_release_extension=("release_extension", "mean"),
                avg_pfx_x=("pfx_x", "mean"),
                avg_pfx_z=("pfx_z", "mean"),
                swings=("swing", "sum"),
                whiffs=("whiff", "sum"),
                out_of_zone_pitches=("out_of_zone", "sum"),
                chases=("chase", "sum"),
                batted_balls=("batted_ball", "sum"),
                hard_hits=("hard_hit", "sum"),
                avg_run_value=("run_value", "mean"),
            )
            .reset_index()
        )
        summary["whiff_rate"] = np.where(
            summary["swings"] > 0, summary["whiffs"] / summary["swings"], np.nan
        )
        summary["chase_rate"] = np.where(
            summary["out_of_zone_pitches"] > 0,
            summary["chases"] / summary["out_of_zone_pitches"],
            np.nan,
        )
        summary["hard_hit_rate"] = np.where(
            summary["batted_balls"] > 0, summary["hard_hits"] / summary["batted_balls"], np.nan
        )
        return summary.sort_values("pitches", ascending=False)

    grouped = _summarize(["pitch_type"])

    pitcher_pitch_type = pd.DataFrame()
    if "pitcher" in df.columns:
        pitcher_pitch_type = _summarize(["pitcher", "pitch_type"]).sort_values(
            ["pitches", "avg_run_value"], ascending=[False, False]
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
        "swings": int(df["swing"].sum()),
        "out_of_zone_pitches": int(df["out_of_zone"].sum()),
        "batted_balls": int(df["batted_ball"].sum()),
        "run_value_rows": int(pd.to_numeric(df["run_value"], errors="coerce").notna().sum()),
    }
    sample_manifest = pd.DataFrame(
        [
            {"field": "rows", "value": sample_summary["rows"]},
            {"field": "date_min", "value": sample_summary["date_min"]},
            {"field": "date_max", "value": sample_summary["date_max"]},
            {"field": "pitch_types", "value": sample_summary["pitch_types"]},
            {"field": "swings", "value": sample_summary["swings"]},
            {"field": "out_of_zone_pitches", "value": sample_summary["out_of_zone_pitches"]},
            {"field": "batted_balls", "value": sample_summary["batted_balls"]},
            {"field": "run_value_rows", "value": sample_summary["run_value_rows"]},
            {"field": "pitcher_pitch_type_rows", "value": int(len(pitcher_pitch_type))},
            {
                "field": "rate_denominators",
                "value": "whiff/swing, chase/out-of-zone, hard-hit/batted-ball",
            },
        ]
    )

    return {
        "pitch_type_summary": grouped,
        "pitcher_pitch_type_summary": pitcher_pitch_type,
        "trait_outcome_correlations": correlations,
        "sample_summary": sample_summary,
        "sample_manifest": sample_manifest,
    }
