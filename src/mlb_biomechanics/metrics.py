from __future__ import annotations

import numpy as np
import pandas as pd


RAW_METRIC_INPUTS = [
    "max_pelvis_rotational_velo",
    "max_torso_rotational_velo",
    "timing_peak_torso_to_peak_pelvis_rot_velo",
    "max_rotation_hip_shoulder_separation",
    "rotation_hip_shoulder_separation_fp",
    "pelvis_lumbar_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
    "shoulder_transfer_fp_br",
    "elbow_transfer_fp_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_extension_from_fp_to_br",
    "lead_grf_z_max",
    "lead_grf_mag_max",
    "rear_grf_mag_max",
    "peak_rfd_lead",
    "cog_velo_pkh",
    "max_shoulder_internal_rotational_velo",
    "max_elbow_extension_velo",
    "arm_slot",
    "stride_angle",
    "stride_length",
]

METRIC_FEATURES = [
    "sequencing_efficiency",
    "hip_shoulder_separation_score",
    "kinetic_chain_transfer_score",
    "lead_leg_block_score",
    "lower_body_force_score",
    "arm_speed_score",
    "release_consistency_score",
]


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


class BiomechanicalMetricTransformer:
    """Train-fit transformer for engineered biomechanical metrics.

    The original MVP computed z-scores on the whole dataframe. For model validation, this
    transformer stores scaling parameters from the training fold and applies them to held-out
    sessions, avoiding test-distribution leakage in engineered metrics.
    """

    def __init__(self, z_clip: float = 6.0) -> None:
        self._params: dict[str, tuple[float, float]] = {}
        self.z_clip = z_clip
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "BiomechanicalMetricTransformer":
        for name in RAW_METRIC_INPUTS:
            self._fit_series(name, _col(df, name))
        timing_abs = _col(df, "timing_peak_torso_to_peak_pelvis_rot_velo").abs()
        self._fit_series("timing_peak_torso_to_peak_pelvis_rot_velo_abs", timing_abs)
        for name, series in self._release_delta_series(df).items():
            self._fit_series(name, series.abs())
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("BiomechanicalMetricTransformer must be fit before transform")

        out = df.copy()
        timing = _col(out, "timing_peak_torso_to_peak_pelvis_rot_velo")

        out["sequencing_efficiency"] = (
            self._z("max_pelvis_rotational_velo", _col(out, "max_pelvis_rotational_velo"))
            + self._z("max_torso_rotational_velo", _col(out, "max_torso_rotational_velo"))
            - self._z("timing_peak_torso_to_peak_pelvis_rot_velo_abs", timing.abs())
        )
        out["hip_shoulder_separation_score"] = (
            self._z(
                "max_rotation_hip_shoulder_separation",
                _col(out, "max_rotation_hip_shoulder_separation"),
            )
            + self._z(
                "rotation_hip_shoulder_separation_fp",
                _col(out, "rotation_hip_shoulder_separation_fp"),
            )
        )
        out["kinetic_chain_transfer_score"] = (
            self._z("pelvis_lumbar_transfer_fp_br", _col(out, "pelvis_lumbar_transfer_fp_br"))
            + self._z("thorax_distal_transfer_fp_br", _col(out, "thorax_distal_transfer_fp_br"))
            + self._z("shoulder_transfer_fp_br", _col(out, "shoulder_transfer_fp_br"))
            + self._z("elbow_transfer_fp_br", _col(out, "elbow_transfer_fp_br"))
        )
        out["lead_leg_block_score"] = (
            self._z(
                "lead_knee_extension_angular_velo_max",
                _col(out, "lead_knee_extension_angular_velo_max"),
            )
            + self._z(
                "lead_knee_extension_from_fp_to_br",
                _col(out, "lead_knee_extension_from_fp_to_br"),
            )
            + self._z("lead_grf_z_max", _col(out, "lead_grf_z_max"))
        )
        out["lower_body_force_score"] = (
            self._z("lead_grf_mag_max", _col(out, "lead_grf_mag_max"))
            + self._z("rear_grf_mag_max", _col(out, "rear_grf_mag_max"))
            + self._z("peak_rfd_lead", _col(out, "peak_rfd_lead"))
            + self._z("cog_velo_pkh", _col(out, "cog_velo_pkh"))
        )
        out["arm_speed_score"] = (
            self._z(
                "max_shoulder_internal_rotational_velo",
                _col(out, "max_shoulder_internal_rotational_velo"),
            )
            + self._z("max_elbow_extension_velo", _col(out, "max_elbow_extension_velo"))
        )

        release_deltas = self._release_delta_series(out)
        out["release_consistency_score"] = -(
            self._z("arm_slot_delta_abs", release_deltas["arm_slot_delta"].abs())
            + self._z("stride_angle_delta_abs", release_deltas["stride_angle_delta"].abs())
            + self._z("stride_length_delta_abs", release_deltas["stride_length_delta"].abs())
        )

        for feature in METRIC_FEATURES:
            out[feature] = out[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _fit_series(self, name: str, series: pd.Series) -> None:
        values = pd.to_numeric(series, errors="coerce")
        mean = float(values.mean()) if not values.dropna().empty else 0.0
        std = float(values.std(ddof=0)) if not values.dropna().empty else 0.0
        if pd.isna(mean):
            mean = 0.0
        if pd.isna(std) or std == 0:
            std = 1.0
        self._params[name] = (mean, std)

    def _z(self, name: str, series: pd.Series) -> pd.Series:
        mean, std = self._params.get(name, (0.0, 1.0))
        values = pd.to_numeric(series, errors="coerce").fillna(mean)
        z = (values - mean) / std
        return z.clip(lower=-self.z_clip, upper=self.z_clip)

    @staticmethod
    def _release_delta_series(df: pd.DataFrame) -> dict[str, pd.Series]:
        arm_slot = _col(df, "arm_slot")
        stride_angle = _col(df, "stride_angle")
        stride_length = _col(df, "stride_length")

        if "session" in df.columns:
            grouped = df.assign(
                arm_slot=arm_slot,
                stride_angle=stride_angle,
                stride_length=stride_length,
            ).groupby("session", dropna=False)
            arm_slot_delta = arm_slot - grouped["arm_slot"].transform("mean")
            stride_angle_delta = stride_angle - grouped["stride_angle"].transform("mean")
            stride_length_delta = stride_length - grouped["stride_length"].transform("mean")
        else:
            arm_slot_delta = arm_slot - arm_slot.mean()
            stride_angle_delta = stride_angle - stride_angle.mean()
            stride_length_delta = stride_length - stride_length.mean()

        return {
            "arm_slot_delta": arm_slot_delta.fillna(0.0),
            "stride_angle_delta": stride_angle_delta.fillna(0.0),
            "stride_length_delta": stride_length_delta.fillna(0.0),
        }


def add_biomechanical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable pitching metrics from OpenBiomechanics-style columns."""
    return BiomechanicalMetricTransformer().fit_transform(df)


def metric_dictionary() -> pd.DataFrame:
    rows = [
        (
            "sequencing_efficiency",
            "Pelvis and torso speed with better peak timing.",
            "Higher should generally support velocity if timing is efficient.",
        ),
        (
            "hip_shoulder_separation_score",
            "Magnitude of hip-shoulder separation near foot plant and max separation.",
            "Higher can indicate useful rotational stretch, within athlete-specific limits.",
        ),
        (
            "kinetic_chain_transfer_score",
            "Pelvis-to-torso, thorax-to-arm, shoulder, and elbow transfer metrics.",
            "Higher suggests more energy moving through the throwing chain.",
        ),
        (
            "lead_leg_block_score",
            "Lead-knee extension and lead-leg force indicators.",
            "Higher can indicate a firmer front side and better rotational braking.",
        ),
        (
            "lower_body_force_score",
            "Ground-reaction force, rate of force development, and center-of-gravity speed.",
            "Higher reflects more lower-half force production.",
        ),
        (
            "arm_speed_score",
            "Shoulder internal rotation and elbow extension angular velocity.",
            "Higher is directly related to arm speed and pitch velocity.",
        ),
        (
            "release_consistency_score",
            "Within-session stability of arm slot, stride angle, and stride length.",
            "Higher means less mechanical variation within the athlete session.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "baseball_interpretation", "expected_direction"])
