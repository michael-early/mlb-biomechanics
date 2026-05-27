from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_FEATURES = [
    "sequencing_efficiency",
    "hip_shoulder_separation_score",
    "kinetic_chain_transfer_score",
    "lead_leg_block_score",
    "lower_body_force_score",
    "arm_speed_score",
    "release_consistency_score",
]


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / std


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def add_biomechanical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable pitching metrics from OpenBiomechanics-style columns."""

    out = df.copy()

    timing = _col(out, "timing_peak_torso_to_peak_pelvis_rot_velo")
    arm_slot = _col(out, "arm_slot")
    stride_angle = _col(out, "stride_angle")
    stride_length = _col(out, "stride_length")

    out["sequencing_efficiency"] = (
        _z(_col(out, "max_pelvis_rotational_velo"))
        + _z(_col(out, "max_torso_rotational_velo"))
        - _z(timing.abs())
    )
    out["hip_shoulder_separation_score"] = (
        _z(_col(out, "max_rotation_hip_shoulder_separation"))
        + _z(_col(out, "rotation_hip_shoulder_separation_fp"))
    )
    out["kinetic_chain_transfer_score"] = (
        _z(_col(out, "pelvis_lumbar_transfer_fp_br"))
        + _z(_col(out, "thorax_distal_transfer_fp_br"))
        + _z(_col(out, "shoulder_transfer_fp_br"))
        + _z(_col(out, "elbow_transfer_fp_br"))
    )
    out["lead_leg_block_score"] = (
        _z(_col(out, "lead_knee_extension_angular_velo_max"))
        + _z(_col(out, "lead_knee_extension_from_fp_to_br"))
        + _z(_col(out, "lead_grf_z_max"))
    )
    out["lower_body_force_score"] = (
        _z(_col(out, "lead_grf_mag_max"))
        + _z(_col(out, "rear_grf_mag_max"))
        + _z(_col(out, "peak_rfd_lead"))
        + _z(_col(out, "cog_velo_pkh"))
    )
    out["arm_speed_score"] = (
        _z(_col(out, "max_shoulder_internal_rotational_velo"))
        + _z(_col(out, "max_elbow_extension_velo"))
    )

    if "session" in out.columns:
        grouped = out.groupby("session", dropna=False)
        arm_slot_delta = arm_slot - grouped["arm_slot"].transform("mean")
        stride_angle_delta = stride_angle - grouped["stride_angle"].transform("mean")
        stride_length_delta = stride_length - grouped["stride_length"].transform("mean")
    else:
        arm_slot_delta = arm_slot - arm_slot.mean()
        stride_angle_delta = stride_angle - stride_angle.mean()
        stride_length_delta = stride_length - stride_length.mean()

    out["release_consistency_score"] = -(
        _z(arm_slot_delta.abs()) + _z(stride_angle_delta.abs()) + _z(stride_length_delta.abs())
    )

    for feature in METRIC_FEATURES:
        out[feature] = out[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


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

