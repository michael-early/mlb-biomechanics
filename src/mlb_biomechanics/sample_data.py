from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_biomechanics(n_sessions: int = 36, pitches_per_session: int = 4) -> pd.DataFrame:
    """Create deterministic baseball-like biomechanics data for offline demos and tests.

    The real MVP path uses OpenBiomechanics CSVs. This sample keeps the package runnable in
    environments without network access and is always labeled as generated sample data.
    """

    rng = np.random.default_rng(7)
    rows: list[dict[str, float | int | str]] = []

    for session in range(1001, 1001 + n_sessions):
        athlete = rng.normal()
        arm_strength = rng.normal()
        lower_body = rng.normal()
        command_variability = rng.uniform(0.4, 1.5)
        throws = "R" if rng.random() > 0.18 else "L"

        for pitch_idx in range(1, pitches_per_session + 1):
            noise = rng.normal(0, 1)
            pelvis_velo = 575 + 48 * lower_body + 18 * rng.normal()
            torso_velo = 900 + 58 * athlete + 32 * lower_body + 24 * rng.normal()
            shoulder_velo = 4300 + 190 * arm_strength + 70 * athlete + 80 * rng.normal()
            elbow_velo = 2350 + 120 * arm_strength + 50 * athlete + 60 * rng.normal()
            hip_sep = 31 + 3.2 * lower_body + 1.1 * rng.normal()
            lead_grf = 2550 + 200 * lower_body + 120 * rng.normal()
            rfd = 150 + 22 * lower_body + 14 * rng.normal()
            timing = 0.02 + 0.012 * rng.normal() - 0.006 * lower_body
            arm_slot = 41 + 2.5 * athlete + command_variability * rng.normal()
            stride_angle = 1.2 + command_variability * rng.normal()
            stride_length = 0.83 + 0.025 * lower_body + 0.015 * rng.normal()
            transfer = 95 + 14 * lower_body + 11 * athlete + 10 * rng.normal()
            pitch_speed = (
                87.5
                + 0.0055 * (shoulder_velo - 4300)
                + 0.0040 * (torso_velo - 900)
                + 0.10 * hip_sep
                + 0.0010 * (lead_grf - 2550)
                - 35 * abs(timing)
                - 0.35 * command_variability
                + noise
            )

            rows.append(
                {
                    "session_pitch": f"{session}_{pitch_idx}",
                    "session": session,
                    "p_throws": throws,
                    "pitch_type": "FF" if rng.random() > 0.16 else "SL",
                    "pitch_speed_mph": round(float(pitch_speed), 2),
                    "max_shoulder_internal_rotational_velo": shoulder_velo,
                    "max_elbow_extension_velo": elbow_velo,
                    "max_torso_rotational_velo": torso_velo,
                    "max_rotation_hip_shoulder_separation": hip_sep,
                    "rotation_hip_shoulder_separation_fp": hip_sep - 1.5 + rng.normal(),
                    "max_pelvis_rotational_velo": pelvis_velo,
                    "lead_knee_extension_angular_velo_max": 470 + 55 * lower_body + 40 * rng.normal(),
                    "lead_knee_extension_from_fp_to_br": 12 + 4.5 * lower_body + rng.normal(),
                    "cog_velo_pkh": 3.1 + 0.22 * lower_body + 0.1 * rng.normal(),
                    "stride_length": stride_length,
                    "stride_angle": stride_angle,
                    "arm_slot": arm_slot,
                    "timing_peak_torso_to_peak_pelvis_rot_velo": timing,
                    "shoulder_transfer_fp_br": transfer + 9 * rng.normal(),
                    "elbow_transfer_fp_br": 12 + 4 * arm_strength + 5 * rng.normal(),
                    "lead_hip_transfer_fp_br": 75 + 18 * lower_body + 8 * rng.normal(),
                    "lead_knee_transfer_fp_br": 110 + 22 * lower_body + 12 * rng.normal(),
                    "pelvis_lumbar_transfer_fp_br": transfer + 4 * rng.normal(),
                    "thorax_distal_transfer_fp_br": transfer + 5 * rng.normal(),
                    "rear_grf_mag_max": 1700 + 150 * lower_body + 110 * rng.normal(),
                    "lead_grf_mag_max": lead_grf,
                    "lead_grf_z_max": 2450 + 180 * lower_body + 110 * rng.normal(),
                    "peak_rfd_rear": 15 + 3 * lower_body + rng.normal(),
                    "peak_rfd_lead": rfd,
                }
            )

    return pd.DataFrame(rows)


def make_sample_statcast(n_pitches: int = 1200) -> pd.DataFrame:
    """Create deterministic Statcast-like pitch-level outcomes for the performance bridge."""

    rng = np.random.default_rng(11)
    release_speed = rng.normal(94.2, 3.1, n_pitches).clip(82, 103)
    pfx_x = rng.normal(0, 7.5, n_pitches)
    pfx_z = rng.normal(13, 4.0, n_pitches)
    release_extension = rng.normal(6.2, 0.55, n_pitches)
    plate_x = rng.normal(0, 0.8, n_pitches)
    plate_z = rng.normal(2.55, 0.72, n_pitches)
    pitch_type = rng.choice(["FF", "SI", "SL", "CH", "CU"], p=[0.42, 0.16, 0.22, 0.13, 0.07], size=n_pitches)

    quality = (
        0.085 * (release_speed - 92)
        + 0.025 * np.abs(pfx_x)
        + 0.035 * (pfx_z - 11)
        + 0.10 * (release_extension - 6)
        - 0.20 * np.abs(plate_x)
    )
    whiff_prob = 1 / (1 + np.exp(-(quality - 0.3)))
    chase_prob = 1 / (1 + np.exp(-(quality - 0.05 + 0.18 * np.abs(plate_x))))
    hard_hit_prob = 1 / (1 + np.exp(quality - 1.1))
    run_value = -0.045 * quality + rng.normal(0, 0.12, n_pitches)

    return pd.DataFrame(
        {
            "game_date": pd.date_range("2025-04-01", periods=n_pitches, freq="h").date,
            "pitcher": rng.integers(500000, 700000, n_pitches),
            "pitch_type": pitch_type,
            "release_speed": release_speed.round(2),
            "release_extension": release_extension.round(2),
            "pfx_x": pfx_x.round(2),
            "pfx_z": pfx_z.round(2),
            "plate_x": plate_x.round(2),
            "plate_z": plate_z.round(2),
            "whiff": rng.binomial(1, whiff_prob).astype(int),
            "chase": rng.binomial(1, chase_prob).astype(int),
            "hard_hit": rng.binomial(1, hard_hit_prob).astype(int),
            "run_value": run_value.round(4),
            "data_source": "generated_statcast_like_sample",
        }
    )

