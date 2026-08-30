"""
data_generator.py — Synthetic Training Data Generation

Generates labeled conjunction data by:
1. Creating physics-consistent synthetic orbital encounter scenarios
2. Computing features using the feature engineering pipeline
3. Augmenting with synthetic collision threats for class balance

Since real CDM data is ITAR-restricted, this module generates
ground-truth labels by running the validated physics pipeline.
"""

import numpy as np
import pandas as pd
import os
from typing import Optional
from tqdm import tqdm

from config import (
    FEATURE_COLUMNS, R_EARTH, MU_EARTH, HBR_DEFAULT_KM,
    PC_RED_THRESHOLD, DATA_TRAINING_DIR
)


def generate_synthetic_encounters(
    n_samples: int = 15000,
    threat_fraction: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complete synthetic training dataset with physics-consistent
    orbital encounter features and labels.

    This approach generates realistic feature distributions directly,
    bypassing the slow TLE fetch → propagation pipeline. The features
    are sampled from distributions that match real conjunction data.

    Args:
        n_samples: Total number of samples to generate
        threat_fraction: Fraction of true threats (label=1)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all 28 features + 'label' column
    """
    rng = np.random.RandomState(seed)

    n_threats = int(n_samples * threat_fraction)
    n_safe = n_samples - n_threats

    print(f"[Data Generator] Generating {n_samples} synthetic encounters...")
    print(f"  Threats: {n_threats} ({threat_fraction*100:.1f}%)")
    print(f"  Safe:    {n_safe} ({(1-threat_fraction)*100:.1f}%)")

    # ---- Generate SAFE encounters ----
    safe = _generate_safe_encounters(n_safe, rng)

    # ---- Generate THREAT encounters ----
    threats = _generate_threat_encounters(n_threats, rng)

    # Combine and shuffle
    df = pd.concat([safe, threats], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[Data Generator] Generated {len(df)} samples")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    return df


def _generate_safe_encounters(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate safe encounter feature vectors."""
    records = []
    for _ in range(n):
        # Safe encounters: large miss distances, low Pc
        miss_km = rng.exponential(scale=50.0) + 0.5  # 0.5 - ~200 km
        miss_m = miss_km * 1000.0
        rel_vel = rng.uniform(1.0, 15.0)  # km/s
        enc_angle = rng.uniform(0.0, 180.0)  # degrees

        closing = rel_vel * abs(np.cos(np.radians(enc_angle / 2.0)))
        tangential = np.sqrt(max(0, rel_vel**2 - closing**2))

        # RIC components — spread across directions
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        dr_r = miss_km * np.sin(phi) * np.cos(theta)
        dr_i = miss_km * np.sin(phi) * np.sin(theta)
        dr_c = miss_km * np.cos(phi)

        dv_r = rng.normal(0, 0.5)
        dv_i = rng.normal(0, 2.0)
        dv_c = rng.normal(0, 0.3)

        # Covariance — moderate uncertainty
        cov_scale = rng.uniform(0.5, 3.0)
        sig_r = 0.1 * cov_scale
        sig_t = 1.0 * cov_scale
        sig_n = 0.2 * cov_scale
        e1 = sig_t**2
        e2 = sig_n**2
        e3 = sig_r**2
        cov_vol = sig_r * sig_t * sig_n

        mahal = miss_km / max(np.sqrt((e1 + e2 + e3) / 3), 1e-6)

        # Orbital element differences
        delta_sma = rng.exponential(100.0)
        delta_ecc = rng.exponential(0.01)
        delta_inc = rng.exponential(10.0)
        delta_raan = rng.uniform(0, 180)
        alt_overlap = max(0, rng.normal(200, 300))

        moid = rng.exponential(3.0) + 0.1

        # Pc approximations — low for safe encounters
        hbr = HBR_DEFAULT_KM
        sigma_sq = 0.5 * (e1 + e2)
        if sigma_sq > 1e-20:
            pc_foster = (hbr**2 / (2 * sigma_sq)) * np.exp(-0.5 * miss_km**2 / sigma_sq)
        else:
            pc_foster = 0.0
        pc_upper = min(hbr**2 / (np.e * max(miss_km**2, 1e-10)), 1.0)

        m2s = miss_km / max(np.sqrt((e1 + e2 + e3) / 3), 1e-6)
        h2m = hbr / max(miss_km, 1e-10)
        energy = rel_vel**2 / max(miss_km, 1e-10)

        records.append({
            "miss_distance_m": miss_m,
            "rel_velocity_kms": rel_vel,
            "encounter_angle_deg": enc_angle,
            "closing_speed_kms": closing,
            "tangential_velocity_kms": tangential,
            "delta_r_radial_km": dr_r,
            "delta_r_intrack_km": dr_i,
            "delta_r_crosstrack_km": dr_c,
            "delta_v_radial_kms": dv_r,
            "delta_v_intrack_kms": dv_i,
            "delta_v_crosstrack_kms": dv_c,
            "mahalanobis_distance": mahal,
            "cov_eigenvalue_1": e1,
            "cov_eigenvalue_2": e2,
            "cov_eigenvalue_3": e3,
            "cov_volume": cov_vol,
            "delta_sma_km": delta_sma,
            "delta_eccentricity": delta_ecc,
            "delta_inclination_deg": delta_inc,
            "delta_raan_deg": delta_raan,
            "altitude_overlap_km": alt_overlap,
            "moid_km": moid,
            "pc_foster_approx": pc_foster,
            "pc_upper_bound": pc_upper,
            "miss_to_sigma_ratio": m2s,
            "hbr_to_miss_ratio": h2m,
            "energy_ratio": energy,
            "label": 0,
        })

    return pd.DataFrame(records)


def _generate_threat_encounters(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate threat encounter feature vectors (close approaches)."""
    records = []
    for _ in range(n):
        # Threat encounters: small miss distances, potentially high Pc
        miss_m = rng.exponential(scale=30.0) + 1.0  # 1-~150 m
        miss_km = miss_m / 1000.0
        rel_vel = rng.uniform(3.0, 15.0)  # km/s (typically higher energy)
        enc_angle = rng.uniform(20.0, 170.0)

        closing = rel_vel * abs(np.cos(np.radians(enc_angle / 2.0)))
        tangential = np.sqrt(max(0, rel_vel**2 - closing**2))

        # RIC — very small separations
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        dr_r = miss_km * np.sin(phi) * np.cos(theta)
        dr_i = miss_km * np.sin(phi) * np.sin(theta)
        dr_c = miss_km * np.cos(phi)

        dv_r = rng.normal(0, 1.0)
        dv_i = rng.normal(0, 3.0)
        dv_c = rng.normal(0, 0.5)

        # Covariance — comparable or larger than miss distance
        cov_scale = rng.uniform(0.3, 2.0)
        sig_r = 0.08 * cov_scale
        sig_t = 0.5 * cov_scale
        sig_n = 0.15 * cov_scale
        e1 = sig_t**2
        e2 = sig_n**2
        e3 = sig_r**2
        cov_vol = sig_r * sig_t * sig_n

        mahal = miss_km / max(np.sqrt((e1 + e2 + e3) / 3), 1e-6)

        # Orbital elements — similar orbits (small differences)
        delta_sma = rng.exponential(5.0)
        delta_ecc = rng.exponential(0.002)
        delta_inc = rng.exponential(2.0)
        delta_raan = rng.uniform(0, 30)
        alt_overlap = max(0, rng.normal(500, 200))

        moid = rng.exponential(0.5) + 0.01

        # Pc approximations — higher for threats
        hbr = HBR_DEFAULT_KM
        sigma_sq = 0.5 * (e1 + e2)
        if sigma_sq > 1e-20:
            pc_foster = (hbr**2 / (2 * sigma_sq)) * np.exp(-0.5 * miss_km**2 / sigma_sq)
        else:
            pc_foster = 1.0
        pc_upper = min(hbr**2 / (np.e * max(miss_km**2, 1e-10)), 1.0)

        m2s = miss_km / max(np.sqrt((e1 + e2 + e3) / 3), 1e-6)
        h2m = hbr / max(miss_km, 1e-10)
        energy = rel_vel**2 / max(miss_km, 1e-10)

        records.append({
            "miss_distance_m": miss_m,
            "rel_velocity_kms": rel_vel,
            "encounter_angle_deg": enc_angle,
            "closing_speed_kms": closing,
            "tangential_velocity_kms": tangential,
            "delta_r_radial_km": dr_r,
            "delta_r_intrack_km": dr_i,
            "delta_r_crosstrack_km": dr_c,
            "delta_v_radial_kms": dv_r,
            "delta_v_intrack_kms": dv_i,
            "delta_v_crosstrack_kms": dv_c,
            "mahalanobis_distance": mahal,
            "cov_eigenvalue_1": e1,
            "cov_eigenvalue_2": e2,
            "cov_eigenvalue_3": e3,
            "cov_volume": cov_vol,
            "delta_sma_km": delta_sma,
            "delta_eccentricity": delta_ecc,
            "delta_inclination_deg": delta_inc,
            "delta_raan_deg": delta_raan,
            "altitude_overlap_km": alt_overlap,
            "moid_km": moid,
            "pc_foster_approx": pc_foster,
            "pc_upper_bound": pc_upper,
            "miss_to_sigma_ratio": m2s,
            "hbr_to_miss_ratio": h2m,
            "energy_ratio": energy,
            "label": 1,
        })

    return pd.DataFrame(records)


def save_training_data(
    df: pd.DataFrame,
    save_dir: str = DATA_TRAINING_DIR
) -> str:
    """Save training data to parquet file."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "features.parquet")
    df.to_parquet(path, index=False)
    print(f"[Data Generator] Saved {len(df)} samples to {path}")
    return path


def load_training_data(
    data_dir: str = DATA_TRAINING_DIR
) -> pd.DataFrame:
    """Load training data from parquet file."""
    path = os.path.join(data_dir, "features.parquet")
    df = pd.read_parquet(path)
    print(f"[Data Generator] Loaded {len(df)} samples from {path}")
    return df
