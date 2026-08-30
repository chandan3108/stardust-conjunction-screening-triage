"""
feature_engineer.py — Orbital Mechanics Feature Extraction

Transforms raw propagated state vectors into ML-ready features.
28 features across 7 categories: kinematics, RIC components,
covariance, orbital element differences, geometry, B-plane, ratios.
"""

import numpy as np
from typing import Dict

from config import HBR_DEFAULT_KM, R_EARTH


def extract_features(
    pos1: np.ndarray, vel1: np.ndarray,
    pos2: np.ndarray, vel2: np.ndarray,
    cov1: np.ndarray, cov2: np.ndarray,
    elements1: Dict, elements2: Dict,
    moid_km: float
) -> Dict[str, float]:
    """
    Extract the complete ML feature vector for a conjunction pair.

    Args:
        pos1: Primary position at TCA (km), shape (3,)
        vel1: Primary velocity at TCA (km/s), shape (3,)
        pos2: Secondary position at TCA (km), shape (3,)
        vel2: Secondary velocity at TCA (km/s), shape (3,)
        cov1: Primary 3x3 position covariance (km²)
        cov2: Secondary 3x3 position covariance (km²)
        elements1: Dict with keys: a_km, ecc, inc_deg, raan_deg
        elements2: Dict with keys: a_km, ecc, inc_deg, raan_deg
        moid_km: Pre-computed MOID for this pair (km)

    Returns:
        Dictionary of 28 named features
    """
    hbr = HBR_DEFAULT_KM  # 0.01 km = 10 m

    # ---- Kinematics at TCA ----
    delta_r = pos1 - pos2
    delta_v = vel1 - vel2

    miss_distance_km = np.linalg.norm(delta_r)
    rel_velocity_kms = np.linalg.norm(delta_v)

    # ---- RIC (Radial, In-track, Cross-track) Frame ----
    r_norm = np.linalg.norm(pos1)
    r_hat = pos1 / max(r_norm, 1e-10)
    h_vec = np.cross(pos1, vel1)
    h_norm = np.linalg.norm(h_vec)
    c_hat = h_vec / max(h_norm, 1e-10)
    i_hat = np.cross(c_hat, r_hat)

    delta_r_ric = np.array([
        np.dot(delta_r, r_hat),
        np.dot(delta_r, i_hat),
        np.dot(delta_r, c_hat),
    ])

    delta_v_ric = np.array([
        np.dot(delta_v, r_hat),
        np.dot(delta_v, i_hat),
        np.dot(delta_v, c_hat),
    ])

    # ---- Encounter Angle ----
    v1_norm = np.linalg.norm(vel1)
    v2_norm = np.linalg.norm(vel2)
    denom = v1_norm * v2_norm
    if denom > 1e-15:
        cos_encounter = np.dot(vel1, vel2) / denom
        encounter_angle_deg = np.degrees(
            np.arccos(np.clip(cos_encounter, -1.0, 1.0))
        )
    else:
        encounter_angle_deg = 0.0

    # Closing speed (radial component of relative velocity)
    if miss_distance_km > 1e-10:
        r_unit = delta_r / miss_distance_km
        closing_speed = abs(float(np.dot(delta_v, r_unit)))
    else:
        closing_speed = rel_velocity_kms

    tangential_velocity = np.sqrt(
        max(0.0, rel_velocity_kms ** 2 - closing_speed ** 2)
    )

    # ---- Covariance Features ----
    cov_combined = cov1 + cov2
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_combined))[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-20)

    # Mahalanobis distance
    try:
        cov_inv = np.linalg.inv(cov_combined)
        mahalanobis = float(np.sqrt(delta_r @ cov_inv @ delta_r))
    except np.linalg.LinAlgError:
        avg_var = max(np.trace(cov_combined) / 3.0, 1e-10)
        mahalanobis = miss_distance_km / np.sqrt(avg_var)

    # Covariance volume
    cov_det = np.linalg.det(cov_combined)
    cov_volume = float(np.sqrt(max(cov_det, 1e-30)))

    # ---- Orbital Element Differences ----
    a1, a2 = elements1['a_km'], elements2['a_km']
    e1, e2 = elements1['ecc'], elements2['ecc']
    i1, i2 = elements1['inc_deg'], elements2['inc_deg']
    raan1, raan2 = elements1['raan_deg'], elements2['raan_deg']

    delta_a = abs(a1 - a2)
    delta_e = abs(e1 - e2)
    delta_inc = abs(i1 - i2)
    delta_raan = abs(raan1 - raan2)
    if delta_raan > 180.0:
        delta_raan = 360.0 - delta_raan

    # Perigee/Apogee overlap
    rp1 = a1 * (1.0 - e1) - R_EARTH
    ra1 = a1 * (1.0 + e1) - R_EARTH
    rp2 = a2 * (1.0 - e2) - R_EARTH
    ra2 = a2 * (1.0 + e2) - R_EARTH
    altitude_overlap = max(0.0, min(ra1, ra2) - max(rp1, rp2))

    # ---- B-Plane / Pc Approximations ----
    sigma_bplane_sq = 0.5 * (cov_combined[1, 1] + cov_combined[2, 2])
    if sigma_bplane_sq > 1e-20:
        pc_foster_approx = float(
            (hbr ** 2 / (2.0 * sigma_bplane_sq)) *
            np.exp(-0.5 * miss_distance_km ** 2 / sigma_bplane_sq)
        )
    else:
        pc_foster_approx = 1.0

    # Akella & Alfriend upper bound
    if miss_distance_km > 1e-10:
        pc_upper_bound = float(hbr ** 2 / (np.e * miss_distance_km ** 2))
        pc_upper_bound = min(pc_upper_bound, 1.0)
    else:
        pc_upper_bound = 1.0

    # ---- Derived Ratios ----
    avg_sigma = max(np.sqrt(np.trace(cov_combined) / 3.0), 1e-10)
    miss_to_sigma = miss_distance_km / avg_sigma
    hbr_to_miss = hbr / max(miss_distance_km, 1e-10)
    energy_ratio = rel_velocity_kms ** 2 / max(miss_distance_km, 1e-10)

    return {
        # Kinematics (6 features)
        "miss_distance_m": miss_distance_km * 1000.0,
        "rel_velocity_kms": rel_velocity_kms,
        "encounter_angle_deg": encounter_angle_deg,
        "closing_speed_kms": closing_speed,
        "tangential_velocity_kms": tangential_velocity,
        # (miss_distance_km stored as miss_distance_m above)

        # RIC Components (6 features)
        "delta_r_radial_km": float(delta_r_ric[0]),
        "delta_r_intrack_km": float(delta_r_ric[1]),
        "delta_r_crosstrack_km": float(delta_r_ric[2]),
        "delta_v_radial_kms": float(delta_v_ric[0]),
        "delta_v_intrack_kms": float(delta_v_ric[1]),
        "delta_v_crosstrack_kms": float(delta_v_ric[2]),

        # Covariance / Uncertainty (5 features)
        "mahalanobis_distance": mahalanobis,
        "cov_eigenvalue_1": float(eigenvalues[0]),
        "cov_eigenvalue_2": float(eigenvalues[1]),
        "cov_eigenvalue_3": float(eigenvalues[2]),
        "cov_volume": cov_volume,

        # Orbital Element Differences (5 features)
        "delta_sma_km": delta_a,
        "delta_eccentricity": delta_e,
        "delta_inclination_deg": delta_inc,
        "delta_raan_deg": delta_raan,
        "altitude_overlap_km": altitude_overlap,

        # Geometry (3 features)
        "moid_km": moid_km,
        "pc_foster_approx": pc_foster_approx,
        "pc_upper_bound": pc_upper_bound,

        # Derived Ratios (3 features)
        "miss_to_sigma_ratio": miss_to_sigma,
        "hbr_to_miss_ratio": hbr_to_miss,
        "energy_ratio": energy_ratio,
    }
