"""
sgp4_propagator.py — Orbit Propagation & TCA Finder

Propagates satellite orbits using SGP4 and finds the Time of
Closest Approach (TCA) between satellite pairs using coarse
grid search + scipy.optimize refinement.
"""

import numpy as np
from sgp4.api import Satrec, jday
from scipy.optimize import minimize_scalar
from typing import Dict, Optional

from config import PROPAGATION_WINDOW_DAYS, PROPAGATION_STEP_SEC


def propagate_single(
    sat: Satrec,
    jd: float,
    fr: float
) -> Dict:
    """
    Propagate a single satellite to a specific time.

    Args:
        sat: sgp4 Satrec object
        jd: Julian date (integer part)
        fr: Julian date (fractional part)

    Returns:
        Dict with position_km (3,), velocity_kms (3,), error_code
    """
    error, position, velocity = sat.sgp4(jd, fr)
    return {
        "position_km": np.array(position) if error == 0 else None,
        "velocity_kms": np.array(velocity) if error == 0 else None,
        "error_code": error,
    }


def propagate_pair(
    sat1: Satrec,
    sat2: Satrec,
    start_jd: float,
    start_fr: float,
    window_days: float = PROPAGATION_WINDOW_DAYS,
    step_seconds: float = PROPAGATION_STEP_SEC,
) -> Optional[Dict]:
    """
    Propagate two satellites and find their closest approach.

    Uses a two-phase approach:
    1. Coarse scan at step_seconds intervals to find approximate TCA
    2. Fine refinement using scipy.optimize.minimize_scalar

    Args:
        sat1: First satellite Satrec object
        sat2: Second satellite Satrec object
        start_jd: Start Julian date (integer part)
        start_fr: Start Julian date (fractional part)
        window_days: Propagation window in days
        step_seconds: Coarse scan step size in seconds

    Returns:
        Dict with TCA information, or None if propagation fails
    """
    # Phase 1: Coarse scan
    n_steps = int(window_days * 86400 / step_seconds)
    min_dist = float('inf')
    min_idx = 0
    found_valid = False

    for i in range(n_steps):
        t_days = i * step_seconds / 86400.0
        fr = start_fr + t_days
        jd = start_jd

        # Handle day rollover
        while fr >= 1.0:
            fr -= 1.0
            jd += 1.0

        e1, r1, v1 = sat1.sgp4(jd, fr)
        e2, r2, v2 = sat2.sgp4(jd, fr)

        if e1 != 0 or e2 != 0:
            continue

        found_valid = True
        r1, r2 = np.array(r1), np.array(r2)
        dist = np.linalg.norm(r1 - r2)

        if dist < min_dist:
            min_dist = dist
            min_idx = i

    if not found_valid:
        return None

    # Phase 2: Fine refinement around the coarse minimum
    t_min = max(0, min_idx - 1) * step_seconds / 86400.0
    t_max = min(n_steps - 1, min_idx + 1) * step_seconds / 86400.0

    def distance_at_t(t_days: float) -> float:
        fr = start_fr + t_days
        jd = start_jd
        while fr >= 1.0:
            fr -= 1.0
            jd += 1.0
        e1, r1, v1 = sat1.sgp4(jd, fr)
        e2, r2, v2 = sat2.sgp4(jd, fr)
        if e1 != 0 or e2 != 0:
            return 1e12
        return np.linalg.norm(np.array(r1) - np.array(r2))

    result = minimize_scalar(
        distance_at_t,
        bounds=(t_min, t_max),
        method='bounded',
        options={'xatol': 1e-8}
    )
    tca_t = result.x

    # Get full state vectors at TCA
    fr_tca = start_fr + tca_t
    jd_tca = start_jd
    while fr_tca >= 1.0:
        fr_tca -= 1.0
        jd_tca += 1.0

    e1, r1, v1 = sat1.sgp4(jd_tca, fr_tca)
    e2, r2, v2 = sat2.sgp4(jd_tca, fr_tca)

    if e1 != 0 or e2 != 0:
        return None

    r1, r2 = np.array(r1), np.array(r2)
    v1, v2 = np.array(v1), np.array(v2)

    delta_r = r1 - r2
    delta_v = v1 - v2

    return {
        "tca_jd": jd_tca,
        "tca_fr": fr_tca,
        "tca_jd_full": jd_tca + fr_tca,
        "tca_offset_days": tca_t,
        "miss_distance_km": np.linalg.norm(delta_r),
        "rel_velocity_km_s": np.linalg.norm(delta_v),
        "pos1_tca_km": r1,
        "pos2_tca_km": r2,
        "vel1_tca_kms": v1,
        "vel2_tca_kms": v2,
        "delta_r_km": delta_r,
        "delta_v_kms": delta_v,
    }


def propagate_pair_short(
    sat1: Satrec,
    sat2: Satrec,
    start_jd: float,
    start_fr: float,
    window_days: float = 3.0,
    step_seconds: float = 120.0,
) -> Optional[Dict]:
    """
    Fast propagation for data generation — shorter window, larger steps.

    Args:
        sat1, sat2: Satrec objects
        start_jd, start_fr: Start epoch
        window_days: Shorter window (default 3 days)
        step_seconds: Larger step (default 120s)

    Returns:
        Dict with TCA info or None
    """
    return propagate_pair(
        sat1, sat2, start_jd, start_fr,
        window_days=window_days,
        step_seconds=step_seconds
    )


def jday_now() -> tuple:
    """Get current Julian date as (jd, fr) tuple."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return jday(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second + now.microsecond / 1e6
    )
