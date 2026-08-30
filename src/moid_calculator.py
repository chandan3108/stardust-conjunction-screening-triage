"""
moid_calculator.py — Minimum Orbit Intersection Distance

Computes MOID between two Keplerian orbits using:
1. Fast perigee/apogee radial overlap pre-filter (O(1))
2. Grid search on parametric distance between orbital ellipses
3. L-BFGS-B local refinement for precision

MOID is purely geometric — it answers "how close can these two
orbits ever get?" regardless of where the objects are at any time.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional
import pandas as pd

from config import MOID_THRESHOLD_KM, R_EARTH


def radial_overlap_check(
    a1: float, e1: float,
    a2: float, e2: float,
    margin_km: float = 0.0
) -> bool:
    """
    Quick O(1) check if two orbits overlap radially.

    If the perigee of one orbit is above the apogee of the other,
    MOID is guaranteed > 0 and the pair can be instantly discarded.

    Args:
        a1, e1: Semi-major axis (km) and eccentricity of orbit 1
        a2, e2: Semi-major axis (km) and eccentricity of orbit 2
        margin_km: Extra margin to add to overlap check

    Returns:
        True if orbits MAY intersect radially (need full MOID)
    """
    r_perigee_1 = a1 * (1.0 - e1)
    r_apogee_1 = a1 * (1.0 + e1)
    r_perigee_2 = a2 * (1.0 - e2)
    r_apogee_2 = a2 * (1.0 + e2)

    overlap = min(r_apogee_1, r_apogee_2) - max(r_perigee_1, r_perigee_2)
    return overlap > -margin_km


def keplerian_to_position(
    a: float, e: float, i: float,
    raan: float, omega: float, nu: float
) -> np.ndarray:
    """
    Convert Keplerian elements + true anomaly to ECI position vector.

    Args:
        a: Semi-major axis (km)
        e: Eccentricity
        i: Inclination (radians)
        raan: Right Ascension of Ascending Node (radians)
        omega: Argument of perigee (radians)
        nu: True anomaly (radians)

    Returns:
        3D position vector in ECI frame (km)
    """
    # Radius at this true anomaly
    r = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))

    # Position in orbital plane (perifocal frame)
    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)

    # Rotation matrix: Perifocal → ECI
    cos_o, sin_o = np.cos(omega), np.sin(omega)
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)

    x_eci = ((cos_O * cos_o - sin_O * sin_o * cos_i) * x_pf +
             (-cos_O * sin_o - sin_O * cos_o * cos_i) * y_pf)
    y_eci = ((sin_O * cos_o + cos_O * sin_o * cos_i) * x_pf +
             (-sin_O * sin_o + cos_O * cos_o * cos_i) * y_pf)
    z_eci = (sin_o * sin_i) * x_pf + (cos_o * sin_i) * y_pf

    return np.array([x_eci, y_eci, z_eci])


def compute_moid(
    elements1: Tuple[float, ...],
    elements2: Tuple[float, ...],
    n_grid: int = 360
) -> float:
    """
    Compute MOID between two orbits using grid search + local refinement.

    Args:
        elements1: (a, e, i_rad, raan_rad, omega_rad) for orbit 1
        elements2: (a, e, i_rad, raan_rad, omega_rad) for orbit 2
        n_grid: Grid resolution for initial search

    Returns:
        MOID in km
    """
    a1, e1, i1, raan1, omega1 = elements1
    a2, e2, i2, raan2, omega2 = elements2

    def distance(params):
        nu1, nu2 = params
        pos1 = keplerian_to_position(a1, e1, i1, raan1, omega1, nu1)
        pos2 = keplerian_to_position(a2, e2, i2, raan2, omega2, nu2)
        return np.linalg.norm(pos1 - pos2)

    # Phase 1: Coarse grid search
    nu_grid = np.linspace(0, 2.0 * np.pi, n_grid, endpoint=False)
    min_dist = float('inf')
    best_nu1, best_nu2 = 0.0, 0.0

    for nu1 in nu_grid:
        pos1 = keplerian_to_position(a1, e1, i1, raan1, omega1, nu1)
        for nu2 in nu_grid:
            pos2 = keplerian_to_position(a2, e2, i2, raan2, omega2, nu2)
            d = np.linalg.norm(pos1 - pos2)
            if d < min_dist:
                min_dist = d
                best_nu1, best_nu2 = nu1, nu2

    # Phase 2: Local refinement with L-BFGS-B
    result = minimize(
        distance,
        x0=[best_nu1, best_nu2],
        method='L-BFGS-B',
        bounds=[(0, 2 * np.pi), (0, 2 * np.pi)]
    )

    return float(result.fun)


def compute_moid_fast(
    elements1: Tuple[float, ...],
    elements2: Tuple[float, ...],
    n_grid: int = 72
) -> float:
    """
    Fast MOID computation with coarser grid (5° resolution).
    Good enough for screening, 25x faster than full resolution.

    Args:
        elements1: (a, e, i_rad, raan_rad, omega_rad) for orbit 1
        elements2: (a, e, i_rad, raan_rad, omega_rad) for orbit 2
        n_grid: Grid resolution (default 72 = 5° steps)

    Returns:
        MOID in km (approximate)
    """
    return compute_moid(elements1, elements2, n_grid=n_grid)


def screen_pairs_by_moid(
    catalog_df: pd.DataFrame,
    moid_threshold_km: float = MOID_THRESHOLD_KM,
    max_pairs: Optional[int] = None,
    use_fast: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Screen all pairs in catalog for MOID below threshold.
    Uses vectorized perigee/apogee pre-filter first.

    Args:
        catalog_df: DataFrame with orbital elements
        moid_threshold_km: Maximum MOID to keep pair (km)
        max_pairs: Optional limit on pairs to check
        use_fast: Use fast (coarser) MOID computation

    Returns:
        List of (idx1, idx2, moid_km) tuples for surviving pairs
    """
    from tqdm import tqdm

    n = len(catalog_df)
    surviving_pairs = []
    pairs_checked = 0
    pairs_filtered_radial = 0

    moid_fn = compute_moid_fast if use_fast else compute_moid

    total_pairs = n * (n - 1) // 2
    if max_pairs:
        total_pairs = min(total_pairs, max_pairs)

    print(f"[MOID Filter] Screening {n} objects "
          f"({total_pairs} pairs to check)...")

    for i in tqdm(range(n), desc="MOID screening"):
        row_i = catalog_df.iloc[i]
        a1 = row_i['SEMI_MAJOR_AXIS_KM']
        e1 = row_i['ECCENTRICITY']

        for j in range(i + 1, n):
            if max_pairs and pairs_checked >= max_pairs:
                break

            row_j = catalog_df.iloc[j]
            a2 = row_j['SEMI_MAJOR_AXIS_KM']
            e2 = row_j['ECCENTRICITY']

            # Quick radial overlap check (O(1))
            if not radial_overlap_check(a1, e1, a2, e2,
                                        margin_km=moid_threshold_km):
                pairs_filtered_radial += 1
                pairs_checked += 1
                continue

            # Full MOID calculation
            elements1 = (
                a1, e1,
                np.radians(row_i['INCLINATION']),
                np.radians(row_i['RA_OF_ASC_NODE']),
                np.radians(row_i['ARG_OF_PERICENTER'])
            )
            elements2 = (
                a2, e2,
                np.radians(row_j['INCLINATION']),
                np.radians(row_j['RA_OF_ASC_NODE']),
                np.radians(row_j['ARG_OF_PERICENTER'])
            )

            moid = moid_fn(elements1, elements2)
            pairs_checked += 1

            if moid <= moid_threshold_km:
                surviving_pairs.append((i, j, moid))

        if max_pairs and pairs_checked >= max_pairs:
            break

    print(f"[MOID Filter] Results:")
    print(f"  Pairs checked:        {pairs_checked}")
    print(f"  Filtered by radial:   {pairs_filtered_radial}")
    print(f"  Surviving pairs:      {len(surviving_pairs)} "
          f"(MOID ≤ {moid_threshold_km} km)")

    return surviving_pairs
