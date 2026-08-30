"""
chan_formula.py — 2D Collision Probability Calculator

Implements the Chan/Foster method for computing probability of collision
at the B-plane (encounter plane) using combined covariance integration.

Reference:
    Chan, F.K. (1997). "Spacecraft Collision Probability." The Aerospace Press.
    Foster, J.L., Estes, H.S. (1992). NASA JSC-25898.
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
from typing import Dict

from config import HBR_DEFAULT_KM, PC_RED_THRESHOLD, PC_YELLOW_THRESHOLD


def compute_bplane_projection(
    delta_r: np.ndarray,
    delta_v: np.ndarray,
    cov_combined: np.ndarray
) -> Dict:
    """
    Project 3D relative state onto the 2D B-plane.

    The B-plane is perpendicular to the relative velocity vector
    at the time of closest approach.

    Args:
        delta_r: Relative position vector at TCA (km), shape (3,)
        delta_v: Relative velocity vector at TCA (km/s), shape (3,)
        cov_combined: Combined 3x3 position covariance (km²)

    Returns:
        Dict with B-plane coordinates and 2D covariance
    """
    # B-plane basis vectors
    e1 = delta_v / np.linalg.norm(delta_v)  # along relative velocity

    cross = np.cross(delta_v, delta_r)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-15:
        # Degenerate case: head-on or nearly so
        if abs(e1[0]) < 0.9:
            e2 = np.cross(e1, np.array([1.0, 0.0, 0.0]))
        else:
            e2 = np.cross(e1, np.array([0.0, 1.0, 0.0]))
        e2 = e2 / np.linalg.norm(e2)
    else:
        e2 = cross / cross_norm

    e3 = np.cross(e1, e2)

    # Projection matrix (2x3)
    P = np.array([e2, e3])

    # B-plane miss vector components
    xi = float(np.dot(delta_r, e2))      # cross-track
    zeta = float(np.dot(delta_r, e3))    # in-plane

    # 2D projected covariance
    cov_2d = P @ cov_combined @ P.T

    # Ensure symmetry
    cov_2d = 0.5 * (cov_2d + cov_2d.T)

    sigma_xi = np.sqrt(max(cov_2d[0, 0], 1e-20))
    sigma_zeta = np.sqrt(max(cov_2d[1, 1], 1e-20))
    corr_denom = sigma_xi * sigma_zeta
    correlation = cov_2d[0, 1] / corr_denom if corr_denom > 1e-20 else 0.0

    return {
        "xi": xi,
        "zeta": zeta,
        "miss_bplane": np.sqrt(xi ** 2 + zeta ** 2),
        "cov_2d": cov_2d,
        "sigma_xi": sigma_xi,
        "sigma_zeta": sigma_zeta,
        "correlation": correlation,
        "basis_e1": e1,
        "basis_e2": e2,
        "basis_e3": e3,
    }


def chan_collision_probability(
    delta_r: np.ndarray,
    delta_v: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    hbr: float = HBR_DEFAULT_KM,
    method: str = "numerical"
) -> Dict:
    """
    Compute 2D collision probability using the Chan/Foster method.

    Args:
        delta_r: Relative position at TCA (km), shape (3,)
        delta_v: Relative velocity at TCA (km/s), shape (3,)
        cov1: Primary 3x3 position covariance (km²)
        cov2: Secondary 3x3 position covariance (km²)
        hbr: Combined Hard-Body Radius (km). Default 10m = 0.01 km
        method: 'numerical' (dblquad) or 'foster' (analytical approx)

    Returns:
        Dict with Pc value and diagnostic info
    """
    # Combined covariance
    cov_combined = cov1 + cov2

    # B-plane projection
    bp = compute_bplane_projection(delta_r, delta_v, cov_combined)

    xi, zeta = bp["xi"], bp["zeta"]
    cov_2d = bp["cov_2d"]
    d_miss_sq = xi ** 2 + zeta ** 2

    if method == "numerical":
        # Numerical 2D integration over HBR disk
        try:
            rv = multivariate_normal(mean=[xi, zeta], cov=cov_2d,
                                     allow_singular=True)

            def integrand(y, x):
                return rv.pdf([x, y])

            def y_lower(x):
                if abs(x) >= hbr:
                    return 0.0
                return -np.sqrt(hbr ** 2 - x ** 2)

            def y_upper(x):
                if abs(x) >= hbr:
                    return 0.0
                return np.sqrt(hbr ** 2 - x ** 2)

            pc, error = dblquad(
                integrand,
                -hbr, hbr,
                y_lower, y_upper,
                epsabs=1e-15, epsrel=1e-12
            )
        except Exception:
            # Fallback to Foster approximation
            pc = _foster_pc(hbr, d_miss_sq, cov_2d)

    elif method == "foster":
        pc = _foster_pc(hbr, d_miss_sq, cov_2d)

    else:
        raise ValueError(f"Unknown method: {method}")

    pc = float(np.clip(pc, 0.0, 1.0))

    # Foster maximum Pc (analytical upper bound)
    pc_foster = _foster_pc(hbr, d_miss_sq, cov_2d)

    # Akella & Alfriend upper bound
    if d_miss_sq > 1e-20:
        pc_upper_bound = float(hbr ** 2 / (np.e * d_miss_sq))
        pc_upper_bound = min(pc_upper_bound, 1.0)
    else:
        pc_upper_bound = 1.0

    # Determine threat level
    if pc > PC_RED_THRESHOLD:
        threat_level = "CRITICAL"
    elif pc > PC_YELLOW_THRESHOLD:
        threat_level = "WARNING"
    else:
        threat_level = "NOMINAL"

    return {
        "pc": pc,
        "pc_foster_approx": pc_foster,
        "pc_upper_bound": pc_upper_bound,
        "miss_distance_km": np.sqrt(d_miss_sq),
        "miss_distance_m": np.sqrt(d_miss_sq) * 1000.0,
        "hbr_km": hbr,
        "bplane_xi_km": xi,
        "bplane_zeta_km": zeta,
        "sigma_xi_km": bp["sigma_xi"],
        "sigma_zeta_km": bp["sigma_zeta"],
        "correlation": bp["correlation"],
        "threat_level": threat_level,
        "is_critical": pc > PC_RED_THRESHOLD,
        "is_warning": pc > PC_YELLOW_THRESHOLD,
    }


def _foster_pc(
    hbr: float,
    d_miss_sq: float,
    cov_2d: np.ndarray
) -> float:
    """
    Compute Foster's approximate maximum Pc.

    Pc ≈ (HBR² / 2σ²) × exp(-d²/2σ²)

    Args:
        hbr: Hard-body radius (km)
        d_miss_sq: Squared miss distance in B-plane (km²)
        cov_2d: 2D covariance matrix (km²)

    Returns:
        Approximate Pc value
    """
    sigma_sq = 0.5 * (cov_2d[0, 0] + cov_2d[1, 1])
    if sigma_sq > 1e-20:
        pc = (hbr ** 2 / (2.0 * sigma_sq)) * np.exp(
            -0.5 * d_miss_sq / sigma_sq
        )
    else:
        pc = 1.0 if d_miss_sq < hbr ** 2 else 0.0
    return float(np.clip(pc, 0.0, 1.0))


def approximate_covariance(
    bstar: float,
    altitude_km: float
) -> np.ndarray:
    """
    Generate an approximate position covariance when real CDM data
    is unavailable. Based on typical TLE-derived uncertainty models.

    Args:
        bstar: BSTAR drag coefficient from TLE
        altitude_km: Orbital altitude in km

    Returns:
        3x3 covariance matrix in km² (diagonal, in RIC-like frame)
    """
    # Scale with drag (lower altitude + higher BSTAR = more uncertainty)
    drag_factor = max(1.0, 5.0 * abs(bstar) * 1e4)

    # Altitude scaling (lower orbits have more drag uncertainty)
    alt_factor = max(0.5, 2.0 - altitude_km / 1000.0)

    combined = drag_factor * alt_factor

    # Typical TLE uncertainty (km)
    sigma_r = 0.1 * combined        # Radial
    sigma_t = 1.0 * combined        # In-track (largest)
    sigma_n = 0.2 * combined        # Cross-track

    return np.diag([sigma_r ** 2, sigma_t ** 2, sigma_n ** 2])
