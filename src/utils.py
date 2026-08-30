"""
utils.py — Coordinate Transforms, Constants & Helper Functions

Provides TEME→ECI conversion, Keplerian element utilities,
and shared helper functions for the STARDUST pipeline.
"""

import numpy as np
from typing import Tuple

# Re-export key constants for convenience
from config import MU_EARTH, R_EARTH, J2


def teme_to_eci_approx(
    pos_teme_km: np.ndarray,
    epoch_jd: float
) -> np.ndarray:
    """
    Approximate TEME → ECI (J2000/GCRS) conversion.

    For TLE-level accuracy, the difference between TEME and J2000 is
    small (< 0.01° rotation). This uses a simplified precession/nutation
    correction sufficient for conjunction screening.

    Args:
        pos_teme_km: Position in TEME frame (km), shape (3,)
        epoch_jd: Julian date of the epoch

    Returns:
        Position in approximate ECI frame (km), shape (3,)
    """
    # Centuries since J2000.0
    T = (epoch_jd - 2451545.0) / 36525.0

    # Precession angles (radians) — simplified IAU 1976
    zeta_A = np.radians((0.6406161 + (0.0000839 + 0.0000050 * T) * T) * T)
    theta_A = np.radians((0.5567530 - (0.0001185 + 0.0000116 * T) * T) * T)
    z_A = np.radians((0.6406161 + (0.0003041 + 0.0000051 * T) * T) * T)

    # Precession rotation matrix
    cos_za, sin_za = np.cos(zeta_A), np.sin(zeta_A)
    cos_ta, sin_ta = np.cos(theta_A), np.sin(theta_A)
    cos_z, sin_z = np.cos(z_A), np.sin(z_A)

    R = np.array([
        [cos_za * cos_ta * cos_z - sin_za * sin_z,
         -sin_za * cos_ta * cos_z - cos_za * sin_z,
         -sin_ta * cos_z],
        [cos_za * cos_ta * sin_z + sin_za * cos_z,
         -sin_za * cos_ta * sin_z + cos_za * cos_z,
         -sin_ta * sin_z],
        [cos_za * sin_ta,
         -sin_za * sin_ta,
         cos_ta]
    ])

    return R @ pos_teme_km


def mean_motion_to_sma(n_rev_per_day: float) -> float:
    """
    Convert mean motion (rev/day) to semi-major axis (km).

    Uses Kepler's third law: a = (μ / (2πn/86400)²)^(1/3)

    Args:
        n_rev_per_day: Mean motion in revolutions per day

    Returns:
        Semi-major axis in km
    """
    n_rad_s = n_rev_per_day * 2.0 * np.pi / 86400.0
    return (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)


def sma_to_period(a_km: float) -> float:
    """
    Convert semi-major axis (km) to orbital period (seconds).

    Args:
        a_km: Semi-major axis in km

    Returns:
        Orbital period in seconds
    """
    return 2.0 * np.pi * np.sqrt(a_km ** 3 / MU_EARTH)


def compute_altitude(a_km: float, ecc: float) -> Tuple[float, float]:
    """
    Compute perigee and apogee altitudes above Earth's surface.

    Args:
        a_km: Semi-major axis (km)
        ecc: Eccentricity

    Returns:
        (perigee_alt_km, apogee_alt_km)
    """
    perigee = a_km * (1.0 - ecc) - R_EARTH
    apogee = a_km * (1.0 + ecc) - R_EARTH
    return perigee, apogee


def keplerian_to_cartesian(
    a: float, e: float, i: float,
    raan: float, omega: float, nu: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian orbital elements to Cartesian state vector (ECI).

    Args:
        a: Semi-major axis (km)
        e: Eccentricity
        i: Inclination (radians)
        raan: Right Ascension of Ascending Node (radians)
        omega: Argument of perigee (radians)
        nu: True anomaly (radians)

    Returns:
        (position_km, velocity_km_s) in ECI frame
    """
    # Radius
    r = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))

    # Position in perifocal frame
    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)

    # Velocity in perifocal frame
    p = a * (1.0 - e ** 2)
    h = np.sqrt(MU_EARTH * p)
    vx_pf = -(MU_EARTH / h) * np.sin(nu)
    vy_pf = (MU_EARTH / h) * (e + np.cos(nu))

    # Rotation matrix: Perifocal → ECI
    cos_o, sin_o = np.cos(omega), np.sin(omega)
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [cos_O * cos_o - sin_O * sin_o * cos_i,
         -cos_O * sin_o - sin_O * cos_o * cos_i,
         sin_O * sin_i],
        [sin_O * cos_o + cos_O * sin_o * cos_i,
         -sin_O * sin_o + cos_O * cos_o * cos_i,
         -cos_O * sin_i],
        [sin_o * sin_i,
         cos_o * sin_i,
         cos_i]
    ])

    pos = R @ np.array([x_pf, y_pf, 0.0])
    vel = R @ np.array([vx_pf, vy_pf, 0.0])

    return pos, vel


def angular_separation_deg(
    vec1: np.ndarray, vec2: np.ndarray
) -> float:
    """
    Compute angular separation between two vectors in degrees.

    Args:
        vec1, vec2: 3D vectors

    Returns:
        Angle in degrees [0, 180]
    """
    cos_angle = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-15
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
