"""
orbit_parser.py — TLE → Satrec Object Parsing

Converts CelesTrak JSON/TLE data into sgp4 Satrec objects
for orbital propagation.
"""

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday
from typing import List, Tuple, Optional, Dict


def parse_tle_to_satrec(tle_line1: str, tle_line2: str) -> Satrec:
    """
    Parse a TLE two-line element set into an sgp4 Satrec object.

    Args:
        tle_line1: TLE line 1 string
        tle_line2: TLE line 2 string

    Returns:
        sgp4 Satrec object ready for propagation
    """
    return Satrec.twoline2rv(tle_line1, tle_line2)


def parse_catalog_to_satrecs(
    df: pd.DataFrame
) -> Tuple[List[Satrec], List[int]]:
    """
    Parse all TLEs in a catalog DataFrame to Satrec objects.

    Args:
        df: DataFrame with 'TLE_LINE1' and 'TLE_LINE2' columns

    Returns:
        (list_of_satrecs, list_of_valid_indices)
    """
    satrecs = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            line1 = str(row['TLE_LINE1']).strip()
            line2 = str(row['TLE_LINE2']).strip()

            if not line1.startswith('1') or not line2.startswith('2'):
                continue

            sat = Satrec.twoline2rv(line1, line2)
            satrecs.append(sat)
            valid_indices.append(idx)

        except Exception as e:
            continue

    print(f"[Orbit Parser] Parsed {len(satrecs)}/{len(df)} TLEs successfully")
    return satrecs, valid_indices


def create_satrec_array(satrecs: List[Satrec]) -> SatrecArray:
    """
    Create a SatrecArray for vectorized batch propagation.

    Args:
        satrecs: List of Satrec objects

    Returns:
        SatrecArray for use with batch sgp4 propagation
    """
    return SatrecArray(satrecs)


def get_epoch_jd(sat: Satrec) -> Tuple[float, float]:
    """
    Extract the epoch Julian date from a Satrec object.

    Args:
        sat: sgp4 Satrec object

    Returns:
        (jd, fr) Julian date integer and fractional parts
    """
    return sat.jdsatepoch, sat.jdsatepochF


def extract_orbital_elements(sat: Satrec) -> Dict[str, float]:
    """
    Extract classical orbital elements from a Satrec object.

    Args:
        sat: sgp4 Satrec object

    Returns:
        Dictionary of orbital elements
    """
    from config import MU_EARTH

    # Mean motion in rad/s
    n_rad_s = sat.no_kozai / 60.0  # sgp4 stores in rad/min

    # Semi-major axis
    a_km = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)

    return {
        'a_km': a_km,
        'ecc': sat.ecco,
        'inc_rad': sat.inclo,
        'inc_deg': np.degrees(sat.inclo),
        'raan_rad': sat.nodeo,
        'raan_deg': np.degrees(sat.nodeo),
        'argp_rad': sat.argpo,
        'argp_deg': np.degrees(sat.argpo),
        'mean_anomaly_rad': sat.mo,
        'mean_anomaly_deg': np.degrees(sat.mo),
        'bstar': sat.bstar,
        'epoch_jd': sat.jdsatepoch + sat.jdsatepochF,
        'norad_id': sat.satnum,
    }
