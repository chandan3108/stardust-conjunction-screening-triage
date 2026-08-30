"""
tle_fetcher.py — CelesTrak TLE Data Ingestion Module

Fetches Two-Line Element sets from CelesTrak's public API
and parses them into structured data for the STARDUST pipeline.
"""

import requests
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from config import (
    CELESTRAK_BASE_URL, CELESTRAK_CACHE_HOURS,
    MU_EARTH, R_EARTH
)


# Cache directory for downloaded TLEs
CACHE_DIR = Path("data/raw")


def fetch_tle_group(
    group: str,
    format: str = "json",
    use_cache: bool = True
) -> List[Dict]:
    """
    Fetch TLE data for a satellite group from CelesTrak.

    Args:
        group: One of 'active', 'debris', 'starlink', etc.
        format: Response format ('json', 'tle', 'csv', etc.)
        use_cache: If True, use cached data if < 2 hours old

    Returns:
        List of satellite data dictionaries (JSON format)
    """
    cache_file = CACHE_DIR / f"{group}_{format}.json"

    # Check cache freshness
    if use_cache and cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=CELESTRAK_CACHE_HOURS):
            print(f"[TLE Fetcher] Using cached data for '{group}' "
                  f"(age: {datetime.now() - mod_time})")
            with open(cache_file, 'r') as f:
                return json.load(f)

    # Fetch from CelesTrak
    url = f"{CELESTRAK_BASE_URL}?GROUP={group}&FORMAT={format}"
    print(f"[TLE Fetcher] Fetching {url} ...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Cache the result
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(data, f)

        print(f"[TLE Fetcher] Downloaded {len(data)} objects "
              f"from group '{group}'")
        return data

    except requests.RequestException as e:
        print(f"[TLE Fetcher] Error fetching {group}: {e}")
        # Fall back to cache if available
        if cache_file.exists():
            print(f"[TLE Fetcher] Falling back to cached data")
            with open(cache_file, 'r') as f:
                return json.load(f)
        raise


def fetch_all_groups(
    groups: Optional[List[str]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch multiple satellite groups and combine into a single DataFrame.

    Args:
        groups: List of group names. Default: ['active', 'debris']
        use_cache: Whether to use cached data

    Returns:
        DataFrame with all orbital elements and metadata
    """
    if groups is None:
        groups = ["active", "debris"]

    all_data = []
    for group in groups:
        try:
            group_data = fetch_tle_group(group, format="json",
                                         use_cache=use_cache)
            for obj in group_data:
                obj["SOURCE_GROUP"] = group
            all_data.extend(group_data)
        except Exception as e:
            print(f"[TLE Fetcher] Skipping group '{group}': {e}")

    if not all_data:
        raise RuntimeError("No TLE data could be fetched from any group")

    df = pd.DataFrame(all_data)
    print(f"[TLE Fetcher] Total catalog: {len(df)} objects")
    return df


def parse_json_to_orbital_elements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CelesTrak JSON fields to standardized orbital elements.
    Adds computed fields: semi-major axis, perigee/apogee altitudes.

    Args:
        df: DataFrame from fetch_all_groups()

    Returns:
        DataFrame with additional computed orbital element columns
    """
    # Ensure numeric columns
    numeric_cols = [
        'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION',
        'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
        'BSTAR', 'MEAN_MOTION_DOT'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert mean motion (rev/day) → semi-major axis (km)
    # n = sqrt(mu / a³)  →  a = (mu / (2πn/86400)²)^(1/3)
    n_rad_s = df["MEAN_MOTION"] * 2.0 * np.pi / 86400.0
    df["SEMI_MAJOR_AXIS_KM"] = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)

    # Compute perigee and apogee altitudes
    df["PERIGEE_ALT_KM"] = (
        df["SEMI_MAJOR_AXIS_KM"] * (1.0 - df["ECCENTRICITY"]) - R_EARTH
    )
    df["APOGEE_ALT_KM"] = (
        df["SEMI_MAJOR_AXIS_KM"] * (1.0 + df["ECCENTRICITY"]) - R_EARTH
    )

    # Orbital period in minutes
    df["PERIOD_MIN"] = (
        2.0 * np.pi * np.sqrt(df["SEMI_MAJOR_AXIS_KM"] ** 3 / MU_EARTH)
        / 60.0
    )

    # Drop rows with invalid data
    valid_mask = (
        df["SEMI_MAJOR_AXIS_KM"].notna() &
        (df["SEMI_MAJOR_AXIS_KM"] > R_EARTH) &
        (df["ECCENTRICITY"] >= 0) &
        (df["ECCENTRICITY"] < 1)
    )
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[TLE Fetcher] Dropped {n_dropped} objects with "
              f"invalid orbital elements")
    df = df[valid_mask].reset_index(drop=True)

    return df


def fetch_single_satellite(norad_id: int) -> Dict:
    """
    Fetch TLE data for a single satellite by NORAD catalog ID.

    Args:
        norad_id: NORAD catalog number (e.g. 25544 for ISS)

    Returns:
        Dictionary with satellite data
    """
    url = f"{CELESTRAK_BASE_URL}?CATNR={norad_id}&FORMAT=json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data:
        return data[0]
    raise ValueError(f"No data found for NORAD ID {norad_id}")
