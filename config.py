"""
config.py — STARDUST Project Constants & Configuration

All physical constants, screening thresholds, API endpoints,
and ML parameters centralized in one location.
"""

# ============================================================
# Earth Physical Constants
# ============================================================
MU_EARTH = 398600.4418          # Gravitational parameter (km³/s²)
R_EARTH = 6378.137              # Equatorial radius (km)
J2 = 1.08263e-3                 # J2 oblateness coefficient
OMEGA_EARTH = 7.2921159e-5      # Earth rotation rate (rad/s)

# ============================================================
# Screening Thresholds
# ============================================================
MOID_THRESHOLD_KM = 10.0        # Max MOID for pair to survive coarse filter
PROPAGATION_WINDOW_DAYS = 7.0   # SGP4 propagation window
PROPAGATION_STEP_SEC = 60.0     # Time step for coarse TCA search (seconds)

# ============================================================
# Collision Probability Thresholds
# ============================================================
PC_RED_THRESHOLD = 1e-4          # CAM required
PC_YELLOW_THRESHOLD = 1e-5      # Enhanced monitoring
PC_GREEN_THRESHOLD = 1e-7       # Routine tracking

# ============================================================
# Hard-Body Radius Defaults
# ============================================================
HBR_DEFAULT_KM = 0.010          # 10 meters (typical LEO satellite)
HBR_ISS_KM = 0.050              # 50 meters (ISS)

# ============================================================
# ML Model Parameters
# ============================================================
ML_TARGET_RECALL = 0.999         # Neyman-Pearson recall constraint
ML_C_FN = 50.0                   # False negative cost multiplier
ML_C_FP = 1.0                    # False positive cost multiplier
ML_DEFAULT_THRESHOLD = 0.08      # Default decision threshold (tuned)

# Feature columns (must match feature_engineer.py output order)
FEATURE_COLUMNS = [
    "miss_distance_m",
    "rel_velocity_kms",
    "encounter_angle_deg",
    "closing_speed_kms",
    "tangential_velocity_kms",
    "delta_r_radial_km",
    "delta_r_intrack_km",
    "delta_r_crosstrack_km",
    "delta_v_radial_kms",
    "delta_v_intrack_kms",
    "delta_v_crosstrack_kms",
    "mahalanobis_distance",
    "cov_eigenvalue_1",
    "cov_eigenvalue_2",
    "cov_eigenvalue_3",
    "cov_volume",
    "delta_sma_km",
    "delta_eccentricity",
    "delta_inclination_deg",
    "delta_raan_deg",
    "altitude_overlap_km",
    "moid_km",
    "pc_foster_approx",
    "pc_upper_bound",
    "miss_to_sigma_ratio",
    "hbr_to_miss_ratio",
    "energy_ratio",
]

# ============================================================
# CelesTrak API Configuration
# ============================================================
CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_CACHE_HOURS = 2        # Don't poll more often than this

# Satellite groups for conjunction screening
CELESTRAK_GROUPS = {
    "active": "Active satellites (ISRO + global)",
    "debris": "Cataloged space debris",
    "starlink": "SpaceX Starlink constellation",
    "stations": "Space stations (ISS, Tiangong)",
}

# ============================================================
# ISRO Assets for Demo
# ============================================================
ISRO_SATELLITES = [
    "CARTOSAT-3",
    "OCEANSAT-3",
    "EOS-06",
    "RISAT-2BR1",
    "RESOURCESAT-2A",
    "ASTROSAT",
    "EMISAT",
    "INSAT-3DR",
    "GSAT-30",
    "IRNSS-1A",
]

# ============================================================
# File Paths
# ============================================================
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_TRAINING_DIR = "data/training"
MODEL_DIR = "models"
