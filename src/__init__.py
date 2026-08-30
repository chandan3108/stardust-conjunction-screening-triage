"""
STARDUST — ML-Accelerated Conjunction Screening Triage Engine

Modules:
    tle_fetcher       — CelesTrak API client
    orbit_parser      — TLE → Satrec object parsing
    sgp4_propagator   — SGP4 propagation + TCA finder
    moid_calculator   — Minimum Orbit Intersection Distance
    chan_formula       — 2D collision probability (Chan/Foster)
    feature_engineer  — Orbital mechanics → ML features
    data_generator    — Synthetic training data pipeline
    ml_model          — XGBoost/LightGBM training + inference
    utils             — Coordinate transforms, constants
"""
