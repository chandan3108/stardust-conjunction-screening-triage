"""
tests/test_propagator.py — SGP4 Propagator Tests
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from sgp4.api import Satrec, jday
from src.sgp4_propagator import propagate_single, propagate_pair


# ISS TLE (representative)
ISS_LINE1 = '1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9006'
ISS_LINE2 = '2 25544  51.6400 208.9163 0006703  35.7025 324.4332 15.49560532431103'


class TestPropagateSingle:
    """Tests for single satellite propagation."""

    def test_iss_propagation(self):
        """ISS should propagate to a valid position."""
        sat = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
        jd, fr = jday(2024, 1, 2, 12, 0, 0)

        result = propagate_single(sat, jd, fr)

        assert result['error_code'] == 0
        assert result['position_km'] is not None
        assert result['velocity_kms'] is not None

        # ISS should be in LEO (~400 km altitude)
        r = np.linalg.norm(result['position_km'])
        assert 6500 < r < 7000  # Earth radius + ~150-600 km

    def test_iss_velocity(self):
        """ISS velocity should be ~7.7 km/s."""
        sat = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
        jd, fr = jday(2024, 1, 2, 12, 0, 0)

        result = propagate_single(sat, jd, fr)
        v = np.linalg.norm(result['velocity_kms'])

        assert 7.0 < v < 8.5  # Typical LEO velocity


class TestPropagatePair:
    """Tests for pair propagation and TCA finding."""

    def test_same_satellite_zero_distance(self):
        """Propagating same satellite should give ~0 miss distance."""
        sat = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
        jd, fr = jday(2024, 1, 1, 0, 0, 0)

        result = propagate_pair(
            sat, sat, jd, fr,
            window_days=1.0,
            step_seconds=300.0
        )

        assert result is not None
        assert result['miss_distance_km'] < 1e-6

    def test_result_fields(self):
        """Result should contain all expected fields."""
        sat = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
        jd, fr = jday(2024, 1, 1, 0, 0, 0)

        result = propagate_pair(
            sat, sat, jd, fr,
            window_days=0.5,
            step_seconds=600.0
        )

        expected = [
            'tca_jd', 'tca_fr', 'miss_distance_km',
            'rel_velocity_km_s', 'pos1_tca_km', 'pos2_tca_km',
        ]
        for key in expected:
            assert key in result, f"Missing key: {key}"
