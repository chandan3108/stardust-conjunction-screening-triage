"""
tests/test_chan_formula.py — Chan/Foster Collision Probability Tests
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.chan_formula import (
    compute_bplane_projection, chan_collision_probability,
    approximate_covariance, _foster_pc
)


class TestBPlaneProjection:
    """Tests for B-plane projection."""

    def test_basic_projection(self):
        """B-plane projection should produce valid 2D coordinates."""
        delta_r = np.array([0.05, 0.1, 0.02])  # km
        delta_v = np.array([5.0, 3.0, 1.0])    # km/s
        cov = np.diag([0.01, 0.5, 0.04])       # km²

        bp = compute_bplane_projection(delta_r, delta_v, cov)

        assert 'xi' in bp
        assert 'zeta' in bp
        assert 'cov_2d' in bp
        assert bp['cov_2d'].shape == (2, 2)
        assert bp['sigma_xi'] > 0
        assert bp['sigma_zeta'] > 0

    def test_miss_bplane_magnitude(self):
        """B-plane miss distance should be <= 3D miss distance."""
        delta_r = np.array([1.0, 2.0, 0.5])
        delta_v = np.array([7.0, 0.0, 0.0])
        cov = np.eye(3) * 0.1

        bp = compute_bplane_projection(delta_r, delta_v, cov)
        miss_3d = np.linalg.norm(delta_r)

        # B-plane miss is the projection — should be <= 3D distance
        assert bp['miss_bplane'] <= miss_3d + 1e-10


class TestChanCollisionProbability:
    """Tests for the Chan/Foster Pc calculation."""

    def test_distant_encounter_low_pc(self):
        """A distant encounter should have very low Pc."""
        delta_r = np.array([100.0, 200.0, 50.0])  # 100+ km miss
        delta_v = np.array([7.0, 3.0, 1.0])
        cov1 = np.diag([0.01, 0.5, 0.04])
        cov2 = np.diag([0.01, 0.5, 0.04])

        result = chan_collision_probability(
            delta_r, delta_v, cov1, cov2,
            hbr=0.01, method='foster'
        )

        assert result['pc'] < 1e-10
        assert result['threat_level'] == 'NOMINAL'

    def test_close_encounter_higher_pc(self):
        """A very close encounter should have higher Pc."""
        delta_r = np.array([0.005, 0.003, 0.001])  # ~6m miss
        delta_v = np.array([10.0, 0.0, 0.0])
        cov1 = np.diag([0.001, 0.01, 0.002])
        cov2 = np.diag([0.001, 0.01, 0.002])

        result = chan_collision_probability(
            delta_r, delta_v, cov1, cov2,
            hbr=0.01, method='foster'
        )

        assert result['pc'] > 1e-6  # Should be non-trivial

    def test_pc_between_zero_and_one(self):
        """Pc should always be in [0, 1]."""
        delta_r = np.array([0.02, 0.05, 0.01])
        delta_v = np.array([8.0, 2.0, 0.5])
        cov1 = np.diag([0.01, 0.5, 0.04])
        cov2 = np.diag([0.01, 0.5, 0.04])

        result = chan_collision_probability(
            delta_r, delta_v, cov1, cov2,
            method='foster'
        )

        assert 0.0 <= result['pc'] <= 1.0

    def test_output_fields(self):
        """Result should contain all expected fields."""
        delta_r = np.array([1.0, 0.5, 0.2])
        delta_v = np.array([7.0, 0.0, 0.0])
        cov1 = np.eye(3) * 0.1
        cov2 = np.eye(3) * 0.1

        result = chan_collision_probability(
            delta_r, delta_v, cov1, cov2,
            method='foster'
        )

        expected_keys = [
            'pc', 'pc_foster_approx', 'pc_upper_bound',
            'miss_distance_km', 'miss_distance_m', 'hbr_km',
            'threat_level', 'is_critical', 'is_warning',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestApproximateCovariance:
    """Tests for TLE-derived covariance approximation."""

    def test_returns_3x3_matrix(self):
        """Should return a 3x3 matrix."""
        cov = approximate_covariance(bstar=1e-4, altitude_km=500)
        assert cov.shape == (3, 3)

    def test_positive_semidefinite(self):
        """Covariance matrix should be positive semidefinite."""
        cov = approximate_covariance(bstar=5e-4, altitude_km=400)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(e >= 0 for e in eigenvalues)

    def test_higher_drag_more_uncertainty(self):
        """Higher BSTAR should give larger covariance."""
        cov_low = approximate_covariance(bstar=1e-5, altitude_km=500)
        cov_high = approximate_covariance(bstar=1e-3, altitude_km=500)
        assert np.trace(cov_high) >= np.trace(cov_low)
