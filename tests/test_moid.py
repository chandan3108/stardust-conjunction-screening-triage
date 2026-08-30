"""
tests/test_moid.py — MOID Calculator Tests
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.moid_calculator import (
    radial_overlap_check, keplerian_to_position, compute_moid
)


class TestRadialOverlap:
    """Tests for the perigee/apogee radial overlap pre-filter."""

    def test_overlapping_orbits(self):
        """Two LEO orbits with overlapping altitude bands should overlap."""
        a1, e1 = 7000.0, 0.01   # perigee 6930, apogee 7070 km
        a2, e2 = 7050.0, 0.01   # perigee 6979.5, apogee 7120.5 km
        assert radial_overlap_check(a1, e1, a2, e2) is True

    def test_non_overlapping_orbits(self):
        """LEO orbit vs GEO orbit should not overlap."""
        a1, e1 = 7000.0, 0.001   # LEO
        a2, e2 = 42164.0, 0.001  # GEO
        assert radial_overlap_check(a1, e1, a2, e2) is False

    def test_eccentric_overlap(self):
        """Eccentric orbit crossing a circular orbit."""
        a1, e1 = 7000.0, 0.001   # Circular LEO
        a2, e2 = 10000.0, 0.35   # Eccentric (perigee ~6500 km)
        assert radial_overlap_check(a1, e1, a2, e2) is True

    def test_identical_orbits(self):
        """Same orbit should always overlap."""
        a, e = 7000.0, 0.01
        assert radial_overlap_check(a, e, a, e) is True


class TestKeplerianToPosition:
    """Tests for Keplerian → Cartesian conversion."""

    def test_circular_equatorial(self):
        """Circular equatorial orbit at nu=0 should be along x-axis."""
        a, e, i = 7000.0, 0.0, 0.0
        raan, omega, nu = 0.0, 0.0, 0.0
        pos = keplerian_to_position(a, e, i, raan, omega, nu)
        assert abs(pos[0] - a) < 1e-6  # x ≈ a
        assert abs(pos[1]) < 1e-6       # y ≈ 0
        assert abs(pos[2]) < 1e-6       # z ≈ 0

    def test_radius_at_perigee(self):
        """Radius at perigee (nu=0) should be a(1-e)."""
        a, e = 8000.0, 0.1
        i, raan, omega, nu = 0.5, 0.3, 0.7, 0.0
        pos = keplerian_to_position(a, e, i, raan, omega, nu)
        r = np.linalg.norm(pos)
        expected_r = a * (1 - e)
        assert abs(r - expected_r) < 1e-4

    def test_radius_at_apogee(self):
        """Radius at apogee (nu=π) should be a(1+e)."""
        a, e = 8000.0, 0.1
        i, raan, omega = 0.5, 0.3, 0.7
        nu = np.pi
        pos = keplerian_to_position(a, e, i, raan, omega, nu)
        r = np.linalg.norm(pos)
        expected_r = a * (1 + e)
        assert abs(r - expected_r) < 1e-4


class TestMOID:
    """Tests for full MOID computation."""

    def test_identical_orbits_zero_moid(self):
        """MOID of identical orbits should be ~0."""
        elements = (7000.0, 0.01, 0.9, 0.5, 1.0)
        moid = compute_moid(elements, elements, n_grid=72)
        assert moid < 1.0  # Should be very close to 0

    def test_distant_orbits_large_moid(self):
        """LEO vs GEO should have large MOID."""
        leo = (7000.0, 0.001, 0.9, 0.0, 0.0)
        geo = (42164.0, 0.001, 0.01, 0.0, 0.0)
        moid = compute_moid(leo, geo, n_grid=72)
        assert moid > 30000  # ~35,000 km apart

    def test_moid_is_positive(self):
        """MOID should always be non-negative."""
        elements1 = (7000.0, 0.05, 0.8, 1.2, 0.5)
        elements2 = (7500.0, 0.03, 0.7, 0.8, 1.5)
        moid = compute_moid(elements1, elements2, n_grid=72)
        assert moid >= 0
