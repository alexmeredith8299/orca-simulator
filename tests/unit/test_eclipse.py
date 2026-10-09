"""M1: dual-cone shadow fraction on synthetic geometry."""

import numpy as np
import pytest

from expand3.env.eclipse import shadow_fraction

R_E = 6378.137
R_SUN = 695700.0
AU = 1.495978707e8
SUN = np.array([1.0, 0.0, 0.0])


def s_at(r: np.ndarray) -> np.ndarray:
    return shadow_fraction(r, SUN, AU, R_E, R_SUN)


def test_sunlit_and_umbra() -> None:
    r = np.array([[7000.0, 0, 0], [0, 7000.0, 0], [-7000.0, 0, 0], [-7000.0, 100.0, 0]])
    np.testing.assert_array_equal(s_at(r), [1, 1, 0, 0])


def test_penumbra_is_monotone_and_bounded() -> None:
    # Walk across the shadow edge behind Earth at |r| = 6903 km.
    phi = np.linspace(np.deg2rad(60), np.deg2rad(80), 20001)
    a = 6903.137
    r = np.stack([-a * np.cos(phi), a * np.sin(phi), np.zeros_like(phi)], axis=-1)
    s = s_at(r)
    assert np.all((s >= 0) & (s <= 1))
    assert s[0] == 0
    assert s[-1] == 1
    assert np.all(np.diff(s) >= -1e-12)
    partial = (s > 0) & (s < 1)
    assert partial.any()
    # Penumbra angular width along the track is about 2 x the Sun's angular radius / sin(edge angle)
    width = phi[partial][-1] - phi[partial][0]
    assert np.rad2deg(width) == pytest.approx(0.68, rel=0.25)


def test_broadcasts_over_time_and_vehicles() -> None:
    r = np.zeros((5, 6, 3))
    r[..., 0] = 7000.0
    sun = np.tile(SUN, (5, 1))[:, None, :]
    s = shadow_fraction(r, sun, np.full((5, 1), AU), R_E, R_SUN)
    assert s.shape == (5, 6)
    assert np.all(s == 1)
