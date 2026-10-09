"""M1: six vehicles for 72 h with the environment products finish under 60 s."""

import pytest

from expand3.constellation import ConstellationRun


@pytest.mark.slow
def test_six_vehicles_72h_under_60s(run72: ConstellationRun) -> None:
    print(f"\nsix-vehicle 72 h propagation + env products: {run72.wall_s:.1f} s")
    assert run72.wall_s < 60
