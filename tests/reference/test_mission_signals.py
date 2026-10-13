"""M6: signals, SAA and crosslink over the 72 h mission thread."""

import numpy as np
import pytest

from expand3.mission.thread import MissionRun, run_mission
from expand3.params import Params
from expand3.units import S_PER_HOUR

pytestmark = pytest.mark.slow


def test_signals_present(params: Params, mission72: MissionRun) -> None:
    print()
    for k, sat in enumerate(mission72.sats):
        s = mission72.signals[sat]
        saa = mission72.saa[sat]
        assert len(s.gnss_r.t_s) == (len(mission72.t_s) + 1) // 2
        np.testing.assert_array_equal(s.gnss_r.truth, mission72.r_km[::2, k])
        assert s.node_temperature.meas.shape == mission72.thermal[sat].T_K.shape
        # the EPS edge-computer load is 0 exactly while the edge computer is off
        edge = mission72.eps[sat].loads_W["edge_computer"]
        assert np.all(edge[saa.edge_off & ~mission72.eps[sat].shed] == 0)
        assert np.all(edge[~saa.edge_off & ~mission72.eps[sat].shed] > 0)
        np.testing.assert_array_equal(s.gates["saa"], saa.in_saa)
        avail = s.star_tracker.available.mean()
        print(f"{sat}: SAA {saa.in_saa.mean():.3f} of time, upsets {saa.upsets.sum()}, "
              f"resets {saa.resets.sum()}, star tracker available {avail:.3f}")
    log = mission72.crosslink
    assert log is not None
    for j, (a, b) in enumerate(log.pairs):
        rate = log.delivered[:, j].sum() / max(1, log.sent[:, j].sum())
        print(f"crosslink {a}->{b}: sent {log.sent[:, j].mean():.3f} of ticks, "
              f"delivered {rate:.3f} of sent")
        assert rate == 1.0  # nominal p = 0


def test_imager_records(params: Params, mission72: MissionRun) -> None:
    for sat, rec in mission72.takes.items():
        th = mission72.thermal[sat]
        n = int(np.flatnonzero(mission72.t_s == rec.t_take_s)[0])
        assert rec.payload_K == th.node("payload")[n]
        expected = 10.0 * 2 ** ((rec.payload_K - 293.15) / 6.5)
        assert rec.dark_current_e_s_px == pytest.approx(expected)


def test_tank_gauge_only_while_burning(mission72: MissionRun) -> None:
    for k, sat in enumerate(mission72.sats):
        g = mission72.signals[sat].tank_gauge
        np.testing.assert_array_equal(g.t_s, mission72.t_s[mission72.gate_burn[:, k]])


def test_signals_deterministic(params: Params) -> None:
    a = run_mission(params, 31, duration_s=6 * S_PER_HOUR)
    b = run_mission(params, 31, duration_s=6 * S_PER_HOUR)
    for sat in a.sats:
        sa, sb = a.signals[sat], b.signals[sat]
        np.testing.assert_array_equal(sa.gnss_r.meas, sb.gnss_r.meas)
        np.testing.assert_array_equal(sa.star_tracker.q_meas, sb.star_tracker.q_meas)
        np.testing.assert_array_equal(sa.soc_estimate.meas, sb.soc_estimate.meas)
        np.testing.assert_array_equal(a.saa[sat].upsets, b.saa[sat].upsets)
    assert a.crosslink is not None
    assert b.crosslink is not None
    np.testing.assert_array_equal(a.crosslink.r_meas_km, b.crosslink.r_meas_km)
