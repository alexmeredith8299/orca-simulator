"""Crosslink received signal strength and the tx_fade fault. Not in the spec
(docs/extended-faults-plan.md, Phase B): an omni link budget, free-space loss at the true range,
a Gaussian reading error, and a loss on the sender's transmit power. A vehicle senses what it
receives and never its own transmit power."""

import numpy as np

from expand3.links import crosslink
from expand3.params import Params, ParamView

SATS = ["SAT-1", "SAT-2", "SAT-3"]
N = 4000
LOSS = "sensor_faults.radio.tx_loss_dB"
C_M_S = 299_792_458.0


def _views(params: Params, fade_db: float = 0.0, sender: str = "SAT-2") -> dict[str, ParamView]:
    views = {s: ParamView(params, s) for s in SATS}
    views[sender].overrides[LOSS] = fade_db
    return views


def _run(params: Params, views: dict[str, ParamView], seed: int = 3):
    t = np.arange(N) * 30.0
    r = np.zeros((N, 3, 3))
    r[:] = [7000.0, 0.0, 0.0]
    r[:, 1, 1] = 1000.0
    r[:, 2, 2] = 400.0
    return crosslink(seed, params, views, SATS, t, r, np.zeros_like(r)), r


def _expected_dbm(params: Params, range_km: float) -> float:
    b = params.links.link_budget
    fspl = 20.0 * np.log10(4.0 * np.pi * range_km * 1000.0 * b.frequency_GHz * 1e9 / C_M_S)
    return b.tx_power_dBm + b.tx_gain_dBi + b.rx_gain_dBi - float(fspl)


def test_rssi_follows_the_link_budget_at_the_true_range(params: Params) -> None:
    log, _ = _run(params, _views(params))
    j = log.pairs.index(("SAT-1", "SAT-2"))
    ok = log.delivered[:, j]
    assert ok.any()
    rssi = log.rssi_dBm[ok, j]
    sigma = params.links.link_budget.rssi_sigma_dB
    assert abs(rssi.mean() - _expected_dbm(params, 1000.0)) < 4 * sigma / np.sqrt(ok.sum())
    assert abs(rssi.std() - sigma) < 0.1 * sigma + 4 * sigma / np.sqrt(2 * ok.sum())
    far, near = log.pairs.index(("SAT-1", "SAT-2")), log.pairs.index(("SAT-1", "SAT-3"))
    assert log.rssi_dBm[log.delivered[:, near], near].mean() > log.rssi_dBm[
        log.delivered[:, far], far].mean()  # 400 km is stronger than 1000 km


def test_no_reading_without_a_delivered_message(params: Params) -> None:
    log, _ = _run(params, _views(params))
    assert np.isnan(log.rssi_dBm[~log.delivered]).all()
    assert np.isfinite(log.rssi_dBm[log.delivered]).all()


def test_tx_fade_lowers_only_the_senders_outgoing_readings(params: Params) -> None:
    base, _ = _run(params, _views(params))
    faded, _ = _run(params, _views(params, fade_db=6.0, sender="SAT-2"))
    for j, (a, _b) in enumerate(base.pairs):
        ok = base.delivered[:, j]
        diff = (faded.rssi_dBm[ok, j] - base.rssi_dBm[ok, j])
        np.testing.assert_allclose(diff, -6.0 if a == "SAT-2" else 0.0, atol=1e-9)


def test_the_fade_leaves_every_other_crosslink_quantity_alone(params: Params) -> None:
    base, _ = _run(params, _views(params))
    faded, _ = _run(params, _views(params, fade_db=6.0))
    np.testing.assert_array_equal(base.delivered, faded.delivered)
    np.testing.assert_array_equal(base.r_meas_km, faded.r_meas_km)
    np.testing.assert_array_equal(base.v_meas_km_s, faded.v_meas_km_s)
