"""Communications availability for the multi-satellite detectors: holds, ages and the relay."""

import numpy as np

from expand3.detect.comms import hold, relay_source


def test_hold_returns_the_latest_available_tick_and_its_age() -> None:
    avail = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    idx, age = hold(avail, dt_s=30.0)
    assert idx.tolist() == [-1, 1, 1, 1, 4, 4]
    assert age[1:].tolist() == [0.0, 30.0, 60.0, 0.0, 30.0]
    assert np.isinf(age[0])


def test_hold_of_an_always_available_signal_is_the_identity() -> None:
    idx, age = hold(np.ones(5, dtype=bool), dt_s=30.0)
    assert idx.tolist() == [0, 1, 2, 3, 4]
    assert np.all(age == 0.0)


def test_hold_of_a_signal_never_available_has_no_source() -> None:
    idx, age = hold(np.zeros(4, dtype=bool), dt_s=30.0)
    assert idx.tolist() == [-1] * 4
    assert np.all(np.isinf(age))


def test_relay_delivers_only_what_the_source_uploaded_before_the_destination_pass() -> None:
    # source passes at ticks 2-3 and 10-11; destination passes at ticks 6-7 and 14-15
    src = np.zeros(18, dtype=bool)
    src[[2, 3, 10, 11]] = True
    dst = np.zeros(18, dtype=bool)
    dst[[6, 7, 14, 15]] = True
    idx = relay_source(src, dst)
    # nothing is known before the destination's first pass
    assert idx[:6].tolist() == [-1] * 6
    # at its first pass the destination gets the source's data as of tick 3, then holds it
    assert idx[6:14].tolist() == [3] * 8
    # at its second pass it gets the source's data as of tick 11
    assert idx[14:].tolist() == [11] * 4


def test_relay_never_returns_data_from_the_future() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        src = rng.random(200) < 0.2
        dst = rng.random(200) < 0.2
        idx = relay_source(src, dst)
        ticks = np.arange(200)
        known = idx >= 0
        assert np.all(idx[known] <= ticks[known])
        assert np.all(src[idx[known]])  # always a tick at which the source was in a pass


def test_relay_is_empty_when_the_destination_never_has_a_pass() -> None:
    src = np.ones(6, dtype=bool)
    assert relay_source(src, np.zeros(6, dtype=bool)).tolist() == [-1] * 6


def test_relay_needs_a_source_pass_before_the_destination_pass() -> None:
    src = np.zeros(8, dtype=bool)
    src[6] = True
    dst = np.zeros(8, dtype=bool)
    dst[2] = True  # the destination passes before the source has uploaded anything
    assert relay_source(src, dst).tolist() == [-1] * 8
