"""The fault posterior: onset prior, calibration of the tempered likelihood ratio, the filter."""

from pathlib import Path

import numpy as np
import pytest
from detect_synth import ARM_S, TAKE_S, make_batch

from expand3.detect.config import DetectConfig
from expand3.detect.harness import run_detectors
from expand3.detect.posterior import (
    Channel,
    Prior,
    combine,
    fit_channel,
    load_priors,
    log_prior_odds,
    onset_cdf,
    posterior_batch,
    reliability,
)

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
LOW, HIGH = 5700.0, 11400.0


def test_priors_come_from_the_params() -> None:
    priors = load_priors(PARAMS_DIR)
    assert set(priors) == {f"F{i}" for i in range(1, 11)}
    assert priors["F1"].vehicle == "SAT-2"
    assert priors["F5"].vehicle == "SAT-4"
    assert priors["F3"].pi == pytest.approx(0.1)
    assert (priors["F1"].lead_low_s, priors["F1"].lead_high_s) == (LOW, HIGH)


def test_extended_priors_are_uniform_over_the_spec_and_extended_faults() -> None:
    from expand3.params import load_params

    p = load_params(PARAMS_DIR)
    priors = load_priors(PARAMS_DIR, "extended")
    n = len(p.faults.faults) + len(p.faults.extended)
    assert set(priors) == set(p.faults.faults) | set(p.faults.extended)
    assert all(v.pi == pytest.approx(1.0 / n) for v in priors.values())
    assert priors["gnss_jump"].vehicle == "SAT-2"


def test_onset_cdf_follows_the_uniform_lead() -> None:
    take = 100000.0
    t = np.array([take - HIGH - 1.0, take - HIGH, take - 0.5 * (LOW + HIGH), take - LOW,
                  take - LOW + 1.0, take + 5000.0])
    f = onset_cdf(t, take, LOW, HIGH)
    assert f == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0, 1.0])


def test_no_take_means_no_fault_is_armed() -> None:
    t = np.linspace(0.0, 1e5, 11)
    assert np.all(onset_cdf(t, None, LOW, HIGH) == 0.0)
    prior = Prior("F2", "SAT-2", 0.1, LOW, HIGH)
    assert np.all(log_prior_odds(t, None, prior, 0.2) == -np.inf)


def test_prior_odds_are_minus_inf_before_onset_and_pi_over_rest_after() -> None:
    prior = Prior("F2", "SAT-2", 0.1, LOW, HIGH)
    take = 100000.0
    early = log_prior_odds(np.array([take - HIGH - 10.0]), take, prior, 0.2)
    late = log_prior_odds(np.array([take]), take, prior, 0.2)
    assert early[0] == -np.inf
    assert late[0] == pytest.approx(np.log(0.1 / (1.0 - 0.2)))


def _simulate(rng: np.random.Generator, n: int, a: float, c: float) -> tuple[np.ndarray, ...]:
    """Scores and labels from a known tempered model: logit P = offset + a S - c."""
    offset = rng.uniform(-3.0, 0.0, n)
    s = rng.gamma(2.0, 3.0, n)
    p = 1.0 / (1.0 + np.exp(-(offset + a * s - c)))
    return s, offset, (rng.uniform(size=n) < p).astype(float)


def test_fit_recovers_the_scale_and_offset() -> None:
    rng = np.random.default_rng(0)
    s, offset, y = _simulate(rng, 40000, a=0.6, c=2.0)
    ch = fit_channel(s, offset, y)
    assert ch.a == pytest.approx(0.6, abs=0.06)
    assert ch.c == pytest.approx(2.0, abs=0.3)


def test_fit_is_calibrated_on_held_out_data() -> None:
    rng = np.random.default_rng(1)
    ch = fit_channel(*_simulate(rng, 30000, a=0.4, c=3.0))
    s, offset, y = _simulate(rng, 30000, a=0.4, c=3.0)
    p = ch.prob(s, offset)
    rel = reliability(p, y, bins=10, p_prior=1.0 / (1.0 + np.exp(-offset)))
    assert rel["ece"] < 0.02
    assert rel["log_loss"] < rel["log_loss_prior"]


def test_fit_with_no_signal_keeps_the_prior() -> None:
    """If the score says nothing the fit must not invent evidence (a goes to 0)."""
    rng = np.random.default_rng(2)
    offset = rng.uniform(-3.0, 0.0, 20000)
    y = (rng.uniform(size=20000) < 1.0 / (1.0 + np.exp(-offset))).astype(float)
    ch = fit_channel(rng.gamma(2.0, 3.0, 20000), offset, y)
    assert ch.a < 0.05


def test_probability_is_monotone_in_the_score() -> None:
    ch = Channel(a=0.5, c=1.0)
    s = np.linspace(0.0, 50.0, 100)
    p = ch.prob(s, np.full(100, -2.0))
    assert np.all(np.diff(p) >= 0.0)
    assert 0.0 < p[0] < p[-1] < 1.0


def test_zero_prior_mass_gives_zero_probability_whatever_the_score() -> None:
    assert Channel(a=1.0, c=0.0).prob(np.array([1e6]), np.array([-np.inf]))[0] == 0.0


def test_combine_normalizes_across_faults_on_a_vehicle() -> None:
    log_odds = {"F1": np.array([0.0, 2.0, -50.0]), "F7": np.array([0.0, 1.0, -np.inf])}
    p, p_none = combine(log_odds)
    total = p["F1"] + p["F7"] + p_none
    assert total == pytest.approx(np.ones(3))
    assert p["F1"][0] == pytest.approx(1.0 / 3.0)
    assert p["F1"][1] > p["F7"][1]
    assert p["F7"][2] == 0.0
    # huge evidence must not overflow
    big, _ = combine({"F1": np.array([1e4]), "F7": np.array([0.0])})
    assert big["F1"][0] == pytest.approx(1.0)


# ------------------------------------------------------------------ batch level, synthetic seeds

TRAIN = ["none"] * 6 + ["F2"] * 4 + ["F8"] * 3 + ["none"] * 3
TEST = ["F2", "F2", "F8", "none", "none", "none"]
CFG = DetectConfig(h_floor=12.0)


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int], list[int]]:
    out = tmp_path_factory.mktemp("post")
    seeds = make_batch(out, [*TRAIN, *TEST])
    return out, seeds[: len(TRAIN)], seeds[len(TRAIN):]


def test_posterior_rises_after_the_fault_and_stays_low_without_one(
        result: tuple[Path, list[int], list[int]]) -> None:
    out, tune, test = result
    keep: dict = {}
    run_detectors(out, ["f2", "f8"], tune, tune + test, CFG, PARAMS_DIR, keep=keep)
    post = posterior_batch(out, ["f2", "f8"], tune, test, keep["traces"], CFG, PARAMS_DIR)
    by_seed = post["posteriors"]
    f2_seed, none_seed = test[0], test[3]
    t = by_seed[f2_seed]["SAT-2"]["t_s"]
    p = by_seed[f2_seed]["SAT-2"]["F2"]
    assert p[t > ARM_S + 4000.0].max() > 0.9
    assert p[t < TAKE_S - HIGH - 1.0].max() == 0.0          # prior mass is zero before onset
    assert by_seed[none_seed]["SAT-2"]["F2"].max() < 0.5
    assert "SAT-1" not in by_seed[f2_seed]                   # no fault is prior-possible there
    assert set(post["channels"]) == {"F2", "F8"}
    m = post["metrics"]["test"]["F2"]
    assert m["log_loss"] < m["log_loss_prior"]


def test_two_detectors_for_one_fault_need_a_selection_rule(
        result: tuple[Path, list[int], list[int]]) -> None:
    out, tune, test = result
    keep: dict = {}
    names = ["f2", "lrn_eps_ridge"]                          # both cover F2
    run_detectors(out, names, tune, tune + test, CFG, PARAMS_DIR, keep=keep)
    with pytest.raises(ValueError, match="two detectors"):
        posterior_batch(out, names, tune, test, keep["traces"], CFG, PARAMS_DIR)
    post = posterior_batch(out, names, tune, test, keep["traces"], CFG, PARAMS_DIR, select=True)
    cand = post["candidates"]["F2"]
    assert set(cand) == {"f2", "lrn_eps_ridge"}
    best = min(cand, key=lambda n: cand[n]["tune_log_loss"])
    assert post["channels"]["F2"]["detector"] == best        # chosen on tune data only
    for m in cand.values():                                  # reported for comparison only
        assert np.isfinite(m["test_log_loss"])
        assert m["test_log_loss"] >= 0.0
