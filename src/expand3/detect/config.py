"""Detector tuning values. These belong to the detection package, not to the simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectConfig:
    """Windows, floors and the false-alarm target shared by the detectors.

    Thresholds are tuned from fault-free data (unfaulted vehicles and the pre-arm part of the
    faulted vehicle) so that each fault id raises at most `target_far_per_vehicle_day` alarm
    episodes per vehicle-day, and never falls below `h_floor`. The floor matters where the
    simulator is noiseless (drops, upsets, wheel fits), so tuned thresholds would otherwise be
    arbitrarily small.
    """

    target_far_per_vehicle_day: float = 0.05
    h_floor: float = 8.0
    # F2: string power over the mean of the others, sunlit only
    f2_window: int = 10
    f2_sunlit_fraction: float = 0.5
    f2_sunlit_quantile: float = 0.99
    f2_sigma_floor: float = 1e-3
    # F8: delivered flags per link
    f8_window: int = 20
    f8_p0_floor: float = 1e-3
    # F9: upsets per tick inside the SAA
    f9_window: int = 20
    f9_lambda_floor_per_s: float = 1e-6
    # F10: log2 of measured over modelled dark current
    f10_sigma_floor: float = 0.05
    # F3/F4: least-squares wheel fit, causal within a slew
    wheel_min_rows: int = 4
    wheel_sigma_floor: float = 1e-3
    wheel_min_conditioning: float = 1e-9

    # context features and learned residual models
    ctx_taus_s: tuple[float, ...] = (300.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0, 36000.0)
    ctx_gate_taus_s: tuple[float, ...] = (900.0, 3600.0, 14400.0)
    ctx_event_cap_s: float = 6000.0
    learn_stride: int = 4
    learn_max_rows: int = 150_000
    learn_alpha: float = 1.0
    learn_window: int = 120
    learn_poly_degree: int = 3
    hgb_max_iter: int = 150
    hgb_learning_rate: float = 0.1
    hgb_max_leaf_nodes: int = 31
    learn_sigma_floor: float = 1e-3
    random_seed: int = 20270
    # unsupervised baselines
    unsup_window_ticks: int = 20
    unsup_pca_variance: float = 0.95
    unsup_trees: int = 100
    # isolation classifier
    clf_step_s: float = 900.0
    clf_lookback_s: float = 21600.0
    clf_max_iter: int = 150
    clf_max_leaf_nodes: int = 15
    clf_folds: int = 5
    clf_floor: float = 0.5
    clf_inputs: tuple[str, ...] = ("f2", "f8", "f9", "f10", "wheels", "lrn_thermal_ridge",
                                   "lrn_eps_ridge", "peer_thermal", "peer_eps", "peer_saa")
    # Monte Carlo dropout forecaster (extra detect-dl)
    mcd_stride: int = 10
    mcd_lags: int = 24
    mcd_hidden: int = 32
    mcd_layers: int = 2
    mcd_dropout: float = 0.2
    mcd_epochs: int = 20
    mcd_batch: int = 512
    mcd_lr: float = 2e-3
    mcd_weight_decay: float = 1e-5
    mcd_samples: int = 20
    mcd_window: int = 12
    mcd_val_mod: int = 5
    mcd_max_rows: int = 200_000
    mcd_threads: int = 8
    mcd_sigma_floor: float = 1e-3
    mcd_device: str = "auto"
    mcdm_age_cap_s: float = 10800.0
    nn_wheel_window: int = 20
    nn_burn_window: int = 36
    nn_gnss_window: int = 36
    nn_txfade_window: int = 60
    nn_mag_window: int = 36
    nn_gyro_window: int = 36
    # raw-signal classifier: trailing windows over the measured signals
    raw_windows_s: tuple[float, ...] = (3600.0, 21600.0)
    raw_count_windows_s: tuple[float, ...] = (21600.0, 86400.0, 1e12)
    raw_min_samples: int = 4
    raw_ratio_floor: float = 1e-9
    # peer differencing
    peer_reference: str = "SAT-1"
    peer_targets: tuple[str, ...] = ("SAT-2", "SAT-3")
    peer_lag_ticks: int = 2400
    peer_max_lag_ticks: int = 300
    peer_saa_window: int = 10

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
