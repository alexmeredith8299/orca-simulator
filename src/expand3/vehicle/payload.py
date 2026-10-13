"""Imager dark current (M6).

Spec: Model: sensors, crosslink, SAA, imager (Imager, F10): D(T_p) = D_0 2^((T_p - T_ref)/dT).
Each take records D at the payload node temperature at take start (CLAUDE.md Known gaps).
"""

from __future__ import annotations

from expand3.params import ParamView


def dark_current_e_s_px(view: ParamView, payload_K: float) -> float:
    """Dark current [e-/s/px]; D_0 is read through ParamView (F10)."""
    d0 = view.get("imager_dark_current.D0_e_s_px")
    t_ref = view.get("imager_dark_current.T_ref_K")
    doubling = view.get("imager_dark_current.delta_T_K")
    return float(d0 * 2.0 ** ((payload_K - t_ref) / doubling))
