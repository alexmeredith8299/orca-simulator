"""Imager dark current (M6)."""

from __future__ import annotations

from expand3.params import ParamView


def dark_current_e_s_px(view: ParamView, payload_K: float) -> float:
    raise NotImplementedError
