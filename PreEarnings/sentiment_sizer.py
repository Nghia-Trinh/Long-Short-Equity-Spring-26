from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class SizingResult(NamedTuple):
    multiplier: float
    w_delta: float
    w_cpr: float
    risk_scale: float


def compute_modifier(
    z_nd: float | None,
    z_cpr: float | None,
    z_vol: float,
    a: float,
    b: float,
    k: float,
) -> SizingResult:
    if z_nd is None or pd.isna(z_nd):
        w_delta = 1.0
    else:
        w_delta = float(np.clip(1.0 - float(a) * float(z_nd), 0.5, 1.5))

    if z_cpr is None or pd.isna(z_cpr):
        w_cpr = 1.0
    else:
        w_cpr = 1.0 + float(b) * float(abs(float(z_cpr)) >= 1.5)

    if pd.isna(z_vol):
        risk_scale = 1.0
    else:
        risk_scale = 1.0 / (1.0 + float(k) * max(0.0, float(z_vol)))

    multiplier = float(w_delta * w_cpr * risk_scale)
    return SizingResult(multiplier=multiplier, w_delta=float(w_delta), w_cpr=float(w_cpr), risk_scale=float(risk_scale))
