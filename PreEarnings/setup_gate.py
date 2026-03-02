from __future__ import annotations

from typing import NamedTuple

import pandas as pd


class GateResult(NamedTuple):
    tradable: bool
    reason: str


def evaluate(
    z_drift: float,
    z_vol: float,
    drift_threshold: float = -1.0,
    vol_threshold: float = 0.5,
) -> GateResult:
    if pd.isna(z_drift) and pd.isna(z_vol):
        return GateResult(False, "missing_drift_and_vol")
    if pd.isna(z_drift):
        return GateResult(False, "missing_drift")
    if pd.isna(z_vol):
        return GateResult(False, "missing_vol")
    if float(z_drift) > float(drift_threshold):
        return GateResult(False, "drift_gate_failed")
    if float(z_vol) < float(vol_threshold):
        return GateResult(False, "vol_gate_failed")
    return GateResult(True, "passed")
