from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, sigma: float = 3.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values

    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad == 0.0:
        return values

    robust_scale = 1.4826 * mad
    lower = median - float(sigma) * robust_scale
    upper = median + float(sigma) * robust_scale
    return values.clip(lower=lower, upper=upper)


def robust_z(series: pd.Series, winsor_sigma: float = 3.0) -> pd.Series:
    clipped = winsorize(series, sigma=winsor_sigma)
    valid = clipped.dropna()
    if valid.empty:
        return clipped.astype(float)

    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad == 0.0:
        return pd.Series(np.nan, index=clipped.index, dtype=float)

    robust_scale = 1.4826 * mad
    return (clipped - median) / robust_scale
