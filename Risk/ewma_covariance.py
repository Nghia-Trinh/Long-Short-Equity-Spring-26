"""
ewma_covariance.py — Exponentially Weighted Moving Average Covariance

Formula (RiskMetrics 1994):
    Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_t * r_t'

where:
    Sigma_t  : N × N covariance matrix at time t
    lambda   : decay factor (lambda_ewma in config.json, default 0.94)
               half-life ≈ 11 trading days at lambda=0.94
    r_t      : N-vector of daily returns at time t (NaN treated as 0)

Inputs:
    utils/data_loader.py  →  get_returns_pivot()  →  pd.DataFrame (T × N)

Outputs:
    - ewma_cov_update()           → single-step update, returns (N, N) np.ndarray
    - compute_ewma_covariances()  → dict{ pd.Timestamp → np.ndarray (N×N) }
                                    + reliable_from date + tickers list
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List


def ewma_cov_update(sigma_prev: Optional[np.ndarray], r_t, lam: float) -> np.ndarray:
    """Single-step EWMA covariance update.

    Parameters
    - sigma_prev: previous covariance (N x N) or None. If None, treated as zero.
    - r_t: 1-D array-like of returns (length N). NaNs are treated as 0.
    - lam: decay factor in [0, 1].

    Returns
    - sigma_t: (N x N) ndarray
    """
    r = np.asarray(r_t, dtype=float)
    # Treat NaNs as 0 (missing returns)
    r = np.nan_to_num(r, nan=0.0)
    outer = np.outer(r, r)

    if sigma_prev is None:
        # Equivalent to starting from zeros: Sigma_t = (1-lam) * r r'
        return (1.0 - lam) * outer

    sigma_prev = np.asarray(sigma_prev, dtype=float)
    if sigma_prev.shape != outer.shape:
        raise ValueError(f"sigma_prev shape {sigma_prev.shape} incompatible with r_t length {r.shape[0]}")

    return lam * sigma_prev + (1.0 - lam) * outer


def compute_ewma_covariances(
    returns: pd.DataFrame, lam: float = 0.94, min_periods: int = 60
) -> Tuple[Dict[pd.Timestamp, np.ndarray], Optional[pd.Timestamp], List[str]]:
    """Compute EWMA covariance matrices for each timestamp in `returns`.

    Parameters
    - returns: T x N DataFrame of returns (index: timestamps, columns: tickers)
    - lam: decay factor
    - min_periods: number of observations required before marking reliable_from

    Returns
    - cov_dict: mapping Timestamp -> (N x N) ndarray
    - reliable_from: pd.Timestamp when at least `min_periods` rows processed, else None
    - tickers: list of column names (tickers)
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    tickers = list(returns.columns)
    cov_dict: Dict[pd.Timestamp, np.ndarray] = {}

    sigma_prev: Optional[np.ndarray] = None
    reliable_from: Optional[pd.Timestamp] = None

    for i, (ts, row) in enumerate(returns.iterrows()):
        r = row.values
        sigma_prev = ewma_cov_update(sigma_prev, r, lam)
        cov_dict[ts] = sigma_prev.copy()
        if reliable_from is None and (i + 1) >= min_periods:
            reliable_from = pd.Timestamp(ts)

    return cov_dict, reliable_from, tickers


