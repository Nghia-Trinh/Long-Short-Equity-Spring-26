"""
risk_matrix.py — Date-Indexed Risk Matrix Builder

Wraps ewma_covariance.py into a class that computes the full covariance
sequence once and exposes a simple date-indexed interface.

Usage:
    builder = RiskMatrixBuilder(lambda_ewma=0.94)
    builder.fit(tickers)           # compute all Sigma_t once
    sigma = builder.get(date)      # returns N×N np.ndarray for a specific date

Inputs:
    Risk/ewma_covariance.py   →  compute_ewma_covariances()
    utils/data_loader.py      →  get_returns_pivot()

Outputs:
    RiskMatrixBuilder class with:
        .fit(tickers)          → self
        .get(date)             → np.ndarray (N, N)
        .tickers               → list[str]
        .reliable_from         → pd.Timestamp (first date after warmup)
"""

import pandas as pd
from typing import Optional, List

from Risk.ewma_covariance import compute_ewma_covariances


class RiskMatrixBuilder:
    """Builds and serves date-indexed EWMA covariance matrices.

    Usage:
        builder = RiskMatrixBuilder(lambda_ewma=0.94)
        builder.fit(returns=returns_df, tickers=[...])
        sigma = builder.get(date)
    """

    def __init__(self, lambda_ewma: float = 0.94, min_periods: int = 60):
        self.lambda_ewma = float(lambda_ewma)
        self.min_periods = int(min_periods)

        self._cov_dict: Optional[dict] = None
        self._tickers: Optional[List[str]] = None
        self._reliable_from: Optional[pd.Timestamp] = None

    def fit(self, tickers: Optional[List[str]] = None, returns: Optional[pd.DataFrame] = None) -> "RiskMatrixBuilder":
        """Compute the full EWMA covariance sequence.

        Parameters
        - tickers: optional list of tickers to restrict the returns dataframe
        - returns: optional pd.DataFrame (T x N) of returns. If None, attempts
                   to load via `utils.data_loader.get_returns_pivot()`.

        Returns
        - self
        """
        if returns is None:
            try:
                from utils.data_loader import get_returns_pivot

                returns = get_returns_pivot()
            except Exception as exc:
                raise RuntimeError(
                    "No `returns` provided and failed to load via utils.data_loader.get_returns_pivot()"
                ) from exc

        if not isinstance(returns, pd.DataFrame):
            raise TypeError("`returns` must be a pandas DataFrame")

        if tickers is not None:
            returns = returns.loc[:, list(tickers)]

        cov_dict, reliable_from, tickers_out = compute_ewma_covariances(
            returns, lam=self.lambda_ewma, min_periods=self.min_periods
        )

        self._cov_dict = cov_dict
        self._reliable_from = reliable_from
        self._tickers = tickers_out

        return self

    def get(self, date) -> pd.DataFrame:
        """Return the N x N covariance matrix for `date`.

        Exact-match on timestamps is used. `date` may be any value
        accepted by `pd.Timestamp`.
        """
        if self._cov_dict is None:
            raise RuntimeError("RiskMatrixBuilder not fitted. Call `.fit()` first.")

        ts = pd.Timestamp(date)
        if ts not in self._cov_dict:
            raise KeyError(f"No covariance matrix available for date {ts}")

        # return a copy to avoid external mutation
        return self._cov_dict[ts].copy()

    @property
    def tickers(self) -> List[str]:
        if self._tickers is None:
            return []
        return list(self._tickers)

    @property
    def reliable_from(self) -> Optional[pd.Timestamp]:
        return self._reliable_from


    