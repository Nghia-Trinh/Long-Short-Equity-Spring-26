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

# TODO: implement class RiskMatrixBuilder:
#           __init__(self, lambda_ewma=0.94, min_periods=60)
#           fit(self, tickers=None) -> RiskMatrixBuilder
#           get(self, date: pd.Timestamp) -> np.ndarray
#           @property tickers -> list[str]
#           @property reliable_from -> pd.Timestamp
