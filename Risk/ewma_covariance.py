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

# TODO: implement ewma_cov_update(sigma_prev, r_t, lam) -> np.ndarray
# TODO: implement compute_ewma_covariances(returns, lam=0.94, min_periods=60)
#         -> tuple[dict[Timestamp, ndarray], Timestamp, list[str]]
