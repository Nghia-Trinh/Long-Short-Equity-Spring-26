"""
optimizer.py — Mean-Variance Portfolio Optimiser with Turnover Penalty

Solves (per rebalance date):
    max_w  alpha' w  -  lambda_risk * w' Sigma w  -  c * ||w - w_prev||_1

Subject to:
    sum(w)     = 0          (dollar neutral — equal long and short notional)
    sum(|w|)  <= L          (gross leverage cap, e.g. 2.0 = 100L / 100S)
    |w_i|     <= w_max      (single-name concentration limit)
    w_i        = 0  for tickers with no valid alpha signal

Solver: cvxpy with CLARABEL backend.
Fallback: rank-based heuristic (long top quintile / short bottom quintile)
          if cvxpy is not installed.

Inputs:
    alpha    : (N,) np.ndarray — cross-sectionally z-scored SUE
    sigma    : (N, N) np.ndarray — EWMA covariance matrix
    w_prev   : (N,) np.ndarray — previous portfolio weights
    Scalar hyperparameters from config.json

Outputs:
    optimize_portfolio(...)  →  (N,) np.ndarray of optimal weights
"""

# TODO: implement optimize_portfolio(alpha, sigma, w_prev,
#                                    lambda_risk, c, max_leverage,
#                                    max_position) -> np.ndarray
# TODO: implement _solve_cvxpy(...)   (primary solver)
# TODO: implement _solve_heuristic(...)  (fallback when cvxpy unavailable)
