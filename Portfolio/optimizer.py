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

from __future__ import annotations

import numpy as np

_CVXPY_AVAILABLE = True
try:
    import cvxpy as cp
except ImportError:
    _CVXPY_AVAILABLE = False


def optimize_portfolio(
    alpha: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray | None = None,
    lambda_risk: float = 1.0,
    transaction_cost: float = 0.001,
    max_leverage: float = 2.0,
    max_position: float = 0.05,
) -> np.ndarray:
    """Solve the MVO + turnover-penalty problem.

    Parameters
    ----------
    alpha : (N,) array
        Blended alpha signal (z-scored).
    sigma : (N, N) array
        EWMA covariance matrix.
    w_prev : (N,) array or None
        Previous weights.  If None, defaults to zeros (no turnover cost
        on first rebalance).
    lambda_risk : float
        Risk-aversion parameter.
    transaction_cost : float
        Per-unit turnover penalty (proportional to |w - w_prev|_1).
    max_leverage : float
        Gross exposure cap (sum of absolute weights).
    max_position : float
        Maximum absolute weight per ticker.

    Returns
    -------
    np.ndarray
        Optimal weight vector of shape (N,).
    """
    n = len(alpha)
    if w_prev is None:
        w_prev = np.zeros(n)

    alpha = np.asarray(alpha, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)

    if _CVXPY_AVAILABLE:
        return _solve_cvxpy(alpha, sigma, w_prev, lambda_risk, transaction_cost, max_leverage, max_position)
    else:
        return _solve_heuristic(alpha, max_leverage, max_position)


def _solve_cvxpy(
    alpha: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    lambda_risk: float,
    tc: float,
    max_leverage: float,
    max_position: float,
) -> np.ndarray:
    n = len(alpha)
    w = cp.Variable(n)

    # Regularise sigma for numerical stability
    sigma_reg = sigma + 1e-6 * np.eye(n)

    # Objective: maximise alpha exposure - risk penalty - turnover cost
    ret_term = alpha @ w
    risk_term = lambda_risk * cp.quad_form(w, sigma_reg)
    tc_term = tc * cp.norm1(w - w_prev)

    objective = cp.Maximize(ret_term - risk_term - tc_term)

    constraints = [
        cp.sum(w) == 0,                   # dollar neutral
        cp.norm1(w) <= max_leverage,       # gross leverage cap
        w >= -max_position,                # per-name lower bound
        w <= max_position,                 # per-name upper bound
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, max_iter=5000)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS, max_iters=10000)
        except cp.SolverError:
            return _solve_heuristic(alpha, max_leverage, max_position)

    if w.value is None:
        return _solve_heuristic(alpha, max_leverage, max_position)

    return np.array(w.value, dtype=float)


def _solve_heuristic(
    alpha: np.ndarray,
    max_leverage: float,
    max_position: float,
) -> np.ndarray:
    """Rank-based fallback: long top quintile, short bottom quintile."""
    n = len(alpha)
    if n == 0:
        return np.array([], dtype=float)

    ranks = np.argsort(np.argsort(alpha))          # 0 = lowest alpha
    quintile = max(1, n // 5)

    w = np.zeros(n)
    bottom = ranks < quintile
    top = ranks >= (n - quintile)

    n_short = bottom.sum()
    n_long = top.sum()

    if n_long > 0:
        w[top] = 1.0 / n_long
    if n_short > 0:
        w[bottom] = -1.0 / n_short

    # Scale to leverage budget
    gross = np.abs(w).sum()
    if gross > 0:
        w *= max_leverage / gross

    # Clip per-name
    w = np.clip(w, -max_position, max_position)

    # Re-centre to dollar neutral
    w -= w.mean()

    return w
