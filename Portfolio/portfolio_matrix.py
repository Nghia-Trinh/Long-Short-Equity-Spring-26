"""
portfolio_matrix.py — T × N Portfolio Weights Matrix

Main loop: iterate over all rebalance dates, call the optimizer at each
step, and assemble the full T × N matrix of portfolio weights.
This matrix is the primary output consumed by backtest.py.

Algorithm per rebalance date t:
    1. Fetch alpha_t from Alpha/alpha_matrix.py      → (N,) vector
    2. Fetch Sigma_t from Risk/risk_matrix.py         → (N, N) matrix
    3. Solve optimisation via Portfolio/optimizer.py  → (N,) weight vector
    4. Store weights and advance w_prev = w_t

Inputs:
    Alpha/alpha_matrix.py     →  build_alpha_matrix()
    Risk/risk_matrix.py       →  RiskMatrixBuilder
    Portfolio/optimizer.py    →  optimize_portfolio()
    utils/universe.py         →  get_universe()
    config.json               →  all hyperparameters

Outputs:
    build_portfolio_matrix(config) →  pd.DataFrame shape (T, N)
                                      index = rebalance_dates
                                      columns = tickers
                                      values = portfolio weights
                                      (sum ≈ 0, sum(|w|) <= max_leverage)
"""

# TODO: implement build_portfolio_matrix(config=None) -> pd.DataFrame
#   Steps:
#     1. Load config
#     2. Build universe via get_universe()
#     3. Fit RiskMatrixBuilder
#     4. Determine rebalance_dates from returns index (post-warmup window)
#     5. Build alpha_matrix via build_alpha_matrix()
#     6. Loop over rebalance_dates: call optimize_portfolio() at each step
#     7. Return pd.DataFrame of weights
