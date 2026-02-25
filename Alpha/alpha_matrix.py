"""
alpha_matrix.py — T × N Alpha Matrix

Assembles a T × N matrix of alpha signals across all rebalance dates and
universe tickers. Each cell is the most recent SUE for that ticker as of
that date.

Optional IC weighting: scale each ticker's SUE by its historical
eps_surprise_vs_1d_return_corr (from Data/summary.csv), so tickers where
SUE has been more predictive receive proportionally larger signal.

Each row is cross-sectionally z-scored before output (zero mean, unit std).

Inputs:
    Alpha/sue.py               →  compute_sue(), get_latest_sue_as_of()
    Data/summary.csv           →  eps_surprise_vs_1d_return_corr per ticker

Outputs:
    build_alpha_matrix()  →  pd.DataFrame shape (T, N)
                             index = rebalance_dates, columns = tickers
"""

# TODO: implement build_alpha_matrix(rebalance_dates, tickers,
#                                    lookback_quarters=8,
#                                    use_ic_weighting=True) -> pd.DataFrame
# TODO: implement _cross_sectional_zscore(row: pd.Series) -> pd.Series
