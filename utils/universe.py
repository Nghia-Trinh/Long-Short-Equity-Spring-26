"""
universe.py — Russell 3000 Universe Filter (ex-large cap)

Defines the investable universe by:
    1. Computing median daily dollar volume (price × volume) over trailing 252 days
    2. Ranking all tickers by dollar volume (largest first)
    3. Excluding the top-N as a large-cap proxy (configurable, default N=500)
    4. Requiring minimum dollar volume threshold (default $1M/day)

This approximates the Russell 3000 ex-large-cap segment since we do not
have direct market-cap data. Dollar volume is a reasonable proxy.

Inputs:
    utils/data_loader.py  →  get_adj_close_pivot(), get_volume_pivot()
    config.json           →  universe_exclude_top_n_largecap,
                              universe_min_dollar_volume

Outputs:
    compute_dollar_volume(lookback_days=252)  →  pd.Series (ticker → median DV)
    get_universe(exclude_top_n, min_dv)       →  list[str] of eligible tickers
"""

# TODO: implement compute_dollar_volume(lookback_days=252) -> pd.Series
# TODO: implement get_universe(exclude_top_n_largecap=500,
#                               min_dollar_volume=1e6) -> list[str]
