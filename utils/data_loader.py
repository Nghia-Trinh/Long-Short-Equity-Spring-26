"""
data_loader.py — Centralised Data Loading (all four CSVs)

All loaders are cached with functools.lru_cache so each file is read
from disk exactly once per process.

Data files (relative to project root):
    Data/earnings.csv              → load_earnings()
    Data/postearnings_results.csv  → load_postearnings()
    Data/prices.csv                → load_prices()
    Data/summary.csv               → load_summary()

Derived helpers (pivot tables):
    get_adj_close_pivot()   →  pd.DataFrame (trade_date × ticker), adj_close
    get_returns_pivot()     →  pd.DataFrame (trade_date × ticker), daily pct_change
    get_volume_pivot()      →  pd.DataFrame (trade_date × ticker), volume
    get_summary_wide()      →  pd.DataFrame (ticker × metric), pivoted summary
    get_tickers()           →  list[str], sorted list of all tickers
"""

# TODO: implement load_earnings()          → pd.DataFrame
# TODO: implement load_postearnings()      → pd.DataFrame
# TODO: implement load_prices()            → pd.DataFrame
# TODO: implement load_summary()           → pd.DataFrame
# TODO: implement get_adj_close_pivot()    → pd.DataFrame (cached)
# TODO: implement get_returns_pivot()      → pd.DataFrame (cached)
# TODO: implement get_volume_pivot()       → pd.DataFrame (cached)
# TODO: implement get_summary_wide()       → pd.DataFrame (cached)
# TODO: implement get_tickers()            → list[str]
