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

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from utils.data_loader import get_adj_close_pivot, get_volume_pivot


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = _project_root() / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_dollar_volume(lookback_days: int = 252) -> pd.Series:
    """Return median daily dollar volume per ticker over the trailing window."""
    adj = get_adj_close_pivot()
    vol = get_volume_pivot()

    # Align shapes
    common_tickers = sorted(set(adj.columns) & set(vol.columns))
    adj = adj[common_tickers]
    vol = vol[common_tickers]

    dollar_vol = adj * vol
    # Use the last `lookback_days` rows
    tail = dollar_vol.tail(lookback_days)
    median_dv = tail.median(axis=0).rename("median_dollar_volume")
    return median_dv.sort_values(ascending=False)


@lru_cache(maxsize=1)
def get_universe(
    exclude_top_n_largecap: int | None = None,
    min_dollar_volume: float | None = None,
) -> list[str]:
    """Return sorted list of eligible tickers after universe filters."""
    cfg = _load_config()
    if exclude_top_n_largecap is None:
        exclude_top_n_largecap = int(cfg.get("universe_exclude_top_n_largecap", 500))
    if min_dollar_volume is None:
        min_dollar_volume = float(cfg.get("universe_min_dollar_volume", 1_000_000))

    dv = compute_dollar_volume()

    # Exclude top-N as large-cap proxy
    if exclude_top_n_largecap > 0 and len(dv) > exclude_top_n_largecap:
        dv = dv.iloc[exclude_top_n_largecap:]

    # Minimum dollar volume filter
    eligible = dv[dv >= min_dollar_volume]
    return sorted(eligible.index.tolist())
