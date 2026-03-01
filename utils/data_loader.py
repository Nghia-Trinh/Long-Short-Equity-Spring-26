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

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_path(filename: str) -> Path:
    return _project_root() / "Data" / filename


@lru_cache(maxsize=1)
def load_earnings() -> pd.DataFrame:
    df = pd.read_csv(_data_path("earnings.csv"))
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_postearnings() -> pd.DataFrame:
    df = pd.read_csv(_data_path("postearnings_results.csv"))
    for col in ("event_date", "trade_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(_data_path("prices.csv"))
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(_data_path("summary.csv"))
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


@lru_cache(maxsize=1)
def get_adj_close_pivot() -> pd.DataFrame:
    prices = load_prices()
    pivot = prices.pivot_table(
        index="trade_date",
        columns="ticker",
        values="adj_close",
        aggfunc="last",
    )
    return pivot.sort_index().sort_index(axis=1)


@lru_cache(maxsize=1)
def get_returns_pivot() -> pd.DataFrame:
    adj_close = get_adj_close_pivot()
    return adj_close.pct_change()


@lru_cache(maxsize=1)
def get_volume_pivot() -> pd.DataFrame:
    prices = load_prices()
    pivot = prices.pivot_table(
        index="trade_date",
        columns="ticker",
        values="volume",
        aggfunc="last",
    )
    return pivot.sort_index().sort_index(axis=1)


@lru_cache(maxsize=1)
def get_summary_wide() -> pd.DataFrame:
    summary = load_summary()
    wide = summary.pivot_table(
        index="ticker",
        columns="metric",
        values="value",
        aggfunc="last",
    )
    return wide.sort_index().sort_index(axis=1)


@lru_cache(maxsize=1)
def get_tickers() -> list[str]:
    ticker_sets = []
    for df in (load_earnings(), load_prices(), load_summary()):
        if "ticker" in df.columns:
            ticker_sets.append(set(df["ticker"].dropna().astype(str).unique()))
    if not ticker_sets:
        return []
    return sorted(set().union(*ticker_sets))
