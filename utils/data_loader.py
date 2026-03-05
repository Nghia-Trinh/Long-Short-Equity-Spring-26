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


def _data_dir() -> Path:
    return _project_root() / "Data"


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_earnings() -> pd.DataFrame:
    df = pd.read_csv(_data_dir() / "earnings.csv")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["eps_estimate"] = pd.to_numeric(df["eps_estimate"], errors="coerce")
    df["eps_actual"] = pd.to_numeric(df["eps_actual"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_postearnings() -> pd.DataFrame:
    df = pd.read_csv(_data_dir() / "postearnings_results.csv")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    for col in ["return_1d", "return_5d", "return_10d", "gap_pct", "rv_10d_annualized"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(_data_dir() / "prices.csv")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(_data_dir() / "summary.csv")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Pivot tables
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_adj_close_pivot() -> pd.DataFrame:
    prices = load_prices()
    pivot = prices.pivot_table(
        index="trade_date", columns="ticker", values="adj_close", aggfunc="last",
    )
    pivot = pivot.sort_index()
    return pivot


@lru_cache(maxsize=1)
def get_returns_pivot() -> pd.DataFrame:
    adj = get_adj_close_pivot()
    returns = adj.pct_change(fill_method=None)
    return returns


@lru_cache(maxsize=1)
def get_volume_pivot() -> pd.DataFrame:
    prices = load_prices()
    pivot = prices.pivot_table(
        index="trade_date", columns="ticker", values="volume", aggfunc="last",
    )
    pivot = pivot.sort_index()
    return pivot


@lru_cache(maxsize=1)
def get_summary_wide() -> pd.DataFrame:
    summary = load_summary()
    if {"ticker", "metric", "value"}.issubset(summary.columns):
        wide = summary.pivot_table(
            index="ticker", columns="metric", values="value", aggfunc="last",
        )
        return wide
    return summary


@lru_cache(maxsize=1)
def get_tickers() -> list[str]:
    adj = get_adj_close_pivot()
    return sorted(adj.columns.tolist())
