"""
volume_signal.py — Earnings-Day Volume Spike Signal

Intuition:
    An earnings-day volume spike signals strong price discovery.
    High-volume repricing events produce larger, more sustained post-earnings
    drifts (see Alpha/sue.py and Data/postearnings_results.csv).

Signal:
    volume_ratio = earnings_day_volume / ADV_20d
    volume_score = log(1 + volume_ratio)     [log-scaled for stability]

    volume_ratio > 2x → high conviction → hold full 10 days, scale up position
    volume_ratio < 1x → low conviction  → exit by day 5, reduce size

    CHANGE: Using call/put ratio instead of volume ratio
Data available:
    Data/prices.csv  →  volume column  ✓  (no additional data required)

Inputs:
    utils/data_loader.py  →  get_volume_pivot(), load_earnings()

Outputs:
    compute_volume_signal(adv_window=20)  →  pd.DataFrame
        columns: ticker, event_date, earnings_day_volume, adv_20d,
                 volume_ratio, volume_score

    get_volume_signal_as_of(date, vol_df)  →  pd.Series (index = ticker)
        most recent volume_score per ticker as of a given date
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_loader import get_volume_pivot, load_earnings


def compute_volume_signal(adv_window: int = 20) -> pd.DataFrame:
    earnings = load_earnings()[["ticker", "event_date"]].copy()
    earnings["ticker"] = earnings["ticker"].astype(str)
    earnings["event_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
    earnings = earnings.dropna(subset=["ticker", "event_date"])

    volume_pivot = get_volume_pivot().sort_index()
    adv = volume_pivot.rolling(window=adv_window, min_periods=adv_window).mean().shift(1)

    event_volume_long = (
        volume_pivot.stack(dropna=False)
        .rename("earnings_day_volume")
        .rename_axis(index=["event_date", "ticker"])
        .reset_index()
    )
    adv_col = f"adv_{adv_window}d"
    adv_long = (
        adv.stack(dropna=False)
        .rename(adv_col)
        .rename_axis(index=["event_date", "ticker"])
        .reset_index()
    )

    out = earnings.merge(event_volume_long, on=["ticker", "event_date"], how="left")
    out = out.merge(adv_long, on=["ticker", "event_date"], how="left")

    out[adv_col] = pd.to_numeric(out[adv_col], errors="coerce")
    out["earnings_day_volume"] = pd.to_numeric(out["earnings_day_volume"], errors="coerce")
    out["volume_ratio"] = np.where(
        out[adv_col] > 0,
        out["earnings_day_volume"] / out[adv_col],
        np.nan,
    )
    out["volume_score"] = np.log1p(out["volume_ratio"])

    return out[
        [
            "ticker",
            "event_date",
            "earnings_day_volume",
            adv_col,
            "volume_ratio",
            "volume_score",
        ]
    ].sort_values(["ticker", "event_date"]).reset_index(drop=True)


def get_volume_signal_as_of(date, vol_df: pd.DataFrame | None = None, adv_window: int = 20) -> pd.Series:
    as_of_date = pd.to_datetime(date)
    signal_df = vol_df.copy() if vol_df is not None else compute_volume_signal(adv_window=adv_window)
    if signal_df.empty:
        return pd.Series(dtype=float, name="volume_score")

    signal_df["event_date"] = pd.to_datetime(signal_df["event_date"], errors="coerce")
    eligible = signal_df[signal_df["event_date"] <= as_of_date].copy()
    if eligible.empty:
        return pd.Series(dtype=float, name="volume_score")

    latest = (
        eligible.sort_values(["ticker", "event_date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")["volume_score"]
    )
    latest.name = "volume_score"
    return latest
