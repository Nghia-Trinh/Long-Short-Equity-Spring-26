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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(config: dict | None = None) -> dict:
    if config is not None:
        return config
    config_path = _project_root() / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_signal_df(df: pd.DataFrame | None, value_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "event_date", value_col])
    if value_col not in df.columns:
        if value_col == "preearnings_score" and {"direction", "position_size"}.issubset(df.columns):
            out = df[["ticker", "event_date", "direction", "position_size"]].copy()
            out["direction"] = out["direction"].astype(str).str.lower().str.strip()
            sign = out["direction"].map({"long": 1.0, "short": -1.0})
            out[value_col] = pd.to_numeric(out["position_size"], errors="coerce") * sign
            out = out.drop(columns=["direction", "position_size"])
        else:
            return pd.DataFrame(columns=["ticker", "event_date", value_col])
    else:
        out = df[["ticker", "event_date", value_col]].copy()

    out["ticker"] = out["ticker"].astype(str)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out.dropna(subset=["ticker", "event_date"]).sort_values(["ticker", "event_date"])


def _latest_signal_as_of(signal_df: pd.DataFrame, date: pd.Timestamp, value_col: str) -> pd.Series:
    if signal_df.empty:
        return pd.Series(dtype=float, name=value_col)
    eligible = signal_df[signal_df["event_date"] <= date]
    if eligible.empty:
        return pd.Series(dtype=float, name=value_col)
    latest = (
        eligible.sort_values(["ticker", "event_date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")[value_col]
    )
    latest.name = value_col
    return latest


def build_alpha_matrix(
    rebalance_dates,
    tickers,
    lookback_quarters: int = 8,
    use_ic_weighting: bool = True,
    sue_df: pd.DataFrame | None = None,
    skew_df: pd.DataFrame | None = None,
    vol_df: pd.DataFrame | None = None,
    preearnings_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    _ = use_ic_weighting  # Preserved for backward-compatible signature.
    cfg = _load_config(config)
    w_sue = float(cfg.get("w_sue", 1.0))
    w_skew = float(cfg.get("w_skew", 0.0))
    w_vol = float(cfg.get("w_vol", 0.0))
    w_preearnings = float(cfg.get("w_preearnings", 0.0))

    if sue_df is None:
        try:
            from Alpha.sue import compute_sue

            sue_df = compute_sue(lookback_quarters=lookback_quarters)
        except Exception:
            sue_df = pd.DataFrame(columns=["ticker", "event_date", "sue"])

    sue_signal = _prepare_signal_df(sue_df, "sue")
    skew_signal = _prepare_signal_df(skew_df, "skew_score")
    vol_signal = _prepare_signal_df(vol_df, "volume_score")
    preearnings_signal = _prepare_signal_df(preearnings_df, "preearnings_score")

    rebalance_index = pd.to_datetime(pd.Index(rebalance_dates))
    ticker_index = pd.Index([str(t) for t in tickers], dtype="object")

    alpha_rows = []
    for date in rebalance_index:
        sue_asof = _latest_signal_as_of(sue_signal, date, "sue")
        skew_asof = _latest_signal_as_of(skew_signal, date, "skew_score")
        vol_asof = _latest_signal_as_of(vol_signal, date, "volume_score")
        preearnings_asof = _latest_signal_as_of(preearnings_signal, date, "preearnings_score")

        components = pd.concat([sue_asof, skew_asof, vol_asof, preearnings_asof], axis=1).reindex(ticker_index)
        weighted_alpha = (
            w_sue * components["sue"].fillna(0.0)
            + w_skew * components["skew_score"].fillna(0.0)
            + w_vol * components["volume_score"].fillna(0.0)
            + w_preearnings * components["preearnings_score"].fillna(0.0)
        )
        weighted_alpha[components.isna().all(axis=1)] = np.nan
        alpha_rows.append(weighted_alpha.rename(date))

    alpha_matrix = pd.DataFrame(alpha_rows, index=rebalance_index, columns=ticker_index)
    return alpha_matrix.apply(_cross_sectional_zscore, axis=1)


def _cross_sectional_zscore(row: pd.Series) -> pd.Series:
    values = pd.to_numeric(row, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=row.index)
    return (values - mean) / std
