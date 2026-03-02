from __future__ import annotations

import numpy as np
import pandas as pd

from .options_loader import load_options_for_events
from utils.data_loader import load_earnings, load_prices


def _safe_div(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(float(numerator) / float(denominator))


def _normalise_options_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "date" in out.columns and "trade_date" not in out.columns:
        out = out.rename(columns={"date": "trade_date"})
    if "expiry" in out.columns and "exdate" not in out.columns:
        out = out.rename(columns={"expiry": "exdate"})

    for col in ["ticker", "trade_date", "exdate", "cp_flag", "delta", "open_interest", "volume"]:
        if col not in out.columns:
            out[col] = np.nan

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["exdate"] = pd.to_datetime(out["exdate"], errors="coerce")
    out["cp_flag"] = out["cp_flag"].astype(str).str.upper().str.strip()
    out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
    out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    return out


def _daily_option_metrics(option_window: pd.DataFrame, event_date: pd.Timestamp) -> tuple[float, float]:
    exdate_candidates = option_window[option_window["exdate"] >= event_date]["exdate"].dropna().sort_values()
    if exdate_candidates.empty:
        return np.nan, np.nan
    target_exdate = exdate_candidates.iloc[0]

    daily_net_delta: list[float] = []
    daily_call_put_ratio: list[float] = []
    scoped = option_window[option_window["exdate"] == target_exdate]

    for _, day_frame in scoped.groupby("trade_date", sort=False):
        nd = (day_frame["delta"] * day_frame["open_interest"]).sum(min_count=1)
        if pd.notna(nd):
            daily_net_delta.append(float(nd))

        put_vol = day_frame.loc[day_frame["cp_flag"] == "P", "volume"].sum(min_count=1)
        call_vol = day_frame.loc[day_frame["cp_flag"] == "C", "volume"].sum(min_count=1)
        ratio = _safe_div(
            float(call_vol) if pd.notna(call_vol) else np.nan,
            float(put_vol) if pd.notna(put_vol) else np.nan,
        )
        if pd.notna(ratio):
            daily_call_put_ratio.append(float(ratio))

    net_delta = float(np.mean(daily_net_delta)) if daily_net_delta else np.nan
    call_put_ratio = float(np.mean(daily_call_put_ratio)) if daily_call_put_ratio else np.nan
    return net_delta, call_put_ratio


def _price_drift_5d(pre_prices: pd.DataFrame, close_col: str) -> float:
    if len(pre_prices) < 6:
        return np.nan
    start_close = pre_prices.iloc[-6][close_col]
    end_close = pre_prices.iloc[-1][close_col]
    if pd.isna(start_close) or pd.isna(end_close) or float(start_close) == 0.0:
        return np.nan
    return float(float(end_close) / float(start_close) - 1.0)


def _rv20d_vs_pre1d(pre_prices: pd.DataFrame, close_col: str) -> float:
    if len(pre_prices) < 21:
        return np.nan
    closes = pd.to_numeric(pre_prices[close_col].tail(21), errors="coerce")
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < 20:
        return np.nan

    rv20d = float(log_returns.std(ddof=1) * np.sqrt(252.0))
    last_close = pre_prices.iloc[-1][close_col]
    prev_close = pre_prices.iloc[-2][close_col]
    if pd.isna(last_close) or pd.isna(prev_close) or float(prev_close) == 0.0:
        return np.nan
    pre_ret_1d = float(float(last_close) / float(prev_close) - 1.0)
    return _safe_div(rv20d, max(abs(pre_ret_1d), 0.001))


def compute_event_features(
    ticker,
    options_df: pd.DataFrame | None,
    prices_df: pd.DataFrame | None,
    earnings_df: pd.DataFrame | None,
    window_days: int = 7,
) -> pd.DataFrame:
    ticker_norm = str(ticker).upper().strip()
    earnings = (earnings_df.copy() if earnings_df is not None else load_earnings())
    prices = (prices_df.copy() if prices_df is not None else load_prices())

    if options_df is None:
        options = load_options_for_events(earnings_df=earnings, delta_targets=None)
    else:
        options = options_df.copy()

    if earnings.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_date",
                "price_drift",
                "rv20d_vs_pre1d",
                "net_delta",
                "call_put_ratio",
                "has_options",
            ]
        )

    earnings.columns = [str(c).strip().lower() for c in earnings.columns]
    earnings["ticker"] = earnings["ticker"].astype(str).str.upper().str.strip()
    earnings["event_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
    earnings = earnings[(earnings["ticker"] == ticker_norm) & earnings["event_date"].notna()]
    earnings = earnings[["ticker", "event_date"]].drop_duplicates().sort_values("event_date")
    if earnings.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_date",
                "price_drift",
                "rv20d_vs_pre1d",
                "net_delta",
                "call_put_ratio",
                "has_options",
            ]
        )

    prices.columns = [str(c).strip().lower() for c in prices.columns]
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
    close_col = "adj_close" if "adj_close" in prices.columns else "close"
    prices[close_col] = pd.to_numeric(prices[close_col], errors="coerce")
    ticker_prices = prices[prices["ticker"] == ticker_norm].sort_values("trade_date").reset_index(drop=True)

    options = _normalise_options_columns(options)
    options = options[options["ticker"] == ticker_norm].sort_values(["trade_date", "exdate"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _, event_row in earnings.iterrows():
        event_date = pd.to_datetime(event_row["event_date"], errors="coerce")
        window_start = event_date - pd.Timedelta(days=int(window_days))
        option_window = options[(options["trade_date"] >= window_start) & (options["trade_date"] <= event_date)]
        pre_prices = ticker_prices[ticker_prices["trade_date"] < event_date].copy()

        net_delta, call_put_ratio = _daily_option_metrics(option_window, event_date)
        rows.append(
            {
                "ticker": ticker_norm,
                "event_date": event_date,
                "price_drift": _price_drift_5d(pre_prices, close_col=close_col),
                "rv20d_vs_pre1d": _rv20d_vs_pre1d(pre_prices, close_col=close_col),
                "net_delta": net_delta,
                "call_put_ratio": call_put_ratio,
                "has_options": bool(not option_window.empty),
            }
        )

    return pd.DataFrame(rows).sort_values(["ticker", "event_date"]).reset_index(drop=True)
