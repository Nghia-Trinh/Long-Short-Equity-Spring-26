"""Macro and sector performance collectors."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from fredapi import Fred

from config import FRED_KEY, FRED_SERIES, SECTOR_ETF_MAP


LOGGER = logging.getLogger(__name__)

def fetch_macro_indicators(months: int) -> dict[str, Any]:
    """Fetch configured FRED indicators over the lookback period."""
    if not FRED_KEY:
        LOGGER.warning("FRED_KEY is empty; macro indicators will be skipped.")
        return {}

    end = datetime.utcnow()
    start = end - timedelta(days=months * 30 + 10)
    fred = Fred(api_key=FRED_KEY)
    output: dict[str, Any] = {}

    for label, series_id in FRED_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start=start.date(), observation_end=end.date())
            series = series.dropna()
            history = [
                {"date": idx.date().isoformat(), "value": float(value)}
                for idx, value in series.items()
            ]
            latest = history[-1]["value"] if history else None
            prior = history[-2]["value"] if len(history) > 1 else None
            output[label] = {
                "series_id": series_id,
                "latest_value": latest,
                "prior_value": prior,
                "history": history,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("FRED fetch failed for %s (%s): %s", label, series_id, exc)
            output[label] = {"series_id": series_id, "error": str(exc), "history": []}

    return output


def compute_sector_performance(close_df) -> dict[str, Any]:
    """Compute total and one-month returns for configured sector ETFs."""
    results: dict[str, Any] = {}

    for sector, etf in SECTOR_ETF_MAP.items():
        if etf not in close_df.columns:
            LOGGER.warning("Sector ETF %s missing from close DataFrame.", etf)
            continue

        prices = close_df[etf].dropna()
        if prices.empty:
            continue

        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1.0
        one_month_return = None
        if len(prices) >= 22:
            one_month_return = (prices.iloc[-1] / prices.iloc[-22]) - 1.0

        results[sector] = {
            "etf": etf,
            "total_return_pct": round(total_return * 100.0, 4),
            "one_month_return_pct": None if one_month_return is None else round(one_month_return * 100.0, 4),
        }

    return results
