"""Fundamentals collector from yfinance and Finnhub."""

from __future__ import annotations

import logging
import time
from typing import Any

import finnhub
import pandas as pd
import yfinance as yf

from config import FINNHUB_DELAY, FINNHUB_KEY


LOGGER = logging.getLogger(__name__)


def safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None when conversion fails."""
    try:
        if value is None:
            return None
        cast = float(value)
        if pd.isna(cast):
            return None
        return cast
    except (TypeError, ValueError):
        return None


def _normalize_quarterly(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Convert yfinance quarterly DataFrame columns to readable quarter labels."""
    if df is None or df.empty:
        return {}

    normalized: dict[str, dict[str, float | None]] = {}
    for column in df.columns:
        try:
            stamp = pd.to_datetime(column)
            quarter_label = f"Q{stamp.quarter} {stamp.year}"
        except Exception:  # noqa: BLE001
            quarter_label = str(column)

        quarter_values = {row_name: safe_float(value) for row_name, value in df[column].items()}
        normalized[quarter_label] = quarter_values
    return normalized


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    """Fetch company fundamentals and optional analyst data."""
    yf_ticker = yf.Ticker(ticker.upper())
    info = yf_ticker.info or {}

    fundamentals: dict[str, Any] = {
        "profile": {
            "symbol": info.get("symbol", ticker.upper()),
            "short_name": info.get("shortName"),
            "long_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "business_summary": info.get("longBusinessSummary"),
            "full_time_employees": safe_float(info.get("fullTimeEmployees")),
        },
        "ratios": {
            "pe_trailing": safe_float(info.get("trailingPE")),
            "pe_forward": safe_float(info.get("forwardPE")),
            "peg_ratio": safe_float(info.get("pegRatio")),
            "price_to_book": safe_float(info.get("priceToBook")),
            "enterprise_to_revenue": safe_float(info.get("enterpriseToRevenue")),
            "enterprise_to_ebitda": safe_float(info.get("enterpriseToEbitda")),
        },
        "profitability": {
            "gross_margins": safe_float(info.get("grossMargins")),
            "operating_margins": safe_float(info.get("operatingMargins")),
            "ebitda_margins": safe_float(info.get("ebitdaMargins")),
            "profit_margins": safe_float(info.get("profitMargins")),
            "return_on_assets": safe_float(info.get("returnOnAssets")),
            "return_on_equity": safe_float(info.get("returnOnEquity")),
        },
        "balance_sheet": {
            "total_cash": safe_float(info.get("totalCash")),
            "total_debt": safe_float(info.get("totalDebt")),
            "debt_to_equity": safe_float(info.get("debtToEquity")),
            "current_ratio": safe_float(info.get("currentRatio")),
            "quick_ratio": safe_float(info.get("quickRatio")),
            "book_value": safe_float(info.get("bookValue")),
        },
        "cash_flow": {
            "operating_cashflow": safe_float(info.get("operatingCashflow")),
            "free_cashflow": safe_float(info.get("freeCashflow")),
            "total_cash_per_share": safe_float(info.get("totalCashPerShare")),
        },
        "growth": {
            "revenue_growth": safe_float(info.get("revenueGrowth")),
            "earnings_growth": safe_float(info.get("earningsGrowth")),
            "earnings_quarterly_growth": safe_float(info.get("earningsQuarterlyGrowth")),
        },
        "analyst_consensus": {
            "target_mean_price": safe_float(info.get("targetMeanPrice")),
            "target_high_price": safe_float(info.get("targetHighPrice")),
            "target_low_price": safe_float(info.get("targetLowPrice")),
            "number_of_analyst_opinions": safe_float(info.get("numberOfAnalystOpinions")),
            "recommendation_key": info.get("recommendationKey"),
            "recommendation_mean": safe_float(info.get("recommendationMean")),
        },
        "quarterly_statements": {
            "income_statement": _normalize_quarterly(yf_ticker.quarterly_financials),
            "balance_sheet": _normalize_quarterly(yf_ticker.quarterly_balance_sheet),
            "cash_flow": _normalize_quarterly(yf_ticker.quarterly_cashflow),
        },
    }

    fundamentals["analyst_api"] = {"recommendation_trends": [], "earnings_history": []}
    if FINNHUB_KEY:
        client = finnhub.Client(api_key=FINNHUB_KEY)
        try:
            fundamentals["analyst_api"]["recommendation_trends"] = client.recommendation_trends(ticker.upper()) or []
            time.sleep(FINNHUB_DELAY)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Finnhub recommendation fetch failed for %s: %s", ticker, exc)
        try:
            fundamentals["analyst_api"]["earnings_history"] = client.company_earnings(
                ticker.upper(),
                limit=12,
            ) or []
            time.sleep(FINNHUB_DELAY)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Finnhub earnings fetch failed for %s: %s", ticker, exc)
    else:
        LOGGER.warning("FINNHUB_KEY is empty; analyst API data skipped for %s.", ticker)

    return fundamentals
