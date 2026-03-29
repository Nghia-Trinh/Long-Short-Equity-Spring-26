"""Price collection and per-ticker splitting."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from config import BENCHMARK_TICKER, SECTOR_ETF_MAP


def _as_multiindex_columns(df: pd.DataFrame, tickers_count: int) -> pd.DataFrame:
    """Normalize yfinance output to a MultiIndex column shape."""
    if isinstance(df.columns, pd.MultiIndex):
        return df
    if tickers_count == 1:
        ticker = df.attrs.get("requested_ticker")
        if not ticker:
            return df
        tuples = [(column, ticker) for column in df.columns]
        df.columns = pd.MultiIndex.from_tuples(tuples)
        return df
    return df


def _extract_close(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return close-price DataFrame from yfinance batch output."""
    if isinstance(ohlcv.columns, pd.MultiIndex):
        close = ohlcv.get("Close")
        if close is None:
            raise ValueError("Missing 'Close' field in downloaded price data.")
        return close.dropna(how="all")

    if "Close" in ohlcv.columns:
        close = pd.DataFrame(ohlcv["Close"]).copy()
        close.columns = [ohlcv.attrs.get("requested_ticker", "TICKER")]
        return close.dropna(how="all")

    raise ValueError("Unexpected price data shape from yfinance.")


def batch_download(tickers: list[str], months: int) -> dict[str, pd.DataFrame]:
    """Download OHLCV in one yfinance batch call."""
    unique_tickers = sorted({ticker.upper() for ticker in tickers})
    symbols = unique_tickers + [BENCHMARK_TICKER] + list(SECTOR_ETF_MAP.values())
    symbols = sorted(set(symbols))
    end = datetime.utcnow()
    start = end - timedelta(days=months * 30 + 10)

    downloaded = yf.download(
        tickers=symbols,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    downloaded.attrs["requested_ticker"] = symbols[0] if len(symbols) == 1 else None
    ohlcv = _as_multiindex_columns(downloaded, len(symbols))
    close = _extract_close(ohlcv)
    close = close.sort_index()
    ohlcv = ohlcv.sort_index()
    return {"close": close, "ohlcv": ohlcv}


def split_ticker_prices(
    ticker: str,
    batch_data: dict[str, Any],
    benchmark: str,
) -> pd.DataFrame:
    """Extract one ticker OHLCV and benchmark close into a clean DataFrame."""
    ohlcv = batch_data["ohlcv"]
    close = batch_data["close"]

    ticker = ticker.upper()
    benchmark = benchmark.upper()

    if not isinstance(ohlcv.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex OHLCV data for ticker split.")

    if ticker not in ohlcv.columns.get_level_values(1):
        raise KeyError(f"Ticker {ticker} not present in downloaded OHLCV data.")
    if benchmark not in close.columns:
        raise KeyError(f"Benchmark {benchmark} not present in close-price data.")

    ticker_ohlcv = ohlcv.xs(ticker, axis=1, level=1).copy()
    benchmark_close = close[benchmark].rename("BenchmarkClose")

    merged = ticker_ohlcv.join(benchmark_close, how="left")
    merged = merged.dropna(how="any")
    merged.index.name = "Date"
    return merged
