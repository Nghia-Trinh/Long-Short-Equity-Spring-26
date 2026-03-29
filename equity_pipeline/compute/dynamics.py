"""Derived price dynamics (pure computation)."""

import numpy as np
import pandas as pd


def _safe_round(value, ndigits=6):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return None
    return round(float(value), ndigits)


def _annualized_return(total_return, periods):
    if periods <= 0:
        return None
    return (1.0 + total_return) ** (252.0 / periods) - 1.0


def _max_drawdown(price_series):
    running_max = price_series.cummax()
    drawdown = (price_series / running_max) - 1.0
    return float(drawdown.min()) if not drawdown.empty else None


def _momentum(price_series, window):
    if len(price_series) < window + 1:
        return None
    return (price_series.iloc[-1] / price_series.iloc[-(window + 1)]) - 1.0


def compute_dynamics(ticker, close_df, benchmark):
    """Compute return/risk dynamics for one ticker versus benchmark."""
    ticker = ticker.upper()
    benchmark = benchmark.upper()

    if ticker not in close_df.columns:
        raise KeyError(f"Ticker {ticker} not found in close DataFrame.")
    if benchmark not in close_df.columns:
        raise KeyError(f"Benchmark {benchmark} not found in close DataFrame.")

    prices = close_df[[ticker, benchmark]].dropna(how="any")
    ticker_prices = prices[ticker]
    benchmark_prices = prices[benchmark]

    if len(ticker_prices) < 2:
        return {
            "ticker": ticker,
            "benchmark": benchmark,
            "observations": int(len(ticker_prices)),
        }

    ticker_returns = ticker_prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    paired_returns = pd.concat([ticker_returns, benchmark_returns], axis=1).dropna(how="any")
    paired_returns.columns = ["ticker", "benchmark"]

    total_return = (ticker_prices.iloc[-1] / ticker_prices.iloc[0]) - 1.0
    benchmark_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1.0
    ann_return = _annualized_return(total_return, len(ticker_returns))
    benchmark_ann_return = _annualized_return(benchmark_return, len(benchmark_returns))

    volatility = ticker_returns.std(ddof=1) * np.sqrt(252.0)
    downside_returns = ticker_returns[ticker_returns < 0]
    downside_vol = downside_returns.std(ddof=1) * np.sqrt(252.0) if not downside_returns.empty else None
    sharpe = (ticker_returns.mean() * 252.0) / volatility if volatility and not np.isnan(volatility) else None
    sortino = (
        (ticker_returns.mean() * 252.0) / downside_vol
        if downside_vol and not np.isnan(downside_vol)
        else None
    )

    beta = None
    correlation = None
    if len(paired_returns) > 1:
        covariance = np.cov(paired_returns["ticker"], paired_returns["benchmark"], ddof=1)[0, 1]
        benchmark_variance = paired_returns["benchmark"].var(ddof=1)
        beta = covariance / benchmark_variance if benchmark_variance and not np.isnan(benchmark_variance) else None
        correlation = paired_returns["ticker"].corr(paired_returns["benchmark"])

    alpha = None
    if ann_return is not None and benchmark_ann_return is not None:
        alpha = ann_return - benchmark_ann_return

    trailing_252 = ticker_prices.tail(252)
    high_52w = trailing_252.max() if not trailing_252.empty else None
    low_52w = trailing_252.min() if not trailing_252.empty else None
    current_price = ticker_prices.iloc[-1]
    pct_from_high = (current_price / high_52w) - 1.0 if high_52w else None

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "observations": int(len(ticker_prices)),
        "total_return": _safe_round(total_return),
        "annualized_return": _safe_round(ann_return),
        "annualized_volatility": _safe_round(volatility),
        "sharpe_ratio": _safe_round(sharpe),
        "sortino_ratio": _safe_round(sortino),
        "max_drawdown": _safe_round(_max_drawdown(ticker_prices)),
        "benchmark_return": _safe_round(benchmark_return),
        "alpha": _safe_round(alpha),
        "correlation": _safe_round(correlation),
        "beta": _safe_round(beta),
        "momentum_1m": _safe_round(_momentum(ticker_prices, 21)),
        "momentum_3m": _safe_round(_momentum(ticker_prices, 63)),
        "momentum_6m": _safe_round(_momentum(ticker_prices, 126)),
        "current_price": _safe_round(current_price),
        "high_52w": _safe_round(high_52w),
        "low_52w": _safe_round(low_52w),
        "pct_from_high_52w": _safe_round(pct_from_high),
    }
