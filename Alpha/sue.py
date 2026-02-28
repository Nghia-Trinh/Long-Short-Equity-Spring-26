"""SUE (Standardized Unexpected Earnings) signal utilities.

This module provides:
1) SUE computation from ``Data/earnings.csv`` without look-ahead bias.
2) Point-in-time retrieval of latest SUE as of a target date.
3) Simple visualization helpers for SUE history and SUE vs. price.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    # Resolve repository root from this file location: Alpha/sue.py -> project root.
    return Path(__file__).resolve().parents[1]


def _charts_dir() -> Path:
    # Centralized chart output folder. Created lazily so callers do not need setup code.
    chart_dir = _project_root() / "outputs" / "charts"
    # Make sure folder exists before returning path.
    chart_dir.mkdir(parents=True, exist_ok=True)
    # Return canonical chart directory path.
    return chart_dir


def _timestamped_filename(base_name: str, timestamped: bool) -> str:
    # Use deterministic names when timestamping is disabled; otherwise keep each run unique.
    if not timestamped:
        # Stable filename mode.
        return f"{base_name}.png"
    # Build timestamp suffix for versioned artifact names.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Return timestamped filename.
    return f"{base_name}_{stamp}.png"


def _load_pyplot(show: bool):
    # Import matplotlib on demand to avoid hard dependency for non-plotting workflows.
    # When show=False we switch to Agg to support headless environments (no Tk/GUI).
    try:
        # Import base matplotlib module first to control backend.
        matplotlib = importlib.import_module("matplotlib")
        if not show:
            # Force non-interactive backend for server/headless runs.
            matplotlib.use("Agg", force=True)
        # Import pyplot API after backend selection.
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        # Raise clear actionable message if plotting dependency is missing.
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc


def _load_earnings() -> pd.DataFrame:
    # Load raw earnings data from disk and normalize schema/types once in one place.
    earnings_path = _project_root() / "Data" / "earnings.csv"
    # Read full earnings CSV into memory.
    earnings = pd.read_csv(earnings_path)

    required = {"ticker", "event_date", "eps_estimate", "eps_actual"}
    missing = required.difference(earnings.columns)
    if missing:
        # Build readable missing-column message for faster debugging.
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"earnings.csv is missing required columns: {missing_cols}")

    earnings = earnings.copy()
    # Standardize identifiers and parse dates so downstream filtering is reliable.
    earnings["ticker"] = earnings["ticker"].astype(str).str.strip().str.upper()
    # Convert event_date to canonical report_date datetime.
    earnings["report_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
    # Coerce numeric EPS columns; any malformed values become NaN then handled below.
    earnings["eps_estimate"] = pd.to_numeric(earnings["eps_estimate"], errors="coerce")
    earnings["eps_actual"] = pd.to_numeric(earnings["eps_actual"], errors="coerce")

    # Drop unusable rows and fill EPS gaps conservatively to preserve row continuity.
    earnings = earnings.dropna(subset=["ticker", "report_date"])
    # Fill missing EPS values with neutral 0.0 fallback.
    earnings[["eps_estimate", "eps_actual"]] = earnings[["eps_estimate", "eps_actual"]].fillna(0.0)
    # Return cleaned earnings frame.
    return earnings


@lru_cache(maxsize=8)
def compute_sue(lookback_quarters: int = 8) -> pd.DataFrame:
    """Compute point-in-time safe SUE values for all tickers.

    Parameters
    ----------
    lookback_quarters : int, default 8
        Rolling window length (in quarterly observations) used to estimate
        surprise volatility. The volatility window excludes the current quarter.

    Returns
    -------
    pd.DataFrame
        Columns include ``ticker``, ``report_date``, ``surprise``,
        ``surprise_std`` and ``sue``.
    """
    if lookback_quarters < 1:
        # Reject invalid rolling-window lengths.
        raise ValueError("lookback_quarters must be >= 1")

    # Sort by ticker/date first so rolling windows follow true event chronology.
    earnings = _load_earnings().sort_values(["ticker", "report_date"]).copy()
    # Surprise (forecast error): actual EPS minus estimated EPS.
    earnings["surprise"] = earnings["eps_actual"] - earnings["eps_estimate"]

    grouped_surprise = earnings.groupby("ticker", group_keys=False)["surprise"]
    # Critical PIT safeguard: shift(1) excludes current surprise from its own sigma.
    # This prevents look-ahead bias in both backtests and live-like simulations.
    earnings["surprise_std"] = grouped_surprise.transform(
        # shift(1) ensures current quarter surprise is excluded from its own denominator.
        lambda series: series.shift(1).rolling(window=lookback_quarters, min_periods=2).std()
    )

    # Standardize surprise by historical volatility to get comparable cross-ticker signal strength.
    earnings["sue"] = earnings["surprise"] / earnings["surprise_std"]
    # Replace invalid arithmetic outputs and fill to stable defaults for downstream matrix operations.
    earnings.replace([np.inf, -np.inf], np.nan, inplace=True)
    # SUE fallback for early-history periods with insufficient lookback.
    earnings["sue"] = earnings["sue"].fillna(0.0)
    # Store 0.0 std where historical sigma is unavailable.
    earnings["surprise_std"] = earnings["surprise_std"].fillna(0.0)

    # Return lean output schema consumed by alpha_matrix and PIT lookup functions.
    return earnings[["ticker", "report_date", "surprise", "surprise_std", "sue"]].reset_index(drop=True)


def get_latest_sue_as_of(ticker: str, target_date: str | pd.Timestamp) -> float:
    """Return the most recent SUE for ``ticker`` known as of ``target_date``.

    If no observation exists for that ticker/date combination, returns ``0.0``.
    """
    if ticker is None:
        # Null ticker -> no valid lookup, return neutral signal.
        return 0.0

    # Canonicalize inputs so lookup is case/format robust.
    ticker_key = str(ticker).strip().upper()
    as_of_date = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(as_of_date):
        # Protect caller from invalid date formats.
        raise ValueError("target_date must be parseable as a datetime")

    # Point-in-time filter: only data published on or before target_date is eligible.
    sue_df = compute_sue()
    candidate_rows = sue_df[(sue_df["ticker"] == ticker_key) & (sue_df["report_date"] <= as_of_date)]
    if candidate_rows.empty:
        # No historical signal available yet for this ticker/date.
        return 0.0

    # Return the latest known value, matching backtest/live PIT behavior.
    latest_sue = candidate_rows.sort_values("report_date").iloc[-1]["sue"]
    if pd.isna(latest_sue):
        # Defensive fallback if row exists but value is missing.
        return 0.0
    # Return numeric scalar SUE value.
    return float(latest_sue)


def plot_sue_distribution(
    ticker: str,
    bins: int = 30,
    show: bool = True,
    save: bool = True,
    save_path: str | Path | None = None,
    timestamped: bool = True,
):
    """Plot a histogram of historical SUE values for ``ticker``.

    Parameters
    ----------
    ticker : str
        Ticker symbol to visualize.
    bins : int, default 30
        Histogram bin count.
    show : bool, default True
        If True, calls ``plt.show()``.
    save : bool, default True
        If True, saves a PNG chart to ``outputs/charts`` unless ``save_path`` is provided.
    save_path : str | Path | None, default None
        Optional explicit file path for the output image.
    timestamped : bool, default True
        If True and ``save_path`` is None, appends a timestamp to the filename.
    """
    plt = _load_pyplot(show=show)

    ticker_key = str(ticker).strip().upper()
    # Pull precomputed SUE panel and isolate ticker series.
    sue_df = compute_sue()
    # Histogram uses historical SUE series for one ticker to inspect distribution shape.
    series = sue_df.loc[sue_df["ticker"] == ticker_key, "sue"].dropna()

    if series.empty:
        # Cannot draw histogram without any samples.
        raise ValueError(f"No SUE history found for ticker '{ticker_key}'")

    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot histogram of SUE values.
    ax.hist(series, bins=bins, alpha=0.75, edgecolor="black")
    # Add mean marker for quick central tendency reference.
    ax.axvline(series.mean(), linestyle="--", linewidth=1.5, label=f"Mean: {series.mean():.2f}")
    ax.set_title(f"SUE Distribution - {ticker_key}")
    ax.set_xlabel("SUE")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    if save:
        # Save chart artifacts by default for reproducibility/reporting.
        output_path = (
            Path(save_path)
            if save_path is not None
            else _charts_dir() / _timestamped_filename(f"sue_distribution_{ticker_key}", timestamped)
        )
        # Persist figure image.
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        # Render interactive window for local analysis.
        plt.show()
    else:
        # Close figure in headless/batch mode to avoid memory accumulation.
        plt.close(fig)
    # Return axis for optional downstream customization/tests.
    return ax


def plot_surprise_vs_returns(
    ticker: str,
    show: bool = True,
    save: bool = True,
    save_path: str | Path | None = None,
    timestamped: bool = True,
):
    """Overlay quarterly SUE bars with daily adjusted close prices.

    Parameters
    ----------
    ticker : str
        Ticker symbol to visualize.
    show : bool, default True
        If True, calls ``plt.show()``.
    save : bool, default True
        If True, saves a PNG chart to ``outputs/charts`` unless ``save_path`` is provided.
    save_path : str | Path | None, default None
        Optional explicit file path for the output image.
    timestamped : bool, default True
        If True and ``save_path`` is None, appends a timestamp to the filename.
    """
    plt = _load_pyplot(show=show)

    ticker_key = str(ticker).strip().upper()
    # Pull daily price data to compare post-earnings signal vs market response.
    prices_path = _project_root() / "Data" / "prices.csv"
    # Read full prices dataset; filtered to ticker below.
    prices = pd.read_csv(prices_path)

    required_price_cols = {"ticker", "trade_date", "adj_close"}
    missing_price_cols = required_price_cols.difference(prices.columns)
    if missing_price_cols:
        # Raise precise schema error if required columns are absent.
        missing_cols = ", ".join(sorted(missing_price_cols))
        raise ValueError(f"prices.csv is missing required columns: {missing_cols}")

    prices = prices.copy()
    # Normalize schema so merges/filters behave consistently.
    prices["ticker"] = prices["ticker"].astype(str).str.strip().str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    # Keep only valid price rows.
    prices = prices.dropna(subset=["trade_date", "adj_close"])
    # Restrict to requested ticker.
    prices = prices[prices["ticker"] == ticker_key].sort_values("trade_date")

    sue_df = compute_sue()
    # Earnings-frequency signal to overlay against higher-frequency price path.
    sue_ticker = sue_df[sue_df["ticker"] == ticker_key].sort_values("report_date")

    if prices.empty:
        # Abort if ticker lacks daily prices.
        raise ValueError(f"No price history found for ticker '{ticker_key}'")
    if sue_ticker.empty:
        # Abort if ticker lacks earnings/SUE records.
        raise ValueError(f"No SUE history found for ticker '{ticker_key}'")

    fig, ax_price = plt.subplots(figsize=(12, 6))
    # Primary axis: adjusted close time series.
    ax_price.plot(prices["trade_date"], prices["adj_close"], color="tab:blue", linewidth=1.5, label="Adj Close")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Adjusted Close", color="tab:blue")
    ax_price.tick_params(axis="y", labelcolor="tab:blue")

    ax_sue = ax_price.twinx()
    # Secondary axis keeps SUE scale readable without distorting price axis.
    ax_sue.bar(sue_ticker["report_date"], sue_ticker["sue"], width=18, alpha=0.35, color="tab:orange", label="SUE")
    ax_sue.axhline(0, color="gray", linewidth=1)
    ax_sue.set_ylabel("SUE", color="tab:orange")
    ax_sue.tick_params(axis="y", labelcolor="tab:orange")

    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_sue, labels_sue = ax_sue.get_legend_handles_labels()
    # Merge legends from both axes into one combined legend.
    ax_price.legend(lines_price + lines_sue, labels_price + labels_sue, loc="upper left")
    ax_price.set_title(f"SUE Signal vs Price - {ticker_key}")

    fig.tight_layout()
    if save:
        # Save chart artifacts by default for reproducibility/reporting.
        output_path = (
            Path(save_path)
            if save_path is not None
            else _charts_dir() / _timestamped_filename(f"sue_vs_price_{ticker_key}", timestamped)
        )
        # Persist figure image.
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        # Render interactive window for local analysis.
        plt.show()
    else:
        # Close figure in headless/batch mode to avoid memory accumulation.
        plt.close(fig)
    # Return axis for optional downstream customization/tests.
    return ax_price


def plot_sue_dashboard(
    ticker: str,
    bins: int = 30,
    show: bool = True,
    timestamped: bool = True,
):
    """Display the two key SUE charts for a ticker.

    This function shows:
    1) SUE distribution histogram.
    2) SUE (bar) overlaid with adjusted close (line).
    """
    # Chart 1: historical SUE distribution.
    plot_sue_distribution(ticker=ticker, bins=bins, show=show, save=True, timestamped=timestamped)
    # Chart 2: SUE vs price overlay.
    plot_surprise_vs_returns(ticker=ticker, show=show, save=True, timestamped=timestamped)
