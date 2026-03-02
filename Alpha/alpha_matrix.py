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

from datetime import datetime
from pathlib import Path
from typing import Iterable
import importlib

import numpy as np
import pandas as pd

try:
    # Package import path (typical usage: import Alpha).
    from Alpha.sue import compute_sue
except ModuleNotFoundError:
    # Fallback for direct script execution from inside Alpha/.
    from sue import compute_sue


def _project_root() -> Path:
    # Resolve repository root from this file location: Alpha/alpha_matrix.py -> project root.
    return Path(__file__).resolve().parents[1]


def _charts_dir() -> Path:
    # Centralized chart output folder. Created lazily so callers do not need setup code.
    chart_dir = _project_root() / "outputs" / "charts"
    # Ensure output path exists before use.
    chart_dir.mkdir(parents=True, exist_ok=True)
    # Return chart path object.
    return chart_dir


def _timestamped_filename(base_name: str, timestamped: bool) -> str:
    # Use deterministic names when timestamping is disabled; otherwise keep each run unique.
    if not timestamped:
        # Stable file name mode.
        return f"{base_name}.png"
    # Build timestamp suffix for versioned artifacts.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Return timestamped filename.
    return f"{base_name}_{stamp}.png"


def _timestamped_csv_filename(base_name: str, timestamped: bool) -> str:
    # Build stable CSV filename when timestamping is disabled.
    if not timestamped:
        return f"{base_name}.csv"
    # Build timestamp suffix for CSV artifact versioning.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Return timestamped CSV filename.
    return f"{base_name}_{stamp}.csv"


def _load_pyplot(show: bool):
    # Import matplotlib on demand to avoid hard dependency for non-plotting workflows.
    # When show=False we switch to Agg to support headless environments (no Tk/GUI).
    try:
        # Import base matplotlib first so backend can be set safely.
        matplotlib = importlib.import_module("matplotlib")
        if not show:
            # Force non-interactive backend for headless/script usage.
            matplotlib.use("Agg", force=True)
        # Return pyplot module for chart creation.
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        # Raise explicit dependency-install hint.
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc


def _load_ic_weights() -> pd.Series:
    # Load per-ticker information coefficient proxies from summary.csv.
    summary_path = _project_root() / "Data" / "summary.csv"
    # Read long-format summary metrics.
    summary = pd.read_csv(summary_path)

    required = {"ticker", "metric", "value"}
    missing = required.difference(summary.columns)
    if missing:
        # Raise precise message if summary schema is incomplete.
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"summary.csv is missing required columns: {missing_cols}")

    ic_rows = summary[summary["metric"] == "eps_surprise_vs_1d_return_corr"].copy()
    if ic_rows.empty:
        # No weights available -> caller logic will default to neutral scaling.
        return pd.Series(dtype=float)

    ic_rows["ticker"] = ic_rows["ticker"].astype(str).str.strip().str.upper()
    # Parse IC values and fill missing with neutral 0.0.
    ic_rows["value"] = pd.to_numeric(ic_rows["value"], errors="coerce").fillna(0.0)

    # Keep latest if duplicates exist.
    ic_rows = ic_rows.drop_duplicates(subset=["ticker"], keep="last")
    # Return ticker-indexed IC series.
    return ic_rows.set_index("ticker")["value"]


def _cross_sectional_zscore(row: pd.Series) -> pd.Series:
    """Z-score normalize a cross-section, safely handling constant rows."""
    # Coerce/clean to avoid propagation of bad values through portfolio construction.
    values = pd.to_numeric(row, errors="coerce").fillna(0.0)
    # Use population std to match deterministic cross-sectional normalization.
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        # If all tickers have identical alpha this date, return neutral zero vector.
        return pd.Series(0.0, index=values.index)
    # Standard z-score transform for non-constant rows.
    return (values - values.mean()) / std


def build_alpha_matrix(
    rebalance_dates: Iterable[str | pd.Timestamp],
    tickers: Iterable[str],
    lookback_quarters: int = 8,
    use_ic_weighting: bool = True,
) -> pd.DataFrame:
    """Build a T × N alpha matrix from point-in-time SUE signals.

    Parameters
    ----------
    rebalance_dates : iterable of datetime-like
        Rebalance timestamps (rows of output matrix).
    tickers : iterable of str
        Universe tickers (columns of output matrix).
    lookback_quarters : int, default 8
        Lookback window used in SUE computation.
    use_ic_weighting : bool, default True
        If True, scales ticker SUE by historical
        ``eps_surprise_vs_1d_return_corr`` from ``summary.csv``.
    """
    # Normalize identifiers upfront so all joins and lookups are consistent.
    ticker_list = [str(ticker).strip().upper() for ticker in tickers]
    if not ticker_list:
        # Return empty matrix when no tickers are requested.
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=[])

    # Parse/sanitize rebalance dates and enforce sorted unique index.
    rebalance_index = pd.to_datetime(pd.Index(list(rebalance_dates)), errors="coerce")
    rebalance_index = rebalance_index.dropna().sort_values().unique()
    if len(rebalance_index) == 0:
        # Return date-empty matrix but preserve requested ticker columns.
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=ticker_list)

    # Compute SUE once, then restrict to requested universe for efficiency.
    sue_df = compute_sue(lookback_quarters=lookback_quarters)
    # Restrict SUE table to requested universe for efficiency.
    sue_df = sue_df[sue_df["ticker"].isin(ticker_list)].copy()

    matrix = pd.DataFrame(0.0, index=rebalance_index, columns=ticker_list)

    # Point-in-time fill per ticker: last known SUE as of each rebalance date.
    for ticker in ticker_list:
        # Pull one ticker's history of report-date SUE values.
        history = sue_df[sue_df["ticker"] == ticker][["report_date", "sue"]].sort_values("report_date")
        if history.empty:
            # Leave default zeros when no history exists for this ticker.
            continue

        # merge_asof with backward direction gives the most recent known signal as-of each date.
        merged = pd.merge_asof(
            pd.DataFrame({"rebalance_date": rebalance_index}).sort_values("rebalance_date"),
            history.rename(columns={"report_date": "rebalance_date"}),
            on="rebalance_date",
            direction="backward",
        )
        # Fill PIT SUE values into matrix column.
        matrix[ticker] = merged["sue"].fillna(0.0).to_numpy()

    if use_ic_weighting:
        # Optional scaling: amplify or damp SUE by historical predictive relationship strength.
        ic_weights = _load_ic_weights()
        # Default weight 1.0 when IC is unavailable.
        aligned_weights = pd.Series(1.0, index=ticker_list)
        aligned_weights.update(ic_weights.reindex(ticker_list).fillna(1.0))
        # Multiply each ticker column by its IC weight.
        matrix = matrix.mul(aligned_weights, axis=1)

    # Standardize each date's cross-section so downstream optimizer receives comparable exposure units.
    matrix = matrix.apply(_cross_sectional_zscore, axis=1)
    # Name index for cleaner exports and downstream readability.
    matrix.index.name = "rebalance_date"
    # Final defensive NaN fill.
    matrix = matrix.fillna(0.0)
    # Return completed alpha matrix.
    return matrix


def plot_alpha_heatmap(
    alpha_matrix: pd.DataFrame,
    max_tickers: int | None = None,
    show: bool = True,
    save: bool = True,
    save_path: str | Path | None = None,
    timestamped: bool = True,
):
    """Plot a heatmap of the alpha matrix.

    Parameters
    ----------
    alpha_matrix : pd.DataFrame
        Output from ``build_alpha_matrix``.
    max_tickers : int | None, default None
        Maximum number of columns to display (left-most subset).
        If None, displays all columns.
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

    if alpha_matrix.empty:
        # Cannot draw heatmap with empty input.
        raise ValueError("alpha_matrix is empty; nothing to plot")

    clipped = alpha_matrix.copy()
    if max_tickers is not None and clipped.shape[1] > max_tickers:
        # Keep chart readable for large universes by plotting a subset.
        clipped = clipped.iloc[:, :max_tickers]

    fig, ax = plt.subplots(figsize=(14, 7))
    # Render matrix as image: rows=time, columns=tickers (transposed for readability).
    # Use rainbow palette to make rank/intensity transitions visually distinct.
    im = ax.imshow(clipped.to_numpy().T, aspect="auto", cmap="rainbow", interpolation="nearest")

    ax.set_title("Alpha Matrix Heatmap (Tickers × Time)")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Ticker")

    x_step = max(1, len(clipped.index) // 10)
    # Downsample x-axis labels so long histories remain legible.
    x_ticks = np.arange(0, len(clipped.index), x_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([pd.Timestamp(clipped.index[i]).strftime("%Y-%m-%d") for i in x_ticks], rotation=45, ha="right")

    y_ticks = np.arange(len(clipped.columns))
    # Label y-axis with ticker symbols.
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(clipped.columns)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Alpha (z-scored)")
    fig.tight_layout()
    if save:
        # Save chart artifacts by default for reproducibility/reporting.
        output_path = (
            Path(save_path)
            if save_path is not None
            else _charts_dir() / _timestamped_filename("alpha_heatmap", timestamped)
        )
        # Persist image file to outputs/charts.
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        # Show interactive chart window.
        plt.show()
    else:
        # Close figure in batch/headless mode.
        plt.close(fig)
    # Return axis handle for optional customization/tests.
    return ax


def plot_alpha_signal_strength(
    alpha_matrix: pd.DataFrame,
    show: bool = True,
    save: bool = True,
    save_path: str | Path | None = None,
    timestamped: bool = True,
):
    """Plot average absolute alpha per rebalance date.

    This is a compact proxy for how strong/dispersed the cross-sectional
    alpha signal is through time.
    """
    plt = _load_pyplot(show=show)

    if alpha_matrix.empty:
        # Cannot compute trend without matrix data.
        raise ValueError("alpha_matrix is empty; nothing to plot")

    # Mean absolute z-score per date summarizes cross-sectional signal intensity.
    strength = alpha_matrix.abs().mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot average absolute alpha per date as signal intensity proxy.
    ax.plot(strength.index, strength.values, color="tab:purple", linewidth=1.8)
    ax.set_title("Average Absolute Alpha Through Time")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Mean |Alpha|")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        # Save chart artifacts by default for reproducibility/reporting.
        output_path = (
            Path(save_path)
            if save_path is not None
            else _charts_dir() / _timestamped_filename("alpha_signal_strength", timestamped)
        )
        # Persist image file to outputs/charts.
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        # Show interactive chart window.
        plt.show()
    else:
        # Close figure in batch/headless mode.
        plt.close(fig)
    # Return axis handle for optional customization/tests.
    return ax


def plot_alpha_long_short_spread(
    alpha_matrix: pd.DataFrame,
    quantile: float = 0.2,
    show: bool = True,
    save: bool = True,
    save_path: str | Path | None = None,
    timestamped: bool = True,
):
    """Plot top-minus-bottom alpha spread through time.

    For each date, this computes:
    mean(alpha of top quantile tickers) - mean(alpha of bottom quantile tickers).
    """
    plt = _load_pyplot(show=show)

    if alpha_matrix.empty:
        # Cannot compute spread with empty matrix.
        raise ValueError("alpha_matrix is empty; nothing to plot")
    if not 0 < quantile <= 0.5:
        # Validate quantile bounds (must define top and bottom buckets).
        raise ValueError("quantile must be in (0, 0.5]")

    def _spread_for_row(row: pd.Series) -> float:
        # Clean row values and drop missing entries.
        clean = pd.to_numeric(row, errors="coerce").dropna()
        if clean.empty:
            # Neutral spread when no values exist.
            return 0.0
        # Number of tickers in each tail bucket.
        k = max(1, int(len(clean) * quantile))
        # Ascending sort so head=bottom and tail=top.
        ranked = clean.sort_values()
        # Bottom quantile mean.
        bottom = ranked.iloc[:k].mean()
        # Top quantile mean.
        top = ranked.iloc[-k:].mean()
        # Long-short spread proxy.
        return float(top - bottom)

    spread = alpha_matrix.apply(_spread_for_row, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot spread trend through time.
    ax.plot(spread.index, spread.values, color="tab:green", linewidth=1.8, label="Top - Bottom Spread")
    ax.axhline(0.0, color="gray", linewidth=1.0)
    ax.set_title(f"Alpha Long/Short Spread (Top {int(quantile * 100)}% - Bottom {int(quantile * 100)}%)")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Spread")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        output_path = (
            Path(save_path)
            if save_path is not None
            else _charts_dir() / _timestamped_filename("alpha_long_short_spread", timestamped)
        )
        # Persist image file to outputs/charts.
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        # Show interactive chart window.
        plt.show()
    else:
        # Close figure in batch/headless mode.
        plt.close(fig)
    # Return axis handle for optional customization/tests.
    return ax


def plot_alpha_matrix_snapshot(
    alpha_matrix: pd.DataFrame,
    max_rows: int = 18,
    max_tickers: int = 10,
    show: bool = True,
    save: bool = True,
    save_path: str | Path | None = None,
    timestamped: bool = True,
):
    """Render a table snapshot of the alpha matrix and save as an image.

    This provides an easy-to-read matrix view in the charts folder.
    """
    plt = _load_pyplot(show=show)

    if alpha_matrix.empty:
        # Cannot render table snapshot for empty matrix.
        raise ValueError("alpha_matrix is empty; nothing to plot")

    clipped = alpha_matrix.iloc[:max_rows, :max_tickers].copy()
    # Convert datetime index to compact display string.
    clipped.index = [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in clipped.index]
    # Round displayed values for readability.
    clipped = clipped.round(3)

    row_count = max(1, clipped.shape[0])
    # Auto-size figure height based on number of table rows.
    fig_height = max(4, min(16, row_count * 0.45 + 1.5))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    # Hide normal axes since we are drawing a table.
    ax.axis("off")
    table = ax.table(
        cellText=clipped.values,
        rowLabels=clipped.index,
        colLabels=clipped.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title("Alpha Matrix Snapshot", pad=12)
    fig.tight_layout()

    if save:
        output_path = (
            Path(save_path)
            if save_path is not None
            else _charts_dir() / _timestamped_filename("alpha_matrix_snapshot", timestamped)
        )
        # Persist image file to outputs/charts.
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        # Show interactive chart window.
        plt.show()
    else:
        # Close figure in batch/headless mode.
        plt.close(fig)
    # Return axis handle for optional customization/tests.
    return ax


def save_alpha_matrix_csv(
    alpha_matrix: pd.DataFrame,
    save_path: str | Path | None = None,
    timestamped: bool = True,
) -> Path:
    """Save the full alpha matrix to charts directory as CSV and return output path."""
    if alpha_matrix.empty:
        # Prevent writing empty artifact files.
        raise ValueError("alpha_matrix is empty; nothing to save")

    output_path = (
        Path(save_path)
        if save_path is not None
        else _charts_dir() / _timestamped_csv_filename("alpha_matrix", timestamped)
    )
    # Persist full matrix to CSV for external analysis.
    alpha_matrix.to_csv(output_path, index=True)
    # Return output path for logging/audit.
    return output_path


def plot_alpha_dashboard(
    rebalance_dates: Iterable[str | pd.Timestamp],
    tickers: Iterable[str],
    lookback_quarters: int = 8,
    use_ic_weighting: bool = True,
    max_tickers: int | None = None,
    show: bool = True,
    timestamped: bool = True,
):
    """Build alpha matrix and save a complete chart/artifact pack.

    Artifacts produced in ``outputs/charts``:
    - Full alpha matrix CSV
    - Heatmap
    - Signal strength line
    - Long/short spread line
    - Matrix snapshot table image
    """
    alpha = build_alpha_matrix(
        rebalance_dates=rebalance_dates,
        tickers=tickers,
        lookback_quarters=lookback_quarters,
        use_ic_weighting=use_ic_weighting,
    )

    # Export full matrix as CSV artifact.
    save_alpha_matrix_csv(alpha, timestamped=timestamped)
    # Save heatmap chart artifact.
    plot_alpha_heatmap(alpha, max_tickers=max_tickers, show=show, save=True, timestamped=timestamped)
    # Save signal-strength trend artifact.
    plot_alpha_signal_strength(alpha, show=show, save=True, timestamped=timestamped)
    # Save long-short spread trend artifact.
    plot_alpha_long_short_spread(alpha, quantile=0.2, show=show, save=True, timestamped=timestamped)
    # Save table-style matrix snapshot artifact.
    snapshot_tickers = 10 if max_tickers is None else min(max_tickers, 10)
    plot_alpha_matrix_snapshot(alpha, max_rows=18, max_tickers=snapshot_tickers, show=show, save=True, timestamped=timestamped)
    # Return matrix for downstream use.
    return alpha
