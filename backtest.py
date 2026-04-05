"""
backtest.py — Offline Backtester (pure Python, no QuantConnect required)

Runs the full pipeline using only Data/ CSV files:
    1. Build alpha matrix  (Alpha/)
    2. Build risk matrix   (Risk/)
    3. Optimise portfolio  (Portfolio/)   — day-by-day
    4. Simulate P&L against forward daily returns
    5. Print performance summary
    6. Save outputs to outputs/

Run from the project root:
    python backtest.py

Outputs written to outputs/:
    portfolio_weights.csv   — T × N weight matrix
    pnl.csv                 — daily P&L series

P&L simulation:
    PnL_t = w_t' · r_{t+1}      (weights at t, forward return at t+1)

Performance metrics printed:
    Total P&L, total return, annualised return, annualised volatility,
    Sharpe ratio, max drawdown, win rate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from LLM.data_fetcher import LLMDataFetcher
from Portfolio.portfolio_matrix import build_portfolio_matrix
from utils.data_loader import get_returns_pivot


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_config() -> dict:
    config_path = _project_root() / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _outputs_dir() -> Path:
    out = _project_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------------
# P&L simulation
# ------------------------------------------------------------------

def simulate_pnl(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Simulate daily P&L from weight matrix and forward returns.

    For each rebalance date t in ``weights.index``:
        Weights are held until the next rebalance date.
        Daily PnL = w_t' @ r_{t+k}  for each trading day t+k in the interval.

    Parameters
    ----------
    weights : pd.DataFrame
        T × N weight matrix (index = rebalance dates).
    returns : pd.DataFrame
        Full daily return matrix (index = trading dates, columns = tickers).

    Returns
    -------
    pd.Series
        Daily portfolio return series.
    """
    # Align columns
    common = sorted(set(weights.columns) & set(returns.columns))
    weights = weights[common]
    returns = returns[common]

    pnl = pd.Series(0.0, index=returns.index, name="portfolio_return")

    rebalance_dates = weights.index.sort_values()
    all_dates = returns.index.sort_values()

    for i, reb_date in enumerate(rebalance_dates):
        w = weights.loc[reb_date].values.astype(float)
        # Hold until next rebalance (or end of data)
        if i + 1 < len(rebalance_dates):
            end_date = rebalance_dates[i + 1]
        else:
            end_date = all_dates[-1]

        mask = (all_dates > reb_date) & (all_dates <= end_date)
        holding_dates = all_dates[mask]

        for d in holding_dates:
            r = returns.loc[d].reindex(common, fill_value=0.0).values.astype(float)
            r = np.nan_to_num(r, nan=0.0)
            pnl.loc[d] = float(w @ r)

    return pnl


# ------------------------------------------------------------------
# Performance summary
# ------------------------------------------------------------------

def print_performance(pnl: pd.Series, initial_capital: float) -> dict:
    """Print and return a dictionary of key performance metrics."""
    pnl_dollar = pnl * initial_capital
    cum = pnl.cumsum()
    total_return = float(cum.iloc[-1]) if len(cum) > 0 else 0.0
    total_pnl = total_return * initial_capital

    n_days = len(pnl)
    ann_factor = 252
    ann_return = float(pnl.mean() * ann_factor) if n_days > 0 else 0.0
    ann_vol = float(pnl.std() * np.sqrt(ann_factor)) if n_days > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum_peak = cum.cummax()
    drawdown = cum - cum_peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Win rate
    wins = (pnl > 0).sum()
    total_days = (pnl != 0).sum()
    win_rate = float(wins / total_days) if total_days > 0 else 0.0

    metrics = {
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }

    print("\n" + "=" * 60)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Total P&L:              ${total_pnl:>14,.2f}")
    print(f"  Total Return:           {total_return:>14.4%}")
    print(f"  Annualised Return:      {ann_return:>14.4%}")
    print(f"  Annualised Volatility:  {ann_vol:>14.4%}")
    print(f"  Sharpe Ratio:           {sharpe:>14.3f}")
    print(f"  Max Drawdown:           {max_dd:>14.4%}")
    print(f"  Win Rate:               {win_rate:>14.2%}")
    print("=" * 60 + "\n")

    return metrics


def _compute_earnings_season_metrics(
    pnl: pd.Series,
    season_windows: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Compute performance slices for each earnings season window.

    Returns a table with one row per season containing total return,
    annualized return/volatility, Sharpe, and season-day count.
    """
    if pnl.empty or season_windows.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "start_date",
                "end_date",
                "days",
                "total_return",
                "annualised_return",
                "annualised_volatility",
                "sharpe_ratio",
                "total_pnl",
            ]
        )

    rows: list[dict] = []
    ann_factor = 252
    for _, row in season_windows.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        season_name = str(row["season"])

        segment = pnl[(pnl.index >= start) & (pnl.index <= end)]
        if segment.empty:
            continue

        total_return = float(segment.sum())
        ann_return = float(segment.mean() * ann_factor)
        ann_vol = float(segment.std() * np.sqrt(ann_factor)) if len(segment) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        rows.append(
            {
                "season": season_name,
                "start_date": start,
                "end_date": end,
                "days": int(len(segment)),
                "total_return": total_return,
                "annualised_return": ann_return,
                "annualised_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "total_pnl": float(total_return * initial_capital),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--live", action="store_true",
                        help="Open live matplotlib dashboard during backtest")
    parser.add_argument(
        "--enable-llm-overlay",
        action="store_true",
        help="Enable thesis-aware LLM overlay adjustments",
    )
    parser.add_argument(
        "--llm-thesis-dir",
        default="Investment Theses",
        help="Directory containing thesis documents",
    )
    parser.add_argument(
        "--llm-member-selections",
        default=None,
        help="Optional JSON/CSV file with member stock selections and base weights",
    )
    args = parser.parse_args()

    print("Loading config...")
    config = _load_config()
    initial_capital = float(config.get("initial_capital", 1_000_000))
    if args.enable_llm_overlay:
        config["llm_overlay_enabled"] = True
        config["llm_thesis_dir"] = args.llm_thesis_dir
        config["llm_member_selection_file"] = args.llm_member_selections
        print("  ▶ LLM thesis overlay enabled for this run")

    print("Building portfolio weights (Alpha + Risk + PreEarnings)...")
    if args.live:
        print("  ▶ Live dashboard mode enabled")
    weights = build_portfolio_matrix(config, live=args.live)
    print(f"  → {weights.shape[0]} rebalance dates × {weights.shape[1]} tickers")

    print("Simulating P&L...")
    returns = get_returns_pivot()
    pnl = simulate_pnl(weights, returns)

    metrics = print_performance(pnl, initial_capital)

    # Earnings-season slice metrics (optional, always attempted).
    fetcher = LLMDataFetcher()
    seasons = fetcher.build_earnings_season_windows()
    season_metrics = _compute_earnings_season_metrics(
        pnl=pnl,
        season_windows=seasons,
        initial_capital=initial_capital,
    )

    # Save outputs
    out_dir = _outputs_dir()
    weights.to_csv(out_dir / "portfolio_weights.csv")
    pnl.to_csv(out_dir / "pnl.csv", header=True)
    if not season_metrics.empty:
        season_metrics.to_csv(out_dir / "earnings_season_metrics.csv", index=False)
        print(f"Earnings season metrics saved to {out_dir / 'earnings_season_metrics.csv'}")
    print(f"Outputs saved to {out_dir}/")
    print("Done.")
