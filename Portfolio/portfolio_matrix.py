"""
portfolio_matrix.py — T × N Portfolio Weights Matrix

Main loop: iterate over all rebalance dates, call the optimizer at each
step, and assemble the full T × N matrix of portfolio weights.
This matrix is the primary output consumed by backtest.py.

Algorithm per rebalance date t:
    1. Fetch alpha_t from Core/signal_blender.py     → (N,) vector
    2. Fetch Sigma_t from Risk/risk_matrix.py         → (N, N) matrix
    3. Solve optimisation via Portfolio/optimizer.py  → (N,) weight vector
    4. Store weights and advance w_prev = w_t

Inputs:
    Core/signal_blender.py    →  SignalBlender
    Risk/risk_matrix.py       →  RiskMatrixBuilder
    Portfolio/optimizer.py    →  optimize_portfolio()
    utils/universe.py         →  get_universe()
    config.json               →  all hyperparameters

Outputs:
    build_portfolio_matrix(config) →  pd.DataFrame shape (T, N)
                                      index = rebalance_dates
                                      columns = tickers
                                      values = portfolio weights
                                      (sum ≈ 0, sum(|w|) <= max_leverage)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from Core.signal_blender import SignalBlender
from Portfolio.optimizer import optimize_portfolio
from Risk.risk_matrix import RiskMatrixBuilder
from utils.data_loader import get_returns_pivot, get_tickers


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = _project_root() / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_portfolio_matrix(config: dict | None = None, live: bool = False) -> pd.DataFrame:
    """Build the full T × N weights matrix.

    Steps:
        1. Load config and determine universe
        2. Fit RiskMatrixBuilder
        3. Determine rebalance dates (weekly, post risk-warmup)
        4. Build SignalBlender (alpha + pre-earnings)
        5. Walk-forward: optimise at each rebalance date
        6. Return pd.DataFrame of weights

    Parameters
    ----------
    config : dict | None
        Override configuration.  If None, reads from config.json.
    live : bool
        If True, open a real-time matplotlib dashboard showing Alpha and
        Risk matrices updating at every rebalance.

    Returns
    -------
    pd.DataFrame
        Shape (T, N), index = rebalance dates, columns = tickers,
        values = portfolio weights.
    """
    if config is None:
        config = _load_config()

    # --- Universe ---
    tickers = get_tickers()
    if not tickers:
        raise RuntimeError("No tickers found in price data")

    # --- Returns & Risk ---
    t0 = time.time()
    print("  Loading price data & building returns...")
    returns = get_returns_pivot()
    # Align tickers to those present in returns
    tickers = sorted(set(tickers) & set(returns.columns))
    returns = returns[tickers]
    print(f"    → {len(tickers)} tickers, {len(returns)} trading days  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("  Fitting EWMA covariance model...")
    risk_builder = RiskMatrixBuilder(
        lambda_ewma=float(config.get("lambda_ewma", 0.94)),
        min_periods=int(config.get("ewma_min_periods", 60)),
    )
    risk_builder.fit(tickers=tickers, returns=returns)
    print(f"    → reliable from {risk_builder.reliable_from}  ({time.time() - t0:.1f}s)")

    # --- Rebalance dates (weekly on Fridays, after risk warmup) ---
    start_date = pd.Timestamp(config.get("start_date", "2018-01-01"))
    end_date = pd.Timestamp(config.get("end_date", "2025-12-31"))

    all_dates = returns.index
    reliable_from = risk_builder.reliable_from
    if reliable_from is not None:
        start_date = max(start_date, reliable_from)

    eligible_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    # Weekly rebalance (every 5th trading day)
    rebalance_dates = eligible_dates[::5]
    if len(rebalance_dates) == 0:
        raise RuntimeError("No valid rebalance dates after risk warmup")

    # --- Signal blender (Alpha + PreEarnings) ---
    t0 = time.time()
    print("  Building alpha signals...")
    blender = SignalBlender(
        tickers=tickers,
        rebalance_dates=pd.DatetimeIndex(rebalance_dates),
        blend_weight_systematic=float(config.get("blend_weight_systematic", 0.6)),
        blend_weight_event=float(config.get("blend_weight_event", 0.25)),
        blend_weight_thesis=float(config.get("blend_weight_thesis", 0.15)),
        pre_earnings_window=int(config.get("pre_earnings_window", int(config.get("holding_period_days", 10)) // 2)),
        config=config,
    )
    print(f"    → alpha matrix built  ({time.time() - t0:.1f}s)")

    # --- Walk-forward optimisation ---
    lambda_risk = float(config.get("lambda_risk_aversion", 1.0))
    tc = float(config.get("transaction_cost", 0.001))
    max_lev = float(config.get("max_leverage", 2.0))
    max_pos = float(config.get("max_position_pct", 0.05))

    n = len(tickers)
    w_prev = np.zeros(n)
    weight_rows: list[np.ndarray] = []
    total = len(rebalance_dates)

    # --- Live dashboard ---
    dashboard = None
    if live:
        from Visualisation.live_dashboard import LiveDashboard
        dashboard = LiveDashboard(tickers=tickers)
        dashboard.open()
        print("  ▶ Live dashboard opened")

    print(f"  Universe: {n} tickers, {total} rebalance dates")
    for idx, date in enumerate(rebalance_dates):
        if idx % 25 == 0 or idx == total - 1:
            print(f"  Rebalancing {idx + 1}/{total}  ({date.strftime('%Y-%m-%d')})...")

        alpha = blender.get_alpha_vector(date)

        try:
            sigma = risk_builder.get(date)
        except KeyError:
            weight_rows.append(w_prev.copy())
            continue

        w_opt = optimize_portfolio(
            alpha=alpha,
            sigma=sigma,
            w_prev=w_prev,
            lambda_risk=lambda_risk,
            transaction_cost=tc,
            max_leverage=max_lev,
            max_position=max_pos,
        )
        weight_rows.append(w_opt)
        w_prev = w_opt

        # --- Update live dashboard ---
        if dashboard is not None:
            dashboard.update(
                date=date, alpha=alpha, sigma=sigma, weights=w_opt,
            )

    # --- Finalise dashboard ---
    if dashboard is not None:
        dashboard.save("live_dashboard_final.png")
        print(f"  ▶ Dashboard snapshot saved to {dashboard.output_dir}/live_dashboard_final.png")
        print("  ▶ Close the dashboard window to continue...")
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()  # blocks until user closes the window

    weights_df = pd.DataFrame(
        weight_rows,
        index=rebalance_dates,
        columns=tickers,
    )
    weights_df.index.name = "rebalance_date"
    return weights_df
