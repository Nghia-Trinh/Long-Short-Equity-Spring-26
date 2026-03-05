"""
live_dashboard.py — Real-time 6-panel matplotlib dashboard

Shows Alpha and Risk matrices updating live during the walk-forward
backtest.  Uses matplotlib interactive mode (plt.ion()) with blitting
for fast redraws.

Layout (2 × 3 grid):
    ┌────────────────────┬──────────────────────┬──────────────────────┐
    │  Alpha Top/Bottom  │  Risk Σ Heatmap      │  Portfolio Weights   │
    │  bar chart (top20) │  (top 20×20 block)   │  bar chart           │
    ├────────────────────┼──────────────────────┼──────────────────────┤
    │  Alpha Dispersion  │  Risk Eigenvalues    │  Cumulative PnL      │
    │  time series       │  top-5 over time     │  (if available)       │
    └────────────────────┴──────────────────────┴──────────────────────┘

Usage:
    dashboard = LiveDashboard(tickers)
    dashboard.open()
    for date in rebalance_dates:
        dashboard.update(date, alpha, sigma, weights, pnl_so_far)
    dashboard.save()
    dashboard.close()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


class LiveDashboard:
    """Real-time 6-panel backtest dashboard."""

    def __init__(self, tickers: list[str], top_k: int = 20,
                 output_dir: str | Path = "outputs/charts"):
        self.tickers = list(tickers)
        self.n = len(tickers)
        self.top_k = min(top_k, self.n)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # history buffers
        self._dates: list[pd.Timestamp] = []
        self._alpha_dispersions: list[float] = []
        self._eigenvalue_history: list[np.ndarray] = []
        self._cum_pnl: list[float] = []
        self._cum_ret: float = 0.0

        self._fig = None
        self._axes = None
        self._opened = False

    # ------------------------------------------------------------------ #
    #  Open / close window                                                 #
    # ------------------------------------------------------------------ #

    def open(self):
        """Create the figure and 6 subplots."""
        plt.ion()
        self._fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        self._fig.canvas.manager.set_window_title("Live Backtest Dashboard")
        self._fig.suptitle("Backtest Dashboard — initialising...",
                           fontsize=14, fontweight="bold")

        self._axes = {
            "alpha_bar":    axes[0, 0],
            "risk_heatmap": axes[0, 1],
            "weight_bar":   axes[0, 2],
            "alpha_disp":   axes[1, 0],
            "risk_eig":     axes[1, 1],
            "cum_pnl":      axes[1, 2],
        }

        # Static axis labels
        self._axes["alpha_bar"].set_title("Alpha — Top/Bottom Tickers", fontsize=10)
        self._axes["risk_heatmap"].set_title("Risk Σ — Correlation Block", fontsize=10)
        self._axes["weight_bar"].set_title("Portfolio Weights", fontsize=10)
        self._axes["alpha_disp"].set_title("Alpha Dispersion (σ_cs)", fontsize=10)
        self._axes["risk_eig"].set_title("Risk — Top 5 Eigenvalues", fontsize=10)
        self._axes["cum_pnl"].set_title("Cumulative PnL", fontsize=10)

        self._fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.pause(0.1)
        self._opened = True

    def close(self):
        """Close the dashboard window."""
        if self._opened:
            plt.ioff()
            plt.close(self._fig)
            self._opened = False

    def save(self, filename: str = "live_dashboard_final.png"):
        """Save the current dashboard state as an image."""
        if self._fig is not None:
            self._fig.savefig(self.output_dir / filename, dpi=150,
                              bbox_inches="tight")

    # ------------------------------------------------------------------ #
    #  Main update — called once per rebalance                             #
    # ------------------------------------------------------------------ #

    def update(
        self,
        date: pd.Timestamp,
        alpha: np.ndarray,
        sigma: np.ndarray,
        weights: np.ndarray,
        daily_pnl: float = 0.0,
    ):
        """Refresh all 6 panels with the latest rebalance data."""
        if not self._opened:
            return

        self._dates.append(date)
        self._cum_ret += daily_pnl
        self._cum_pnl.append(self._cum_ret)

        self._fig.suptitle(
            f"Live Backtest — {date.strftime('%Y-%m-%d')}  "
            f"({len(self._dates)} rebalances)",
            fontsize=14, fontweight="bold",
        )

        self._update_alpha_bar(alpha)
        self._update_risk_heatmap(sigma)
        self._update_weight_bar(weights)
        self._update_alpha_dispersion(alpha)
        self._update_risk_eigenvalues(sigma)
        self._update_cum_pnl()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

    # ------------------------------------------------------------------ #
    #  Panel updaters                                                      #
    # ------------------------------------------------------------------ #

    def _update_alpha_bar(self, alpha: np.ndarray):
        """Horizontal bar chart of top-10 + bottom-10 alpha scores."""
        ax = self._axes["alpha_bar"]
        ax.clear()

        k = min(10, self.n // 2)
        order = np.argsort(alpha)
        bot_idx = order[:k]
        top_idx = order[-k:][::-1]
        show_idx = np.concatenate([top_idx, bot_idx])

        labels = [self.tickers[i] for i in show_idx]
        values = alpha[show_idx]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

        ax.barh(range(len(labels)), values, color=colors, height=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("Alpha — Top/Bottom Tickers", fontsize=10)
        ax.invert_yaxis()

    def _update_risk_heatmap(self, sigma: np.ndarray):
        """Show a correlation matrix heatmap for the top-20 variance tickers."""
        ax = self._axes["risk_heatmap"]
        ax.clear()

        k = min(self.top_k, sigma.shape[0])
        # Pick tickers with highest marginal variance
        diag = np.diag(sigma)
        top_var_idx = np.argsort(diag)[-k:][::-1]

        # Extract sub-block and convert to correlation
        sub_cov = sigma[np.ix_(top_var_idx, top_var_idx)]
        stds = np.sqrt(np.diag(sub_cov))
        stds[stds == 0] = 1.0
        corr = sub_cov / np.outer(stds, stds)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1,
                        interpolation="nearest", aspect="auto")
        labels = [self.tickers[i] for i in top_var_idx]
        ax.set_xticks(range(k))
        ax.set_xticklabels(labels, fontsize=5, rotation=90)
        ax.set_yticks(range(k))
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_title("Risk Σ — Correlation Block (top var)", fontsize=10)

    def _update_weight_bar(self, weights: np.ndarray):
        """Bar chart of current portfolio weights (non-zero only)."""
        ax = self._axes["weight_bar"]
        ax.clear()

        nonzero_mask = np.abs(weights) > 1e-6
        if not np.any(nonzero_mask):
            ax.text(0.5, 0.5, "No positions", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("Portfolio Weights", fontsize=10)
            return

        # Show top 10 + bottom 10 by weight
        k = min(10, nonzero_mask.sum() // 2, self.n)
        order = np.argsort(weights)
        bot_idx = order[:k]
        top_idx = order[-k:][::-1]
        show_idx = np.concatenate([top_idx, bot_idx])

        labels = [self.tickers[i] for i in show_idx]
        vals = weights[show_idx]
        colors = ["#27ae60" if v > 0 else "#c0392b" for v in vals]

        ax.barh(range(len(labels)), vals, color=colors, height=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("Portfolio Weights", fontsize=10)
        ax.invert_yaxis()

    def _update_alpha_dispersion(self, alpha: np.ndarray):
        """Running plot of cross-sectional alpha standard deviation."""
        ax = self._axes["alpha_disp"]
        ax.clear()

        valid = alpha[~np.isnan(alpha)]
        disp = float(np.std(valid)) if len(valid) > 1 else 0.0
        self._alpha_dispersions.append(disp)

        ax.fill_between(self._dates, 0, self._alpha_dispersions,
                         alpha=0.3, color="steelblue")
        ax.plot(self._dates, self._alpha_dispersions,
                color="steelblue", linewidth=1.2)
        ax.set_title("Alpha Dispersion (σ_cs)", fontsize=10)
        ax.set_ylabel("Std")
        if len(self._dates) > 1:
            ax.tick_params(axis="x", rotation=30, labelsize=7)

    def _update_risk_eigenvalues(self, sigma: np.ndarray):
        """Running plot of top-5 eigenvalues of Σ."""
        ax = self._axes["risk_eig"]
        ax.clear()

        try:
            eigvals = np.sort(np.linalg.eigvalsh(sigma))[::-1]
        except Exception:
            eigvals = np.zeros(5)

        self._eigenvalue_history.append(eigvals[:5].copy())

        eig_arr = np.array(self._eigenvalue_history)  # (T, 5)
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
        for i in range(min(5, eig_arr.shape[1])):
            ax.plot(self._dates, eig_arr[:, i], color=colors[i],
                    linewidth=1.2, label=f"λ{i+1}")

        ax.set_yscale("log")
        ax.set_title("Risk — Top 5 Eigenvalues", fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=3)
        if len(self._dates) > 1:
            ax.tick_params(axis="x", rotation=30, labelsize=7)

    def _update_cum_pnl(self):
        """Running cumulative PnL line."""
        ax = self._axes["cum_pnl"]
        ax.clear()

        if len(self._cum_pnl) < 2:
            ax.text(0.5, 0.5, "Waiting for PnL data...", ha="center",
                    va="center", transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title("Cumulative PnL", fontsize=10)
            return

        vals = np.array(self._cum_pnl)
        color = "#27ae60" if vals[-1] >= 0 else "#e74c3c"
        ax.fill_between(self._dates, 0, vals, alpha=0.2, color=color)
        ax.plot(self._dates, vals, color=color, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Cumulative PnL — {vals[-1]:.4%}", fontsize=10)
        ax.set_ylabel("Cumulative Return")
        if len(self._dates) > 1:
            ax.tick_params(axis="x", rotation=30, labelsize=7)
