"""
signal_blender.py — Blends Systematic Alpha with Event-Driven PreEarnings
and discretionary thesis NLP overlay.

Produces a single (N,) alpha vector per rebalance date that the portfolio
optimizer consumes.

Logic per date:
    1. Start from the systematic SUE alpha row (cross-sectionally z-scored).
    2. For tickers within `pre_earnings_window` days of an earnings event,
       overlay the PreEarnings direction × size as an additive event signal.
    3. Overlay discretionary thesis NLP scores where provided.
    4. Blend weights: w_sys, w_evt, w_thesis (automatically re-scaled across
       the active components; the configured weights do not need to sum to 1).

When no overlays are active on a given date, the blended alpha collapses to
the systematic SUE signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from Alpha.alpha_matrix import build_alpha_matrix
from Core.thesis_nlp import ThesisNLPOverlay
from utils.data_loader import load_earnings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = _project_root() / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SignalBlender:
    """Combines systematic SUE alpha with event-driven and thesis overlays.

    Parameters
    ----------
    tickers : list[str]
        Canonical ordered ticker universe (shared with Risk and Portfolio).
    rebalance_dates : pd.DatetimeIndex
        Full set of rebalance dates (used to pre-build the alpha matrix).
    blend_weight_systematic : float
        Weight on the systematic (SUE) signal.  Default 0.6.
    blend_weight_event : float
        Weight on the event (PreEarnings) overlay.  Default 0.25.
    blend_weight_thesis : float
        Weight on discretionary thesis NLP overlay. Default 0.15.
    pre_earnings_window : int
        Number of calendar days before an earnings event during which the
        pre-earnings overlay is active.  Default 5.
    config : dict | None
        Override config dict; if None, read from config.json.
    """

    def __init__(
        self,
        tickers: list[str],
        rebalance_dates: pd.DatetimeIndex,
        blend_weight_systematic: float = 0.6,
        blend_weight_event: float = 0.25,
        blend_weight_thesis: float = 0.15,
        pre_earnings_window: int = 5,
        config: dict | None = None,
    ):
        self.tickers = list(tickers)
        self.n = len(self.tickers)
        self.w_sys = float(blend_weight_systematic)
        self.w_evt = float(blend_weight_event)
        self.w_thesis = float(blend_weight_thesis)
        self.window = int(pre_earnings_window)
        self.config = config or _load_config()

        # Pre-build the full systematic alpha matrix once.
        self._alpha_matrix: pd.DataFrame = build_alpha_matrix(
            rebalance_dates=rebalance_dates,
            tickers=self.tickers,
            lookback_quarters=int(self.config.get("sue_lookback_quarters", 8)),
            use_ic_weighting=True,
        )

        # Pre-build earnings lookup dict: {ticker -> sorted list of dates}
        earnings = load_earnings()[["ticker", "event_date"]].copy()
        earnings["ticker"] = earnings["ticker"].astype(str).str.strip().str.upper()
        earnings["event_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
        earnings = earnings.dropna(subset=["event_date"])
        self._earnings_dict: dict[str, list[pd.Timestamp]] = (
            earnings.groupby("ticker")["event_date"]
            .apply(lambda s: sorted(s.tolist()))
            .to_dict()
        )

        # Lazy-initialise PreEarningsSignal (may not be available if options
        # data is missing — in that case fall back to pure systematic).
        self._pre_earnings: Optional[object] = None
        self._pre_earnings_available: Optional[bool] = None
        self._pre_earnings_tried: bool = False

        # Optional discretionary overlay from investment theses.
        self._thesis_overlay = ThesisNLPOverlay.from_config(
            config=self.config,
            tickers=self.tickers,
        )

    def _get_pre_earnings_signal(self):
        """Lazy-load PreEarningsSignal; returns None if import or init fails."""
        if self._pre_earnings_available is False:
            return None
        if self._pre_earnings is not None:
            return self._pre_earnings
        try:
            from PreEarnings.pre_earnings_signal import PreEarningsSignal
            self._pre_earnings = PreEarningsSignal(config=self.config)
            self._pre_earnings_available = True
            return self._pre_earnings
        except Exception:
            self._pre_earnings_available = False
            return None

    def _next_earnings(self, ticker: str, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Return the next earnings date for *ticker* on or after *date*."""
        dates = self._earnings_dict.get(ticker)
        if not dates:
            return None
        # Binary search for the first date >= target
        import bisect
        idx = bisect.bisect_left(dates, date)
        if idx < len(dates):
            return pd.Timestamp(dates[idx])
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_alpha_vector(self, date: pd.Timestamp) -> np.ndarray:
        """Return the (N,) blended alpha vector for a given rebalance date.

        If the date is not in the alpha matrix, the nearest earlier date is
        used.  If no pre-earnings overlay is active, the pure systematic
        vector is returned.
        """
        date = pd.Timestamp(date)

        # --- Systematic alpha (SUE) ---
        if date in self._alpha_matrix.index:
            alpha_row = self._alpha_matrix.loc[date].values.astype(float)
        else:
            # Fall back to the most recent available row.
            earlier = self._alpha_matrix.index[self._alpha_matrix.index <= date]
            if len(earlier) == 0:
                alpha_row = np.zeros(self.n)
            else:
                alpha_row = self._alpha_matrix.loc[earlier[-1]].values.astype(float)

        alpha_row = np.asarray(alpha_row, dtype=float)
        weighted_sum = np.zeros_like(alpha_row)
        total_weight = 0.0  # scalar sum of component weights applied; re-scaled across active components
        if self.w_sys > 0:
            weighted_sum += self.w_sys * alpha_row
            total_weight += self.w_sys

        # --- Event overlay (PreEarnings) ---
        # After the first failed attempt, stop retrying to avoid
        # per-ticker overhead when options data is missing.
        event_overlay = None
        if not (self._pre_earnings_tried and not self._pre_earnings_available):
            pre_earnings = self._get_pre_earnings_signal()
            if pre_earnings is not None:
                overlay_vec = np.zeros(self.n)
                has_overlay = False
                failed_count = 0

                for i, tk in enumerate(self.tickers):
                    # Skip tickers with no earnings in the calendar
                    if tk not in self._earnings_dict:
                        continue
                    upcoming = self._next_earnings(tk, date)
                    if upcoming is None:
                        continue
                    days_to_event = (upcoming - date).days
                    if 0 < days_to_event <= self.window:
                        try:
                            output = pre_earnings.generate(
                                ticker=tk,
                                event_date=upcoming,
                                options_df=None,
                                prices_df=None,
                                earnings_df=None,
                            )
                            if output.direction is not None and output.position_size > 0:
                                direction_sign = 1.0 if output.direction == "long" else -1.0
                                overlay_vec[i] = direction_sign * output.position_size
                                has_overlay = True
                            else:
                                failed_count += 1
                        except Exception:
                            failed_count += 1

                        # If the first 3 attempts all fail, options data is missing;
                        # bail out for all future dates.
                        if failed_count >= 3 and not has_overlay:
                            self._pre_earnings_tried = True
                            self._pre_earnings_available = False
                            break

                if has_overlay:
                    ev_nonzero = overlay_vec[overlay_vec != 0]
                    ev_std = float(np.std(ev_nonzero)) if len(ev_nonzero) > 1 else 1.0
                    if ev_std > 0:
                        overlay_vec = overlay_vec / ev_std
                    event_overlay = overlay_vec

        if event_overlay is not None:
            weighted_sum += self.w_evt * event_overlay
            total_weight += self.w_evt

        # --- Thesis overlay (NLP) ---
        if self._thesis_overlay is not None:
            thesis_vector = self._thesis_overlay.get_overlay(date)
            if thesis_vector is not None:
                weighted_sum += self.w_thesis * thesis_vector
                total_weight += self.w_thesis

        if total_weight <= 1e-10:
            return alpha_row
        return weighted_sum / total_weight
