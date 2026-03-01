from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .feature_engine import compute_event_features
from .iv_skew import IVSkewSignal
from .normalizer import robust_z
from .options_loader import load_options_for_events
from .sentiment_sizer import compute_modifier
from .setup_gate import evaluate
from .volume_signal import compute_volume_signal
from utils.data_loader import load_earnings, load_prices


class PreEarningsOutput(NamedTuple):
    direction: str | None
    position_size: float
    diagnostics: dict


class PreEarningsSignal:
    def __init__(self, config: dict):
        self.config = dict(config or {})
        self.iv_skew_signal = IVSkewSignal(
            skew_lookback_days=int(self.config.get("skew_lookback_days", 5)),
            zscore_window=int(self.config.get("zscore_window", 252)),
        )

    @staticmethod
    def _direction_from_skew(skew_score: float) -> str | None:
        if pd.isna(skew_score):
            return None
        # rr_norm > 0 means puts are rich vs calls; use contrarian long mapping.
        return "long" if float(skew_score) > 0.0 else "short"

    @staticmethod
    def _lookup_volume_score(ticker: str, event_date: pd.Timestamp) -> float:
        try:
            vol_df = compute_volume_signal()
        except Exception:
            return np.nan
        if vol_df.empty:
            return np.nan
        view = vol_df[
            (vol_df["ticker"].astype(str) == str(ticker))
            & (pd.to_datetime(vol_df["event_date"], errors="coerce") == event_date)
        ]
        if view.empty:
            return np.nan
        return float(pd.to_numeric(view.iloc[-1]["volume_score"], errors="coerce"))

    def generate(
        self,
        ticker,
        event_date,
        options_df: pd.DataFrame | None,
        prices_df: pd.DataFrame | None,
        earnings_df: pd.DataFrame | None,
    ) -> PreEarningsOutput:
        ticker_norm = str(ticker).upper().strip()
        event_ts = pd.to_datetime(event_date, errors="coerce")
        if pd.isna(event_ts):
            return PreEarningsOutput(
                direction=None,
                position_size=0.0,
                diagnostics={"error": "invalid_event_date"},
            )

        earnings = earnings_df.copy() if earnings_df is not None else load_earnings()
        prices = prices_df.copy() if prices_df is not None else load_prices()
        if options_df is None:
            options = load_options_for_events(earnings_df=earnings, delta_targets=None)
        else:
            options = options_df.copy()

        feature_df = compute_event_features(
            ticker=ticker_norm,
            options_df=options,
            prices_df=prices,
            earnings_df=earnings,
            window_days=int(self.config.get("preearnings_window_days", 7)),
        )
        if feature_df.empty:
            return PreEarningsOutput(
                direction=None,
                position_size=0.0,
                diagnostics={"error": "no_features"},
            )

        feature_df["event_date"] = pd.to_datetime(feature_df["event_date"], errors="coerce")
        feature_df = feature_df.sort_values("event_date").reset_index(drop=True)
        if event_ts not in set(feature_df["event_date"]):
            return PreEarningsOutput(
                direction=None,
                position_size=0.0,
                diagnostics={"error": "event_not_found"},
            )

        feature_df["z_drift"] = robust_z(feature_df["price_drift"], winsor_sigma=float(self.config.get("winsor_sigma", 3.0)))
        feature_df["z_vol"] = robust_z(feature_df["rv20d_vs_pre1d"], winsor_sigma=float(self.config.get("winsor_sigma", 3.0)))
        feature_df["z_nd"] = robust_z(feature_df["net_delta"], winsor_sigma=float(self.config.get("winsor_sigma", 3.0)))

        cpr_series = pd.to_numeric(feature_df["call_put_ratio"], errors="coerce")
        cpr_log = np.where(cpr_series > 0, np.log(cpr_series), np.nan)
        feature_df["z_cpr"] = robust_z(pd.Series(cpr_log, index=feature_df.index), winsor_sigma=float(self.config.get("winsor_sigma", 3.0)))

        event_row = feature_df[feature_df["event_date"] == event_ts].iloc[-1]
        if not bool(event_row.get("has_options", False)):
            return PreEarningsOutput(
                direction=None,
                position_size=0.0,
                diagnostics={
                    "gate_result": {"tradable": False, "reason": "no_options_coverage"},
                    "coverage_flags": {
                        "has_options": False,
                        "missing_net_delta": True,
                        "missing_call_put_ratio": True,
                    },
                },
            )

        gate = evaluate(
            z_drift=float(event_row["z_drift"]) if pd.notna(event_row["z_drift"]) else np.nan,
            z_vol=float(event_row["z_vol"]) if pd.notna(event_row["z_vol"]) else np.nan,
            drift_threshold=float(self.config.get("gate_drift_threshold", -1.0)),
            vol_threshold=float(self.config.get("gate_vol_threshold", 0.5)),
        )

        single_event_earnings = pd.DataFrame([{"ticker": ticker_norm, "event_date": event_ts}])
        skew_df = self.iv_skew_signal.compute(single_event_earnings, options)
        skew_score = np.nan
        if not skew_df.empty:
            skew_match = skew_df[
                (skew_df["ticker"].astype(str) == ticker_norm)
                & (pd.to_datetime(skew_df["event_date"], errors="coerce") == event_ts)
            ]
            if not skew_match.empty:
                skew_score = float(pd.to_numeric(skew_match.iloc[-1]["skew_score"], errors="coerce"))
        volume_score = self._lookup_volume_score(ticker=ticker_norm, event_date=event_ts)

        direction = self._direction_from_skew(skew_score)
        if not gate.tradable or direction is None:
            return PreEarningsOutput(
                direction=None,
                position_size=0.0,
                diagnostics={
                    "gate_result": {"tradable": gate.tradable, "reason": gate.reason},
                    "core_signal": {"skew_score": skew_score, "volume_score": volume_score},
                    "coverage_flags": {
                        "has_options": True,
                        "missing_net_delta": bool(pd.isna(event_row["net_delta"])),
                        "missing_call_put_ratio": bool(pd.isna(event_row["call_put_ratio"])),
                    },
                },
            )

        sizing = compute_modifier(
            z_nd=float(event_row["z_nd"]) if pd.notna(event_row["z_nd"]) else None,
            z_cpr=float(event_row["z_cpr"]) if pd.notna(event_row["z_cpr"]) else None,
            z_vol=float(event_row["z_vol"]) if pd.notna(event_row["z_vol"]) else np.nan,
            a=float(self.config.get("sizer_delta_a", 0.3)),
            b=float(self.config.get("sizer_cpr_b", 0.2)),
            k=float(self.config.get("sizer_vol_k", 0.5)),
        )

        base_position_size = float(self.config.get("base_position_size", 1.0))
        position_size = float(base_position_size * sizing.multiplier)
        return PreEarningsOutput(
            direction=direction,
            position_size=position_size,
            diagnostics={
                "gate_result": {"tradable": gate.tradable, "reason": gate.reason},
                "sizing_result": {
                    "multiplier": sizing.multiplier,
                    "w_delta": sizing.w_delta,
                    "w_cpr": sizing.w_cpr,
                    "risk_scale": sizing.risk_scale,
                },
                "core_signal": {"skew_score": skew_score, "volume_score": volume_score},
                "coverage_flags": {
                    "has_options": True,
                    "missing_net_delta": bool(pd.isna(event_row["net_delta"])),
                    "missing_call_put_ratio": bool(pd.isna(event_row["call_put_ratio"])),
                },
            },
        )
