"""
iv_skew.py — Pre-Earnings Options IV Skew Signal  [STUB — not yet implemented]

Intuition:
    Fear before earnings drives investors to buy protective puts, inflating
    downside IV relative to upside IV. Extreme skew is a contrarian signal.

Signal:
    25-delta risk reversal:  RR = put_25d_IV - call_25d_IV
    Normalised by ATM IV  →  dimensionless skew score
    Extreme negative (expensive puts)  → contrarian long bias
    Extreme positive (expensive calls) → contrarian short bias

Data required (NOT YET AVAILABLE):
    Options chain per ticker per day:
        columns: ticker, trade_date, strike, expiry, call_iv, put_iv, delta

Expected outputs:
    IVSkewSignal.compute(earnings_dates) → pd.DataFrame
        columns: ticker, event_date, skew_score

Integration:
    See PreEarnings/README.md for composite alpha weighting plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IVSkewSignal:
    def __init__(self, skew_lookback_days: int = 5, zscore_window: int = 252):
        self.skew_lookback_days = int(skew_lookback_days)
        self.zscore_window = int(zscore_window)

    def load_options_data(self, path) -> pd.DataFrame:
        path = str(path)
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        elif "date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["date"], errors="coerce")

        if "exdate" in df.columns:
            df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
        elif "expiry" in df.columns:
            df["exdate"] = pd.to_datetime(df["expiry"], errors="coerce")

        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str)

        return df

    def _to_long_iv(self, options_df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in options_df.columns}
        df = options_df.copy()

        if "trade_date" not in df.columns and "date" in cols:
            df["trade_date"] = pd.to_datetime(df[cols["date"]], errors="coerce")
        if "exdate" not in df.columns and "expiry" in cols:
            df["exdate"] = pd.to_datetime(df[cols["expiry"]], errors="coerce")
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        if "exdate" in df.columns:
            df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str)

        if {"option_type", "iv", "delta"}.issubset(df.columns):
            out = df[["ticker", "trade_date", "exdate", "option_type", "iv", "delta"]].copy()
            out["option_type"] = out["option_type"].astype(str).str.lower()
            out["iv"] = pd.to_numeric(out["iv"], errors="coerce")
            out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
            return out.dropna(subset=["ticker", "trade_date", "exdate", "iv", "delta"])

        has_call_put = {"call_iv", "put_iv", "delta"}.issubset(df.columns)
        if not has_call_put:
            return pd.DataFrame(columns=["ticker", "trade_date", "exdate", "option_type", "iv", "delta"])

        base_cols = ["ticker", "trade_date", "exdate", "delta"]
        base = df[base_cols].copy()
        base["delta"] = pd.to_numeric(base["delta"], errors="coerce")

        calls = base.copy()
        calls["option_type"] = "call"
        calls["iv"] = pd.to_numeric(df["call_iv"], errors="coerce")
        calls["delta"] = calls["delta"].abs()

        puts = base.copy()
        puts["option_type"] = "put"
        puts["iv"] = pd.to_numeric(df["put_iv"], errors="coerce")
        puts["delta"] = -puts["delta"].abs()

        out = pd.concat([calls, puts], ignore_index=True)
        return out.dropna(subset=["ticker", "trade_date", "exdate", "iv", "delta"])

    def compute_risk_reversal(self, options_df) -> pd.DataFrame:
        long_df = self._to_long_iv(options_df)
        if long_df.empty:
            return pd.DataFrame(
                columns=["ticker", "trade_date", "put_25d_iv", "call_25d_iv", "atm_iv", "rr", "rr_norm"]
            )

        records = []
        grouped = long_df.groupby(["ticker", "trade_date"], sort=False)
        for (ticker, trade_date), group in grouped:
            puts = group[group["option_type"] == "put"]
            calls = group[group["option_type"] == "call"]

            put_iv = np.nan
            if not puts.empty:
                put_idx = (puts["delta"].abs() - 0.25).abs().idxmin()
                put_iv = float(puts.loc[put_idx, "iv"])

            call_iv = np.nan
            if not calls.empty:
                call_idx = (calls["delta"].abs() - 0.25).abs().idxmin()
                call_iv = float(calls.loc[call_idx, "iv"])

            atm_iv = np.nan
            if not group.empty:
                atm_idx = (group["delta"].abs() - 0.50).abs().idxmin()
                atm_iv = float(group.loc[atm_idx, "iv"])

            rr = put_iv - call_iv if pd.notna(put_iv) and pd.notna(call_iv) else np.nan
            rr_norm = np.nan
            if pd.notna(rr) and pd.notna(atm_iv) and atm_iv != 0:
                rr_norm = rr / atm_iv

            records.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "put_25d_iv": put_iv,
                    "call_25d_iv": call_iv,
                    "atm_iv": atm_iv,
                    "rr": rr,
                    "rr_norm": rr_norm,
                }
            )

        out = pd.DataFrame(records)
        return out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    def compute(self, earnings_dates, options_df) -> pd.DataFrame:
        earnings = earnings_dates[["ticker", "event_date"]].copy()
        earnings["ticker"] = earnings["ticker"].astype(str)
        earnings["event_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
        earnings = earnings.dropna(subset=["ticker", "event_date"]).sort_values(["ticker", "event_date"])

        if earnings.empty or options_df is None or options_df.empty:
            return earnings.assign(skew_score=np.nan)[["ticker", "event_date", "skew_score"]]

        long_df = self._to_long_iv(options_df)
        if long_df.empty:
            return earnings.assign(skew_score=np.nan)[["ticker", "event_date", "skew_score"]]

        event_scores: list[dict] = []
        for ticker, ticker_events in earnings.groupby("ticker", sort=False):
            ticker_opts = long_df[long_df["ticker"] == ticker]
            if ticker_opts.empty:
                for event_date in ticker_events["event_date"]:
                    event_scores.append({"ticker": ticker, "event_date": event_date, "skew_raw": np.nan})
                continue

            for event_date in ticker_events["event_date"]:
                event_chain = ticker_opts[ticker_opts["exdate"] >= event_date]
                if event_chain.empty:
                    event_scores.append({"ticker": ticker, "event_date": event_date, "skew_raw": np.nan})
                    continue

                nearest_exdate = event_chain["exdate"].min()
                window_start = event_date - pd.Timedelta(days=self.skew_lookback_days)
                window = event_chain[
                    (event_chain["exdate"] == nearest_exdate)
                    & (event_chain["trade_date"] <= event_date)
                    & (event_chain["trade_date"] > window_start)
                ]

                if window.empty:
                    event_scores.append({"ticker": ticker, "event_date": event_date, "skew_raw": np.nan})
                    continue

                rr_df = self.compute_risk_reversal(window)
                skew_raw = rr_df["rr_norm"].mean(skipna=True) if not rr_df.empty else np.nan
                event_scores.append({"ticker": ticker, "event_date": event_date, "skew_raw": skew_raw})

        skew_df = pd.DataFrame(event_scores).sort_values(["ticker", "event_date"]).reset_index(drop=True)
        grouped = skew_df.groupby("ticker", sort=False)["skew_raw"]
        rolling_mean = grouped.transform(
            lambda s: s.shift(1).rolling(window=self.zscore_window, min_periods=20).mean()
        )
        rolling_std = grouped.transform(
            lambda s: s.shift(1).rolling(window=self.zscore_window, min_periods=20).std()
        )
        skew_df["skew_score"] = (skew_df["skew_raw"] - rolling_mean) / rolling_std
        skew_df.loc[rolling_std == 0, "skew_score"] = np.nan

        return skew_df[["ticker", "event_date", "skew_score"]]

