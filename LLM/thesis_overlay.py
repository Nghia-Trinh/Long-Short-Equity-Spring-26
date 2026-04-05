"""
Thesis-aware LLM overlay for long/short portfolio adjustments.

This module adds a lightweight, optional layer on top of the existing
systematic alpha + optimizer flow:

1) Score investment-thesis text per ticker (Hugging Face IBM model if available;
   deterministic lexical fallback otherwise)
2) Build ticker-level risk penalties from Russell-style event training data
3) Adjust alpha and final optimized weights using thesis conviction + risk budget
4) Keep output constrained (dollar-neutral, leverage cap, single-name cap)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from LLM.data_fetcher import LLMDataFetcher


_JSON_BLOCK = re.compile(r"\{.*\}", flags=re.DOTALL)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    sd = float(x.std(ddof=0))
    if sd <= 0:
        return pd.Series(0.0, index=x.index)
    return (x - float(x.mean())) / sd


@dataclass
class ThesisScore:
    score: float
    confidence: float
    risk_penalty: float


class GraniteThesisOverlay:
    """
    Optional thesis-aware overlay for alpha and final weights.

    The overlay is intentionally robust in offline backtests:
    - If `transformers` or the IBM model is unavailable, it falls back to a
      deterministic lexical scorer.
    - If no thesis data exists, it behaves as a no-op.
    """

    def __init__(
        self,
        config: dict,
        tickers: list[str],
        thesis_dir: str = "Investment Theses",
        member_selection_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.config = dict(config or {})
        self.tickers = [str(t).upper().strip() for t in tickers]
        self.ticker_set = set(self.tickers)
        self.enabled = bool(self.config.get("llm_overlay_enabled", False))

        self.llm_model_id = str(
            self.config.get("llm_model_id", "ibm-granite/granite-3.3-2b-instruct")
        )
        self.use_hf_model = bool(self.config.get("llm_use_hf_model", False))
        self.max_prompt_chars = int(self.config.get("llm_max_prompt_chars", 3500))

        self.thesis_alpha_scale = float(self.config.get("llm_thesis_alpha_scale", 0.35))
        self.risk_alpha_penalty = float(self.config.get("llm_risk_alpha_penalty", 0.30))
        self.conviction_weight_boost = float(self.config.get("llm_conviction_weight_boost", 0.20))
        self.conflict_weight_cut = float(self.config.get("llm_conflict_weight_cut", 0.25))
        self.member_weight_blend = float(self.config.get("llm_member_weight_blend", 0.20))
        self.risk_target_vol = float(self.config.get("llm_risk_target_vol", 0.020))

        self.max_leverage = float(self.config.get("max_leverage", 2.0))
        self.max_position = float(self.config.get("max_position_pct", 0.05))
        self.risk_lookback_days = int(self.config.get("llm_risk_lookback_days", 365))

        self._fetcher = LLMDataFetcher()
        payload = self._fetcher.fetch_llm_inputs(
            thesis_dir=thesis_dir,
            member_selection_file=member_selection_file,
            start_date=start_date,
            end_date=end_date,
        )
        self.theses_df: pd.DataFrame = payload["theses"]
        self.member_df: pd.DataFrame = payload["member_selections"]
        self.training_df: pd.DataFrame = payload["training_frame"]
        self.earnings_seasons: pd.DataFrame = payload["earnings_seasons"]

        self._hf_generator = self._build_hf_generator() if self.use_hf_model else None
        self._ticker_scores = self._build_ticker_scores()
        self._member_bias = self._build_member_bias()
        self._risk_stats = self._build_risk_stats()
        self._date_cache: dict[str, pd.Series] = {}

        self._diagnostics: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Model scoring
    # ------------------------------------------------------------------
    def _build_hf_generator(self):
        try:
            from transformers import pipeline  # type: ignore
        except Exception:
            return None

        try:
            return pipeline(
                task="text-generation",
                model=self.llm_model_id,
                tokenizer=self.llm_model_id,
                trust_remote_code=True,
            )
        except Exception:
            return None

    def _lexical_score(self, text: str) -> ThesisScore:
        positive = {
            "undervalued",
            "outperform",
            "strong",
            "moat",
            "growth",
            "upside",
            "buy",
            "long",
            "rebound",
            "beat",
            "quality",
            "conviction",
        }
        negative = {
            "overvalued",
            "underperform",
            "weak",
            "decline",
            "downside",
            "sell",
            "short",
            "miss",
            "deteriorating",
            "headwind",
            "risk",
        }
        risk_words = {
            "debt",
            "liquidity",
            "volatility",
            "uncertain",
            "cyclical",
            "commodity",
            "geopolitical",
            "lawsuit",
            "regulatory",
        }

        tokens = re.findall(r"[A-Za-z']+", text.lower())
        pos_hits = sum(tok in positive for tok in tokens)
        neg_hits = sum(tok in negative for tok in tokens)
        risk_hits = sum(tok in risk_words for tok in tokens)

        raw = (pos_hits - neg_hits) / max(1.0, pos_hits + neg_hits)
        score = float(np.clip(raw, -1.0, 1.0))
        confidence = float(np.clip((pos_hits + neg_hits) / 20.0, 0.1, 1.0))
        risk_penalty = float(np.clip(risk_hits / 15.0, 0.0, 1.0))
        return ThesisScore(score=score, confidence=confidence, risk_penalty=risk_penalty)

    def _score_text(self, ticker: str, text: str) -> ThesisScore:
        clipped = text[: self.max_prompt_chars]
        if self._hf_generator is None:
            return self._lexical_score(clipped)

        prompt = (
            "You are a long-short equity research assistant.\n"
            f"Ticker: {ticker}\n"
            "Read the investment thesis and return a single JSON object with keys:\n"
            "score (float in [-1,1], positive=long bias, negative=short bias),\n"
            "confidence (float in [0,1]), risk_penalty (float in [0,1]).\n"
            "Return JSON only.\n"
            f"Thesis:\n{clipped}\n"
        )
        try:
            out = self._hf_generator(
                prompt,
                max_new_tokens=120,
                do_sample=False,
                temperature=0.0,
            )
            txt = str(out[0].get("generated_text", "")) if out else ""
            block_match = _JSON_BLOCK.search(txt)
            if not block_match:
                return self._lexical_score(clipped)
            payload = json.loads(block_match.group(0))
            return ThesisScore(
                score=float(np.clip(_safe_float(payload.get("score"), 0.0), -1.0, 1.0)),
                confidence=float(np.clip(_safe_float(payload.get("confidence"), 0.25), 0.0, 1.0)),
                risk_penalty=float(np.clip(_safe_float(payload.get("risk_penalty"), 0.0), 0.0, 1.0)),
            )
        except Exception:
            return self._lexical_score(clipped)

    def _build_ticker_scores(self) -> pd.DataFrame:
        cols = ["ticker", "thesis_score", "confidence", "risk_penalty", "doc_count"]
        if self.theses_df.empty:
            return pd.DataFrame(columns=cols)

        frame = self.theses_df.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frame = frame[frame["ticker"].isin(self.ticker_set)]
        if frame.empty:
            return pd.DataFrame(columns=cols)

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            tk = str(row["ticker"]).strip().upper()
            txt = str(row["thesis_text"])
            if not tk or not txt:
                continue
            score = self._score_text(tk, txt)
            rows.append(
                {
                    "ticker": tk,
                    "thesis_score": score.score,
                    "confidence": score.confidence,
                    "risk_penalty": score.risk_penalty,
                }
            )
        if not rows:
            return pd.DataFrame(columns=cols)

        scored = pd.DataFrame(rows)
        grouped = scored.groupby("ticker", as_index=False).agg(
            thesis_score=("thesis_score", "mean"),
            confidence=("confidence", "mean"),
            risk_penalty=("risk_penalty", "mean"),
            doc_count=("ticker", "size"),
        )
        return grouped

    def _build_member_bias(self) -> pd.Series:
        if self.member_df.empty:
            return pd.Series(dtype=float)
        grouped = self.member_df.groupby("ticker")["base_weight"].mean()
        grouped.index = grouped.index.astype(str).str.upper().str.strip()
        grouped = grouped.reindex(self.tickers).fillna(0.0)
        return grouped

    def _build_risk_stats(self) -> pd.DataFrame:
        cols = ["ticker", "rv_mean", "rv_std", "ret10_mean", "events"]
        if self.training_df.empty:
            return pd.DataFrame(columns=cols)

        train = self.training_df.copy()
        train["ticker"] = train["ticker"].astype(str).str.upper().str.strip()
        train = train[train["ticker"].isin(self.ticker_set)]
        if train.empty:
            return pd.DataFrame(columns=cols)

        grouped = train.groupby("ticker", as_index=False).agg(
            rv_mean=("rv_10d_annualized", "mean"),
            rv_std=("rv_10d_annualized", "std"),
            ret10_mean=("return_10d", "mean"),
            events=("ticker", "size"),
        )
        grouped["rv_std"] = grouped["rv_std"].fillna(0.0)
        return grouped

    def _risk_vector_for_date(self, date: pd.Timestamp, tickers: list[str]) -> pd.Series:
        key = f"{pd.Timestamp(date).date()}::{len(tickers)}"
        if key in self._date_cache:
            return self._date_cache[key].reindex(tickers).fillna(0.0)

        if self.training_df.empty:
            out = pd.Series(0.0, index=tickers)
            self._date_cache[key] = out
            return out

        ts = pd.Timestamp(date)
        lb = ts - pd.Timedelta(days=self.risk_lookback_days)

        window = self.training_df[
            (self.training_df["event_date"] < ts) & (self.training_df["event_date"] >= lb)
        ].copy()
        if window.empty:
            out = pd.Series(0.0, index=tickers)
            self._date_cache[key] = out
            return out

        risk = (
            window.groupby("ticker")["rv_10d_annualized"]
            .mean()
            .reindex(tickers)
            .fillna(window["rv_10d_annualized"].mean())
        )
        risk = _zscore(risk)
        out = risk.clip(-3.0, 3.0)
        self._date_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Public API used by portfolio builder
    # ------------------------------------------------------------------
    def has_signal(self) -> bool:
        return self.enabled and (not self._ticker_scores.empty or not self._member_bias.empty)

    def adjust_alpha(self, date: pd.Timestamp, alpha: np.ndarray, tickers: list[str]) -> np.ndarray:
        if not self.has_signal():
            return np.asarray(alpha, dtype=float)

        tickers = [str(t).upper().strip() for t in tickers]
        base = pd.Series(np.asarray(alpha, dtype=float), index=tickers).fillna(0.0)

        thesis = (
            self._ticker_scores.set_index("ticker")["thesis_score"]
            if not self._ticker_scores.empty
            else pd.Series(dtype=float)
        )
        confidence = (
            self._ticker_scores.set_index("ticker")["confidence"]
            if not self._ticker_scores.empty
            else pd.Series(dtype=float)
        )
        risk_pen = (
            self._ticker_scores.set_index("ticker")["risk_penalty"]
            if not self._ticker_scores.empty
            else pd.Series(dtype=float)
        )
        thesis = thesis.reindex(tickers).fillna(0.0)
        confidence = confidence.reindex(tickers).fillna(0.0)
        risk_pen = risk_pen.reindex(tickers).fillna(0.0)

        member_bias = self._member_bias.reindex(tickers).fillna(0.0)
        thesis_component = thesis * (0.5 + 0.5 * confidence) + np.sign(member_bias) * np.abs(member_bias)

        risk_vec = self._risk_vector_for_date(pd.Timestamp(date), tickers)
        risk_cut = risk_pen + risk_vec.clip(lower=0.0)

        adjusted = base + self.thesis_alpha_scale * thesis_component - self.risk_alpha_penalty * risk_cut
        adjusted = _zscore(adjusted).fillna(0.0)
        return adjusted.values.astype(float)

    def _project_to_constraints(self, w: pd.Series) -> pd.Series:
        w = w.clip(lower=-self.max_position, upper=self.max_position)
        # Keep dollar-neutrality exactly.
        w = w - w.mean()
        gross = float(np.abs(w).sum())
        if gross > self.max_leverage and gross > 0:
            w = w * (self.max_leverage / gross)
        return w

    def _enforce_risk_target(self, w: pd.Series, sigma: np.ndarray | None) -> pd.Series:
        if sigma is None or len(w) == 0:
            return w
        try:
            vec = w.values.astype(float)
            sigma_arr = np.asarray(sigma, dtype=float)
            pvar = float(vec.T @ sigma_arr @ vec)
            pvol = float(np.sqrt(max(pvar, 0.0)))
            if pvol > self.risk_target_vol > 0:
                scale = self.risk_target_vol / pvol
                w = w * scale
        except Exception:
            return w
        return w

    def adjust_weights(
        self,
        date: pd.Timestamp,
        weights: np.ndarray,
        tickers: list[str],
        sigma: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.has_signal():
            return np.asarray(weights, dtype=float)

        tickers = [str(t).upper().strip() for t in tickers]
        w = pd.Series(np.asarray(weights, dtype=float), index=tickers).fillna(0.0)

        if self._ticker_scores.empty:
            w = self._project_to_constraints(w)
            w = self._enforce_risk_target(w, sigma)
            return w.values.astype(float)

        score_df = self._ticker_scores.set_index("ticker")
        thesis_score = score_df["thesis_score"].reindex(tickers).fillna(0.0)
        conf = score_df["confidence"].reindex(tickers).fillna(0.0)
        risk_pen = score_df["risk_penalty"].reindex(tickers).fillna(0.0)
        risk_vec = self._risk_vector_for_date(pd.Timestamp(date), tickers).clip(lower=-2.0, upper=2.0)

        desired_dir = np.sign(thesis_score).astype(float)
        current_dir = np.sign(w).astype(float)
        aligned = (desired_dir == 0.0) | (desired_dir == current_dir)

        boost = self.conviction_weight_boost * np.abs(thesis_score) * conf
        cut = self.conflict_weight_cut * np.abs(thesis_score) * conf
        mult = np.where(aligned, 1.0 + boost, 1.0 - cut)
        w = w * mult

        # De-risk high-risk names.
        risk_scale = 1.0 / (1.0 + np.maximum(0.0, risk_pen + risk_vec))
        w = w * risk_scale

        if len(self._member_bias) > 0:
            mb = self._member_bias.reindex(tickers).fillna(0.0)
            w = (1.0 - self.member_weight_blend) * w + self.member_weight_blend * mb

        w = self._project_to_constraints(w)
        w = self._enforce_risk_target(w, sigma)

        self._diagnostics.append(
            {
                "rebalance_date": pd.Timestamp(date),
                "gross_exposure": float(np.abs(w).sum()),
                "net_exposure": float(w.sum()),
                "thesis_coverage": int((thesis_score != 0).sum()),
                "risk_penalty_avg": float((risk_pen + risk_vec).mean()),
            }
        )
        return w.values.astype(float)

    def diagnostics_frame(self) -> pd.DataFrame:
        if not self._diagnostics:
            return pd.DataFrame(
                columns=[
                    "rebalance_date",
                    "gross_exposure",
                    "net_exposure",
                    "thesis_coverage",
                    "risk_penalty_avg",
                ]
            )
        return pd.DataFrame(self._diagnostics).sort_values("rebalance_date")
