"""
thesis_nlp.py — Lightweight NLP overlay for discretionary investment theses.

This module converts free-form investment theses into a cross-sectional
overlay that can be blended with the systematic SUE alpha. It is deliberately
lightweight (regex + keyword heuristics) to avoid extra dependencies while
still capturing directional and conviction cues from text.

Usage:
    overlay = ThesisNLPOverlay.from_config(config, tickers)
    vector = overlay.get_overlay(date)  # np.ndarray aligned to tickers order

Notes:
- Date precedence when parsing: ``thesis_date`` → ``date`` → ``as_of``.
- Confidence should be provided via ``confidence`` (0-5); ``score`` is
  accepted as a legacy alias.
- Thesis scores are clipped symmetrically to ``[-score_clip, score_clip]``
  before cross-sectional z-scoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "config.json").exists():
            return parent
    # Fallback to previous behaviour if marker search fails.
    return here.parents[1]


_MIN_DECAY_HALF_LIFE = 1
_MIN_CONFIDENCE = 0.05
_MIN_SCORE_CLIP = 0.5


_POSITIVE_CUES = {
    "beat",
    "beats",
    "strong",
    "improving",
    "expanding",
    "growing",
    "lead",
    "leadership",
    "tailwind",
    "accelerate",
    "accelerating",
    "momentum",
    "outperform",
    "upside",
    "undervalued",
    "cheap",
    "compelling",
    "resilient",
    "moat",
    "re-rating",
}

_NEGATIVE_CUES = {
    "miss",
    "weak",
    "deteriorating",
    "shrinking",
    "decline",
    "declining",
    "slowdown",
    "headwind",
    "downside",
    "overvalued",
    "rich",
    "expensive",
    "competitive",
    "risk",
    "fragile",
    "pressure",
    "margin compression",
    "downgrade",
    "short",
    "overhang",
}

_CONVICTION_CUES = {
    "high conviction": 1.5,
    "core position": 1.3,
    "add": 1.1,
    "reduce": 0.9,
    "trim": 0.85,
    "speculative": 0.7,
}


@dataclass
class ThesisRecord:
    ticker: str
    text: str
    thesis_date: Optional[pd.Timestamp]
    confidence: float
    horizon_days: Optional[int]
    direction: Optional[str]


class ThesisNLPOverlay:
    """Transform discretionary theses into an additive alpha overlay."""

    def __init__(
        self,
        tickers: Iterable[str],
        records: List[ThesisRecord],
        decay_half_life: int = 45,
        default_confidence: float = 0.6,
        score_clip: float = 5.0,
    ):
        self.tickers = [str(t).upper() for t in tickers]
        self.decay_half_life = max(int(decay_half_life), _MIN_DECAY_HALF_LIFE)
        self.default_confidence = max(float(default_confidence), _MIN_CONFIDENCE)
        self.score_clip = max(float(score_clip), _MIN_SCORE_CLIP)

        filtered_records = [r for r in records if r.ticker in self.tickers]
        self._records_by_ticker: dict[str, list[ThesisRecord]] = {}
        for rec in filtered_records:
            self._records_by_ticker.setdefault(rec.ticker, []).append(rec)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, tickers: Iterable[str]) -> Optional["ThesisNLPOverlay"]:
        """Create overlay if any thesis input is provided in config."""
        if config is None:
            return None

        records: list[ThesisRecord] = []

        default_conf = float(config.get("thesis_default_confidence", 0.6))
        score_clip = float(config.get("thesis_score_clip", 5.0))

        inline = config.get("investment_theses")
        if isinstance(inline, list):
            records.extend(cls._normalise_records(inline, default_confidence=default_conf))

        path_str = config.get("investment_theses_file_path")
        if not path_str:
            path_str = config.get("investment_theses_file")
        if isinstance(path_str, str) and path_str.strip():
            path = Path(path_str)
            if not path.is_absolute():
                path = _project_root() / path
            if path.exists():
                records.extend(cls._load_from_file(path, default_confidence=default_conf))

        if not records:
            return None

        return cls(
            tickers=tickers,
            records=records,
            decay_half_life=int(config.get("thesis_decay_half_life", 45)),
            default_confidence=default_conf,
            score_clip=score_clip,
        )

    @classmethod
    def _load_from_file(cls, path: Path, default_confidence: float) -> list[ThesisRecord]:
        suffix = path.suffix.lower()
        if suffix in {".json", ".txt"}:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "theses" in payload:
                payload = payload.get("theses", [])
            if not isinstance(payload, list):
                return []
            return cls._normalise_records(payload, default_confidence=default_confidence)
        if suffix == ".csv":
            df = pd.read_csv(path)
            return cls._normalise_records(df.to_dict(orient="records"), default_confidence=default_confidence)
        return []

    @staticmethod
    def _normalise_records(raw: Iterable[dict], default_confidence: float) -> list[ThesisRecord]:
        records: list[ThesisRecord] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).strip().upper()
            text = str(item.get("thesis", item.get("text", ""))).strip()
            if not ticker or not text:
                continue
            date_val = None
            for key in ("thesis_date", "date", "as_of"):
                if key in item:
                    date_val = item.get(key)
                    break
            thesis_date = pd.to_datetime(date_val, errors="coerce") if date_val is not None else pd.NaT
            raw_conf = item.get("confidence")
            if raw_conf is None:
                raw_conf = item.get("score")
            try:
                confidence = float(raw_conf) if raw_conf is not None else float("nan")
            except (TypeError, ValueError):
                confidence = float("nan")
            if np.isnan(confidence):
                confidence = default_confidence
            horizon = item.get("horizon_days") or item.get("horizon")
            horizon_int = int(horizon) if horizon is not None and pd.notna(horizon) else None
            direction = item.get("direction")
            if isinstance(direction, str):
                direction = direction.lower().strip()
                if direction not in {"long", "short"}:
                    direction = None
            records.append(
                ThesisRecord(
                    ticker=ticker,
                    text=text,
                    thesis_date=thesis_date if pd.notna(thesis_date) else None,
                    confidence=max(min(confidence, 5.0), 0.0),
                    horizon_days=horizon_int,
                    direction=direction,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", text.lower())

    def _base_sentiment(self, text: str) -> float:
        tokens = self._tokenise(text)
        if not tokens:
            return 0.0
        pos = sum(token in _POSITIVE_CUES for token in tokens)
        neg = sum(token in _NEGATIVE_CUES for token in tokens)
        if pos == 0 and neg == 0:
            return 0.0
        return (pos - neg) / float(pos + neg)

    def _conviction_boost(self, text: str) -> float:
        text_lower = text.lower()
        boost = 1.0
        for phrase, weight in _CONVICTION_CUES.items():
            if phrase in text_lower:
                boost *= weight
        return boost

    def _decay(self, as_of: pd.Timestamp, thesis_date: Optional[pd.Timestamp], horizon_days: Optional[int]) -> float:
        if thesis_date is None or pd.isna(thesis_date):
            return 1.0
        days_elapsed = max((as_of - thesis_date).days, 0)
        if horizon_days is not None and days_elapsed > horizon_days:
            return 0.0
        # exponential half-life decay
        return 0.5 ** (days_elapsed / float(self.decay_half_life))

    def _score_record(self, record: ThesisRecord, as_of: pd.Timestamp) -> float:
        base = self._base_sentiment(record.text)
        if base == 0.0 and record.direction is None:
            return 0.0
        if record.direction is not None:
            base = abs(base) if base != 0 else self.default_confidence
            base = base if record.direction == "long" else -base
        boost = self._conviction_boost(record.text)
        decay = self._decay(as_of, record.thesis_date, record.horizon_days)
        confidence = record.confidence if record.confidence > 0 else self.default_confidence
        return float(base * boost * confidence * decay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_overlay(self, as_of: pd.Timestamp) -> Optional[np.ndarray]:
        """Return z-scored overlay vector aligned to `self.tickers`."""
        as_of = pd.Timestamp(as_of)
        scores = np.zeros(len(self.tickers), dtype=float)
        has_signal = False

        for idx, ticker in enumerate(self.tickers):
            records = self._records_by_ticker.get(ticker, [])
            if not records:
                continue
            rec_scores = [self._score_record(rec, as_of) for rec in records]
            rec_scores = [s for s in rec_scores if s != 0.0]
            if rec_scores:
                scores[idx] = float(np.clip(np.mean(rec_scores), -self.score_clip, self.score_clip))
                has_signal = True

        if not has_signal:
            return None

        std = float(scores.std())
        if std > 0:
            scores = (scores - scores.mean()) / std
        return scores
