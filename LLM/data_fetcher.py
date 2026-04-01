"""
LLM data fetcher for thesis-aware equity overlays.

This module centralizes all data collection needed by the LLM overlay:
1) Investment theses (txt/json/docx/pdf) from a directory
2) Optional member stock selections from JSON/CSV
3) Russell-style training tables from Data/ CSVs
4) Earnings-season windows for slice backtests
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd

from utils.data_loader import load_earnings, load_postearnings, get_returns_pivot


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


_TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


@dataclass
class ThesisRecord:
    source_path: str
    ticker: str | None
    thesis_text: str


class LLMDataFetcher:
    """Collects model inputs for thesis-aware portfolio adjustments."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or _project_root()

    def _read_text_like_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return json.dumps(raw, ensure_ascii=True)
        if suffix == ".docx":
            try:
                import docx  # type: ignore
            except Exception:
                return ""
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text)
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                return ""
            try:
                reader = PdfReader(str(path))
                parts: list[str] = []
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    if txt:
                        parts.append(txt)
                return "\n".join(parts)
            except Exception:
                return ""
        return ""

    def _guess_ticker(self, path: Path, text: str, known_tickers: set[str]) -> str | None:
        stem_tokens = _TICKER_PATTERN.findall(path.stem.upper())
        for tok in stem_tokens:
            if tok in known_tickers:
                return tok

        text_upper = text.upper()
        for match in _TICKER_PATTERN.finditer(text_upper):
            tok = match.group(0)
            if tok in known_tickers:
                return tok
        return None

    def load_investment_theses(self, thesis_dir: str = "Investment Theses") -> pd.DataFrame:
        """Load thesis files from a directory and infer ticker tags."""
        root = (self.project_root / thesis_dir).resolve()
        if not root.exists():
            return pd.DataFrame(columns=["source_path", "ticker", "thesis_text"])

        known_tickers = set(get_returns_pivot().columns.astype(str))
        records: list[ThesisRecord] = []
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            text = self._read_text_like_file(path).strip()
            if not text:
                continue
            ticker = self._guess_ticker(path, text, known_tickers)
            records.append(
                ThesisRecord(
                    source_path=str(path.relative_to(self.project_root)),
                    ticker=ticker,
                    thesis_text=text,
                )
            )

        if not records:
            return pd.DataFrame(columns=["source_path", "ticker", "thesis_text"])
        return pd.DataFrame([r.__dict__ for r in records])

    def load_member_selections(self, path: str | None = None) -> pd.DataFrame:
        """
        Load optional member selections.

        Supported formats:
        - JSON: [{"ticker": "AAPL", "member": "alice", "base_weight": 0.03}, ...]
        - CSV:  columns should include ticker; member/base_weight are optional.
        """
        if not path:
            return pd.DataFrame(columns=["ticker", "member", "base_weight"])

        resolved = (self.project_root / path).resolve()
        if not resolved.exists():
            return pd.DataFrame(columns=["ticker", "member", "base_weight"])

        if resolved.suffix.lower() == ".json":
            raw = json.loads(resolved.read_text(encoding="utf-8"))
            df = pd.DataFrame(raw)
        else:
            df = pd.read_csv(resolved)

        if "ticker" not in df.columns:
            return pd.DataFrame(columns=["ticker", "member", "base_weight"])
        if "member" not in df.columns:
            df["member"] = "unknown"
        if "base_weight" not in df.columns:
            df["base_weight"] = 0.0

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["member"] = df["member"].astype(str).str.strip()
        df["base_weight"] = pd.to_numeric(df["base_weight"], errors="coerce").fillna(0.0)
        return df[["ticker", "member", "base_weight"]]

    def load_russell_training_frame(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Build a joined event-level frame from earnings + post-earnings tables."""
        earnings = load_earnings().copy()
        post = load_postearnings().copy()
        merged = earnings.merge(
            post[
                [
                    "ticker",
                    "event_date",
                    "return_1d",
                    "return_5d",
                    "return_10d",
                    "rv_10d_annualized",
                    "gap_pct",
                ]
            ],
            on=["ticker", "event_date"],
            how="left",
        )
        if start_date:
            merged = merged[merged["event_date"] >= pd.Timestamp(start_date)]
        if end_date:
            merged = merged[merged["event_date"] <= pd.Timestamp(end_date)]
        merged = merged.sort_values(["event_date", "ticker"]).reset_index(drop=True)
        return merged

    def build_earnings_season_windows(
        self,
        pre_days: int = 5,
        post_days: int = 15,
    ) -> pd.DataFrame:
        """
        Build earnings-season windows by calendar quarter.

        Each quarter's season starts `pre_days` before the first event and ends
        `post_days` after the last event in that quarter.
        """
        earnings = load_earnings().copy()
        if earnings.empty:
            return pd.DataFrame(columns=["season", "start_date", "end_date", "event_count"])

        earnings["quarter"] = earnings["event_date"].dt.to_period("Q").astype(str)
        grouped = earnings.groupby("quarter")["event_date"].agg(["min", "max", "count"]).reset_index()
        grouped["start_date"] = grouped["min"] - pd.Timedelta(days=pre_days)
        grouped["end_date"] = grouped["max"] + pd.Timedelta(days=post_days)
        grouped = grouped.rename(columns={"quarter": "season", "count": "event_count"})
        return grouped[["season", "start_date", "end_date", "event_count"]].sort_values("season")

    def fetch_llm_inputs(
        self,
        thesis_dir: str = "Investment Theses",
        member_selection_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Convenience method returning all key LLM inputs."""
        return {
            "theses": self.load_investment_theses(thesis_dir=thesis_dir),
            "member_selections": self.load_member_selections(path=member_selection_file),
            "training_frame": self.load_russell_training_frame(
                start_date=start_date,
                end_date=end_date,
            ),
            "earnings_seasons": self.build_earnings_season_windows(),
        }
