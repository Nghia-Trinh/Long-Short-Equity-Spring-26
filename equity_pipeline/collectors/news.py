"""Finnhub news collector."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import finnhub

from config import FINNHUB_DELAY, FINNHUB_KEY


LOGGER = logging.getLogger(__name__)


def fetch_news(ticker: str, months: int) -> list[dict[str, Any]]:
    """Fetch and deduplicate Finnhub company news in 30-day windows."""
    if not FINNHUB_KEY:
        LOGGER.warning("FINNHUB_KEY is empty; skipping news for %s.", ticker)
        return []

    client = finnhub.Client(api_key=FINNHUB_KEY)
    end = datetime.utcnow().date()
    start = (datetime.utcnow() - timedelta(days=months * 30)).date()
    cursor = start
    deduped: dict[str, dict[str, Any]] = {}

    while cursor <= end:
        window_end = min(cursor + timedelta(days=29), end)
        try:
            articles = client.company_news(
                ticker.upper(),
                _from=cursor.isoformat(),
                to=window_end.isoformat(),
            )
            for article in articles or []:
                headline = (article.get("headline") or "").strip()
                key = headline or f"{article.get('id')}::{article.get('datetime')}"
                if not key:
                    continue
                deduped[key] = article
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Finnhub news fetch failed for %s window %s-%s: %s",
                ticker,
                cursor,
                window_end,
                exc,
            )
        time.sleep(FINNHUB_DELAY)
        cursor = window_end + timedelta(days=1)

    items = list(deduped.values())
    items.sort(key=lambda x: x.get("datetime", 0), reverse=True)
    return items
