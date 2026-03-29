"""SEC filing metadata collector via EdgarTools."""

from __future__ import annotations

import logging
from typing import Any

from config import SEC_IDENTITY


LOGGER = logging.getLogger(__name__)


def _extract_attr(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _normalize_filing(record: Any, fallback_form: str) -> dict[str, Any]:
    return {
        "form_type": _extract_attr(record, "form") or fallback_form,
        "filed_date": _extract_attr(record, "filing_date") or _extract_attr(record, "filed"),
        "accession_number": _extract_attr(record, "accession_number")
        or _extract_attr(record, "accessionNo"),
    }


def fetch_sec_filings(ticker: str) -> dict[str, Any]:
    """Fetch latest 10-K and 10-Q filing metadata for a ticker."""
    try:
        from edgar import Company, set_identity
    except ImportError:
        LOGGER.warning("edgartools is not installed; SEC filing collection skipped.")
        return {}

    if SEC_IDENTITY:
        try:
            set_identity(SEC_IDENTITY)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to set SEC identity: %s", exc)

    output: dict[str, Any] = {"10-K": [], "10-Q": []}
    try:
        company = Company(ticker.upper())
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to initialize Edgar company for %s: %s", ticker, exc)
        return output

    for form in ("10-K", "10-Q"):
        try:
            filings = company.get_filings(form=form)
            records = []
            for filing in filings:
                records.append(_normalize_filing(filing, form))
                if len(records) >= 4:
                    break
            output[form] = records
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch %s filings for %s: %s", form, ticker, exc)
            output[form] = []

    return output
