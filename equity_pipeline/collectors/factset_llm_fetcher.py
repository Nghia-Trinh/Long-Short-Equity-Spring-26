"""FactSet-backed data fetcher for LLM workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from config import (
    FACTSET_API_KEY,
    FACTSET_AUTH_METHOD,
    FACTSET_DEFAULT_EXCHANGE,
    FACTSET_OAUTH_CONFIG_PATH,
    FACTSET_USERNAME,
)


LOGGER = logging.getLogger(__name__)


class FactSetConfigurationError(RuntimeError):
    """Raised when FactSet authentication settings are invalid."""


class FactSetDependencyError(RuntimeError):
    """Raised when required FactSet SDK dependencies are missing."""


def _import_factset_entity_modules() -> tuple[Any, Any]:
    """Import FactSet Entity SDK modules lazily."""
    try:
        import fds.sdk.FactSetEntity as factset_entity
        from fds.sdk.FactSetEntity.api import entity_reference_api
    except ImportError as exc:
        raise FactSetDependencyError(
            "Missing FactSet SDK dependencies. Install `fds.sdk.utils` and "
            "`fds.sdk.FactSetEntity` before using this fetcher."
        ) from exc
    return factset_entity, entity_reference_api


def _normalize_text(value: str | None) -> str:
    """Normalize string inputs."""
    return (value or "").strip()


def _build_configuration(factset_entity: Any) -> Any:
    """Create a FactSet SDK configuration for supported auth methods."""
    auth_method = _normalize_text(FACTSET_AUTH_METHOD).lower() or "oauth"

    if auth_method == "oauth":
        config_path_value = _normalize_text(FACTSET_OAUTH_CONFIG_PATH)
        if not config_path_value:
            raise FactSetConfigurationError(
                "FACTSET_OAUTH_CONFIG_PATH is required when FACTSET_AUTH_METHOD=oauth."
            )
        config_path = Path(config_path_value)
        if not config_path.exists():
            raise FactSetConfigurationError(
                f"FactSet OAuth config file not found: {config_path}"
            )
        try:
            from fds.sdk.utils.authentication import ConfidentialClient
        except ImportError as exc:
            raise FactSetDependencyError(
                "Missing `fds.sdk.utils`; install it to use OAuth authentication."
            ) from exc
        return factset_entity.Configuration(
            fds_oauth_client=ConfidentialClient(config_path=str(config_path))
        )

    if auth_method == "apikey":
        username = _normalize_text(FACTSET_USERNAME)
        api_key = _normalize_text(FACTSET_API_KEY)
        if not username or not api_key:
            raise FactSetConfigurationError(
                "FACTSET_USERNAME and FACTSET_API_KEY are required when "
                "FACTSET_AUTH_METHOD=apikey."
            )
        return factset_entity.Configuration(username=username, password=api_key)

    raise FactSetConfigurationError(
        f"Unsupported FACTSET_AUTH_METHOD={auth_method!r}. "
        "Use 'oauth' or 'apikey'."
    )


def _normalize_entity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map the raw FactSet entity record to an LLM-friendly shape."""
    return {
        "request_id": record.get("request_id") or record.get("requestId"),
        "fsym_id": record.get("fsym_id") or record.get("fsymId"),
        "entity_name": record.get("entity_name") or record.get("entityName"),
        "ticker_exchange": record.get("ticker_exchange") or record.get("tickerExchange"),
        "entity_type_description": record.get("entity_type_description")
        or record.get("entityTypeDescription"),
        "country_name": record.get("country_name") or record.get("countryName"),
        "iso_country": record.get("iso_country") or record.get("isoCountry"),
        "raw": record,
    }


def _clean_ids(ids: Iterable[str]) -> list[str]:
    """Normalize request ids while keeping original order."""
    cleaned: list[str] = []
    for item in ids:
        token = _normalize_text(item).upper()
        if token:
            cleaned.append(token)
    return cleaned


def fetch_factset_entity_context(ids: Iterable[str], *, include_raw: bool = False) -> dict[str, Any]:
    """Fetch entity references from FactSet and normalize for LLM consumption."""
    cleaned_ids = _clean_ids(ids)
    if not cleaned_ids:
        raise ValueError("At least one identifier is required.")

    factset_entity, entity_reference_api = _import_factset_entity_modules()
    configuration = _build_configuration(factset_entity)

    with factset_entity.ApiClient(configuration) as api_client:
        api_instance = entity_reference_api.EntityReferenceApi(api_client)
        response = api_instance.get_entity_references(cleaned_ids)

    payload = response.to_dict() if hasattr(response, "to_dict") else {}
    data = payload.get("data") or []
    errors = payload.get("errors") or []

    normalized_records = [_normalize_entity_record(item) for item in data if isinstance(item, dict)]

    result: dict[str, Any] = {
        "provider": "FactSet",
        "dataset": "FactSet Entity API",
        "requested_ids": cleaned_ids,
        "returned_records": len(normalized_records),
        "data": normalized_records,
        "errors": errors,
    }
    if include_raw:
        result["raw_response"] = payload
    return result


def fetch_factset_entity_context_for_ticker(
    ticker: str,
    *,
    exchange: str | None = None,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Fetch FactSet entity context for a ticker using ticker-exchange format."""
    symbol = _normalize_text(ticker).upper()
    if not symbol:
        raise ValueError("Ticker is required.")
    exchange_suffix = _normalize_text(exchange or FACTSET_DEFAULT_EXCHANGE).upper() or "US"
    identifier = f"{symbol}-{exchange_suffix}"
    return fetch_factset_entity_context([identifier], include_raw=include_raw)
