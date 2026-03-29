"""Filesystem persistence utilities."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if missing and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_ticker_dir(base_dir: str | Path, ticker: str) -> Path:
    """Create and return the per-ticker output directory."""
    return ensure_dir(Path(base_dir) / ticker.upper())


def _maybe_numpy_scalar(value: Any) -> Any:
    """Convert numpy scalar-like objects without importing numpy directly."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _sanitize(value: Any) -> Any:
    """Convert unsupported values into JSON-serializable primitives."""
    value = _maybe_numpy_scalar(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    return value


def save_json(data: Any, path: str | Path) -> None:
    """Save a Python object to JSON with stable formatting."""
    target = Path(path)
    ensure_dir(target.parent)
    sanitized = _sanitize(data)
    target.write_text(
        json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_csv(df: Any, path: str | Path) -> None:
    """Save a DataFrame-like object to CSV."""
    target = Path(path)
    ensure_dir(target.parent)
    df.to_csv(target, index=True)


def save_run_metadata(metadata: dict[str, Any], output_dir: str | Path) -> None:
    """Save run metadata into the root output directory."""
    save_json(metadata, Path(output_dir) / "run_metadata.json")
