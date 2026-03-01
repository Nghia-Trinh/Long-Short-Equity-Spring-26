
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.data_loader import load_earnings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = _project_root() / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_options_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    rename_map = {}
    if "date" in out.columns and "trade_date" not in out.columns:
        rename_map["date"] = "trade_date"
    if "quotedate" in out.columns and "trade_date" not in out.columns:
        rename_map["quotedate"] = "trade_date"
    if "expiry" in out.columns and "exdate" not in out.columns:
        rename_map["expiry"] = "exdate"
    if "expiry_date" in out.columns and "exdate" not in out.columns:
        rename_map["expiry_date"] = "exdate"
    if rename_map:
        out = out.rename(columns=rename_map)

    return out


KEEP_COLS_RAW = [
    "ticker", "date", "quotedate", "exdate", "expiry", "expiry_date",
    "cp_flag", "delta", "volume", "open_interest", "impl_volatility",
]


def _usecols_for_file(csv_path: Path) -> list[str] | None:
    with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
        header = [c.strip().lower() for c in fh.readline().split(",")]
    keep = [c for c in KEEP_COLS_RAW if c in header]
    return keep if {"ticker", "delta"}.issubset(keep) else None


def _filter_chunks(
    csv_path: Path,
    ticker_ranges: pd.DataFrame,
    delta_targets: list[float] | None,
    delta_tolerance: float,
    chunksize: int,
) -> pd.DataFrame:
    usecols = _usecols_for_file(csv_path)
    read_kw: dict = {"chunksize": chunksize}
    if usecols is not None:
        read_kw["usecols"] = usecols

    file_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, **read_kw):
        chunk = _normalise_options_columns(chunk)
        required = {"ticker", "trade_date", "exdate", "delta"}
        if not required.issubset(chunk.columns):
            continue

        chunk["ticker"] = chunk["ticker"].astype(str)
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"], errors="coerce")
        chunk["exdate"] = pd.to_datetime(chunk["exdate"], errors="coerce")
        chunk["delta"] = pd.to_numeric(chunk["delta"], errors="coerce")
        chunk = chunk.dropna(subset=["ticker", "trade_date", "exdate", "delta"])

        chunk = chunk.merge(ticker_ranges, on="ticker", how="inner")
        if chunk.empty:
            continue

        in_window = (chunk["trade_date"] >= chunk["min_date"]) & (
            chunk["trade_date"] <= chunk["max_date"]
        )
        if delta_targets is None:
            kept = chunk.loc[in_window].copy()
        else:
            abs_delta = chunk["delta"].abs()
            delta_mask = pd.Series(False, index=chunk.index)
            for target in delta_targets:
                delta_mask = delta_mask | ((abs_delta - float(target)).abs() <= delta_tolerance)
            kept = chunk.loc[in_window & delta_mask].copy()
        if kept.empty:
            continue

        kept = kept.drop(columns=["min_date", "max_date"], errors="ignore")
        file_chunks.append(kept)

    if not file_chunks:
        return pd.DataFrame()
    return pd.concat(file_chunks, ignore_index=True)


def load_options_for_events(
    earnings_df: Optional[pd.DataFrame] = None,
    options_data_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    delta_targets: Optional[list[float]] = [0.25, 0.50],
    delta_tolerance: float = 0.05,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load options data filtered to earnings-relevant rows."""
    config = _load_config()
    root = _project_root()

    if earnings_df is None:
        earnings_df = load_earnings()

    if earnings_df.empty:
        return pd.DataFrame()

    earnings = earnings_df[["ticker", "event_date"]].copy()
    earnings["ticker"] = earnings["ticker"].astype(str)
    earnings["event_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
    earnings = earnings.dropna(subset=["ticker", "event_date"])

    if earnings.empty:
        return pd.DataFrame()

    skew_lookback_days = int(config.get("skew_lookback_days", 5))
    ticker_ranges = (
        earnings.groupby("ticker", as_index=False)["event_date"]
        .agg(min_date="min", max_date="max")
        .assign(min_date=lambda x: x["min_date"] - pd.Timedelta(days=skew_lookback_days))
    )

    default_cache_name = "options_filtered.parquet" if delta_targets is not None else "options_all_deltas.parquet"
    cache_file = Path(cache_path) if cache_path else root / "Data" / default_cache_name
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if "trade_date" in cached.columns:
            cached["trade_date"] = pd.to_datetime(cached["trade_date"], errors="coerce")
        if "exdate" in cached.columns:
            cached["exdate"] = pd.to_datetime(cached["exdate"], errors="coerce")
        return cached

    base_dir = options_data_dir or config.get("options_data_dir", "OptionData2.26")
    options_dir = Path(base_dir)
    if not options_dir.is_absolute():
        options_dir = root / options_dir

    csv_files = sorted(options_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    per_file_dfs: list[pd.DataFrame] = []
    for csv_path in csv_files:
        print(f"  Loading {csv_path.name} ...")
        file_df = _filter_chunks(
            csv_path, ticker_ranges, delta_targets, delta_tolerance, chunksize,
        )
        if not file_df.empty:
            per_file_dfs.append(file_df)
            print(f"    kept {len(file_df):,} rows")
        else:
            print(f"    0 rows matched")

    if not per_file_dfs:
        return pd.DataFrame()

    options_df = pd.concat(per_file_dfs, ignore_index=True)
    del per_file_dfs
    options_df = options_df.sort_values(["ticker", "trade_date", "exdate"]).reset_index(drop=True)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        options_df.to_parquet(cache_file, index=False)
    except Exception:
        pass

    return options_df
