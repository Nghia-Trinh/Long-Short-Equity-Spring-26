"""Pipeline orchestrator and CLI entry point."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collectors.fundamentals import fetch_fundamentals
from collectors.macro import compute_sector_performance, fetch_macro_indicators
from collectors.news import fetch_news
from collectors.prices import batch_download, split_ticker_prices
from collectors.sec_filings import fetch_sec_filings
from compute.dynamics import compute_dynamics
from config import BENCHMARK_TICKER, MONTHS, OUTPUT_DIR, TICKERS_FILE
from persistence.writer import ensure_dir, ensure_ticker_dir, save_csv, save_json, save_run_metadata


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equity data pipeline")
    parser.add_argument("--tickers", default=str(TICKERS_FILE), help="Path to tickers.txt")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--months", type=int, default=MONTHS, help="Lookback window in months")
    parser.add_argument("--skip-news", action="store_true", help="Skip Finnhub news collection")
    return parser.parse_args()


def _read_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().upper()
        if not token or token.startswith("#"):
            continue
        tickers.append(token)
    return sorted(set(tickers))


def _record_error(errors: dict[str, list[str]], key: str, message: str) -> None:
    errors.setdefault(key, []).append(message)


def run_pipeline() -> dict[str, Any]:
    args = _parse_args()
    start_time = datetime.now(timezone.utc)

    tickers_path = Path(args.tickers)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    errors: dict[str, list[str]] = {}
    tickers = _read_tickers(tickers_path)
    if not tickers:
        raise ValueError(f"No tickers found in {tickers_path}")

    LOGGER.info("Loaded %d ticker(s).", len(tickers))
    LOGGER.info("Phase 1/3: global fetches")

    batch_data = {}
    try:
        batch_data = batch_download(tickers, args.months)
    except Exception as exc:  # noqa: BLE001
        _record_error(errors, "_global", f"batch_download failed: {exc}")
        LOGGER.exception("Global price download failed.")
        raise

    macro_dir = ensure_dir(output_dir / "_macro")
    try:
        macro_indicators = fetch_macro_indicators(args.months)
        save_json(macro_indicators, macro_dir / "macro_indicators.json")
    except Exception as exc:  # noqa: BLE001
        _record_error(errors, "_macro", f"fetch_macro_indicators failed: {exc}")
        LOGGER.exception("Macro indicator fetch failed.")
        save_json({}, macro_dir / "macro_indicators.json")

    try:
        sector_performance = compute_sector_performance(batch_data["close"])
        save_json(sector_performance, macro_dir / "sector_performance.json")
    except Exception as exc:  # noqa: BLE001
        _record_error(errors, "_macro", f"compute_sector_performance failed: {exc}")
        LOGGER.exception("Sector performance computation failed.")
        save_json({}, macro_dir / "sector_performance.json")

    LOGGER.info("Phase 2/3: per-ticker collection")
    successful_tickers = 0

    for index, ticker in enumerate(tickers, start=1):
        LOGGER.info("[%d/%d] -- %s --", index, len(tickers), ticker)
        ticker_dir = ensure_ticker_dir(output_dir, ticker)
        ticker_failed = False

        try:
            ticker_prices = split_ticker_prices(ticker, batch_data, BENCHMARK_TICKER)
            save_csv(ticker_prices, ticker_dir / "prices.csv")
        except Exception as exc:  # noqa: BLE001
            ticker_failed = True
            _record_error(errors, ticker, f"prices failed: {exc}")
            LOGGER.exception("Price split failed for %s", ticker)

        if not args.skip_news:
            try:
                news = fetch_news(ticker, args.months)
                save_json(news, ticker_dir / "news.json")
            except Exception as exc:  # noqa: BLE001
                ticker_failed = True
                _record_error(errors, ticker, f"news failed: {exc}")
                LOGGER.exception("News fetch failed for %s", ticker)
                save_json([], ticker_dir / "news.json")
        else:
            save_json([], ticker_dir / "news.json")

        try:
            fundamentals = fetch_fundamentals(ticker)
            save_json(fundamentals, ticker_dir / "fundamentals.json")
        except Exception as exc:  # noqa: BLE001
            ticker_failed = True
            _record_error(errors, ticker, f"fundamentals failed: {exc}")
            LOGGER.exception("Fundamentals fetch failed for %s", ticker)
            save_json({}, ticker_dir / "fundamentals.json")

        try:
            filings = fetch_sec_filings(ticker)
            save_json(filings, ticker_dir / "sec_filings.json")
        except Exception as exc:  # noqa: BLE001
            ticker_failed = True
            _record_error(errors, ticker, f"sec_filings failed: {exc}")
            LOGGER.exception("SEC filings fetch failed for %s", ticker)
            save_json({}, ticker_dir / "sec_filings.json")

        try:
            dynamics = compute_dynamics(ticker, batch_data["close"], BENCHMARK_TICKER)
            save_json(dynamics, ticker_dir / "dynamics.json")
        except Exception as exc:  # noqa: BLE001
            ticker_failed = True
            _record_error(errors, ticker, f"dynamics failed: {exc}")
            LOGGER.exception("Dynamics compute failed for %s", ticker)
            save_json({}, ticker_dir / "dynamics.json")

        if not ticker_failed:
            successful_tickers += 1

    LOGGER.info("Phase 3/3: metadata")
    end_time = datetime.now(timezone.utc)
    metadata = {
        "started_at_utc": start_time.isoformat(),
        "finished_at_utc": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 3),
        "tickers_total": len(tickers),
        "tickers_successful": successful_tickers,
        "tickers_with_errors": len([name for name in tickers if name in errors]),
        "months": args.months,
        "skip_news": bool(args.skip_news),
        "benchmark": BENCHMARK_TICKER,
        "errors": errors,
    }
    save_run_metadata(metadata, output_dir)
    LOGGER.info("Pipeline completed. Successful tickers: %d/%d", successful_tickers, len(tickers))
    return metadata


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
