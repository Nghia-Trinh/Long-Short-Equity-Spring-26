# Equity pipeline

A small Python CLI that pulls public-market and macro data for a list of tickers, then writes JSON/CSV artifacts under a single output directory for downstream analysis or LLM workflows.

## Summary

The pipeline runs in three phases:

1. **Global** — Batch download of adjusted OHLCV (and related series) for all tickers plus the benchmark, via **yfinance**. Fetches **FRED** macro series and computes **sector ETF** performance from the batch close panel.
2. **Per ticker** — For each symbol: saves **prices** (with benchmark alignment), **Finnhub** news (optional skip), **yfinance** fundamentals-style snapshot, **SEC** 10-K / 10-Q filing metadata via **EdgarTools** (`edgar`), and derived **dynamics** metrics.
3. **Metadata** — Writes `run_metadata.json` with timing, success counts, and per-stage error notes.

Default benchmark is `^GSPC` (S&P 500). Defaults for paths and lookback live in `config.py` (`MONTHS`, `OUTPUT_DIR`, `TICKERS_FILE`, etc.).

## Requirements

- **Python**: 3.9+ recommended (matches common conda stacks; use whatever your environment supports for the listed wheels).
- **Install from file** (from this directory):

  ```bash
  pip install -r requirements.txt
  ```

  > **Note:** `requirements.txt` currently lists some packages twice with different pins. `pip` applies the last pin per package. For reproducible installs, consider deduplicating that file to one pin per dependency.

### Python packages

| Package | Role in this project |
|--------|------------------------|
| `pandas` | Tabular data for prices and joins |
| `numpy` | Numerics |
| `yfinance` | Prices and fundamentals |
| `finnhub-python` | News API |
| `fredapi` | FRED macro series |
| `edgartools` | SEC filing metadata (`import edgar`) |

`requirements.txt` also lists `PyYAML` and `python-dotenv`; they are not imported by the current modules (you can remove them from the file if you want a minimal dependency set). Exact pins live in `requirements.txt`.

### API keys and identity (configuration)

Edit **`config.py`** before running:

- **Finnhub**: `FINNHUB_API_KEY` (and related URLs if you customize them).
- **FRED**: `FRED_API_KEY` (alias `FRED_KEY` for collectors).
- **SEC / EdgarTools**: `SEC_IDENTITY` is built from name / email / organization; SEC expects a truthful `User-Agent`-style identity string.

If `FRED_KEY` is empty, macro indicators are skipped (warnings only). If **EdgarTools** is not installed, SEC filing collection logs a warning and returns empty objects.

**Before pushing to GitHub:** replace real tokens with placeholders or move secrets to environment variables and keep `config.py` out of the repo (or use a `config.local.py` pattern). Do not commit live keys.

## Usage

1. Put one ticker per line in **`tickers.txt`** (lines starting with `#` are ignored).
2. From the **`equity_pipeline`** directory (so imports resolve), run:

   ```bash
   python pipeline.py
   ```

### CLI options

| Option | Description |
|--------|-------------|
| `--tickers PATH` | Ticker file (default: `tickers.txt` next to `config.py`) |
| `--output DIR` | Output root (default: `equity_data` under this package) |
| `--months N` | Lookback window in months (default: `MONTHS` in `config.py`) |
| `--skip-news` | Skip Finnhub news collection (still writes empty `news.json`) |

## Output layout

```
equity_data/
  run_metadata.json
  _macro/
    macro_indicators.json
    sector_performance.json
  <TICKER>/
    prices.csv
    news.json
    fundamentals.json
    sec_filings.json
    dynamics.json
```

## Project layout

| Path | Purpose |
|------|---------|
| `pipeline.py` | CLI entry and orchestration |
| `config.py` | Keys, URLs, FRED series map, sector ETF map, defaults |
| `tickers.txt` | Input universe |
| `collectors/` | Data fetchers (prices, news, fundamentals, macro, SEC) |
| `compute/` | Derived metrics (e.g. dynamics) |
| `persistence/` | Directory helpers and file writers |

## License

Add a `LICENSE` file in the repository if you intend open-source distribution; none is included here by default.
