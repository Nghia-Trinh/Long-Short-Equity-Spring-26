"""Central configuration for the equity pipeline.

Edit this file directly before running the pipeline.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# API provider inputs (paste values directly)
# ---------------------------------------------------------------------------
#
# Finnhub docs indicate standard REST/WebSocket auth is token-based.
# A separate "secret" field is provided for account variants that expose one.
FINNHUB_REST_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_WEBSOCKET_URL = "wss://ws.finnhub.io?token=TOKEN_ID"
FINNHUB_API_KEY = "TOKEN_ID"
FINNHUB_API_SECRET = "TOKEN_HERE"

# FRED API uses a single API key.
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY = "TOKEN_ID"

# SEC EDGAR identity for EdgarTools (typically "Name email@domain.com").
SEC_IDENTITY_NAME = "NAME"
SEC_IDENTITY_EMAIL = "EMAIL_ID"
SEC_IDENTITY_ORGANIZATION = "ORGANIZATION_NAME"
SEC_IDENTITY = " ".join(
    part for part in [SEC_IDENTITY_NAME, SEC_IDENTITY_EMAIL, SEC_IDENTITY_ORGANIZATION] if part
).strip()

# Backward-compatible aliases used by current collectors
FINNHUB_KEY = FINNHUB_API_KEY
FRED_KEY = FRED_API_KEY

# FactSet SDK configuration (for LLM-oriented enrichment fetches)
# Supported auth methods: "oauth" (preferred) or "apikey"
FACTSET_AUTH_METHOD = os.getenv("FACTSET_AUTH_METHOD", "oauth").strip().lower()
FACTSET_OAUTH_CONFIG_PATH = os.getenv("FACTSET_OAUTH_CONFIG_PATH", "").strip()
FACTSET_USERNAME = os.getenv("FACTSET_USERNAME", "").strip()
FACTSET_API_KEY = os.getenv("FACTSET_API_KEY", "").strip()
FACTSET_DEFAULT_EXCHANGE = os.getenv("FACTSET_DEFAULT_EXCHANGE", "US").strip().upper()

# Pipeline defaults (paths relative to this package directory, not the shell cwd)
_PACKAGE_DIR = Path(__file__).resolve().parent
MONTHS = 6
OUTPUT_DIR = _PACKAGE_DIR / "equity_data"
TICKERS_FILE = _PACKAGE_DIR / "tickers.txt"
BENCHMARK_TICKER = "^GSPC"

# Rate limits
FINNHUB_DELAY = 1.1

# FRED series map: readable label -> FRED series id
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "ten_year_treasury_yield": "DGS10",
    "cpi_all_items": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "real_gdp": "GDPC1",
    "vix_index": "VIXCLS",
}

# Sector map: readable sector -> representative ETF ticker
SECTOR_ETF_MAP = {
    "technology": "XLK",
    "health_care": "XLV",
    "financials": "XLF",
    "consumer_discretionary": "XLY",
    "communication_services": "XLC",
    "industrials": "XLI",
    "energy": "XLE",
    "utilities": "XLU",
    "materials": "XLB",
    "real_estate": "XLRE",
    "consumer_staples": "XLP",
}
