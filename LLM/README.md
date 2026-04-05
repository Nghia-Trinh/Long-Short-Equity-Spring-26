## LLM Thesis Overlay (IBM Granite + Russell 3000 backtesting)

This module adds an optional thesis-aware overlay to the existing long/short pipeline:

1. Fetches thesis/member/training inputs via `LLMDataFetcher`
2. Scores thesis conviction per ticker via IBM Granite on Hugging Face (or lexical fallback)
3. Applies risk-aware alpha and weight adjustments
4. Supports earnings-season slice backtests

### Components

- `LLM/data_fetcher.py`
  - Loads thesis docs from `Investment Theses/` (`.txt`, `.md`, `.json`, `.docx`, `.pdf`)
  - Loads optional member selections from JSON/CSV
  - Builds Russell-style training frame from `Data/earnings.csv` + `Data/postearnings_results.csv`
  - Builds per-quarter earnings season windows

- `LLM/thesis_overlay.py`
  - `GraniteThesisOverlay` with:
    - `adjust_alpha(...)`
    - `adjust_weights(...)`
    - `diagnostics_frame()`

### Local usage (CLI)

Run baseline:

`python3 backtest.py`

Run with LLM overlay:

`python3 backtest.py --enable-llm-overlay --llm-thesis-dir "Investment Theses" --llm-member-selections "Investment Theses/member_selections.sample.json"`

Outputs:

- `outputs/portfolio_weights.csv`
- `outputs/pnl.csv`
- `outputs/earnings_season_metrics.csv`
- `outputs/llm_overlay_diagnostics.csv` (when overlay has thesis/member coverage)

### Colab usage

Use `granite_timeseries_patchtsmixer.ipynb` in repo root (updated for this flow).
