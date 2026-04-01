# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a single-repo Python quantitative trading strategy (Earnings Sentiment Long-Short Equity). No databases, Docker, or external services are required to run the core backtest.

### Key commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Run backtest | `python3 backtest.py` |
| Run Alpha artifacts | `python3 Alpha/__init__.py --max-tickers 25 --no-show --keep-latest-charts 1 --keep-latest-excel 1` |
| Run Streamlit dashboard | `streamlit run Visualisation/dashboard.py --server.port 8501 --server.headless true` |

See `README.md` for full details on configuration and available flags.

### Dependency version gotcha

The project's `requirements.txt` specifies minimum versions (e.g. `cvxpy>=1.4`). However, `cvxpy>=1.5` with the latest `scipy` triggers an ARPACK convergence failure in the PSD check for large covariance matrices (502x502). The working combination is:

- `cvxpy==1.4.4` (uses a different internal PSD-check code path)
- `scipy>=1.11,<1.15`

The update script pins these versions. If you see `ArpackNoConvergence` errors during `python3 backtest.py`, this is the cause.

### Streamlit dashboard

- Requires backtest outputs (`outputs/portfolio_weights.csv`, `outputs/pnl.csv`) to exist first. Run `python3 backtest.py` before starting the dashboard.
- Requires `streamlit` and `plotly` (not in the root `requirements.txt`).
- Use `--server.headless true` for cloud/headless environments.
- The `.style` accessor in pandas DataFrames used by the dashboard requires `jinja2>=3.1.4`.

### No tests or linting

This repository has no test suite, linting config, or CI pipeline. There is no `pyproject.toml` or `setup.cfg`.
