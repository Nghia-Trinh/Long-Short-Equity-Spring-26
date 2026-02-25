# Earnings Sentiment Long-Short Equity Strategy

A quantitative long-short equity strategy targeting post-earnings price drift in the Russell 3000 (ex-large cap).

## Strategy Overview

We exploit the well-documented post-earnings announcement drift (PEAD) phenomenon:
- **Long** stocks with positive earnings surprise (SUE > 0)
- **Short** stocks with negative earnings surprise (SUE < 0)
- Enter after earnings announcement, exit when drift is exhausted (~1-2 weeks)

## Project Structure

```
Long-Short-Equity-Spring-26/
├── config.json              ← all hyperparameters (lambda_ewma, lambda_risk, c, etc.)
├── lean.json                ← QuantConnect LEAN CLI project config
├── requirements.txt
├── main.py                  ← QCAlgorithm subclass (LEAN / QC Cloud entry point)
├── backtest.py              ← pure-Python offline runner: python backtest.py
├── Data/                    ← CSV data files (do not modify)
│   ├── earnings.csv         ← EPS estimates vs actuals per (ticker, event_date)
│   ├── postearnings_results.csv  ← post-event returns (1d, 5d, 10d) + realized vol
│   ├── prices.csv           ← daily OHLCV for all tickers (2016-2025)
│   └── summary.csv          ← per-ticker SUE predictiveness diagnostics
├── Alpha/
│   ├── sue.py               ← SUE = (EPS_actual - EPS_estimate) / sigma_i
│   └── alpha_matrix.py      ← T×N alpha matrix (cross-sectionally z-scored, IC-weighted)
├── Risk/
│   ├── ewma_covariance.py   ← Sigma_t = lambda*Sigma_{t-1} + (1-lambda)*r_t*r_t'
│   └── risk_matrix.py       ← date-indexed RiskMatrixBuilder wrapper
├── Portfolio/
│   ├── optimizer.py         ← cvxpy: max alpha'w - lambda*w'Sigma*w - c*||w-w_prev||_1
│   └── portfolio_matrix.py  ← full T×N weights loop over all rebalance dates
├── PreEarnings/
│   ├── README.md            ← signal specs and integration plan
│   ├── iv_skew.py           ← options IV skew signal stub (needs options data)
│   └── volume_signal.py     ← earnings-day volume spike signal (uses prices.csv)
├── utils/
│   ├── data_loader.py       ← cached loaders for all 4 CSVs + pivot helpers
│   └── universe.py          ← Russell 3000 ex-large-cap filter
└── outputs/                 ← portfolio_weights.csv and pnl.csv written here
```

## Matrices

### 1. Alpha Matrix (Alpha/)
Signal: **Standardised Unexpected Earnings (SUE)**

```
SUE_{i,t} = (EPS_actual_{i,t} - EPS_estimate_{i,t}) / sigma_i
```

`sigma_i` is the rolling std of past forecast errors over the last 8 quarters.
The T×N matrix holds the most recent SUE for each ticker as of each rebalance date, cross-sectionally z-scored.

### 2. Risk Matrix (Risk/)
**EWMA Covariance** — RiskMetrics (1994):

```
Sigma_t = lambda_ewma * Sigma_{t-1} + (1 - lambda_ewma) * r_t * r_t'
```

- `lambda_ewma = 0.94` (configurable in config.json), half-life ≈ 11 trading days
- `r_t` is the N-vector of daily returns across the universe
- Updated every trading day

### 3. Portfolio Matrix (Portfolio/)
**Mean-Variance Optimisation with Turnover Penalty:**

```
max_w  alpha' w  -  lambda_risk * w' Sigma w  -  c * ||w - w_prev||_1
```

Subject to:
- `sum(w) = 0`  (dollar neutral)
- `sum(|w|) <= max_leverage`  (2.0 = 100% long / 100% short)
- `|w_i| <= max_position_pct`  (5% single-name cap)

Solved via `cvxpy` (CLARABEL solver). Falls back to rank-based heuristic if unavailable.

## Pre-Earnings Signals (PreEarnings/)

Two planned signals to trade **before** earnings:
1. **Options IV Skew** — 25-delta risk reversal detects over-hedged fear
2. **Volume Spike** — earnings-day volume vs 20-day ADV measures conviction

See [PreEarnings/README.md](PreEarnings/README.md) for full specs.

## Running the Strategy

### Option A: Pure Python (offline, no QuantConnect)
```bash
pip install -r requirements.txt
python backtest.py
```
Reads directly from `Data/` CSVs, outputs to `outputs/`.

### Option B: LEAN CLI (local QuantConnect backtest)
```bash
pip install lean
lean backtest "Long-Short-Equity-Spring-26"
```
Uses `main.py` as the algorithm entry point.

### Option C: QuantConnect Cloud
Upload the entire project directory via quantconnect.com → Projects → Upload.

## Configuration

All strategy parameters are in `config.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_ewma` | 0.94 | EWMA decay for covariance (≠ risk aversion λ) |
| `lambda_risk_aversion` | 1.0 | MVO risk aversion coefficient |
| `transaction_cost` | 0.001 | L1 turnover penalty coefficient `c` |
| `holding_period_days` | 10 | Hard exit after N trading days |
| `max_leverage` | 2.0 | Max gross exposure (100L / 100S) |
| `max_position_pct` | 0.05 | Max single-name weight |
| `sue_lookback_quarters` | 8 | Rolling window for SUE sigma estimation |
| `start_date` | 2018-01-01 | Backtest start |
| `end_date` | 2025-12-31 | Backtest end |
| `initial_capital` | 1,000,000 | Starting capital ($) |
| `universe_exclude_top_n_largecap` | 500 | Top-N tickers excluded (large-cap filter) |
| `universe_min_dollar_volume` | 1,000,000 | Min daily dollar volume filter |

## Data

| File | Rows | Description |
|------|------|-------------|
| `earnings.csv` | 17,685 | EPS estimates vs actuals, ~503 tickers, Q1 2016 – Q4 2025 |
| `postearnings_results.csv` | 17,685 | Post-event returns (1d, 5d, 10d), realized vol, gap % |
| `prices.csv` | ~1M+ | Daily OHLCV + adjusted close, all tickers |
| `summary.csv` | 4,023 | Per-ticker SUE-to-return IC and data quality diagnostics |

## Team

Long/Short Equity Sector — Quant Finance Organisation, Spring 2026
