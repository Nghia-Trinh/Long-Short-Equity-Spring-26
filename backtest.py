"""
backtest.py — Offline Backtester (pure Python, no QuantConnect required)

Runs the full pipeline using only Data/ CSV files:
    1. Build alpha matrix  (Alpha/)
    2. Build risk matrix   (Risk/)
    3. Optimise portfolio  (Portfolio/)   — day-by-day
    4. Simulate P&L against forward daily returns
    5. Print performance summary
    6. Save outputs to outputs/

Run from the project root:
    python backtest.py

Outputs written to outputs/:
    portfolio_weights.csv   — T × N weight matrix
    pnl.csv                 — daily P&L series

P&L simulation:
    PnL_t = w_t' · r_{t+1}      (weights at t, forward return at t+1)

Performance metrics printed:
    Total P&L, total return, annualised return, annualised volatility,
    Sharpe ratio, max drawdown, win rate
"""

# TODO: implement simulate_pnl(weights, returns) -> pd.Series
#   For each date in weights.index:
#       w_t    = weights.loc[date]
#       r_t+1  = returns.iloc[returns.index.searchsorted(date) + 1]
#       pnl[t+1] = w_t' @ r_t+1

# TODO: implement print_performance(pnl, initial_capital)
#   Compute: total P&L, total return, ann return (x252), ann vol, Sharpe,
#            max drawdown, win rate

# TODO: implement __main__ block:
#   1. Load config.json
#   2. weights_pct = build_portfolio_matrix(config)
#   3. weights_dollar = weights_pct * initial_capital
#   4. pnl = simulate_pnl(weights_dollar, get_returns_pivot())
#   5. print_performance(pnl, initial_capital)
#   6. Save outputs to outputs/portfolio_weights.csv and outputs/pnl.csv
