"""
main.py — QuantConnect LEAN Algorithm Entry Point

This is the top-level algorithm file for running the Earnings Sentiment
Long-Short strategy on QuantConnect (cloud or LEAN CLI local backtest).

To run locally with LEAN CLI:
    lean backtest "Long-Short-Equity-Spring-26"

To run on QuantConnect Cloud:
    Upload this directory via quantconnect.com → Projects → Upload

Architecture:
    OnData / Rebalance  →  called daily by LEAN scheduler
        1. Update EWMA covariance from QC price history (Risk/)
        2. Fetch latest SUE for active universe (Alpha/)
        3. Solve MVO + turnover penalty optimisation (Portfolio/)
        4. Execute trades via SetHoldings()
        5. Apply hard exit for positions past holding_period_days

Universe:
    QC coarse-fundamental filter → Russell 3000 ex-large-cap proxy
    (top-N tickers by dollar volume excluded; configurable in config.json)

Exit logic:
    Soft: turnover penalty in the optimiser naturally decays stale positions
    Hard: force-liquidate any position held > holding_period_days

See README.md for full strategy description and configuration reference.
"""

# TODO: import QuantConnect AlgorithmImports (LEAN runtime)
# TODO: import Alpha, Risk, Portfolio modules

# TODO: implement class EarningsSentimentAlgorithm(QCAlgorithm):
#
#   def Initialize(self):
#       Load config.json
#       SetStartDate / SetEndDate / SetCash
#       SetBrokerageModel (Interactive Brokers, Margin)
#       AddUniverse(self._select_universe)
#       Pre-load SUE cache via compute_sue()
#       Schedule.On(EveryDay, AfterMarketOpen("SPY", 30), self.Rebalance)
#
#   def _select_universe(self, fundamental):
#       Sort by DollarVolume descending
#       Exclude top-N (large-cap proxy)
#       Require DollarVolume > min_dollar_volume
#       Return up to 500 symbols
#
#   def Rebalance(self):
#       1. Collect active tickers from self.ActiveSecurities
#       2. Pull 63-day price History, compute returns, update EWMA Sigma
#       3. get_latest_sue_as_of(self.Time) → alpha vector
#       4. Cross-sectional z-score alpha
#       5. optimize_portfolio(alpha, sigma, w_prev, ...) → w_opt
#       6. SetHoldings for each ticker
#       7. Hard-exit positions past holding_period_days
#       8. Update self._w_prev and self._entry_dates
