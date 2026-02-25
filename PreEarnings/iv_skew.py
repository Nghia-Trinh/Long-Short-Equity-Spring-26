"""
iv_skew.py — Pre-Earnings Options IV Skew Signal  [STUB — not yet implemented]

Intuition:
    Fear before earnings drives investors to buy protective puts, inflating
    downside IV relative to upside IV. Extreme skew is a contrarian signal.

Signal:
    25-delta risk reversal:  RR = put_25d_IV - call_25d_IV
    Normalised by ATM IV  →  dimensionless skew score
    Extreme negative (expensive puts)  → contrarian long bias
    Extreme positive (expensive calls) → contrarian short bias

Data required (NOT YET AVAILABLE):
    Options chain per ticker per day:
        columns: ticker, trade_date, strike, expiry, call_iv, put_iv, delta

Expected outputs:
    IVSkewSignal.compute(earnings_dates) → pd.DataFrame
        columns: ticker, event_date, skew_score

Integration:
    See PreEarnings/README.md for composite alpha weighting plan.
"""

# TODO: implement class IVSkewSignal:
#     __init__(self, skew_lookback_days=5, zscore_window=252)
#     load_options_data(self, path) -> pd.DataFrame
#     compute_risk_reversal(self, options_df) -> pd.DataFrame
#         25-delta RR = put_25d_IV - call_25d_IV, normalised by ATM IV
#     compute(self, earnings_dates) -> pd.DataFrame
#         columns: ticker, event_date, skew_score
