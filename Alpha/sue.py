"""
sue.py — Standardised Unexpected Earnings (SUE)

Formula:
    SUE_{i,t} = (EPS_actual_{i,t} - EPS_estimate_{i,t}) / sigma_i

where sigma_i is the rolling std of past forecast errors for ticker i
over the last `lookback_quarters` quarters (default: 8).

Inputs:
    Data/earnings.csv  →  columns: ticker, event_date, eps_estimate, eps_actual

Outputs:
    - compute_sue()            → pd.DataFrame (ticker, event_date, forecast_error,
                                               sigma_forecast_error, sue)
    - get_latest_sue_as_of()   → pd.Series indexed by ticker, most recent SUE
                                 as of a given date

Reference:
    Bernard & Thomas (1989), "Post-Earnings-Announcement Drift:
    Delayed Price Response or Risk Premium?"
"""

# TODO: implement compute_sue(lookback_quarters: int = 8) -> pd.DataFrame
# TODO: implement get_latest_sue_as_of(date, sue_df) -> pd.Series
