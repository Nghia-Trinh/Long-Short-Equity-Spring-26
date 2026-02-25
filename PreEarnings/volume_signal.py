"""
volume_signal.py — Earnings-Day Volume Spike Signal

Intuition:
    An earnings-day volume spike signals strong price discovery.
    High-volume repricing events produce larger, more sustained post-earnings
    drifts (see Alpha/sue.py and Data/postearnings_results.csv).

Signal:
    volume_ratio = earnings_day_volume / ADV_20d
    volume_score = log(1 + volume_ratio)     [log-scaled for stability]

    volume_ratio > 2x → high conviction → hold full 10 days, scale up position
    volume_ratio < 1x → low conviction  → exit by day 5, reduce size

Data available:
    Data/prices.csv  →  volume column  ✓  (no additional data required)

Inputs:
    utils/data_loader.py  →  get_volume_pivot(), load_earnings()

Outputs:
    compute_volume_signal(adv_window=20)  →  pd.DataFrame
        columns: ticker, event_date, earnings_day_volume, adv_20d,
                 volume_ratio, volume_score

    get_volume_signal_as_of(date, vol_df)  →  pd.Series (index = ticker)
        most recent volume_score per ticker as of a given date
"""

# TODO: implement compute_volume_signal(adv_window=20) -> pd.DataFrame
#   For each (ticker, event_date) in earnings.csv:
#       1. Look up volume on event_date from prices.csv
#       2. Compute ADV over the prior adv_window trading days
#       3. volume_ratio = event_volume / adv
#       4. volume_score = log1p(volume_ratio)

# TODO: implement get_volume_signal_as_of(date, vol_df=None, adv_window=20)
#           -> pd.Series (index = ticker)
#   Returns most recent volume_score per ticker as of `date`
