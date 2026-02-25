# Pre-Earnings Signal Module

Signals to trade **before** the earnings announcement, based on options market
data and volume patterns. These are additive to the SUE-based post-earnings
alpha. Implementation is pending until options chain data is sourced.

---

## Signal 1 — Options IV Skew (`iv_skew.py`)

**Intuition:**
Fear before earnings drives investors to buy protective puts, inflating
downside implied volatility relative to upside IV. Extreme skew is a
contrarian signal — the market is over-hedging a move that often doesn't
materialise, or under-hedging one that does.

**Signal construction:**
- 25-delta risk reversal: `RR = put_25d_IV − call_25d_IV`
- Normalised by ATM IV → dimensionless skew score
- Extreme negative (puts expensive) → contrarian long bias
- Extreme positive (calls expensive) → contrarian short bias

**Trade construction:**
Enter long/short position ~3-5 days before earnings in the direction of
the contrarian signal. Exit at or shortly after the announcement.

**Data required:**
Options chain per ticker per day: strike, expiry, call_iv, put_iv, delta

---

## Signal 2 — Volume Spike Detector (`volume_signal.py`)

**Intuition:**
An earnings-day volume spike signals strong price discovery and investor
attention. High-volume repricing events tend to produce larger, more
sustained post-earnings drifts.

**Signal construction:**
```
volume_ratio = earnings_day_volume / ADV_20d
volume_score = log(1 + volume_ratio)
```
- `volume_ratio > 2×` → high conviction → scale up position, hold full 10 days
- `volume_ratio < 1×` → weak conviction → reduce size, tighten stop or exit day 5

**Trade construction:**
Use `volume_score` as a position-size multiplier on top of the SUE signal.

**Data required:**
Daily volume is already available in `Data/prices.csv` ✓

---

## Integration Plan

Once signals are ready, combine with SUE into a composite alpha:

```
alpha_composite = w_sue * alpha_sue + w_skew * alpha_skew + w_vol * alpha_vol
```

Weights `w_*` estimated via time-series cross-validation (walk-forward) on
the 2016-2022 training set, evaluated on 2023-2025 holdout.
