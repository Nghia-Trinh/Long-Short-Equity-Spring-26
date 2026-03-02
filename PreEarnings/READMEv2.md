# PreEarnings v2

## Architecture

The v2 PreEarnings block uses a three-stage pipeline:

1. **Setup Gate** (`setup_gate.py`): checks if the event is tradable.
2. **Core Signal** (`iv_skew.py`, `volume_signal.py`): determines directional context.
3. **Sentiment Sizer** (`sentiment_sizer.py`): adjusts position size continuously.

The orchestrator (`pre_earnings_signal.py`) runs the pipeline event-by-event and returns:
- `direction`: `"long" | "short" | None`
- `position_size`: `base_position_size * multiplier`
- `diagnostics`: gate/sizer/core/coverage context

## New Feature Layer

`feature_engine.py` computes per-event:
- `price_drift` (5-day pre-event return)
- `rv20d_vs_pre1d` (`rv20d_annualized / max(abs(pre_ret_1d), 0.001)`)
- `net_delta` (mean daily `delta * open_interest`)
- `call_put_ratio` (mean daily `call_volume / put_volume`)
- `has_options` (coverage flag for event window)

`normalizer.py` provides:
- `winsorize(series, sigma)`
- `robust_z(series, winsor_sigma)`

`call_put_ratio` is log-transformed before z-scoring in the orchestrator due to heavy tails.

## Setup Gate Logic

Gate uses strict AND logic:
- `z_drift <= gate_drift_threshold` (contrarian setup; default `-1.0`)
- `z_vol >= gate_vol_threshold` (high-vol regime; default `0.5`)

If either check fails, the event is non-tradable.

## Sentiment Sizer Logic

Sizing uses:

`w_delta = clip(1 - a * z_nd, 0.5, 1.5)`  
`w_cpr = 1 + b * I(|z_cpr| >= 1.5)`  
`risk_scale = 1 / (1 + k * max(0, z_vol))`  
`multiplier = w_delta * w_cpr * risk_scale`

Defaults:
- `a = 0.3`
- `b = 0.2`
- `k = 0.5`

## Coverage-Aware Fallback Rules

- Missing `net_delta` → `w_delta = 1.0`
- Missing `call_put_ratio` → `w_cpr = 1.0`
- Missing both → multiplier reduces to `risk_scale`
- Missing options coverage (`has_options = False`) → skip event (`direction=None`, `position_size=0`)

## OOS Kill-Switch (Configuration)

Configuration keys are added for an out-of-sample kill-switch controller:
- `oos_killswitch_quarters` (default `8`)
- `oos_killswitch_min_edge_bps` (default `10`)

These are included in `config.json` for conservative staged rollout and monitoring.

## Alpha Matrix Integration

`Alpha/alpha_matrix.py` now accepts:
- `preearnings_df` with columns `ticker`, `event_date`, `preearnings_score`
- weight `w_preearnings` from config (default `0.0` for shadow mode)

Composite alpha now supports:

`w_sue * sue + w_skew * skew_score + w_vol * volume_score + w_preearnings * preearnings_score`

Set `w_preearnings > 0` only after validation.
