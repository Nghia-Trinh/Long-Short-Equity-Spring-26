"""
dashboard.py — Streamlit live backtest dashboard

Displays the Alpha matrix, Risk (covariance) matrix, and Portfolio
weights evolving through time.  Because there are ~500 tickers, we
show only the most interesting slices at each rebalance date:

  • Alpha:  Top-10 and Bottom-10 tickers by alpha score
  • Risk:   20×20 correlation heatmap of the highest-variance cluster
  • Weights: Long/Short book + exposure KPIs
  • PnL:    Running cumulative return curve

Usage (from project root):
    streamlit run Visualisation/dashboard.py
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Core.signal_blender import SignalBlender
from Risk.risk_matrix import RiskMatrixBuilder
from utils.data_loader import get_returns_pivot, get_tickers, load_earnings

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="L/S Equity — Live Matrices",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Caching: heavy one-time computation
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    p = PROJECT_ROOT / "config.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner="Loading price data …")
def load_returns():
    returns = get_returns_pivot()
    tickers = sorted(set(get_tickers()) & set(returns.columns))
    returns = returns[tickers]
    return returns, tickers


@st.cache_resource(show_spinner="Fitting EWMA covariance model …")
def fit_risk(lam: float, min_periods: int):
    returns, tickers = load_returns()
    builder = RiskMatrixBuilder(lambda_ewma=lam, min_periods=min_periods)
    builder.fit(tickers=tickers, returns=returns)
    return builder


@st.cache_resource(show_spinner="Building alpha matrix …")
def build_blender(rebalance_dates_tuple, tickers_tuple, config_json: str):
    cfg = json.loads(config_json)
    rebalance_dates = pd.DatetimeIndex(list(rebalance_dates_tuple))
    tickers = list(tickers_tuple)
    blender = SignalBlender(
        tickers=tickers,
        rebalance_dates=rebalance_dates,
        blend_weight_systematic=cfg.get("blend_weight_systematic", 0.6),
        blend_weight_event=cfg.get("blend_weight_event", 0.25),
        blend_weight_thesis=cfg.get("blend_weight_thesis", 0.15),
        pre_earnings_window=int(cfg.get("pre_earnings_window", 5)),
        config=cfg,
    )
    return blender


@st.cache_data(show_spinner="Loading portfolio weights …")
def load_weights():
    p = PROJECT_ROOT / "outputs" / "portfolio_weights.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df


@st.cache_data(show_spinner="Loading earnings calendar …")
def load_earnings_calendar():
    e = load_earnings()[["ticker", "event_date"]].copy()
    e["ticker"] = e["ticker"].astype(str).str.strip().str.upper()
    e["event_date"] = pd.to_datetime(e["event_date"], errors="coerce")
    e = e.dropna(subset=["event_date"]).sort_values("event_date")
    return e


@st.cache_data(show_spinner="Loading PnL …")
def load_pnl():
    p = PROJECT_ROOT / "outputs" / "pnl.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.squeeze()


# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
config = _load_config()
returns, tickers = load_returns()
risk_builder = fit_risk(
    lam=config.get("lambda_ewma", 0.94),
    min_periods=config.get("ewma_min_periods", 60),
)

# Rebalance dates (from saved weights for consistency)
weights_df = load_weights()
rebalance_dates = weights_df.index
tickers = sorted(set(tickers) & set(weights_df.columns))

blender = build_blender(
    rebalance_dates_tuple=tuple(rebalance_dates),
    tickers_tuple=tuple(tickers),
    config_json=json.dumps(config),
)

pnl_series = load_pnl()

N_DATES = len(rebalance_dates)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️  Controls")

# Session-state defaults
if "_step" not in st.session_state:
    st.session_state["_step"] = 0
if "playing" not in st.session_state:
    st.session_state["playing"] = False

# Play / Pause / Reset row  (before slider so we can advance _step first)
col_play, col_reset = st.sidebar.columns(2)
with col_play:
    label = "⏸  Pause" if st.session_state["playing"] else "▶  Play"
    if st.button(label, use_container_width=True):
        st.session_state["playing"] = not st.session_state["playing"]
        st.rerun()
with col_reset:
    if st.button("⏮  Reset", use_container_width=True):
        st.session_state["playing"] = False
        st.session_state["_step"] = 0
        st.rerun()

# Date slider (value driven by _step so auto-play can move it)
date_idx = st.sidebar.slider(
    "Rebalance step",
    min_value=0,
    max_value=N_DATES - 1,
    value=st.session_state["_step"],
)
# Sync back: if user drags the slider manually, update _step
st.session_state["_step"] = date_idx

speed = st.sidebar.select_slider(
    "Playback speed", options=["Slow", "Normal", "Fast", "Turbo"],
    value="Fast",
)
speed_map = {"Slow": 1.0, "Normal": 0.3, "Fast": 0.08, "Turbo": 0.01}

# Top-N controls
top_n = st.sidebar.slider("Top/Bottom N tickers (Alpha)", 5, 30, 10)
heatmap_n = st.sidebar.slider("Risk heatmap size", 10, 40, 20)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Universe**: {len(tickers)} tickers  \n"
    f"**Rebalance dates**: {N_DATES}  \n"
    f"**Range**: {rebalance_dates[0].strftime('%Y-%m-%d')} → "
    f"{rebalance_dates[-1].strftime('%Y-%m-%d')}"
)

# ---------------------------------------------------------------------------
# Compute data for the selected date
# ---------------------------------------------------------------------------
current_date = rebalance_dates[date_idx]

# Alpha vector
alpha = blender.get_alpha_vector(current_date)
alpha_s = pd.Series(alpha, index=tickers)

# Risk matrix
try:
    sigma = risk_builder.get(current_date)
except KeyError:
    sigma = np.eye(len(tickers))

# Weights
w = weights_df.loc[current_date].reindex(tickers, fill_value=0.0)

# PnL up to this date
pnl_to_date = pnl_series.loc[:current_date]
cum_pnl = pnl_to_date.cumsum()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊  Long-Short Equity — Live Backtest Dashboard")
st.markdown(
    f"### Rebalance date: **{current_date.strftime('%Y-%m-%d')}**  "
    f"(step {date_idx + 1} / {N_DATES})"
)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
longs = (w > 1e-6).sum()
shorts = (w < -1e-6).sum()
gross = float(w.abs().sum())
net = float(w.sum())
cum_ret = float(cum_pnl.iloc[-1]) if len(cum_pnl) > 0 else 0.0
ann_vol = float(pnl_to_date.std() * np.sqrt(252)) if len(pnl_to_date) > 10 else 0.0

k1.metric("Long", int(longs))
k2.metric("Short", int(shorts))
k3.metric("Gross Exp.", f"{gross:.2f}×")
k4.metric("Net Exp.", f"{net:+.4f}")
k5.metric("Cum Return", f"{cum_ret:.2%}")
k6.metric("Ann Vol", f"{ann_vol:.2%}")

# ---------------------------------------------------------------------------
# Upcoming Earnings (next 30 days from current rebalance date)
# ---------------------------------------------------------------------------
earnings_cal = load_earnings_calendar()
earnings_cal = earnings_cal[earnings_cal["ticker"].isin(tickers)]

# Look forward: earnings in the next 30 days from the current rebalance date
fwd_start = current_date
fwd_end = current_date + pd.Timedelta(days=30)
upcoming = earnings_cal[
    (earnings_cal["event_date"] > fwd_start) & (earnings_cal["event_date"] <= fwd_end)
].copy()
upcoming["days_away"] = (upcoming["event_date"] - current_date).dt.days
upcoming = upcoming.sort_values("event_date")
# Deduplicate (keep earliest per ticker)
upcoming = upcoming.drop_duplicates(subset="ticker", keep="first")

st.divider()
st.subheader(f"📅  Upcoming Earnings — next 30 days ({fwd_start.strftime('%b %d')} → {fwd_end.strftime('%b %d, %Y')})")

if upcoming.empty:
    st.info("No earnings events in the next 30 days for this rebalance date.")
else:
    ue_k1, ue_k2, ue_k3 = st.columns(3)
    ue_k1.metric("Tickers reporting", len(upcoming))
    this_week = upcoming[upcoming["days_away"] <= 7]
    ue_k2.metric("This week", len(this_week))
    next_week = upcoming[(upcoming["days_away"] > 7) & (upcoming["days_away"] <= 14)]
    ue_k3.metric("Next week", len(next_week))

    # Merge with alpha & weight info for context
    upcoming_display = upcoming[["ticker", "event_date", "days_away"]].copy()
    upcoming_display["alpha"] = upcoming_display["ticker"].map(alpha_s)
    upcoming_display["weight"] = upcoming_display["ticker"].map(w)
    upcoming_display["event_date"] = upcoming_display["event_date"].dt.strftime("%Y-%m-%d")
    upcoming_display = upcoming_display.rename(columns={
        "ticker": "Ticker", "event_date": "Earnings Date",
        "days_away": "Days Away", "alpha": "Alpha", "weight": "Weight",
    })

    col_tbl, col_timeline = st.columns([3, 2])
    with col_tbl:
        st.dataframe(
            upcoming_display.style
            .format({"Alpha": "{:+.3f}", "Weight": "{:+.2%}"})
            .background_gradient(subset=["Alpha"], cmap="RdYlGn", vmin=-2, vmax=2)
            .background_gradient(subset=["Weight"], cmap="RdYlGn", vmin=-0.02, vmax=0.02),
            use_container_width=True,
            height=min(len(upcoming_display) * 35 + 40, 500),
            hide_index=True,
        )
    with col_timeline:
        # Scatter: days away vs alpha, colored by weight direction
        fig_sched = go.Figure(go.Scatter(
            x=upcoming_display["Days Away"],
            y=upcoming_display["Alpha"],
            mode="markers+text",
            text=upcoming_display["Ticker"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=10,
                color=upcoming_display["Weight"],
                colorscale="RdYlGn",
                cmin=-0.02, cmax=0.02,
                colorbar=dict(title="Weight"),
                line=dict(width=0.5, color="gray"),
            ),
        ))
        fig_sched.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis_title="Days until earnings",
            yaxis_title="Current alpha score",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sched, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 1: Alpha + Risk
# ---------------------------------------------------------------------------
col_alpha, col_risk = st.columns(2)

# --- Alpha panel ---
with col_alpha:
    st.subheader("🔵  Alpha Matrix — Top / Bottom movers")

    top = alpha_s.nlargest(top_n)
    bottom = alpha_s.nsmallest(top_n)
    combined = pd.concat([bottom, top])
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in combined.values]

    fig_alpha = go.Figure(go.Bar(
        x=combined.values,
        y=combined.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in combined.values],
        textposition="outside",
    ))
    fig_alpha.update_layout(
        height=max(400, top_n * 22),
        margin=dict(l=80, r=20, t=10, b=10),
        xaxis_title="Alpha score (z-scored)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

    # Alpha dispersion metric
    st.caption(
        f"Cross-sectional std: **{alpha_s.std():.4f}** · "
        f"Mean: {alpha_s.mean():.4f} · "
        f"Skew: {float(alpha_s.skew()):.2f}"
    )

# --- Risk panel (correlation heatmap) ---
with col_risk:
    st.subheader("🔴  Risk Matrix — Correlation heatmap")

    # Pick the top-N tickers by marginal variance (diagonal)
    diag = np.diag(sigma)
    diag_s = pd.Series(diag, index=tickers)
    top_var_tickers = diag_s.nlargest(heatmap_n).index.tolist()
    idx_mask = [tickers.index(t) for t in top_var_tickers]

    sub_sigma = sigma[np.ix_(idx_mask, idx_mask)]
    # Convert covariance to correlation
    vol = np.sqrt(np.diag(sub_sigma))
    vol[vol == 0] = 1e-8
    corr = sub_sigma / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)

    fig_risk = px.imshow(
        corr,
        x=top_var_tickers,
        y=top_var_tickers,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="equal",
    )
    fig_risk.update_layout(
        height=max(400, heatmap_n * 22),
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar_title="ρ",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Eigenvalue summary
    eigs = np.sort(np.linalg.eigvalsh(sigma))[::-1][:5]
    st.caption(
        f"Top eigenvalues: " +
        " · ".join(f"λ{i+1}={v:.2e}" for i, v in enumerate(eigs))
    )

st.divider()

# ---------------------------------------------------------------------------
# Row 2: Weights + PnL
# ---------------------------------------------------------------------------
col_wt, col_pnl = st.columns(2)

# --- Weights panel ---
with col_wt:
    st.subheader("📦  Portfolio Weights")

    w_nonzero = w[w.abs() > 1e-6].sort_values()
    if len(w_nonzero) > 40:
        # Show top-20 longs and bottom-20 shorts
        show = pd.concat([w_nonzero.head(20), w_nonzero.tail(20)])
    else:
        show = w_nonzero

    colors_w = ["#ef4444" if v < 0 else "#22c55e" for v in show.values]
    fig_w = go.Figure(go.Bar(
        x=show.values,
        y=show.index,
        orientation="h",
        marker_color=colors_w,
        text=[f"{v:+.2%}" for v in show.values],
        textposition="outside",
    ))
    fig_w.update_layout(
        height=max(400, len(show) * 18),
        margin=dict(l=80, r=20, t=10, b=10),
        xaxis_title="Weight",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_w, use_container_width=True)

# --- PnL panel ---
with col_pnl:
    st.subheader("📈  Cumulative PnL")

    fig_pnl = go.Figure()
    # Convert index to strings to avoid plotly Timestamp arithmetic bug
    pnl_x = [d.strftime("%Y-%m-%d") for d in cum_pnl.index]
    fig_pnl.add_trace(go.Scatter(
        x=pnl_x,
        y=cum_pnl.values,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59,130,246,0.15)",
        name="Cum Return",
    ))
    # Add a vertical line at the current date
    date_str = current_date.strftime("%Y-%m-%d")
    fig_pnl.add_shape(
        type="line", x0=date_str, x1=date_str, y0=0, y1=1,
        yref="paper", line=dict(color="red", dash="dash", width=1.5),
    )
    fig_pnl.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=10, b=30),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".1%",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Drawdown
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    sharpe = (float(pnl_to_date.mean()) * 252) / ann_vol if ann_vol > 0 else 0.0
    st.caption(
        f"Sharpe: **{sharpe:.2f}** · "
        f"Max drawdown: **{max_dd:.2%}** · "
        f"Win rate: **{(pnl_to_date > 0).mean():.1%}**"
    )

# ---------------------------------------------------------------------------
# Row 3: Alpha delta (change from previous rebalance)
# ---------------------------------------------------------------------------
if date_idx > 0:
    st.divider()
    st.subheader("🔄  Alpha Changes — Biggest moves since last rebalance")

    prev_date = rebalance_dates[date_idx - 1]
    alpha_prev = blender.get_alpha_vector(prev_date)
    alpha_delta = pd.Series(alpha - alpha_prev, index=tickers)

    biggest_up = alpha_delta.nlargest(top_n)
    biggest_down = alpha_delta.nsmallest(top_n)
    delta_show = pd.concat([biggest_down, biggest_up])

    col_dup, col_ddown = st.columns(2)
    with col_dup:
        st.markdown("**⬆️ Biggest alpha upgrades**")
        st.dataframe(
            biggest_up.to_frame("Δ alpha").style.format("{:+.4f}")
            .background_gradient(cmap="Greens"),
            use_container_width=True,
        )
    with col_ddown:
        st.markdown("**⬇️ Biggest alpha downgrades**")
        st.dataframe(
            biggest_down.to_frame("Δ alpha").style.format("{:+.4f}")
            .background_gradient(cmap="Reds_r"),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Row 4: Risk variance changes
# ---------------------------------------------------------------------------
if date_idx > 0:
    st.subheader("🔄  Risk Changes — Variance shifts since last rebalance")

    try:
        sigma_prev = risk_builder.get(prev_date)
        var_now = pd.Series(np.diag(sigma), index=tickers)
        var_prev = pd.Series(np.diag(sigma_prev), index=tickers)
        var_delta = var_now - var_prev
        var_pct = (var_delta / var_prev.replace(0, np.nan)).fillna(0)

        col_vup, col_vdown = st.columns(2)
        with col_vup:
            st.markdown("**⬆️ Variance increased most**")
            top_v = var_pct.nlargest(top_n)
            st.dataframe(
                pd.DataFrame({
                    "σ² now": var_now[top_v.index],
                    "σ² prev": var_prev[top_v.index],
                    "Δ%": var_pct[top_v.index],
                }).style.format({"σ² now": "{:.6f}", "σ² prev": "{:.6f}", "Δ%": "{:+.1%}"})
                .background_gradient(subset=["Δ%"], cmap="Reds"),
                use_container_width=True,
            )
        with col_vdown:
            st.markdown("**⬇️ Variance decreased most**")
            bot_v = var_pct.nsmallest(top_n)
            st.dataframe(
                pd.DataFrame({
                    "σ² now": var_now[bot_v.index],
                    "σ² prev": var_prev[bot_v.index],
                    "Δ%": var_pct[bot_v.index],
                }).style.format({"σ² now": "{:.6f}", "σ² prev": "{:.6f}", "Δ%": "{:+.1%}"})
                .background_gradient(subset=["Δ%"], cmap="Greens_r"),
                use_container_width=True,
            )
    except Exception:
        st.info("Could not compute risk delta for this date.")

# ---------------------------------------------------------------------------
# Auto-play: advance one step per rerun while playing
# ---------------------------------------------------------------------------
if st.session_state["playing"]:
    if st.session_state["_step"] < N_DATES - 1:
        time.sleep(speed_map[speed])
        st.session_state["_step"] += 1
        st.rerun()
    else:
        # Reached the end — stop
        st.session_state["playing"] = False
        st.toast("✅ Playback complete!", icon="🏁")
