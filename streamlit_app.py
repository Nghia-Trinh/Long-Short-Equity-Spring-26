"""Streamlit deployment for the thesis-aware long/short equity model."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
FRONTEND_DIR = PROJECT_ROOT / "frontend"

from LLM.long_short_model import export_model_run, run_long_short_model


def _read_frontend_file(filename: str) -> str:
    return (FRONTEND_DIR / filename).read_text(encoding="utf-8")


def _build_embedded_page() -> str:
    index_html = _read_frontend_file("index.html")
    css = _read_frontend_file("styles.css")
    js = _read_frontend_file("app.js")
    index_html = index_html.replace('<link rel="stylesheet" href="./styles.css" />', f"<style>{css}</style>")
    index_html = index_html.replace('<script src="./app.js"></script>', f"<script>{js}</script>")
    return index_html


@st.cache_data(show_spinner="Running four-month LLM long/short model...")
def load_model(months: int, use_finnhub: bool):
    run = run_long_short_model(months=months, use_finnhub=use_finnhub)
    export_model_run(run)
    return run


def fmt_pct(value: float | int | str) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def fmt_num(value: float | int | str) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def render_metric_row(metrics: dict[str, float | str]) -> None:
    cols = st.columns(6)
    cols[0].metric("4M Return", fmt_pct(metrics["cumulative_return"]))
    cols[1].metric("Sharpe", fmt_num(metrics["sharpe"]))
    cols[2].metric("Ann. Vol", fmt_pct(metrics["annualized_volatility"]))
    cols[3].metric("Max DD", fmt_pct(metrics["max_drawdown"]))
    cols[4].metric("Gross", fmt_num(metrics["gross_exposure"]))
    cols[5].metric("Net", fmt_num(metrics["net_exposure"]))


def render_results(run) -> None:
    render_metric_row(run.metrics)
    st.caption(
        f"Paper-trade window: {run.metrics['start_date']} to {run.metrics['end_date']} | "
        f"Finnhub status: {run.data_status['finnhub_news_status']} | "
        f"1-3 week proxy hold: {run.data_status['holding_period_trading_days']} trading days"
    )

    pnl = run.pnl.reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pnl["trade_date"],
            y=pnl["cumulative_return"],
            mode="lines",
            name="Cumulative return",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig.update_layout(height=360, yaxis_tickformat=".1%", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    latest_date = run.weights.index.max()
    latest = pd.DataFrame(
        {
            "Ticker": run.weights.columns,
            "Alpha": run.alpha_matrix.loc[latest_date].values,
            "Risk": run.risk_matrix.loc[latest_date].values,
            "Weight": run.weights.loc[latest_date].values,
        }
    ).sort_values("Alpha", ascending=False)
    latest["Side"] = latest["Weight"].map(lambda w: "Long" if w > 0 else "Short" if w < 0 else "Flat")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Latest ranking: long top / short bottom")
        st.dataframe(
            latest.style.format({"Alpha": "{:+.2f}", "Risk": "{:.5f}", "Weight": "{:+.2%}"}),
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.subheader("Weight sizing matrix")
        fig_w = px.bar(latest, x="Ticker", y="Weight", color="Side", color_discrete_map={"Long": "#16a34a", "Short": "#dc2626", "Flat": "#64748b"})
        fig_w.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_w, use_container_width=True)


def render_matrices(run) -> None:
    st.subheader("Alpha Matrix")
    st.write("Built from call/put imbalance proxy, net delta proxy, IV skew proxy, Sharpe rank, QMJ proxy, thesis score, and Finnhub news sentiment.")
    st.dataframe(run.alpha_matrix.style.format("{:+.2f}"), use_container_width=True)
    fig_a = px.imshow(run.alpha_matrix.T, color_continuous_scale="RdYlGn", aspect="auto", title="Alpha Matrix")
    fig_a.update_layout(height=360)
    st.plotly_chart(fig_a, use_container_width=True)

    st.subheader("Risk Matrix")
    st.write("EWMA marginal variance by rebalance date; full covariance is used inside weight sizing.")
    st.dataframe(run.risk_matrix.style.format("{:.6f}"), use_container_width=True)
    fig_r = px.imshow(run.risk_matrix.T, color_continuous_scale="Reds", aspect="auto", title="Risk Matrix")
    fig_r.update_layout(height=360)
    st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("Signal snapshot")
    st.dataframe(
        run.signal_snapshot.tail(40).style.format(
            {
                "call_put_imbalance": "{:+.2f}",
                "net_delta": "{:+.2f}",
                "iv_skew": "{:+.2f}",
                "sharpe_z": "{:+.2f}",
                "qmj_z": "{:+.2f}",
                "alpha": "{:+.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_theses_and_news(run) -> None:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Investment thesis basket")
        st.dataframe(
            run.thesis_basket.style.format({"thesis_score": "{:+.2f}", "thesis_confidence": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )
        excluded = run.data_status.get("excluded_tickers", [])
        if excluded:
            st.info(f"Excluded from paper trade because Data/prices.csv has no history: {', '.join(excluded)}")
    with c2:
        st.subheader("LLM overlay role")
        st.markdown(
            "- Human analysts define the thesis basket and long/short intent.\n"
            "- The LLM layer scores thesis tone and Finnhub news tone.\n"
            "- Risk calculations and sizing are automated into alpha, risk, and weight matrices."
        )

    st.subheader("Finnhub financial news and earnings-data access")
    if run.news.empty:
        st.warning("No Finnhub token is configured, so the model used thesis and market-data proxies only.")
    else:
        st.dataframe(run.news.sort_values("datetime", ascending=False).head(30), use_container_width=True, hide_index=True)


def render_copilot() -> None:
    if not FRONTEND_DIR.exists():
        st.error("Frontend assets not found. Expected folder: frontend/")
        return
    html(_build_embedded_page(), height=1150, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="LLM Long/Short Equity", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.title("LLM-Assisted Long/Short Equity Trading Model")
    st.caption("Alpha Matrix + Risk Matrix + Portfolio Weight Sizing Matrix for thesis-driven U.S. equities.")

    st.sidebar.header("Model Controls")
    months = st.sidebar.slider("Result window (months)", min_value=1, max_value=6, value=4)
    use_finnhub = st.sidebar.checkbox("Use Finnhub API when token is configured", value=True)
    st.sidebar.caption("Set FINNHUB_API_KEY, FINNHUB_KEY, or FINNHUB_TOKEN to enrich with live Finnhub company news.")

    run = load_model(months=months, use_finnhub=use_finnhub)
    tabs = st.tabs(["Model Results", "Matrices", "Theses + Finnhub", "Investment Thesis Copilot"])
    with tabs[0]:
        render_results(run)
    with tabs[1]:
        render_matrices(run)
    with tabs[2]:
        render_theses_and_news(run)
    with tabs[3]:
        render_copilot()


if __name__ == "__main__":
    main()
