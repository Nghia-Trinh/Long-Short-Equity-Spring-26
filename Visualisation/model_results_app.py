"""
Interactive model results UI for investment thesis testing.

Run:
    streamlit run Visualisation/model_results_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEEDBACK_PATH = OUTPUTS_DIR / "model_ui_feedback.csv"


@dataclass(frozen=True)
class ModelPreset:
    name: str
    signal_strength: float
    risk_penalty: float
    turnover_penalty: float


PRESETS = {
    "Baseline (production-like)": ModelPreset(
        name="Baseline (production-like)",
        signal_strength=1.0,
        risk_penalty=1.0,
        turnover_penalty=1.0,
    ),
    "Aggressive Growth Tilt": ModelPreset(
        name="Aggressive Growth Tilt",
        signal_strength=1.25,
        risk_penalty=0.8,
        turnover_penalty=0.75,
    ),
    "Defensive Quality Tilt": ModelPreset(
        name="Defensive Quality Tilt",
        signal_strength=0.85,
        risk_penalty=1.2,
        turnover_penalty=1.25,
    ),
}


def apply_theme(selected_theme: str) -> None:
    dark = selected_theme == "Dark"
    bg = "#0e1117" if dark else "#f8fafc"
    surface = "#151925" if dark else "#ffffff"
    text = "#e5e7eb" if dark else "#111827"
    subtext = "#cbd5e1" if dark else "#4b5563"
    border = "#334155" if dark else "#dbe2ea"
    accent = "#60a5fa" if dark else "#2563eb"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text};
        }}
        [data-testid="stSidebar"] {{
            background-color: {surface};
            border-right: 1px solid {border};
        }}
        .card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1rem 1rem 0.8rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }}
        .kpi {{
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin-top: 0.15rem;
            margin-bottom: 0.2rem;
            color: {text};
        }}
        .kpi-label {{
            color: {subtext};
            font-size: 0.88rem;
            font-weight: 500;
            margin-bottom: 0.2rem;
        }}
        .section-title {{
            margin-top: 0.35rem;
            margin-bottom: 0.6rem;
            color: {text};
            font-size: 1.05rem;
            font-weight: 600;
        }}
        .theme-note {{
            color: {subtext};
            font-size: 0.88rem;
            padding-top: 0.25rem;
            padding-bottom: 0.2rem;
        }}
        .accent {{
            color: {accent};
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights_path = OUTPUTS_DIR / "portfolio_weights.csv"
    pnl_path = OUTPUTS_DIR / "pnl.csv"

    if not weights_path.exists() or not pnl_path.exists():
        raise FileNotFoundError(
            "Missing outputs data. Expected outputs/portfolio_weights.csv and outputs/pnl.csv."
        )

    weights = pd.read_csv(weights_path)
    pnl = pd.read_csv(pnl_path)

    date_col_weights = weights.columns[0]
    weights = weights.rename(columns={date_col_weights: "rebalance_date"})
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"], errors="coerce")
    weights = weights.dropna(subset=["rebalance_date"]).sort_values("rebalance_date")
    weights = weights.set_index("rebalance_date")

    date_col_pnl = pnl.columns[0]
    ret_col = pnl.columns[1]
    pnl = pnl.rename(columns={date_col_pnl: "trade_date", ret_col: "portfolio_return"})
    pnl["trade_date"] = pd.to_datetime(pnl["trade_date"], errors="coerce")
    pnl["portfolio_return"] = pd.to_numeric(pnl["portfolio_return"], errors="coerce").fillna(0.0)
    pnl = pnl.dropna(subset=["trade_date"]).sort_values("trade_date")
    return weights, pnl


def parse_uploaded_thesis(file_name: str, raw_text: str) -> dict[str, float | str]:
    lowered = raw_text.lower()
    line_count = len([line for line in raw_text.splitlines() if line.strip()])
    has_risk = any(token in lowered for token in ("risk", "drawdown", "volatility", "hedge"))
    has_growth = any(token in lowered for token in ("growth", "upside", "expansion", "accelerat"))
    has_value = any(token in lowered for token in ("value", "multiple", "margin of safety", "discount"))

    sentiment_proxy = (
        lowered.count("buy")
        + lowered.count("overweight")
        + lowered.count("upside")
        - lowered.count("sell")
        - lowered.count("underweight")
        - lowered.count("downside")
    )

    tone_score = max(min(sentiment_proxy / 8.0, 2.0), -2.0)

    thesis_style = "Balanced"
    if has_growth and not has_value:
        thesis_style = "Growth-led"
    elif has_value and not has_growth:
        thesis_style = "Value-led"
    elif has_risk and (has_growth or has_value):
        thesis_style = "Risk-aware"

    return {
        "file_name": file_name,
        "line_count": float(line_count),
        "tone_score": float(tone_score),
        "has_risk_language": float(1 if has_risk else 0),
        "thesis_style": thesis_style,
    }


def parse_uploaded_files(uploaded_files: Iterable) -> pd.DataFrame:
    parsed_rows: list[dict[str, float | str]] = []
    for file in uploaded_files:
        content = file.getvalue()
        file_name = file.name
        suffix = Path(file_name).suffix.lower()

        if suffix in {".txt", ".md"}:
            text = content.decode("utf-8", errors="ignore")
            parsed_rows.append(parse_uploaded_thesis(file_name, text))
            continue

        if suffix == ".csv":
            try:
                text = content.decode("utf-8", errors="ignore")
                df = pd.read_csv(StringIO(text), nrows=1)
                cols = ", ".join(df.columns.astype(str).tolist()[:8])
            except Exception:
                cols = "unreadable columns"
            parsed_rows.append(
                {
                    "file_name": file_name,
                    "line_count": 0.0,
                    "tone_score": 0.0,
                    "has_risk_language": 0.0,
                    "thesis_style": f"Structured CSV ({cols})",
                }
            )
            continue

        if suffix == ".json":
            try:
                payload = json.loads(content.decode("utf-8", errors="ignore"))
                key_count = len(payload) if isinstance(payload, dict) else 0
                style = f"JSON ({key_count} keys)"
            except Exception:
                style = "JSON (unparseable)"
            parsed_rows.append(
                {
                    "file_name": file_name,
                    "line_count": 0.0,
                    "tone_score": 0.0,
                    "has_risk_language": 0.0,
                    "thesis_style": style,
                }
            )
            continue

        parsed_rows.append(
            {
                "file_name": file_name,
                "line_count": 0.0,
                "tone_score": 0.0,
                "has_risk_language": 0.0,
                "thesis_style": "Unsupported format (processed as metadata only)",
            }
        )

    return pd.DataFrame(parsed_rows)


def model_adjusted_returns(
    base_pnl: pd.DataFrame,
    thesis_features: pd.DataFrame,
    preset: ModelPreset,
    signal_boost: float,
    risk_aversion: float,
    thesis_weight: float,
) -> pd.DataFrame:
    df = base_pnl.copy()
    raw_returns = df["portfolio_return"].astype(float)
    positive = raw_returns.clip(lower=0.0)
    negative = raw_returns.clip(upper=0.0)

    thesis_tone = 0.0
    thesis_risk_pref = 0.0
    if not thesis_features.empty:
        thesis_tone = float(thesis_features["tone_score"].mean())
        thesis_risk_pref = float(thesis_features["has_risk_language"].mean())

    thesis_modifier = 1.0 + thesis_weight * 0.05 * thesis_tone
    risk_modifier = 1.0 + max(risk_aversion, 0.0) * 0.35 + thesis_risk_pref * 0.1
    upside_modifier = max(signal_boost, 0.0) * preset.signal_strength * thesis_modifier

    adjusted = positive * upside_modifier + negative / max(risk_modifier * preset.risk_penalty, 0.25)
    turnover_drag = (preset.turnover_penalty - 1.0) * 0.00015
    df["model_return"] = adjusted - turnover_drag
    df["cum_return"] = (1.0 + df["model_return"]).cumprod() - 1.0
    return df


def summary_stats(returns: pd.Series) -> dict[str, float]:
    s = returns.fillna(0.0).astype(float)
    cumulative = float((1.0 + s).prod() - 1.0)
    ann_return = float(s.mean() * 252)
    ann_vol = float(s.std() * (252 ** 0.5))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    wealth = (1.0 + s).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }


def save_feedback_row(row: dict[str, str | float]) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if FEEDBACK_PATH.exists():
        prior = pd.read_csv(FEEDBACK_PATH)
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(FEEDBACK_PATH, index=False)


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">{label}</div>
            <div class="kpi">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Model Results UI — Investment Theses",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("App Controls")
    selected_theme = st.sidebar.radio("Theme", options=["Light", "Dark"], index=0)
    apply_theme(selected_theme)

    st.title("Model Results UI for Investment Thesis Testing")
    st.markdown(
        "Upload your thesis files, test model versions, adjust parameters, and capture feedback in one interface."
    )

    st.sidebar.markdown("### Model Version")
    preset_name = st.sidebar.selectbox("Version preset", options=list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    st.sidebar.markdown("### Adjustment Levers")
    signal_boost = st.sidebar.slider("Signal boost", min_value=0.5, max_value=1.8, value=1.0, step=0.05)
    risk_aversion = st.sidebar.slider("Risk aversion", min_value=0.2, max_value=2.2, value=1.0, step=0.1)
    thesis_weight = st.sidebar.slider("Thesis influence", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    with st.sidebar.expander("Preset details", expanded=False):
        st.write(f"Signal strength: `{preset.signal_strength:.2f}`")
        st.write(f"Risk penalty: `{preset.risk_penalty:.2f}`")
        st.write(f"Turnover penalty: `{preset.turnover_penalty:.2f}`")
    st.sidebar.markdown(
        '<div class="theme-note">Theme and controls update interactively.</div>',
        unsafe_allow_html=True,
    )

    try:
        weights_df, pnl_df = load_base_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([1.25, 1.75], gap="large")

    with left:
        st.markdown('<div class="section-title">1) Import investment theses</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload thesis files (.txt, .md, .csv, .json)",
            type=["txt", "md", "csv", "json"],
            accept_multiple_files=True,
        )

        thesis_features = pd.DataFrame()
        if uploaded_files:
            thesis_features = parse_uploaded_files(uploaded_files)
            st.dataframe(thesis_features, use_container_width=True, hide_index=True)
        else:
            st.info("No thesis files uploaded yet. The model will run using baseline assumptions.")

        st.markdown('<div class="section-title">2) Portfolio focus tuning</div>', unsafe_allow_html=True)
        universe_size = weights_df.shape[1]
        top_n = st.slider(
            "Top holdings shown",
            min_value=5,
            max_value=min(40, universe_size),
            value=15,
            step=1,
        )

    with right:
        st.markdown('<div class="section-title">3) Run model version test</div>', unsafe_allow_html=True)
        if st.button("Run simulation", type="primary", use_container_width=True):
            st.session_state["run_clicked"] = True

        run_clicked = st.session_state.get("run_clicked", False)
        if not run_clicked:
            st.info("Click **Run simulation** to compute model results for your selected thesis and version.")
            st.stop()

        modeled = model_adjusted_returns(
            base_pnl=pnl_df,
            thesis_features=thesis_features,
            preset=preset,
            signal_boost=signal_boost,
            risk_aversion=risk_aversion,
            thesis_weight=thesis_weight,
        )
        base_stats = summary_stats(pnl_df["portfolio_return"])
        adjusted_stats = summary_stats(modeled["model_return"])

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("Cumulative Return", f"{adjusted_stats['cumulative_return']:.2%}")
        with k2:
            render_kpi_card("Annualized Return", f"{adjusted_stats['annualized_return']:.2%}")
        with k3:
            render_kpi_card("Sharpe", f"{adjusted_stats['sharpe']:.2f}")
        with k4:
            render_kpi_card("Max Drawdown", f"{adjusted_stats['max_drawdown']:.2%}")

        comparison = pd.DataFrame(
            [
                {
                    "Metric": "Cumulative Return",
                    "Format": "pct",
                    "Baseline": base_stats["cumulative_return"],
                    "Selected Version": adjusted_stats["cumulative_return"],
                },
                {
                    "Metric": "Annualized Return",
                    "Format": "pct",
                    "Baseline": base_stats["annualized_return"],
                    "Selected Version": adjusted_stats["annualized_return"],
                },
                {
                    "Metric": "Annualized Volatility",
                    "Format": "pct",
                    "Baseline": base_stats["annualized_volatility"],
                    "Selected Version": adjusted_stats["annualized_volatility"],
                },
                {
                    "Metric": "Sharpe",
                    "Format": "num",
                    "Baseline": base_stats["sharpe"],
                    "Selected Version": adjusted_stats["sharpe"],
                },
                {
                    "Metric": "Max Drawdown",
                    "Format": "pct",
                    "Baseline": base_stats["max_drawdown"],
                    "Selected Version": adjusted_stats["max_drawdown"],
                },
            ]
        )
        comparison["Delta"] = comparison["Selected Version"] - comparison["Baseline"]
        comparison_display = comparison.copy()
        for col in ["Baseline", "Selected Version", "Delta"]:
            comparison_display[col] = comparison_display.apply(
                lambda row: f"{row[col]:+.2%}" if row["Format"] == "pct" else f"{row[col]:+.2f}",
                axis=1,
            )

        st.markdown('<div class="section-title">4) Version comparison</div>', unsafe_allow_html=True)
        st.dataframe(
            comparison_display.drop(columns=["Format"]),
            use_container_width=True,
            hide_index=True,
        )

        chart_df = modeled[["trade_date", "model_return", "cum_return"]].copy()
        baseline = pnl_df[["trade_date", "portfolio_return"]].copy()
        baseline["baseline_cum_return"] = (1.0 + baseline["portfolio_return"]).cumprod() - 1.0
        merged = chart_df.merge(
            baseline[["trade_date", "baseline_cum_return"]],
            on="trade_date",
            how="left",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=merged["trade_date"],
                y=merged["baseline_cum_return"],
                mode="lines",
                name="Baseline",
                line=dict(width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["trade_date"],
                y=merged["cum_return"],
                mode="lines",
                name=preset.name,
                line=dict(width=2.5),
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis_tickformat=".1%",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            legend=dict(orientation="h", y=1.05, x=0.0),
        )
        st.plotly_chart(fig, use_container_width=True)

        latest_weights = weights_df.iloc[-1].sort_values(ascending=False)
        top_holdings = latest_weights.head(top_n).rename_axis("Ticker").reset_index(name="Weight")
        fig_holdings = px.bar(
            top_holdings,
            x="Ticker",
            y="Weight",
            title=f"Top {top_n} Holdings Snapshot",
            color="Weight",
            color_continuous_scale="Blues",
        )
        fig_holdings.update_layout(height=320, margin=dict(l=10, r=10, t=42, b=20))
        st.plotly_chart(fig_holdings, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">5) Feedback and readjustment log</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.2, 2.0], gap="large")
    with col1:
        feedback_rating = st.select_slider("Outcome quality", options=[1, 2, 3, 4, 5], value=3)
        feedback_action = st.selectbox(
            "Needed adjustment",
            options=["None", "Increase growth tilt", "Increase risk control", "Reduce turnover", "Custom"],
            index=0,
        )
    with col2:
        user_name = st.text_input("Analyst name", value="")
        desired_change = st.text_area("Requested change", value="", height=110)
    with col3:
        rationale = st.text_area(
            "Feedback rationale",
            value="",
            height=145,
            placeholder="Describe why this version works or what should be adjusted.",
        )

    if st.button("Save feedback", use_container_width=False):
        row = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "analyst": user_name.strip(),
            "version": preset.name,
            "theme": selected_theme,
            "signal_boost": signal_boost,
            "risk_aversion": risk_aversion,
            "thesis_weight": thesis_weight,
            "quality_rating": int(feedback_rating),
            "adjustment_action": feedback_action,
            "requested_change": desired_change.strip(),
            "rationale": rationale.strip(),
        }
        save_feedback_row(row)
        st.success(f"Feedback saved to {FEEDBACK_PATH.relative_to(PROJECT_ROOT)}")

    if FEEDBACK_PATH.exists():
        st.markdown("#### Recent feedback entries")
        log_df = pd.read_csv(FEEDBACK_PATH).tail(10)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    st.caption(
        "Tip: switch Light/Dark mode from the sidebar to adapt the app for presentation or deep-work sessions."
    )


if __name__ == "__main__":
    main()
