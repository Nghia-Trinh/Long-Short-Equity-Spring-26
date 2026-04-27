"""Finnhub-aware LLM long/short model for Streamlit.

Human-written investment theses define the basket and direction. This module
automates the repeatable parts: Finnhub news enrichment, signal matrices,
EWMA risk estimates, portfolio sizing, and four-month paper-trade PnL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
THESIS_DIR = PROJECT_ROOT / "Investment Theses"

THESIS_FILE_TICKERS = {
    "Applied Digital.docx": "APLD",
    "CIEN_Thesis.docx": "CIEN",
    "CLMT_Thesis.docx": "CLMT",
    "Cheniere Energy Stock Pitch.pdf": "LNG",
    "Constilation Energy.docx": "CEG",
    "Delta Airlines.docx": "DAL",
    "Energy Transfer Stock Pitch.pdf": "ET",
    "Enterprise Products Stock Pitch.pdf": "EPD",
    "General Dynamics.docx": "GD",
    "Plug Power Stock Pitch.pdf": "PLUG",
    "Sunrun Stock Pitch.pdf": "RUN",
}

BUILT_IN_THESIS_TEXT = {
    "APLD": "Applied Digital data center growth thesis with execution and financing risk.",
    "CEG": "Constellation Energy long thesis: nuclear baseload scarcity, clean power demand, and improving cash flow quality.",
    "CIEN": "Ciena optical networking thesis driven by AI bandwidth demand and telecom upgrade cycles.",
    "CLMT": "Calumet thesis tied to specialty refining cash flow and renewable diesel execution.",
    "DAL": "Delta Air Lines thesis balances premium travel demand, loyalty economics, fuel risk, and cyclicality.",
    "ET": "Energy Transfer midstream thesis with fee-based cash flows and commodity-cycle risk.",
    "EPD": "Enterprise Products thesis focused on stable midstream distributions and disciplined capital allocation.",
    "GD": "General Dynamics defense thesis supported by backlog durability, Gulfstream demand, and federal budget risk.",
    "LNG": "Cheniere Energy long thesis: contracted LNG export cash flows, capacity expansion, and global energy-security demand.",
    "PLUG": "Plug Power short thesis: funding needs, hydrogen adoption uncertainty, and margin pressure.",
    "RUN": "Sunrun short thesis: residential solar demand pressure from rates, NEM 3.0 policy, and high financing needs.",
}

POSITIVE_TERMS = {
    "long",
    "buy",
    "growth",
    "upside",
    "demand",
    "margin",
    "cash",
    "contracted",
    "quality",
    "backlog",
    "expansion",
    "stable",
    "strong",
}
NEGATIVE_TERMS = {
    "short",
    "sell",
    "risk",
    "pressure",
    "decline",
    "weak",
    "funding",
    "uncertainty",
    "cyclical",
    "volatile",
    "policy",
    "debt",
}


@dataclass(frozen=True)
class ModelRun:
    alpha_matrix: pd.DataFrame
    risk_matrix: pd.DataFrame
    weights: pd.DataFrame
    pnl: pd.DataFrame
    signal_snapshot: pd.DataFrame
    thesis_basket: pd.DataFrame
    news: pd.DataFrame
    metrics: dict[str, float | str]
    data_status: dict[str, Any]


def _zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


def _read_docx(path: Path) -> str:
    try:
        with ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return ""
    text = re.sub(r"<[^>]+>", " ", xml)
    return re.sub(r"\s+", " ", text).strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def _read_thesis_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    return ""


def _thesis_score(text: str, ticker: str) -> tuple[float, float, str]:
    text = text or BUILT_IN_THESIS_TEXT.get(ticker, "")
    tokens = re.findall(r"[a-z']+", text.lower())
    pos = sum(token in POSITIVE_TERMS for token in tokens)
    neg = sum(token in NEGATIVE_TERMS for token in tokens)
    score = (pos - neg) / max(pos + neg, 1)
    confidence = min(1.0, max(0.15, (pos + neg) / 18.0))
    lowered = text.lower()
    if re.search(r"\bshort\b|recommendation\s*:\s*s\s*h\s*o\s*r\s*t", lowered):
        score = min(score, -0.45)
    if re.search(r"\blong\b|\bbuy\b|recommendation\s*:\s*b\s*u\s*y", lowered):
        score = max(score, 0.45)
    return float(np.clip(score, -1.0, 1.0)), float(confidence), text[:420]


def load_thesis_basket() -> pd.DataFrame:
    """Load repo investment theses and map them to available price history."""
    prices = pd.read_csv(DATA_DIR / "prices.csv", usecols=["ticker"])
    available = set(prices["ticker"].astype(str).str.upper())
    rows: list[dict[str, Any]] = []
    for filename, ticker in THESIS_FILE_TICKERS.items():
        path = THESIS_DIR / filename
        text = _read_thesis_file(path) if path.exists() else ""
        score, confidence, preview = _thesis_score(text, ticker)
        rows.append(
            {
                "ticker": ticker,
                "source_file": filename,
                "in_price_data": ticker in available,
                "thesis_score": score,
                "thesis_confidence": confidence,
                "thesis_preview": preview or BUILT_IN_THESIS_TEXT.get(ticker, ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["in_price_data", "ticker"], ascending=[False, True])


def _finnhub_key() -> str:
    return (
        os.getenv("FINNHUB_API_KEY")
        or os.getenv("FINNHUB_KEY")
        or os.getenv("FINNHUB_TOKEN")
        or ""
    ).strip()


def _finnhub_get(path: str, params: dict[str, Any], timeout: float = 8.0) -> Any:
    token = _finnhub_key()
    if not token:
        return None
    query = urlencode({**params, "token": token})
    url = f"https://finnhub.io/api/v1/{path}?{query}"
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host.
        return json.loads(response.read().decode("utf-8"))


def fetch_finnhub_news(tickers: list[str], end_date: pd.Timestamp, months: int = 4) -> tuple[pd.DataFrame, str]:
    """Fetch Finnhub news when a token is configured."""
    if not _finnhub_key():
        columns = ["ticker", "datetime", "headline", "source", "url", "sentiment"]
        return pd.DataFrame(columns=columns), "missing_token"

    start_date = (end_date - pd.DateOffset(months=months)).date().isoformat()
    end = end_date.date().isoformat()
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            payload = _finnhub_get("company-news", {"symbol": ticker, "from": start_date, "to": end}) or []
        except Exception:
            payload = []
        for item in payload[:80]:
            headline = str(item.get("headline") or "")
            summary = str(item.get("summary") or "")
            text = f"{headline} {summary}".lower()
            sentiment = sum(term in text for term in POSITIVE_TERMS) - sum(term in text for term in NEGATIVE_TERMS)
            rows.append(
                {
                    "ticker": ticker,
                    "datetime": pd.to_datetime(item.get("datetime"), unit="s", errors="coerce"),
                    "headline": headline,
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "sentiment": float(np.clip(sentiment / 4.0, -1.0, 1.0)),
                }
            )
    status = "live_finnhub" if rows else "live_finnhub_no_rows"
    return pd.DataFrame(rows), status


def _load_returns(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "prices.csv", usecols=["trade_date", "ticker", "adj_close"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices[prices["ticker"].isin(tickers)]
    close = prices.pivot_table(index="trade_date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    close = close.loc[(close.index >= start - pd.Timedelta(days=100)) & (close.index <= end)]
    returns = close.pct_change(fill_method=None).fillna(0.0)
    return returns.dropna(how="all")


def _build_signal_matrix(
    returns: pd.DataFrame,
    thesis: pd.DataFrame,
    news: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = returns.columns.tolist()
    thesis_score = thesis.set_index("ticker")["thesis_score"].reindex(tickers).fillna(0.0)
    news_sentiment = pd.Series(0.0, index=tickers)
    if not news.empty:
        news_sentiment.update(news.groupby("ticker")["sentiment"].mean())

    volume_proxy = returns.abs().rolling(20, min_periods=5).mean() / returns.abs().rolling(60, min_periods=10).mean()
    momentum = returns.rolling(15, min_periods=5).mean()
    rolling_vol = returns.rolling(20, min_periods=5).std()
    long_vol = returns.rolling(60, min_periods=10).std()
    rolling_sharpe = momentum / rolling_vol.replace(0, np.nan)
    iv_skew_proxy = (rolling_vol - long_vol).fillna(0.0)
    drawdown = returns.cumsum() - returns.cumsum().rolling(60, min_periods=10).max()

    rows: list[pd.Series] = []
    snapshots: list[pd.DataFrame] = []
    for date in rebalance_dates:
        cp_imbalance = _zscore(volume_proxy.loc[:date].iloc[-1].reindex(tickers).fillna(0.0)) * np.sign(thesis_score.replace(0, 1))
        net_delta = _zscore(momentum.loc[:date].iloc[-1].reindex(tickers).fillna(0.0) / rolling_vol.loc[:date].iloc[-1].reindex(tickers).replace(0, np.nan))
        iv_skew = -_zscore(iv_skew_proxy.loc[:date].iloc[-1].reindex(tickers).fillna(0.0))
        sharpe_rank = _zscore(rolling_sharpe.loc[:date].iloc[-1].reindex(tickers).fillna(0.0))
        qmj = _zscore(
            sharpe_rank
            + thesis_score.reindex(tickers).fillna(0.0)
            + news_sentiment.reindex(tickers).fillna(0.0)
            + _zscore(drawdown.loc[:date].iloc[-1].reindex(tickers).fillna(0.0))
        )
        alpha = _zscore(0.25 * cp_imbalance + 0.20 * net_delta + 0.15 * iv_skew + 0.25 * sharpe_rank + 0.15 * qmj)
        alpha.name = date
        rows.append(alpha)
        snapshots.append(
            pd.DataFrame(
                {
                    "rebalance_date": date,
                    "ticker": tickers,
                    "call_put_imbalance": cp_imbalance.values,
                    "net_delta": net_delta.values,
                    "iv_skew": iv_skew.values,
                    "sharpe_z": sharpe_rank.values,
                    "qmj_z": qmj.values,
                    "alpha": alpha.values,
                }
            )
        )

    alpha_matrix = pd.DataFrame(rows)
    alpha_matrix.index.name = "rebalance_date"
    return alpha_matrix, pd.concat(snapshots, ignore_index=True)


def _ewma_covariance(returns_window: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    clean = returns_window.fillna(0.0)
    cov = clean.cov()
    for _, row in clean.tail(60).iterrows():
        x = row.values.reshape(-1, 1)
        cov = lam * cov + (1.0 - lam) * pd.DataFrame(x @ x.T, index=clean.columns, columns=clean.columns)
    return cov.fillna(0.0)


def _size_weights(alpha: pd.Series, cov: pd.DataFrame, gross: float = 1.0, max_position: float = 0.30) -> pd.Series:
    ranked = alpha.sort_values()
    n_side = max(1, min(3, len(ranked) // 2))
    shorts = ranked.head(n_side).index
    longs = ranked.tail(n_side).index
    vol = pd.Series(np.sqrt(np.diag(cov.reindex(index=alpha.index, columns=alpha.index).fillna(0.0))), index=alpha.index)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.replace([np.inf, -np.inf], np.nan).median())
    weights = pd.Series(0.0, index=alpha.index)
    weights.loc[longs] = inv_vol.loc[longs] / inv_vol.loc[longs].sum() * (gross / 2.0)
    weights.loc[shorts] = -inv_vol.loc[shorts] / inv_vol.loc[shorts].sum() * (gross / 2.0)
    weights = weights.clip(-max_position, max_position)
    weights = weights - weights.mean()
    if weights.abs().sum() > gross:
        weights *= gross / weights.abs().sum()
    return weights.fillna(0.0)


def run_long_short_model(months: int = 4, end_date: str | None = None, use_finnhub: bool = True) -> ModelRun:
    thesis = load_thesis_basket()
    active = thesis[thesis["in_price_data"]].copy()
    tickers = active["ticker"].tolist()
    if not tickers:
        raise RuntimeError("No thesis basket tickers are available in Data/prices.csv.")

    prices = pd.read_csv(DATA_DIR / "prices.csv", usecols=["trade_date"])
    latest = pd.to_datetime(prices["trade_date"], errors="coerce").max()
    end = pd.Timestamp(end_date) if end_date else latest
    start = end - pd.DateOffset(months=months)
    returns = _load_returns(tickers, start, end)
    trade_returns = returns.loc[(returns.index >= start) & (returns.index <= end), tickers].copy()
    rebalance_dates = pd.DatetimeIndex(trade_returns.index[::10])
    if len(rebalance_dates) < 2:
        raise RuntimeError("Not enough trading days to run the four-month model.")

    news, news_status = fetch_finnhub_news(tickers, end, months=months) if use_finnhub else (
        pd.DataFrame(columns=["ticker", "datetime", "headline", "source", "url", "sentiment"]),
        "disabled",
    )
    alpha_matrix, signal_snapshot = _build_signal_matrix(returns[tickers], active, news, rebalance_dates)

    weights: list[pd.Series] = []
    risk_rows: list[pd.Series] = []
    for date in rebalance_dates:
        cov = _ewma_covariance(returns.loc[:date, tickers].tail(60))
        risk_rows.append(pd.Series(np.diag(cov), index=tickers, name=date))
        weights.append(_size_weights(alpha_matrix.loc[date], cov))
    weight_matrix = pd.DataFrame(weights, index=rebalance_dates)
    weight_matrix.index.name = "rebalance_date"
    risk_matrix = pd.DataFrame(risk_rows)
    risk_matrix.index.name = "rebalance_date"

    daily_weights = weight_matrix.reindex(trade_returns.index, method="ffill").fillna(0.0)
    portfolio_return = (daily_weights.shift(1).fillna(0.0) * trade_returns).sum(axis=1)
    pnl = pd.DataFrame(
        {
            "portfolio_return": portfolio_return,
            "cumulative_return": (1.0 + portfolio_return).cumprod() - 1.0,
        }
    )
    pnl.index.name = "trade_date"

    ann_return = float(portfolio_return.mean() * 252)
    ann_vol = float(portfolio_return.std() * np.sqrt(252))
    wealth = (1.0 + portfolio_return).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    final_weights = weight_matrix.iloc[-1]
    metrics: dict[str, float | str] = {
        "start_date": trade_returns.index.min().strftime("%Y-%m-%d"),
        "end_date": trade_returns.index.max().strftime("%Y-%m-%d"),
        "cumulative_return": float(pnl["cumulative_return"].iloc[-1]),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": float(ann_return / ann_vol) if ann_vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "gross_exposure": float(final_weights.abs().sum()),
        "net_exposure": float(final_weights.sum()),
        "long_count": float((final_weights > 1e-8).sum()),
        "short_count": float((final_weights < -1e-8).sum()),
    }

    status = {
        "finnhub_news_status": news_status,
        "finnhub_token_present": bool(_finnhub_key()),
        "universe_size": len(tickers),
        "excluded_tickers": thesis.loc[~thesis["in_price_data"], "ticker"].tolist(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holding_period_trading_days": 10,
    }
    return ModelRun(
        alpha_matrix=alpha_matrix,
        risk_matrix=risk_matrix,
        weights=weight_matrix,
        pnl=pnl,
        signal_snapshot=signal_snapshot,
        thesis_basket=thesis,
        news=news,
        metrics=metrics,
        data_status=status,
    )


def export_model_run(run: ModelRun, output_dir: Path | None = None) -> dict[str, Path]:
    out = output_dir or PROJECT_ROOT / "outputs" / "llm_long_short"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "alpha_matrix": out / "alpha_matrix.csv",
        "risk_matrix": out / "risk_matrix.csv",
        "portfolio_weights": out / "portfolio_weights.csv",
        "pnl": out / "pnl.csv",
        "signals": out / "signal_snapshot.csv",
        "thesis_basket": out / "thesis_basket.csv",
        "news": out / "finnhub_news.csv",
        "summary": out / "summary.json",
    }
    run.alpha_matrix.to_csv(paths["alpha_matrix"])
    run.risk_matrix.to_csv(paths["risk_matrix"])
    run.weights.to_csv(paths["portfolio_weights"])
    run.pnl.to_csv(paths["pnl"])
    run.signal_snapshot.to_csv(paths["signals"], index=False)
    run.thesis_basket.to_csv(paths["thesis_basket"], index=False)
    run.news.to_csv(paths["news"], index=False)
    paths["summary"].write_text(json.dumps({"metrics": run.metrics, "data_status": run.data_status}, indent=2), encoding="utf-8")
    return paths


if __name__ == "__main__":
    model_run = run_long_short_model(months=4, use_finnhub=True)
    outputs = export_model_run(model_run)
    print(json.dumps({"metrics": model_run.metrics, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
