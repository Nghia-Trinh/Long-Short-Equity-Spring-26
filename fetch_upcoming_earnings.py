"""
fetch_upcoming_earnings.py — Pull REAL upcoming earnings dates from Nasdaq
for every ticker in our basket.

Uses the Nasdaq earnings calendar API to get confirmed earnings dates
for the next 30 days, then cross-references with our 502-ticker universe.

Usage:
    python fetch_upcoming_earnings.py
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.data_loader import get_tickers

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TODAY = pd.Timestamp.now().normalize()
WINDOW_END = TODAY + pd.Timedelta(days=30)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}


def fetch_earnings_for_date(date: pd.Timestamp) -> list[dict]:
    """Fetch all earnings events for a single date from Nasdaq."""
    url = "https://api.nasdaq.com/api/calendar/earnings"
    params = {"date": date.strftime("%Y-%m-%d")}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        rows = data.get("data", {}).get("rows", [])
        if rows is None:
            return []
        return rows
    except Exception:
        return []


def main():
    tickers = set(get_tickers())
    print(f"Universe: {len(tickers)} tickers")
    print(f"Fetching real earnings dates from Nasdaq: {TODAY.date()} -> {WINDOW_END.date()}")
    print()

    # Generate all business days in the 30-day window
    dates = pd.bdate_range(TODAY, WINDOW_END)
    all_events = []

    for i, d in enumerate(dates):
        events = fetch_earnings_for_date(d)
        our_events = [e for e in events if e.get("symbol", "").upper() in tickers]
        all_events.extend(
            {
                "ticker": e["symbol"].upper(),
                "company": e.get("name", ""),
                "earnings_date": d.strftime("%Y-%m-%d"),
                "time": e.get("time", "").replace("time-", ""),
                "fiscal_quarter": e.get("fiscalQuarterEnding", ""),
                "eps_forecast": e.get("epsForecast", ""),
                "n_estimates": e.get("noOfEsts", ""),
                "last_year_eps": e.get("lastYearEPS", ""),
                "last_year_date": e.get("lastYearRptDt", ""),
                "market_cap": e.get("marketCap", ""),
            }
            for e in our_events
        )
        print(f"  {d.strftime('%Y-%m-%d')}: {len(events):>3d} total, {len(our_events):>2d} in our basket")
        # Small delay to be polite
        time.sleep(0.15)

    print(f"\n  Total events in our basket: {len(all_events)}")

    if not all_events:
        print("No upcoming earnings found in our basket!")
        return

    df = pd.DataFrame(all_events)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df["days_away"] = (df["earnings_date"] - TODAY).dt.days
    df = df.sort_values(["earnings_date", "ticker"])
    # Deduplicate
    df = df.drop_duplicates(subset=["ticker", "earnings_date"], keep="first")

    # ── Console output ────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"  UPCOMING EARNINGS -- {TODAY.date()} -> {WINDOW_END.date()}")
    print(f"  {len(df)} companies in our basket reporting (REAL dates from Nasdaq)")
    print("=" * 80)

    for _, r in df.iterrows():
        timing = r["time"].replace("not-supplied", "TBD")
        eps = r["eps_forecast"] if r["eps_forecast"] else "n/a"
        print(
            f"  {r['ticker']:<8s}  "
            f"{r['earnings_date'].strftime('%Y-%m-%d')}  "
            f"({r['days_away']:+3d}d)  "
            f"{timing:<14s}  "
            f"EPS est: {eps:<8s}  "
            f"{r['company'][:35]}"
        )

    # ── By week ───────────────────────────────────────────────────────────
    print()
    df["week"] = df["earnings_date"].dt.isocalendar().week
    for wk, grp in df.groupby("week"):
        tks = ", ".join(grp["ticker"].tolist())
        print(f"  Week {wk} ({len(grp)} companies): {tks}")

    # ── Save outputs ──────────────────────────────────────────────────────
    df_out = df.drop(columns=["week"], errors="ignore")
    df_out["earnings_date"] = df_out["earnings_date"].dt.strftime("%Y-%m-%d")

    csv_path = OUTPUT_DIR / "upcoming_earnings_30d.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  CSV saved  -> {csv_path}")

    # Full text summary
    txt_path = OUTPUT_DIR / "upcoming_earnings_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"UPCOMING EARNINGS REPORT\n")
        f.write(f"Source: Nasdaq Earnings Calendar (real dates)\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Window: {TODAY.date()} -> {WINDOW_END.date()}\n")
        f.write(f"Companies reporting: {len(df)}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"{'Ticker':<8s}  {'Date':<12s}  {'Days':<6s}  {'Time':<14s}  {'EPS Est':<8s}  {'Company'}\n")
        f.write(f"{'-' * 80}\n")
        for _, r in df.iterrows():
            timing = r["time"].replace("not-supplied", "TBD")
            eps = r["eps_forecast"] if r["eps_forecast"] else "n/a"
            f.write(
                f"{r['ticker']:<8s}  "
                f"{r['earnings_date'].strftime('%Y-%m-%d'):<12s}  "
                f"{r['days_away']:<+6d}  "
                f"{timing:<14s}  "
                f"{eps:<8s}  "
                f"{r['company']}\n"
            )
    print(f"  Text saved -> {txt_path}")


if __name__ == "__main__":
    main()
