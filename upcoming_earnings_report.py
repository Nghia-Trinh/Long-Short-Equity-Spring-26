"""
upcoming_earnings_report.py — Generate a report of companies in our
basket with upcoming earnings announcements.

Since our earnings data runs through 2026-01-30, this script:
  1. Shows the latest known earnings dates per ticker
  2. Estimates the next reporting date based on each company's
     historical quarterly cadence
  3. Outputs a CSV + text summary to outputs/

Usage:
    python upcoming_earnings_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.data_loader import load_earnings, get_tickers

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    tickers = set(get_tickers())
    earnings = load_earnings()[["ticker", "event_date"]].copy()
    earnings["ticker"] = earnings["ticker"].astype(str).str.strip().str.upper()
    earnings["event_date"] = pd.to_datetime(earnings["event_date"], errors="coerce")
    earnings = earnings.dropna(subset=["event_date"])
    earnings = earnings[earnings["ticker"].isin(tickers)].sort_values("event_date")

    data_end = earnings["event_date"].max()
    today = pd.Timestamp("2026-03-04")
    forecast_end = today + pd.Timedelta(days=30)

    print(f"Earnings data range : {earnings['event_date'].min().date()} → {data_end.date()}")
    print(f"Forecast window     : {today.date()} → {forecast_end.date()}")
    print(f"Universe tickers    : {len(tickers)}")
    print()

    # ── Build per-ticker history ──────────────────────────────────────────
    rows = []
    for tk, grp in earnings.groupby("ticker"):
        dates = grp["event_date"].sort_values().tolist()
        last_date = dates[-1]

        # Estimate quarterly cadence (median gap between consecutive reports)
        if len(dates) >= 2:
            gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
            median_gap = int(np.median(gaps))
        else:
            median_gap = 90  # default quarterly

        # Estimate next earnings date
        est_next = last_date + pd.Timedelta(days=median_gap)

        # How many quarters of history
        n_reports = len(dates)

        rows.append({
            "ticker": tk,
            "last_known_earnings": last_date,
            "n_reports": n_reports,
            "median_gap_days": median_gap,
            "estimated_next_earnings": est_next,
            "est_days_away": (est_next - today).days,
        })

    df = pd.DataFrame(rows).sort_values("estimated_next_earnings")

    # ── Filter: estimated next earnings within the next 30 days ───────────
    upcoming = df[
        (df["estimated_next_earnings"] >= today)
        & (df["estimated_next_earnings"] <= forecast_end)
    ].copy()

    past_due = df[df["estimated_next_earnings"] < today].copy()
    past_due = past_due.sort_values("estimated_next_earnings", ascending=False)

    further_out = df[df["estimated_next_earnings"] > forecast_end].copy()

    # ── Console output ────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  UPCOMING EARNINGS — Next 30 days ({today.date()} → {forecast_end.date()})")
    print(f"  {len(upcoming)} companies expected to report")
    print("=" * 70)

    if not upcoming.empty:
        for _, r in upcoming.iterrows():
            print(
                f"  {r['ticker']:<8s}  "
                f"est. {r['estimated_next_earnings'].strftime('%Y-%m-%d')}  "
                f"({r['est_days_away']:+3d}d)  "
                f"last: {r['last_known_earnings'].strftime('%Y-%m-%d')}  "
                f"gap: {r['median_gap_days']}d  "
                f"({r['n_reports']} reports)"
            )
    else:
        print("  (none)")

    print()
    print("=" * 70)
    print(f"  LIKELY ALREADY REPORTED (est. date < today) — may need updated data")
    print(f"  {len(past_due)} companies")
    print("=" * 70)
    if not past_due.empty:
        for _, r in past_due.head(30).iterrows():
            print(
                f"  {r['ticker']:<8s}  "
                f"est. {r['estimated_next_earnings'].strftime('%Y-%m-%d')}  "
                f"({r['est_days_away']:+3d}d)  "
                f"last: {r['last_known_earnings'].strftime('%Y-%m-%d')}"
            )
        if len(past_due) > 30:
            print(f"  ... and {len(past_due) - 30} more")

    print()
    print("=" * 70)
    print(f"  FURTHER OUT (> 30 days)")
    print(f"  {len(further_out)} companies")
    print("=" * 70)
    if not further_out.empty:
        for _, r in further_out.head(20).iterrows():
            print(
                f"  {r['ticker']:<8s}  "
                f"est. {r['estimated_next_earnings'].strftime('%Y-%m-%d')}  "
                f"({r['est_days_away']:+3d}d)  "
                f"last: {r['last_known_earnings'].strftime('%Y-%m-%d')}"
            )
        if len(further_out) > 20:
            print(f"  ... and {len(further_out) - 20} more")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    # Full report
    df["last_known_earnings"] = df["last_known_earnings"].dt.strftime("%Y-%m-%d")
    df["estimated_next_earnings"] = df["estimated_next_earnings"].dt.strftime("%Y-%m-%d")
    full_path = OUTPUT_DIR / "earnings_calendar_full.csv"
    df.to_csv(full_path, index=False)

    # Upcoming only
    if not upcoming.empty:
        upcoming_out = upcoming.copy()
        upcoming_out["last_known_earnings"] = upcoming_out["last_known_earnings"].dt.strftime("%Y-%m-%d")
        upcoming_out["estimated_next_earnings"] = upcoming_out["estimated_next_earnings"].dt.strftime("%Y-%m-%d")
        upcoming_path = OUTPUT_DIR / "upcoming_earnings_30d.csv"
        upcoming_out.to_csv(upcoming_path, index=False)
        print(f"\n✅ Upcoming 30d report saved → {upcoming_path}")

    print(f"✅ Full calendar saved       → {full_path}")

    # ── Summary text file ─────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "upcoming_earnings_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"UPCOMING EARNINGS REPORT\n")
        f.write(f"Generated: {today.date()}\n")
        f.write(f"Forecast window: {today.date()} → {forecast_end.date()}\n")
        f.write(f"Data through: {data_end.date()}\n")
        f.write(f"{'=' * 70}\n\n")

        f.write(f"NEXT 30 DAYS — {len(upcoming)} companies expected\n")
        f.write(f"{'-' * 70}\n")
        if not upcoming.empty:
            f.write(f"{'Ticker':<8s}  {'Est. Date':<12s}  {'Days':<6s}  {'Last Known':<12s}  {'Gap':<5s}  {'Reports'}\n")
            for _, r in upcoming.iterrows():
                f.write(
                    f"{r['ticker']:<8s}  "
                    f"{r['estimated_next_earnings'].strftime('%Y-%m-%d'):<12s}  "
                    f"{r['est_days_away']:<+6d}  "
                    f"{r['last_known_earnings'].strftime('%Y-%m-%d'):<12s}  "
                    f"{r['median_gap_days']:<5d}  "
                    f"{r['n_reports']}\n"
                )
        else:
            f.write("  (none)\n")

        f.write(f"\nLIKELY ALREADY REPORTED — {len(past_due)} companies\n")
        f.write(f"{'-' * 70}\n")
        for _, r in past_due.iterrows():
            f.write(
                f"{r['ticker']:<8s}  "
                f"est. {r['estimated_next_earnings'].strftime('%Y-%m-%d')}  "
                f"last: {r['last_known_earnings'].strftime('%Y-%m-%d')}\n"
            )

        f.write(f"\nFURTHER OUT (> 30 days) — {len(further_out)} companies\n")
        f.write(f"{'-' * 70}\n")
        for _, r in further_out.iterrows():
            f.write(
                f"{r['ticker']:<8s}  "
                f"est. {r['estimated_next_earnings'].strftime('%Y-%m-%d')}  "
                f"last: {r['last_known_earnings'].strftime('%Y-%m-%d')}\n"
            )

    print(f"✅ Summary text saved        → {summary_path}")


if __name__ == "__main__":
    main()
