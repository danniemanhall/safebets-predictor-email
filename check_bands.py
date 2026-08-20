"""
check_bands.py — one question only: is WTI/COPPER's 14.3% Bull's Eye rate real,
or an artifact of Yahoo's continuous futures series?

Two hypotheses, and they leave different fingerprints:

  A. The bands are simply generous for these assets, the same way LINK's and
     AVAX's are. Fingerprint: band/sigma lands in the loose group (~0.12-0.13),
     and the history looks clean.

  B. The CL=F / HG=F series understates real movement -- stale repeated closes,
     missing days, or contract rolls stitched in a way that flattens the move
     distribution. Fingerprint: duplicate closes, calendar gaps, or a daily
     return distribution with outsized single-day jumps against a too-small body.

Prints one row per asset at 30D, sorted by band/sigma. Nothing is written and
nothing in the ranking changes -- this only reports.
"""

import numpy as np
import yfinance as yf

from safebets_core import (
    asset_map, SYMBOL_MAP, THRESHOLDS, HISTORY_PROXY, HISTORY_PERIOD,
)

PERIOD = "DAYS_30"
DAYS = 30


def band_for(sb_symbol, period, tier="BULLS_EYE"):
    for row in THRESHOLDS.get(sb_symbol, []):
        if row["periodName"] == period and row["tier"] == tier:
            return float(row["deviation"])
    return None


def diagnose(ticker):
    hist_ticker = HISTORY_PROXY.get(ticker, ticker)
    try:
        raw = yf.Ticker(hist_ticker).history(period=HISTORY_PERIOD)
    except Exception as exc:
        return {"error": str(exc)[:40]}
    if raw.empty or len(raw) < DAYS + 60:
        return {"error": "no usable history"}

    close = raw["Close"].dropna()
    daily = (close / close.shift(1) - 1.0).dropna()
    moves = (close.shift(-DAYS) / close - 1.0).dropna().abs() * 100.0

    # Staleness: how often does the close not move at all from the day before?
    flat_days = int((daily == 0).sum())

    # Calendar gaps: largest hole in the index, in days.
    idx = close.index
    gaps = (idx[1:] - idx[:-1]).days
    max_gap = int(gaps.max()) if len(gaps) else 0

    # Roll fingerprint: single-day moves far outside the body of the
    # distribution. A cleanly stitched series has few; a badly stitched one
    # shows isolated spikes where one contract was swapped for the next.
    sd_daily = float(daily.std())
    jumps_5sd = int((daily.abs() > 5 * sd_daily).sum())

    return {
        "hist_ticker": hist_ticker,
        "n_bars": len(close),
        "n_windows": len(moves),
        "sigma_30d": float(np.std(close.shift(-DAYS) / close - 1.0).__float__()) * 100.0,
        "median_move": float(np.median(moves)),
        "flat_days": flat_days,
        "max_gap": max_gap,
        "jumps_5sd": jumps_5sd,
        "moves": moves.to_numpy(),
    }


def main():
    rows = []
    for asset_name, ticker in asset_map.items():
        sb = SYMBOL_MAP.get(ticker)
        if sb not in THRESHOLDS:
            continue
        band = band_for(sb, PERIOD)
        d = diagnose(ticker)
        if "error" in d:
            rows.append({"asset": asset_name, "error": d["error"]})
            continue
        p_be = float(np.mean(d["moves"] <= band)) if band else float("nan")
        rows.append({
            "asset": asset_name,
            "hist": d["hist_ticker"],
            "band": band,
            "sigma": d["sigma_30d"],
            "ratio": band / d["sigma_30d"] if d["sigma_30d"] else float("nan"),
            "p_be": p_be,
            "median": d["median_move"],
            "bars": d["n_bars"],
            "flat": d["flat_days"],
            "gap": d["max_gap"],
            "jumps": d["jumps_5sd"],
        })

    good = [r for r in rows if "error" not in r]
    good.sort(key=lambda r: r["ratio"], reverse=True)

    print(f"30D Bull's Eye diagnostics — band width relative to each asset's own 30D sigma\n")
    print(f"{'ASSET':<15}{'HIST':<12}{'BAND%':>7}{'SIG30%':>8}{'B/SIG':>7}"
          f"{'P(BE)':>7}{'MED%':>7}{'BARS':>6}{'FLAT':>6}{'GAP':>5}{'JMP':>5}")
    for r in good:
        print(f"{r['asset']:<15}{r['hist']:<12}{r['band']:>7.2f}{r['sigma']:>8.2f}"
              f"{r['ratio']:>7.3f}{r['p_be']:>7.1%}{r['median']:>7.2f}"
              f"{r['bars']:>6}{r['flat']:>6}{r['gap']:>5}{r['jumps']:>5}")

    for r in rows:
        if "error" in r:
            print(f"{r['asset']:<15}skipped: {r['error']}")

    print("\nFLAT = days the close did not move at all (staleness).")
    print("GAP  = largest gap between consecutive bars, in calendar days.")
    print("JMP  = single-day moves beyond 5 sigma (contract-roll fingerprint).")
    print("\nIf WTI/COPPER sit with LINK/AVAX on B/SIG and their FLAT/GAP/JMP look")
    print("like the equities', the high hit rate is genuine band generosity.")


if __name__ == "__main__":
    main()
