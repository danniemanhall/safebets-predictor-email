"""
class_strategy.py — per-asset-class slot strategy across all four horizons.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It does NOT produce a different prediction *value* per asset class. There
isn't one. Spot beat every alternative in every class and horizon under
purged walk-forward validation, and the deviation bands are symmetric about
the submitted number, so the value you type is the current price for every
slot regardless of class.

What varies by class is which slots are worth a coin, and by how much. This
module answers that from two independent sources and blends them:

  1. MODELLED EV  — each asset's own historical move distribution measured
     against its own bands and the confirmed payout table.
  2. MEASURED EV  — Daniel's 134 resolved predictions, grouped by class and
     horizon.

CALIBRATION HISTORY — READ BEFORE CHANGING THE BLEND
----------------------------------------------------
Measured 24H Bull's Eye rate was 20.8% against roughly 8% modelled. The
first hypothesis was that the model derived 24-hour moves from daily CLOSES
while a slot entered at 15:46 resolves at 15:46 the next day. That was
tested directly (calib_check.py) against 17,000 hourly bars per crypto
asset. It made almost no difference: BTC 8.2% -> 7.9%, ETH 7.9% -> 7.1%,
AVAX 10.4% -> 12.1%. The hypothesis is dead. Hourly bars are retained here
because they are marginally the more faithful measurement, not because they
fixed anything.

The real explanation is clustering. The 16 Bull's Eyes fall on six days, six
of them on 12 August alone, when LINK, DOGE, SOL, ETH, COPPER and AVGO all
hit the tightest band together. Slots resolving on the same day share one
market move, so 48 crypto records are closer to 8 independent observations.
The shrink below therefore weights by DISTINCT DAYS, not record count. Do
not revert that without a reason: weighting by records over-trusts a calm
fortnight and would inflate every estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from safebets_core import (
    asset_map,
    HISTORY_PROXY,
    PAYOUTS,
    PERIOD_DAYS,
    THRESHOLDS,
    empirical_deviations,
)

# --------------------------------------------------------------------------
# Asset classes
# --------------------------------------------------------------------------

CRYPTO = {'BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'HYPE'}
COMMODITY = {'WTI', 'COPPER', 'GOLD', 'SILVER'}
# Everything else in asset_map is treated as an equity, including SPCX.

CLASS_ORDER = ['crypto', 'commodity', 'equity']
PERIOD_ORDER = ['HOURS_24', 'DAYS_7', 'DAYS_14', 'DAYS_30']
PERIOD_LABEL = {
    'HOURS_24': '24H',
    'DAYS_7': '7D',
    'DAYS_14': '14D',
    'DAYS_30': '30D',
}

STAKE = 1.0          # unicoins per slot; fixed by the platform
SHRINK_K = 12.0      # pseudo-DAYS of weight given to the model prior
MIN_SAMPLES = 200    # minimum history points before an EV is trusted


def asset_class(sb_symbol: str) -> str:
    """Return 'crypto', 'commodity' or 'equity' for a SafeBets symbol."""
    if sb_symbol in CRYPTO:
        return 'crypto'
    if sb_symbol in COMMODITY:
        return 'commodity'
    return 'equity'


def symbol_ticker_pairs():
    """Yield (sb_symbol, yahoo_ticker) for every asset we can rank."""
    for label, ticker in asset_map.items():
        sb_symbol = label.split('-')[-1].strip()
        if sb_symbol in THRESHOLDS:
            yield sb_symbol, ticker


# --------------------------------------------------------------------------
# Measured results — Daniel's 134 resolved predictions
# --------------------------------------------------------------------------
# (records, total unicoins returned, distinct resolution days).
#
# The third field is what the shrink uses. See the calibration note at the
# top of this file. Nothing at 30D has ever resolved, so those cells are
# absent and the model runs unblended there.

MEASURED = {
    ('crypto',    'HOURS_24'): (48, 491.4, 8),
    ('crypto',    'DAYS_7'):   (27, 972.0, 5),
    ('crypto',    'DAYS_14'):  (4,  400.0, 1),
    ('commodity', 'HOURS_24'): (10, 42.5,  4),
    ('commodity', 'DAYS_7'):   (10, 152.0, 4),
    ('commodity', 'DAYS_14'):  (1,  0.0,   1),
    ('equity',    'HOURS_24'): (19, 36.0,  2),
    ('equity',    'DAYS_7'):   (15, 50.0,  1),
}


def measured_ev(cls: str, period: str):
    """
    Realised unicoins per coin staked, the record count, and the number of
    distinct days those records came from. Returns (None, 0, 0) when the
    class and horizon have never resolved.
    """
    row = MEASURED.get((cls, period))
    if not row:
        return None, 0, 0
    n, total, days = row
    return (total / n if n else None), n, days


# --------------------------------------------------------------------------
# Move distributions
# --------------------------------------------------------------------------

_intraday_cache: dict = {}


def intraday_deviations(ticker_symbol: str, hours: int = 24):
    """
    Absolute percentage move over `hours`, measured from hourly bars so that
    entry hour and resolution hour match. Returns None when hourly history
    is unavailable or too short, so the caller can fall back to daily bars.

    Yahoo caps 60-minute history at roughly 730 days, which is ample: even a
    single year of hourly bars gives thousands of overlapping observations.
    """
    ticker_symbol = HISTORY_PROXY.get(ticker_symbol, ticker_symbol)

    key = (ticker_symbol, hours)
    if key in _intraday_cache:
        return _intraday_cache[key]

    result = None
    try:
        raw = yf.Ticker(ticker_symbol).history(period='730d', interval='60m')
        if not raw.empty and len(raw) >= MIN_SAMPLES:
            close = raw['Close'].dropna()
            target = close.index + pd.Timedelta(hours=hours)
            # For 24/7 crypto the match is exact; for equities the nearest
            # bar within 12 hours stands in for an overnight/weekend gap,
            # which is what the platform itself resolves against.
            later = close.reindex(target, method='nearest',
                                  tolerance=pd.Timedelta(hours=12))
            moves = (later.to_numpy() / close.to_numpy() - 1.0)
            moves = np.abs(moves[~np.isnan(moves)]) * 100.0
            if len(moves) >= MIN_SAMPLES:
                result = moves
    except Exception:
        result = None

    _intraday_cache[key] = result
    return result


def moves_for(ticker_symbol: str, period: str):
    """
    Best available move distribution for this asset and horizon, plus a label
    saying where it came from so the caller can report calibration honestly.
    """
    if period == 'HOURS_24':
        intraday = intraday_deviations(ticker_symbol, hours=24)
        if intraday is not None:
            return intraday, 'hourly'
    daily = empirical_deviations(ticker_symbol, PERIOD_DAYS[period])
    if daily is None or len(daily) < MIN_SAMPLES:
        return None, None
    return daily, 'daily'


# --------------------------------------------------------------------------
# EV
# --------------------------------------------------------------------------

def slot_ev(sb_symbol: str, ticker_symbol: str, period: str):
    """
    Modelled expected unicoins for one slot, using nested exclusive bands.
    Returns a dict, or None when the asset has no usable history.
    """
    bands = [
        (float(r['deviation']), r['tier'])
        for r in THRESHOLDS.get(sb_symbol, [])
        if r['periodName'] == period
    ]
    if not bands or period not in PAYOUTS:
        return None

    moves, source = moves_for(ticker_symbol, period)
    if moves is None:
        return None

    ev, prev_cum, probs = 0.0, 0.0, {}
    for dev, tier in sorted(bands):
        cum = float(np.mean(moves <= dev))
        excl = max(0.0, cum - prev_cum)
        ev += excl * PAYOUTS[period].get(tier, 0)
        probs[tier] = excl
        prev_cum = cum

    return {
        'symbol': sb_symbol,
        'period': period,
        'ev': ev,
        'p_win': prev_cum,
        'p_bulls_eye': probs.get('BULLS_EYE', 0.0),
        'n': len(moves),
        'source': source,
    }


def blended_ev(model_ev, cls: str, period: str):
    """
    Shrink the measured per-coin return toward the modelled EV, weighting by
    distinct resolution days rather than record count.

    With SHRINK_K = 12 pseudo-days, crypto 24H (8 days) lands at 40% measured
    and equity 7D (1 day) at 8% — which is roughly the confidence a single
    day's results deserve.
    """
    meas, n, days = measured_ev(cls, period)
    if meas is None or model_ev is None:
        return model_ev, meas, n, days
    weight = days / (days + SHRINK_K)
    return weight * meas + (1 - weight) * model_ev, meas, n, days


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------

def class_strategies(verbose: bool = False):
    """
    Build the per-class, per-horizon strategy table.

    Returns a list of dicts, one per (class, horizon), each carrying the
    modelled EV, the measured EV, the blend, the slot count, and a verdict.
    """
    slots = []
    for sb_symbol, ticker in symbol_ticker_pairs():
        for period in PERIOD_ORDER:
            row = slot_ev(sb_symbol, ticker, period)
            if row:
                row['class'] = asset_class(sb_symbol)
                slots.append(row)
            elif verbose:
                print(f'  no history: {sb_symbol} {PERIOD_LABEL[period]}')

    out = []
    for cls in CLASS_ORDER:
        for period in PERIOD_ORDER:
            members = [s for s in slots
                       if s['class'] == cls and s['period'] == period]
            if not members:
                continue
            model = float(np.mean([s['ev'] for s in members]))
            blend, meas, n, days = blended_ev(model, cls, period)

            if blend is None:
                verdict, note = 'UNKNOWN', 'no usable estimate'
            elif blend < STAKE:
                verdict = 'SKIP'
                note = 'below the 1-coin stake'
            elif days == 0:
                verdict = 'ENTER'
                note = 'modelled only — no resolved result yet'
            else:
                verdict = 'ENTER'
                note = f'{n} records over {days} day{"s" if days != 1 else ""}'

            out.append({
                'class': cls,
                'period': period,
                'label': PERIOD_LABEL[period],
                'slots': len(members),
                'model_ev': model,
                'measured_ev': meas,
                'measured_n': n,
                'measured_days': days,
                'blended_ev': blend,
                'verdict': verdict,
                'note': note,
                'source': members[0]['source'],
                'best': sorted(members, key=lambda s: -s['ev'])[:3],
            })
    return out


def print_report():
    rows = class_strategies(verbose=True)

    print()
    print(f"{'class':11}{'horizon':9}{'slots':>6}{'model':>9}"
          f"{'measured':>10}{'n':>5}{'days':>6}{'blended':>9}  verdict")
    print('-' * 84)
    for r in rows:
        meas = f"{r['measured_ev']:.2f}" if r['measured_ev'] is not None else '  --'
        print(f"{r['class']:11}{r['label']:9}{r['slots']:6}"
              f"{r['model_ev']:9.2f}{meas:>10}{r['measured_n']:5}"
              f"{r['measured_days']:6}{r['blended_ev']:9.2f}"
              f"  {r['verdict']} ({r['note']})")

    print()
    print('Best slots in each class and horizon, by modelled EV:')
    for r in rows:
        names = ', '.join(f"{s['symbol']} {s['ev']:.0f}" for s in r['best'])
        print(f"  {r['class']:10} {r['label']:4} {names}")

    print()
    total_slots = sum(r['slots'] for r in rows if r['verdict'] == 'ENTER')
    total_ev = sum(r['slots'] * r['blended_ev']
                   for r in rows if r['verdict'] == 'ENTER')
    print(f'Slots to enter: {total_slots}   blended expected return: '
          f'{total_ev:,.0f} unicoins for {total_slots} staked')

    skips = [r for r in rows if r['verdict'] == 'SKIP']
    print(f'Slots to skip: {sum(r["slots"] for r in skips)}'
          + (f" ({', '.join(r['class'] + ' ' + r['label'] for r in skips)})"
             if skips else ' — every class clears the stake at every horizon'))

    unverified = [r for r in rows if r['measured_days'] == 0]
    if unverified:
        print()
        print('Unverified — modelled only, never resolved in your history:')
        for r in unverified:
            print(f"  {r['class']:10} {r['label']:4} "
                  f"{r['slots']} slots at {r['model_ev']:.0f} each")

    print()
    print('Every slot above submits the same value: the current price shown '
          'on the SafeBets tile.')


if __name__ == '__main__':
    print_report()
