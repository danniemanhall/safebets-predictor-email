"""
calib_check.py — did the hourly-bar fix actually change the 24H estimate?

Prints, for a handful of assets, the modelled 24H Bull's Eye probability and
EV computed two ways: from daily closes (the original method) and from
hourly bars (the proposed fix). If the two columns are identical, the hourly
fetch is silently falling back to daily and the calibration hypothesis was
never tested. If they differ but neither approaches the measured rate, the
hypothesis is wrong.

Measured for comparison: 20.8% Bull's Eye across 77 resolved 24H slots,
10.24 unicoins per coin for crypto specifically.
"""

import numpy as np

from safebets_core import PAYOUTS, PERIOD_DAYS, THRESHOLDS, empirical_deviations
from class_strategy import asset_class, intraday_deviations, symbol_ticker_pairs

SAMPLE = ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'HYPE',
          'WTI', 'COPPER', 'GOLD', 'SILVER',
          'AAPL', 'MSFT', 'NVDA', 'INTC']


def ev_from_moves(sb_symbol, moves, period='HOURS_24'):
    """Expected unicoins and Bull's Eye probability for one move distribution."""
    bands = [
        (float(r['deviation']), r['tier'])
        for r in THRESHOLDS.get(sb_symbol, [])
        if r['periodName'] == period
    ]
    if not bands or moves is None or len(moves) == 0:
        return None, None

    ev, prev_cum, p_be = 0.0, 0.0, 0.0
    for dev, tier in sorted(bands):
        cum = float(np.mean(moves <= dev))
        excl = max(0.0, cum - prev_cum)
        ev += excl * PAYOUTS[period].get(tier, 0)
        if tier == 'BULLS_EYE':
            p_be = excl
        prev_cum = cum
    return ev, p_be


def main():
    tickers = dict(symbol_ticker_pairs())

    print(f"{'sym':8}{'class':10}"
          f"{'daily n':>9}{'dailyBE%':>10}{'dailyEV':>9}"
          f"{'hourly n':>10}{'hourBE%':>9}{'hourEV':>8}   status")
    print('-' * 82)

    for sb in SAMPLE:
        ticker = tickers.get(sb)
        if not ticker:
            print(f'{sb:8}not in asset_map')
            continue

        daily = empirical_deviations(ticker, PERIOD_DAYS['HOURS_24'])
        hourly = intraday_deviations(ticker, hours=24)

        d_ev, d_be = ev_from_moves(sb, daily)
        h_ev, h_be = ev_from_moves(sb, hourly)

        if hourly is None:
            status = 'FELL BACK to daily'
        elif d_be is not None and abs((h_be or 0) - d_be) < 1e-9:
            status = 'identical'
            
        else:
            status = 'hourly differs'

        dn = len(daily) if daily is not None else 0
        hn = len(hourly) if hourly is not None else 0
        dbe = f'{100*d_be:.1f}' if d_be is not None else '--'
        hbe = f'{100*h_be:.1f}' if h_be is not None else '--'
        dev = f'{d_ev:.2f}' if d_ev is not None else '--'
        hev = f'{h_ev:.2f}' if h_ev is not None else '--'

        print(f'{sb:8}{asset_class(sb):10}{dn:9}{dbe:>10}{dev:>9}'
              f'{hn:10}{hbe:>9}{hev:>8}   {status}')

    print()
    print('Measured across your 77 resolved 24H slots: 20.8% Bull\'s Eye.')
    print('Measured crypto 24H return: 10.24 unicoins per coin staked.')


if __name__ == '__main__':
    main()
