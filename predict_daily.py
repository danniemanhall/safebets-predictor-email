"""
predict_daily.py — the day's 104 SafeBets predictions, by email.

Run once a day by GitHub Actions. It fetches live prices (and, when needed,
news), works out a number for every asset and timeframe, and emails the list.
It never touches SafeBets itself: the predictions are entered by hand.

HOW THE NUMBERS ARE CHOSEN
  The method comes from backtest_result.json, which news_bakeoff.py writes
  each time the test runs:

    use = "tile"  The tile price for everything. This is the default, and it
                  stays the method until a model beats it in testing by a
                  margin that chance does not explain.
    use = "lin"   The news-only linear model.
    use = "ens"   The news + price ensemble.

  When a model is in use it is fitted fresh each morning on the latest two
  years of data, exactly as it was in the test, and its output is a small
  percentage move away from the current price for each of the 104 slots.

  The test can also find slot types that stop paying on heavy- or quiet-news
  days. Those are marked "skip" on the days they apply.

Environment (GitHub repo secrets), same as send_daily.py:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_TO, MAIL_FROM (optional)

Local test, sends nothing:
  python predict_daily.py --dry-run          # prints the HTML
  python predict_daily.py --dry-run --text   # prints the plain-text part
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bakeoff import fetch
from class_strategy import asset_class
from news_bakeoff import (BUCKETS, HORIZONS, MIN_TRAIN, NEWS_COLS, TRAIN_WINDOW,
                          download_news, load_news, make_ensemble, make_linear,
                          news_age_days, news_features, prepare)
from safebets_core import asset_map, get_live_price
from send_daily import load_config, send

VERDICT_FILE = 'backtest_result.json'
DAILY_TREES = 150
MAX_NEWS_AGE = 7         # days; older than this and a model is not trusted

# Same order as the SafeBets tabs, so the email can be worked top to bottom.
TABS = [
    ('Crypto', ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'HYPE']),
    ('Big Tech', ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL',
                  'NFLX', 'SPCX']),
    ('AI Chips', ['AMD', 'MU', 'SNDK', 'AVGO', 'INTC', 'ARM']),
    ('Commodities', ['GOLD', 'SILVER', 'WTI', 'COPPER']),
]
NAMES = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'DOGE': 'Dogecoin',
    'AVAX': 'Avalanche', 'LINK': 'Chainlink', 'HYPE': 'Hyperliquid',
    'NVDA': 'Nvidia', 'TSLA': 'Tesla', 'AAPL': 'Apple', 'MSFT': 'Microsoft',
    'AMZN': 'Amazon', 'META': 'Meta', 'GOOGL': 'Alphabet', 'NFLX': 'Netflix',
    'SPCX': 'SpaceX', 'AMD': 'AMD', 'MU': 'Micron', 'SNDK': 'SanDisk',
    'AVGO': 'Broadcom', 'INTC': 'Intel', 'ARM': 'Arm Holdings',
    'GOLD': 'Gold', 'SILVER': 'Silver', 'WTI': 'US Crude Oil', 'COPPER': 'Copper',
}
SLOT_LABEL = {'HOURS_24': '1D', 'DAYS_7': '7D', 'DAYS_14': '14D', 'DAYS_30': '30D'}
METHOD_NAME = {'tile': 'the tile price', 'lin': 'the news-only model',
               'ens': 'the news + price model'}


def ticker_for(sb: str):
    for label, ticker in asset_map.items():
        if label.split('-')[-1].strip() == sb:
            return ticker
    return None


def format_price(price):
    if price is None:
        return '—'
    if price >= 1000:
        return f'{price:,.2f}'
    if price >= 1:
        return f'{price:,.4f}'
    return f'{price:.6f}'


# --------------------------------------------------------------------------
# The verdict from the latest test
# --------------------------------------------------------------------------

def load_verdict(path: str = VERDICT_FILE) -> dict:
    """The latest test result, or the safe default when there is none."""
    default = {'use': 'tile', 'skip': [], 'models': {}, 'generated_at': None,
               'tile_per_coin': None, 'predictions': 0, 'data': 'live'}
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding='utf-8') as fh:
            verdict = {**default, **json.load(fh)}
    except (OSError, ValueError):
        return default
    # A result from made-up data, or an unknown method, must never drive
    # real predictions.
    if verdict.get('data') != 'live' or verdict.get('use') not in METHOD_NAME:
        verdict['use'] = 'tile'
    if verdict.get('data') != 'live':
        verdict['skip'] = []
    return verdict


# --------------------------------------------------------------------------
# Model output for one asset (only used when the test has earned it)
# --------------------------------------------------------------------------

def model_view(sb: str, ticker: str, use: str):
    """
    Returns ({period: predicted move as a fraction}, news bucket today).
    Either part is None/empty when the data for this asset is missing.
    """
    news = load_news(sb)
    age = news_age_days(sb)
    df = fetch(ticker)
    if (news is None or df is None or len(news) < 400
            or age is None or age > MAX_NEWS_AGE):
        return {}, None             # no fresh news: this asset uses the tile price

    lag = 0 if asset_class(sb) == 'crypto' else 1
    feats, close, nf = prepare(df, news_features(news, lag))

    # Work from the last finished day: today's bar and today's news are
    # still filling in when this runs.
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    table = feats.join(nf)
    table = table[table.index < today].dropna()
    if table.empty:
        return {}, None
    latest = table.iloc[[-1]]

    ratio = float(latest['news_vol_ratio'].iloc[0])
    bucket = next((name for name, lo, hi in BUCKETS if lo <= ratio < hi), None)

    moves = {}
    if use in ('lin', 'ens'):
        cols = NEWS_COLS if use == 'lin' else list(feats.columns) + NEWS_COLS
        for period, h in HORIZONS.items():
            target = (close.shift(-h) / close - 1.0).rename('y')
            train = table.join(target).dropna().tail(TRAIN_WINDOW)
            if len(train) < MIN_TRAIN:
                continue
            model = make_linear() if use == 'lin' else make_ensemble(DAILY_TREES)
            model.fit(train[cols].to_numpy(dtype=float),
                      train['y'].to_numpy(dtype=float))
            moves[period] = float(
                model.predict(latest[cols].to_numpy(dtype=float))[0])
    return moves, bucket


def build_plan(verdict: dict):
    """One entry per asset: reference price plus four slots."""
    use = verdict['use']
    skip_rules = {(s['cls'], s['period'], s['bucket']) for s in verdict['skip']}
    need_model = use != 'tile' or bool(skip_rules)
    if need_model:
        download_news([sb for _, symbols in TABS for sb in symbols],
                      budget_min=15, max_age_days=1)

    plan = []
    for tab, symbols in TABS:
        for sb in symbols:
            ticker = ticker_for(sb)
            price = None
            if ticker:
                try:
                    price, _ = get_live_price(ticker)
                except Exception:
                    price = None

            moves, bucket = {}, None
            if need_model and ticker:
                try:
                    moves, bucket = model_view(sb, ticker, use)
                except Exception as exc:
                    print(f'  {sb}: model data unavailable ({exc})', file=sys.stderr)

            slots = []
            for period in HORIZONS:
                move = moves.get(period)
                skip = (asset_class(sb), period, bucket) in skip_rules
                slots.append({
                    'label': SLOT_LABEL[period],
                    'move': move,                       # None = use the tile price
                    'value': (price * (1.0 + move)
                              if (price is not None and move is not None) else None),
                    'skip': skip,
                })
            plan.append({'tab': tab, 'symbol': sb, 'name': NAMES.get(sb, sb),
                         'price': price, 'slots': slots})
    return plan


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def slot_text(slot):
    """What to type for one slot, as short text."""
    if slot['skip']:
        return 'skip'
    if slot['move'] is None:
        return 'tile'
    pct = f"{slot['move'] * 100:+.2f}%"
    return f"{format_price(slot['value'])} ({pct})" if slot['value'] else f'tile {pct}'


def test_summary(verdict):
    """One or two sentences on what the latest test found."""
    if not verdict['generated_at']:
        return ('No news test has been run yet, so every number is the tile '
                'price, which won the August price test.')
    when = verdict['generated_at']
    if not verdict['models']:
        return f'Latest test ({when}) covered the skip check only.'
    parts = [f"tile price {verdict['tile_per_coin']:.2f}"]
    for model in verdict['models'].values():
        parts.append(f"{model['label'].lower()} {model['per_coin']:.2f}")
    return (f"Latest test ({when}), unicoins returned per coin over "
            f"{verdict['predictions']:,} predictions on "
            f"{len(verdict.get('assets') or [])} of 26 assets: {', '.join(parts)}.")


def headline(verdict, plan):
    use = verdict['use']
    skips = sum(1 for a in plan for s in a['slots'] if s['skip'])
    total = sum(len(a['slots']) for a in plan) - skips
    if use == 'tile':
        how = ('For each asset, type the price on its SafeBets tile into '
               'every timeframe.')
    else:
        how = (f'Numbers come from {METHOD_NAME[use]}. Type the number shown. '
               'If the tile price has moved since this email, apply the '
               'percentage to the tile price instead.')
    if skips:
        how += f' {skips} slot(s) are marked skip today.'
    return total, how


def build_html(verdict, plan, stamp):
    total, how = headline(verdict, plan)
    model_mode = any(s['move'] is not None or s['skip']
                     for a in plan for s in a['slots'])

    blocks, current_tab = [], None
    for asset in plan:
        if asset['tab'] != current_tab:
            current_tab = asset['tab']
            blocks.append(
                '<tr><td style="padding:16px 4px 6px;font-size:11px;'
                'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;'
                f'color:#6b7280;">{current_tab}</td></tr>')
        flag = '' if asset['price'] is not None else ' *'
        cells = ''
        if model_mode:
            cells = ''.join(
                "<td width='25%' style='padding:0;vertical-align:top;'>"
                f"<span style='font-size:11px;color:#6b7280;'>{s['label']}</span><br>"
                f"<span style='font-size:12px;font-family:ui-monospace,Menlo,monospace;"
                f"color:{'#b45309' if s['skip'] else '#111827'};'>{slot_text(s)}</span>"
                '</td>' for s in asset['slots'])
            cells = ("<table width='100%' style='border-collapse:collapse;"
                     f"margin-top:6px;'><tr>{cells}</tr></table>")
        blocks.append(f"""
        <tr>
          <td style="padding:9px 4px;border-bottom:1px solid #e5e7eb;">
            <table width="100%" style="border-collapse:collapse;">
              <tr>
                <td style="padding:0;font-weight:600;font-size:14px;">{asset['name']}
                  <span style="font-weight:400;font-size:12px;color:#6b7280;">&nbsp;{asset['symbol']}</span></td>
                <td style="padding:0;text-align:right;font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#374151;">{format_price(asset['price'])}{flag}</td>
              </tr>
            </table>
            {cells}
          </td>
        </tr>""")

    missing = any(a['price'] is None for a in plan)
    flag_note = ('<p style="margin:12px 0 0;color:#b45309;font-size:12px;">'
                 '* No reference price available. Read the tile.</p>'
                 if missing else '')

    return f"""<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:12px;background:#f9fafb;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#111827;">
  <div style="max-width:460px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;">

    <h1 style="margin:0 0 3px;font-size:17px;">Today&rsquo;s predictions — {total} slots</h1>
    <p style="margin:0 0 12px;color:#6b7280;font-size:12px;">{stamp} · one coin each</p>

    <div style="margin:0 0 6px;padding:10px 12px;background:#eff6ff;border-radius:6px;">
      <p style="margin:0;font-size:13px;color:#1e3a8a;font-weight:600;">{how}</p>
    </div>

    <table style="width:100%;border-collapse:collapse;">
      <tbody>{''.join(blocks)}
      </tbody>
    </table>
    {flag_note}

    <p style="margin:16px 0 0;padding-top:12px;border-top:1px solid #e5e7eb;font-size:11px;color:#6b7280;">
      {test_summary(verdict)} The method only changes when a model beats the
      tile price by more than chance explains. Prices shown are reference
      prices as of {stamp}; SafeBets scores against its own tile. SafeBets
      does not guarantee the value of unicoins.
    </p>

  </div>
</body>
</html>"""


def build_text(verdict, plan, stamp):
    total, how = headline(verdict, plan)
    model_mode = any(s['move'] is not None or s['skip']
                     for a in plan for s in a['slots'])
    lines = [f"Today's predictions — {total} slots, one coin each", stamp, '',
             how, '']
    current_tab = None
    for asset in plan:
        if asset['tab'] != current_tab:
            current_tab = asset['tab']
            lines += ['', current_tab.upper()]
        flag = '' if asset['price'] is not None else ' *'
        lines.append(f"{asset['name']:<14}{asset['symbol']:<8}"
                     f"{format_price(asset['price']):>14}{flag}")
        if model_mode:
            lines.append('    ' + '   '.join(
                f"{s['label']} {slot_text(s)}" for s in asset['slots']))
    if any(a['price'] is None for a in plan):
        lines += ['', '* No reference price available. Read the tile.']
    lines += ['', test_summary(verdict),
              'The method only changes when a model beats the tile price by '
              'more than chance explains.',
              f'Prices shown are reference prices as of {stamp}; SafeBets '
              'scores against its own tile.',
              'SafeBets does not guarantee the value of unicoins.']
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Email today's 104 SafeBets predictions.")
    ap.add_argument('--dry-run', action='store_true',
                    help='print the email instead of sending it')
    ap.add_argument('--text', action='store_true',
                    help='with --dry-run, print the plain-text part')
    ap.add_argument('--verdict', default=VERDICT_FILE,
                    help='test result to read the method from')
    args = ap.parse_args()

    verdict = load_verdict(args.verdict)
    plan = build_plan(verdict)
    if all(a['price'] is None for a in plan):
        raise SystemExit('No prices fetched — refusing to send a table of dashes.')

    now = datetime.now(timezone.utc)
    stamp = now.strftime('%H:%M UTC on %d %b %Y')
    html = build_html(verdict, plan, stamp)
    text = build_text(verdict, plan, stamp)
    if args.dry_run:
        print(text if args.text else html)
        return

    slots = sum(1 for a in plan for s in a['slots'] if not s['skip'])
    send(f"SafeBets — {slots} predictions — {now.strftime('%d %b')}",
         html, text, load_config())
    print(f"Sent {slots} predictions using {METHOD_NAME[verdict['use']]}.",
          file=sys.stderr)


if __name__ == '__main__':
    main()
