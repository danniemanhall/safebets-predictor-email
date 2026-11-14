"""
news_bakeoff.py — does news help? Tested the way SafeBets pays.

The August bake-off tested price history and chart indicators against typing
the tile price. This adds the piece that was never tested: news.

NEWS SOURCE
  GDELT, a free archive of world news coverage. For each asset it gives, for
  every day since 1 January 2017, how many articles mentioned it and the
  average tone of that coverage. Downloaded once and saved in news_cache/.

TWO QUESTIONS, ANSWERED SEPARATELY
  1. SKIP TEST (minutes, no model). On days when an asset is unusually heavily
     in the news, does typing the tile price still pay more than the 1 unicoin
     it costs? If a bucket falls below 1, that is a slot worth skipping.
  2. NUMBER TEST (slow). Can a model that reads news produce a better number
     than the tile price? Two models are tried against it:
       - news + price ensemble: the August ensemble with news features added
       - news-only linear: a cautious model that only nudges the tile price
     All three are scored on unicoins returned per coin, using each asset's
     own SafeBets bands and payouts. Purged walk-forward, as in August.

WHAT COUNTS AS A WIN
  A model wins only if all three hold:
    - it returns more per coin than the tile price;
    - the 95% range of that difference stays above zero;
    - it is ahead in both the first and the second half of the test period,
      so one lucky stretch cannot carry it.
  The range is worked out from month-by-month results, because predictions
  made in the same weeks rise and fall together and must not be counted as
  independent evidence.

NO PEEKING
  Crypto closes its day at midnight UTC, the same moment a GDELT day ends, so
  crypto uses the same day's news. Stocks and commodities close hours before
  the GDELT day ends, so they use the previous day's news.

  As in August, horizons for stocks and commodities are counted in trading
  days, which is the same for every contender.

USAGE
  python news_bakeoff.py --quick            # 8 assets, both tests
  python news_bakeoff.py --skip-only        # all 26 assets, skip test only
  python news_bakeoff.py                    # all 26 assets, both tests (hours)
  python news_bakeoff.py --assets BTC WTI   # named assets
  python news_bakeoff.py --synthetic noise  # control: news is pure noise
  python news_bakeoff.py --synthetic signal # control: news truly predicts price
  python news_bakeoff.py --json backtest_result.json   # also save the verdict

The saved verdict is what predict_daily.py reads to decide how to produce the
day's numbers: the tile price, unless a model wins by the rule below. If a
later test finds the model no longer wins, the daily numbers go back to the
tile price.

Keep this file in the same folder as bakeoff.py, safebets_core.py and
class_strategy.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from bakeoff import (compute_features, fetch, HORIZONS, PERIOD_LABEL,
                         TRAIN_WINDOW, MIN_TRAIN, REFIT_EVERY, TEST_DAYS)
    from safebets_core import asset_map, PAYOUTS, THRESHOLDS
    from class_strategy import asset_class
except ImportError as exc:
    if exc.name not in ('bakeoff', 'safebets_core', 'class_strategy'):
        raise
    raise SystemExit(
        f'Could not find {exc.name}.py. Put news_bakeoff.py in the same folder '
        'as bakeoff.py, safebets_core.py and class_strategy.py.')

QUICK_ASSETS = {'BTC', 'ETH', 'LINK', 'AVAX', 'WTI', 'COPPER', 'MSFT', 'AAPL'}
CLASS_ORDER = ['crypto', 'commodity', 'equity']

# What to search the news archive for. Phrases go in double quotes; a list of
# alternatives goes in brackets with OR between them.
NEWS_QUERY = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'DOGE': 'dogecoin',
    'AVAX': '(avax OR "avalanche network" OR "avalanche blockchain")',
    'LINK': 'chainlink',
    'HYPE': 'hyperliquid',
    'NVDA': 'nvidia',
    'TSLA': 'tesla',
    'AAPL': '("apple inc" OR iphone OR "tim cook")',
    'MSFT': 'microsoft',
    'AMZN': 'amazon',
    'META': '("meta platforms" OR facebook)',
    'GOOGL': 'google',
    'NFLX': 'netflix',
    'SPCX': 'spacex',
    'AMD': '("advanced micro devices" OR ryzen)',
    'MU': '"micron technology"',
    'SNDK': 'sandisk',
    'AVGO': 'broadcom',
    'INTC': 'intel',
    'ARM': '"arm holdings"',
    'GOLD': '("gold price" OR "gold prices" OR bullion)',
    'SILVER': '("silver price" OR "silver prices")',
    'WTI': '("crude oil" OR "oil prices")',
    'COPPER': '("copper price" OR "copper prices")',
}

NEWS_COLS = ['news_vol_ratio', 'news_vol_3d', 'tone_7d', 'tone_z', 'tone_trend']

# News-volume buckets for the skip test: today's share of all news coverage
# divided by its own average over the previous 30 days.
BUCKETS = [('quiet', 0.0, 0.8), ('normal', 0.8, 1.5), ('heavy', 1.5, np.inf)]


# --------------------------------------------------------------------------
# News download
# --------------------------------------------------------------------------

GDELT_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'
CACHE_DIR = 'news_cache'
MIN_GAP = 8.0            # seconds between requests; the archive asks for 5+
_last_call = [0.0]


def _gdelt(query: str, mode: str):
    """One timeline (daily since 2017) from GDELT, with polite retries."""
    params = {
        'query': query, 'mode': mode, 'format': 'json',
        'startdatetime': '20170101000000',
        'enddatetime': datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'),
    }
    url = GDELT_URL + '?' + urllib.parse.urlencode(params)
    reason = 'no answer'
    for attempt in range(1, 7):
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        text = ''
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0 (news_bakeoff)'})
            with urllib.request.urlopen(req, timeout=90) as resp:
                text = resp.read().decode('utf-8-sig', errors='replace')
            data = json.loads(text, strict=False)
            series = (data.get('timeline') or [{}])[0].get('data') or []
            if series:
                return series
            raise RuntimeError('the archive returned no rows for this search')
        except urllib.error.HTTPError as exc:
            reason = f'HTTP {exc.code}'
            if exc.code != 429:
                raise RuntimeError(reason)
        except json.JSONDecodeError:
            reason = text.strip()[:140] or 'unreadable answer'
            if 'limit requests' not in text:
                raise RuntimeError(reason)
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = f'network problem ({type(exc).__name__})'
        time.sleep(20 * attempt)          # asked to slow down, or a blip
    raise RuntimeError(reason)


def load_news(sym: str, refresh: bool = False) -> pd.DataFrame | None:
    """Daily article count, total articles and tone for one asset."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{sym}.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    query = NEWS_QUERY.get(sym)
    if not query:
        return None
    vol = _gdelt(query, 'timelinevolraw')
    tone = _gdelt(query, 'timelinetone')

    def frame(rows, fields):
        idx = pd.to_datetime([r['date'] for r in rows],
                             format='%Y%m%dT%H%M%SZ').normalize()
        return pd.DataFrame({name: [r.get(key) for r in rows]
                             for name, key in fields.items()}, index=idx)

    news = frame(vol, {'count': 'value', 'norm': 'norm'}).join(
        frame(tone, {'tone': 'value'}), how='outer')
    news = news[~news.index.duplicated(keep='last')].sort_index()
    news.to_csv(path)
    return news


def news_features(news: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Turn daily counts and tone into the five features the models see. Every
    baseline is taken from earlier days only, and the whole table is then
    pushed forward by `lag` days.
    """
    days = pd.date_range(news.index.min(), news.index.max(), freq='D')
    n = news.reindex(days)
    share = (n['count'] / n['norm'].replace(0, np.nan)).ffill(limit=3)
    tone = n['tone'].ffill(limit=3)

    base = share.rolling(30, min_periods=15).mean().shift(1)
    tone_mean = tone.rolling(60, min_periods=30).mean().shift(1)
    tone_sd = tone.rolling(60, min_periods=30).std().shift(1)

    out = pd.DataFrame(index=days)
    out['news_vol_ratio'] = (share / base).clip(0, 20)
    out['news_vol_3d'] = (share.rolling(3).mean() / base).clip(0, 20)
    out['tone_7d'] = tone.rolling(7, min_periods=4).mean()
    out['tone_z'] = ((tone - tone_mean) / tone_sd.replace(0, np.nan)).clip(-8, 8)
    out['tone_trend'] = (tone.rolling(3).mean()
                         - tone.rolling(30, min_periods=15).mean())
    return out.replace([np.inf, -np.inf], np.nan).shift(lag)


# --------------------------------------------------------------------------
# Made-up data for the two controls
# --------------------------------------------------------------------------

def synthetic_pair(seed: int, signal: bool, n: int = 1500):
    """
    A price series and a news series. With signal=False the news is unrelated
    to the price, so no model should beat the tile price. With signal=True
    tomorrow's price move is partly driven by today's news tone, so a working
    test must show the models winning.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2019-01-01', periods=n, freq='D')

    mood = np.zeros(n)                      # slow-moving, unit variance
    shocks = rng.normal(0, np.sqrt(1 - 0.97 ** 2), n)
    for t in range(1, n):
        mood[t] = 0.97 * mood[t - 1] + shocks[t]

    drift = 0.006 * np.roll(mood, 1) if signal else 0.0
    ret = 0.0002 + drift + rng.normal(0, 0.02, n)
    close = 100 * np.exp(np.cumsum(ret))
    wiggle = np.abs(rng.normal(0, 0.005, n)) * close
    price = pd.DataFrame({'Open': close, 'High': close + wiggle,
                          'Low': close - wiggle, 'Close': close,
                          'Volume': rng.lognormal(15, 0.4, n)}, index=idx)

    news = pd.DataFrame({
        'count': rng.lognormal(6, 0.5, n).round(),
        'norm': 500000.0,
        'tone': -1.0 + 1.5 * (mood + 0.3 * rng.normal(0, 1, n)),
    }, index=idx)
    return price, news


# --------------------------------------------------------------------------
# Scoring — SafeBets' own bands and payouts
# --------------------------------------------------------------------------

def payouts(sb: str, period: str, predicted, actual) -> np.ndarray:
    """Unicoins returned for each prediction (nested bands, best one wins)."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(predicted - actual) / predicted * 100.0
    out = np.zeros(len(predicted))
    bands = sorted(((float(r['deviation']), r['tier'])
                    for r in THRESHOLDS.get(sb, [])
                    if r['periodName'] == period), reverse=True)
    for limit, tier in bands:               # widest first, tightest last
        out[dev <= limit] = PAYOUTS[period].get(tier, 0)
    out[~np.isfinite(dev) | (predicted <= 0)] = 0.0
    return out


def day_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def prepare(df: pd.DataFrame, news_feats: pd.DataFrame):
    """Price features and news features on one shared daily index."""
    feats = compute_features(df)
    feats.index = day_index(df.index)
    feats = feats[~feats.index.duplicated(keep='last')]
    close = pd.Series(df['Close'].to_numpy(), index=day_index(df.index))
    close = close[~close.index.duplicated(keep='last')]
    return feats, close, news_feats.reindex(feats.index)


# --------------------------------------------------------------------------
# Test 1 — should heavy-news days be skipped?
# --------------------------------------------------------------------------

def skip_test(sb, close, news_feats):
    """Tile-price payout on every day since 2017, split by news volume."""
    rows = []
    ratio = news_feats['news_vol_ratio']
    for period, h in HORIZONS.items():
        actual = close.shift(-h)
        ok = actual.notna() & ratio.notna()
        if ok.sum() < 200:
            continue
        pay = payouts(sb, period, close[ok].to_numpy(), actual[ok].to_numpy())
        r = ratio[ok].to_numpy()
        for name, lo, hi in BUCKETS:
            pick = (r >= lo) & (r < hi)
            rows.append((asset_class(sb), period, name,
                         int(pick.sum()), float(pay[pick].sum())))
    return rows


def print_skip_table(records):
    agg = defaultdict(lambda: [0, 0.0])
    for cls, period, bucket, n, total in records:
        agg[(cls, period, bucket)][0] += n
        agg[(cls, period, bucket)][1] += total

    print('\n' + '=' * 76)
    print('TEST 1 — TILE PRICE, UNICOINS PER COIN, BY HOW MUCH NEWS THERE WAS')
    print('=' * 76)
    print(f"{'type':11}{'horizon':9}"
          + ''.join(f"{name + ' news':>18}" for name, _, _ in BUCKETS))
    print('-' * 76)
    below, lowest = [], None
    for cls in CLASS_ORDER:
        for period in HORIZONS:
            cells = []
            for name, _, _ in BUCKETS:
                n, total = agg.get((cls, period, name), (0, 0.0))
                if n == 0:
                    cells.append(f"{'--':>18}")
                    continue
                per = total / n
                cells.append(f"{per:10.2f} ({n:5})")
                if n >= 100:
                    if per < 1.0:
                        below.append((cls, period, name, per, n))
                    if lowest is None or per < lowest[3]:
                        lowest = (cls, period, name, per, n)
            if any('--' not in c for c in cells):
                print(f'{cls:11}{PERIOD_LABEL[period]:9}' + ''.join(cells))
    print('(number in brackets = days in that bucket)')
    if below:
        print('\n=> BELOW THE 1-COIN COST:')
        for cls, period, name, per, n in below:
            print(f'   {cls} {PERIOD_LABEL[period]} on {name}-news days: '
                  f'{per:.2f} per coin over {n} days')
    elif lowest:
        cls, period, name, per, n = lowest
        print(f'\n=> NOTHING TO SKIP. The lowest bucket is {cls} '
              f'{PERIOD_LABEL[period]} on {name}-news days, '
              f'at {per:.2f} per coin.')
    return [{'cls': cls, 'period': period, 'bucket': name,
             'per_coin': round(per, 3), 'days': n}
            for cls, period, name, per, n in below]


# --------------------------------------------------------------------------
# Test 2 — can a news model beat the tile price?
# --------------------------------------------------------------------------

def make_ensemble(trees: int):
    return VotingRegressor([
        ('xgb', XGBRegressor(n_estimators=trees, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             n_jobs=2, verbosity=0, random_state=0)),
        ('rf', RandomForestRegressor(n_estimators=trees, max_depth=12,
                                     n_jobs=2, random_state=0)),
        ('gb', GradientBoostingRegressor(n_estimators=trees, max_depth=6,
                                         learning_rate=0.03, random_state=0)),
    ])


def make_linear():
    # Heavy shrinkage: with no real signal this predicts almost no move and
    # so lands on the tile price; it only departs when the news earns it.
    return make_pipeline(StandardScaler(), Ridge(alpha=200.0))


def number_test(sb, period, feats, close, news_feats, args):
    """Walk forward through the last `test_days`, refitting as in August."""
    h = HORIZONS[period]
    target = close.shift(-h) / close - 1.0
    data = feats.join(news_feats).join(target.rename('y')).dropna()
    if len(data) < MIN_TRAIN + 60:
        return None

    price_cols = list(feats.columns)
    x_all = data[price_cols + NEWS_COLS].to_numpy(dtype=float)
    x_news = data[NEWS_COLS].to_numpy(dtype=float)
    y = data['y'].to_numpy(dtype=float)
    px = close.reindex(data.index).to_numpy(dtype=float)
    actual = px * (1.0 + y)

    n = len(data)
    start = max(MIN_TRAIN + h, n - args.test_days)
    if start >= n:
        return None

    ens = lin = None
    fitted_at = -10 ** 9
    keep, p_ens, p_lin = [], [], []
    for i in range(start, n):
        # Purge: the newest training row must have resolved before day i.
        train_end = i - h
        train_start = max(0, train_end - TRAIN_WINDOW)
        if train_end - train_start < MIN_TRAIN:
            continue
        if ens is None or i - fitted_at >= args.refit:
            sl = slice(train_start, train_end)
            ens = make_ensemble(args.trees).fit(x_all[sl], y[sl])
            lin = make_linear().fit(x_news[sl], y[sl])
            fitted_at = i
        keep.append(i)
        p_ens.append(px[i] * (1.0 + float(ens.predict(x_all[i:i + 1])[0])))
        p_lin.append(px[i] * (1.0 + float(lin.predict(x_news[i:i + 1])[0])))

    if not keep:
        return None
    keep = np.array(keep)
    return pd.DataFrame({
        'date': data.index[keep],
        'cls': asset_class(sb),
        'period': period,
        'spot': payouts(sb, period, px[keep], actual[keep]),
        'ens': payouts(sb, period, p_ens, actual[keep]),
        'lin': payouts(sb, period, p_lin, actual[keep]),
    })


def month_range(diff: pd.Series, dates: pd.Series):
    """
    Mean difference and its 95% range from month-by-month results, plus
    whether the difference is positive in both halves of the period.
    Returns (mean, lo, hi, both_halves, months).
    """
    monthly = diff.groupby(pd.DatetimeIndex(dates).to_period('M')).mean()
    m = len(monthly)
    mean = float(diff.mean())
    if m < 4:
        return mean, np.nan, np.nan, False, m
    se = float(monthly.std(ddof=1) / np.sqrt(m))
    half = m // 2
    both = bool(monthly.iloc[:half].mean() > 0 and monthly.iloc[half:].mean() > 0)
    return mean, mean - 2.1 * se, mean + 2.1 * se, both, m


def print_number_table(res: pd.DataFrame):
    print('\n' + '=' * 76)
    print('TEST 2 — UNICOINS PER COIN: TILE PRICE VS MODELS THAT READ THE NEWS')
    print('=' * 76)
    print(f"{'type':11}{'horizon':9}{'n':>7}{'tile price':>12}"
          f"{'news+price':>12}{'news-only':>11}   best")
    print('-' * 76)
    names = {'spot': 'tile price', 'ens': 'news+price', 'lin': 'news-only'}
    for cls in CLASS_ORDER + ['ALL']:
        for period in HORIZONS:
            part = res[res['period'] == period]
            if cls != 'ALL':
                part = part[part['cls'] == cls]
            if part.empty:
                continue
            means = {k: part[k].mean() for k in names}
            best = max(means, key=means.get)
            print(f"{cls:11}{PERIOD_LABEL[period]:9}{len(part):7}"
                  f"{means['spot']:12.2f}{means['ens']:12.2f}"
                  f"{means['lin']:11.2f}   {names[best]}")

    print('\n' + '=' * 76)
    print('VERDICT')
    print('=' * 76)
    print(f"Predictions scored per contender : {len(res):,}")
    print(f"Tile price                       : {res['spot'].mean():.2f} per coin")
    winners, models = [], {}
    for key, label in (('ens', 'News + price ensemble'), ('lin', 'News-only linear')):
        mean, lo, hi, both, months = month_range(res[key] - res['spot'], res['date'])
        rng = (f'95% range [{lo:+.2f}, {hi:+.2f}] over {months} months'
               if np.isfinite(lo) else f'too few months ({months}) for a range')
        print(f"{label:22}: {res[key].mean():.2f} per coin   "
              f"vs tile price {mean:+.2f}   {rng}   "
              f"ahead in both halves: {'yes' if both else 'no'}")
        won = bool(np.isfinite(lo) and lo > 0 and both)
        if won:
            winners.append(key)
        models[key] = {
            'label': label, 'per_coin': round(float(res[key].mean()), 3),
            'diff': round(mean, 3), 'months': months,
            'lo': round(lo, 3) if np.isfinite(lo) else None,
            'hi': round(hi, 3) if np.isfinite(hi) else None,
            'both_halves': both, 'won': won,
        }

    # The daily numbers use the tile price unless a model won. If both
    # models won, the one further ahead.
    use = max(winners, key=lambda k: models[k]['diff']) if winners else 'tile'
    if winners:
        print(f"\n=> A NEWS MODEL WINS: "
              f"{', '.join(models[k]['label'] for k in winners)}.")
    else:
        print('\n=> THE TILE PRICE HOLDS. No news model beat it by more than '
              'chance would explain.')
    print(f"Daily numbers will use: "
          f"{'the tile price' if use == 'tile' else models[use]['label']}")
    return {'predictions': int(len(res)),
            'tile_per_coin': round(float(res['spot'].mean()), 3),
            'models': models, 'use': use}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Test whether news beats the tile price.')
    ap.add_argument('--quick', action='store_true', help='8 assets, lighter models')
    ap.add_argument('--assets', nargs='*', help='named SafeBets symbols')
    ap.add_argument('--skip-only', action='store_true',
                    help='run the skip test only (minutes)')
    ap.add_argument('--synthetic', choices=['noise', 'signal'],
                    help='no downloads; made-up data with a known answer')
    ap.add_argument('--refresh-news', action='store_true',
                    help='download the news again instead of using news_cache/')
    ap.add_argument('--trees', type=int, help='trees per model (default 300, quick 150)')
    ap.add_argument('--test-days', type=int, default=TEST_DAYS)
    ap.add_argument('--refit', type=int, default=REFIT_EVERY)
    ap.add_argument('--json', metavar='PATH',
                    help='also save the verdict to this file')
    args = ap.parse_args()
    if args.trees is None:
        args.trees = 150 if args.quick else 300

    pairs = []
    for label, ticker in asset_map.items():
        sb = label.split('-')[-1].strip()
        if sb in THRESHOLDS:
            pairs.append((sb, ticker))
    if args.assets:
        want = {a.upper() for a in args.assets}
        pairs = [p for p in pairs if p[0] in want]
    elif args.synthetic:
        pairs = [p for p in pairs if p[0] in {'BTC', 'WTI'}]
    elif args.quick:
        pairs = [p for p in pairs if p[0] in QUICK_ASSETS]

    mode = f'MADE-UP DATA ({args.synthetic})' if args.synthetic else 'live data'
    print(f'Assets: {len(pairs)}   {mode}')

    # ---- gather prices and news --------------------------------------
    ready = []
    if not args.synthetic:
        need = [sb for sb, _ in pairs
                if args.refresh_news
                or not os.path.exists(os.path.join(CACHE_DIR, f'{sb}.csv'))]
        if need:
            print(f'Downloading news for {len(need)} assets '
                  f'(about {len(need) * 2 * MIN_GAP / 60:.0f} minutes, once only)...')
    for k, (sb, ticker) in enumerate(pairs, 1):
        if args.synthetic:
            df, news = synthetic_pair(seed=k, signal=(args.synthetic == 'signal'))
        else:
            try:
                news = load_news(sb, refresh=args.refresh_news)
            except RuntimeError as exc:
                print(f'  {sb:8} news download failed: {exc}')
                continue
            if news is None or len(news) < 400:
                print(f'  {sb:8} too little news history — skipped')
                continue
            df = fetch(ticker)
            if df is None:
                print(f'  {sb:8} no usable price history — skipped')
                continue
        lag = 0 if asset_class(sb) == 'crypto' else 1
        feats, close, nf = prepare(df, news_features(news, lag))
        covered = int(nf['news_vol_ratio'].notna().sum())
        print(f'  {sb:8} {len(news):5} days of news, {covered:5} matched to prices')
        ready.append((sb, feats, close, nf))

    if not ready:
        raise SystemExit('Nothing to test. If the news download failed for every '
                         'asset, wait ten minutes and run the same command again.')

    # ---- test 1 -------------------------------------------------------
    records = []
    for sb, feats, close, nf in ready:
        records += skip_test(sb, close, nf)
    result = {
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'data': 'made-up' if args.synthetic else 'live',
        'assets': [sb for sb, *_ in ready],
        'settings': {'trees': args.trees, 'test_days': args.test_days,
                     'refit': args.refit},
        'skip': print_skip_table(records),
        'predictions': 0, 'tile_per_coin': None, 'models': {}, 'use': 'tile',
    }

    def save():
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as fh:
                json.dump(result, fh, indent=2)
            print(f'\nSaved the verdict to {args.json}')

    if args.skip_only:
        save()
        return

    # ---- test 2 -------------------------------------------------------
    print('\nNow the slow part: fitting models. Leave it running.\n')
    parts = []
    for sb, feats, close, nf in ready:
        for period in HORIZONS:
            out = number_test(sb, period, feats, close, nf, args)
            if out is None:
                continue
            parts.append(out)
            print(f"  {sb:8} {PERIOD_LABEL[period]:4} n={len(out):4}  "
                  f"tile {out['spot'].mean():8.2f}   "
                  f"news+price {out['ens'].mean():8.2f}   "
                  f"news-only {out['lin'].mean():8.2f}")
    if not parts:
        raise SystemExit('No asset had enough history for the model test.')
    result.update(print_number_table(pd.concat(parts, ignore_index=True)))
    save()


if __name__ == '__main__':
    main()
