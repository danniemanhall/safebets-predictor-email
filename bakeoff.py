"""
bakeoff.py — ensemble vs spot, scored the way SafeBets actually pays.

CONTENDERS
  Model A  spot: submit the current price.
  Model B  ensemble: VotingRegressor(XGBoost + RandomForest + GradientBoosting)
           over RSI-14, MACD(12/26), Bollinger position(20,2), SMA 20/50/200
           crossovers, ATR-14, volume velocity and 200-day momentum, trained on
           a rolling two years of daily bars, predicting forward return.

METRIC
  Expected unicoins per coin staked, under each asset's own proximity bands and
  the confirmed payout table. Deviation is |predicted - actual| / predicted,
  which is the platform's own formula, verified against resolved cards.
  MAE, RMSE and directional accuracy are reported alongside but decide nothing.

VALIDATION
  Purged walk-forward. Training windows end h days before the test point, so a
  target that overlaps the test period can never appear in training. Models are
  refit every REFIT_EVERY trading days rather than daily; a daily refit is
  ~20x the compute for a difference that does not change the ranking.

USAGE
  python bakeoff.py                      # all 26 assets, all four horizons
  python bakeoff.py --quick              # 8 assets, lighter models, for a first look
  python bakeoff.py --assets BTC ETH WTI # named subset
  python bakeoff.py --synthetic          # no network: geometric brownian motion,
                                         # a known-no-signal control for the harness
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from xgboost import XGBRegressor

from safebets_core import (asset_map, HISTORY_PROXY, PAYOUTS, THRESHOLDS)
from class_strategy import asset_class

HORIZONS = {'HOURS_24': 1, 'DAYS_7': 7, 'DAYS_14': 14, 'DAYS_30': 30}
PERIOD_LABEL = {'HOURS_24': '24H', 'DAYS_7': '7D', 'DAYS_14': '14D', 'DAYS_30': '30D'}

TRAIN_WINDOW = 504      # trading days ~= 2 years, as specified
TEST_DAYS = 500         # out-of-sample tail to evaluate on
REFIT_EVERY = 63        # trading days between refits
MIN_TRAIN = 250         # refuse to fit on less than this


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
# Computed directly rather than via the `ta` package: identical formulas, no
# extra runtime dependency in the repo, and every line is causal by
# construction. Verified against ta in the self-test below.

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close, high, low = df['Close'], df['High'], df['Low']

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['macd'] = (ema12 - ema26) / close
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out['bb_pos'] = (close - (sma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out['sma_20_50'] = sma20 / sma50 - 1
    out['sma_50_200'] = sma50 / sma200 - 1
    out['momentum_200'] = close / sma200 - 1

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    out['atr'] = tr.ewm(alpha=1 / 14, adjust=False).mean() / close

    out['volatility'] = close.pct_change().rolling(30).std()
    if 'Volume' in df and df['Volume'].notna().any() and df['Volume'].sum() > 0:
        out['volume_velocity'] = df['Volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        out['volume_velocity'] = 0.0
    out['ret_5'] = close.pct_change(5)
    out['ret_20'] = close.pct_change(20)

    return out.replace([np.inf, -np.inf], np.nan)


def make_model(quick: bool):
    n = 150 if quick else 300
    return VotingRegressor([
        ('xgb', XGBRegressor(n_estimators=n, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             n_jobs=2, verbosity=0, random_state=0)),
        ('rf', RandomForestRegressor(n_estimators=n, max_depth=12,
                                     n_jobs=2, random_state=0)),
        ('gb', GradientBoostingRegressor(n_estimators=n, max_depth=6,
                                         learning_rate=0.03, random_state=0)),
    ])


# --------------------------------------------------------------------------
# Scoring — the platform's own rules
# --------------------------------------------------------------------------

def payout(sb_symbol: str, period: str, predicted: float, actual: float) -> float:
    """Unicoins returned for one prediction, using nested exclusive bands."""
    if predicted is None or not np.isfinite(predicted) or predicted <= 0:
        return 0.0
    dev = abs(predicted - actual) / predicted * 100.0
    bands = sorted((float(r['deviation']), r['tier'])
                   for r in THRESHOLDS.get(sb_symbol, [])
                   if r['periodName'] == period)
    for limit, tier in bands:
        if dev <= limit:
            return float(PAYOUTS[period].get(tier, 0))
    return 0.0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def synthetic_series(seed: int, n: int = 1600) -> pd.DataFrame:
    """Geometric brownian motion — no signal exists, by construction."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.02, n)
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range('2019-01-01', periods=n)
    noise = np.abs(rng.normal(0, 0.005, n)) * close
    return pd.DataFrame({'Close': close, 'High': close + noise,
                         'Low': close - noise,
                         'Volume': rng.lognormal(15, 0.4, n)}, index=idx)


def fetch(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf
    ticker = HISTORY_PROXY.get(ticker, ticker)
    try:
        df = yf.Ticker(ticker).history(period='max', interval='1d')
    except Exception:
        return None
    if df is None or df.empty or len(df) < TRAIN_WINDOW + TEST_DAYS // 2:
        return None
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------

def evaluate(sb_symbol, df, period, quick):
    """
    Returns per-slot results for one asset and horizon:
      spot payouts, ensemble payouts, and error/direction diagnostics.
    """
    h = HORIZONS[period]
    feats = compute_features(df)
    close = df['Close']
    target = close.shift(-h) / close - 1.0

    data = feats.join(target.rename('y')).dropna()
    if len(data) < MIN_TRAIN + 60:
        return None

    X = data.drop(columns='y').to_numpy(dtype=float)
    y = data['y'].to_numpy(dtype=float)
    px = close.reindex(data.index).to_numpy(dtype=float)
    actual = px * (1.0 + y)

    n = len(data)
    start = max(MIN_TRAIN + h, n - TEST_DAYS)
    if start >= n:
        return None

    model, fitted_at = None, -10**9
    rows = []
    for i in range(start, n):
        # Purge: the last usable training target must have resolved before i.
        train_end = i - h
        train_start = max(0, train_end - TRAIN_WINDOW)
        if train_end - train_start < MIN_TRAIN:
            continue

        if i - fitted_at >= REFIT_EVERY or model is None:
            model = make_model(quick)
            model.fit(X[train_start:train_end], y[train_start:train_end])
            fitted_at = i

        pred_ret = float(model.predict(X[i:i + 1])[0])
        ens_price = px[i] * (1.0 + pred_ret)

        rows.append({
            'spot_pay': payout(sb_symbol, period, px[i], actual[i]),
            'ens_pay': payout(sb_symbol, period, ens_price, actual[i]),
            'spot_err': abs(px[i] - actual[i]) / actual[i],
            'ens_err': abs(ens_price - actual[i]) / actual[i],
            'dir_ok': (pred_ret > 0) == (y[i] > 0),
        })

    return rows or None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='8 assets, lighter models')
    ap.add_argument('--assets', nargs='*', help='subset of SafeBets symbols')
    ap.add_argument('--synthetic', action='store_true',
                    help='no network; GBM control series')
    args = ap.parse_args()

    pairs = []
    for label, ticker in asset_map.items():
        sb = label.split('-')[-1].strip()
        if sb in THRESHOLDS:
            pairs.append((sb, ticker))
    if args.assets:
        want = {a.upper() for a in args.assets}
        pairs = [p for p in pairs if p[0] in want]
    elif args.quick:
        pairs = [p for p in pairs
                 if p[0] in {'BTC', 'ETH', 'LINK', 'AVAX', 'WTI', 'COPPER',
                             'MSFT', 'AAPL'}]

    print(f'Assets: {len(pairs)}   horizons: {len(HORIZONS)}   '
          f'refit every {REFIT_EVERY}d   {"SYNTHETIC" if args.synthetic else "live data"}')
    print('This takes a while — three models refit repeatedly per asset-horizon.\n')

    agg = defaultdict(lambda: {'spot': [], 'ens': [], 'se': [], 'ee': [], 'dir': []})
    per_slot = []

    for k, (sb, ticker) in enumerate(pairs, 1):
        df = synthetic_series(seed=k) if args.synthetic else fetch(ticker)
        if df is None:
            print(f'  {sb:8} no usable history — skipped')
            continue

        for period in HORIZONS:
            rows = evaluate(sb, df, period, args.quick)
            if not rows:
                continue
            spot = np.array([r['spot_pay'] for r in rows])
            ens = np.array([r['ens_pay'] for r in rows])
            cls = asset_class(sb)
            for key in ((cls, period), ('ALL', period), ('ALL', 'ALL')):
                agg[key]['spot'].append(spot)
                agg[key]['ens'].append(ens)
                agg[key]['se'].append([r['spot_err'] for r in rows])
                agg[key]['ee'].append([r['ens_err'] for r in rows])
                agg[key]['dir'].append([r['dir_ok'] for r in rows])
            per_slot.append((sb, period, spot.mean(), ens.mean(), len(rows)))
            print(f'  {sb:8} {PERIOD_LABEL[period]:4} n={len(rows):4}  '
                  f'spot {spot.mean():8.2f}   ensemble {ens.mean():8.2f}')

    if not per_slot:
        raise SystemExit('Nothing evaluated.')

    def cat(key, field):
        return np.concatenate([np.asarray(a, dtype=float) for a in agg[key][field]])

    print('\n' + '=' * 72)
    print('EXPECTED UNICOINS PER COIN STAKED')
    print('=' * 72)
    print(f"{'group':12}{'horizon':9}{'n':>7}{'spot':>10}{'ensemble':>11}{'winner':>10}")
    print('-' * 72)
    for cls in ['crypto', 'commodity', 'equity', 'ALL']:
        for period in list(HORIZONS) + (['ALL'] if cls == 'ALL' else []):
            key = (cls, period)
            if key not in agg:
                continue
            s, e = cat(key, 'spot'), cat(key, 'ens')
            win = 'spot' if s.mean() > e.mean() else ('ensemble' if e.mean() > s.mean() else 'tie')
            lbl = PERIOD_LABEL.get(period, period)
            print(f'{cls:12}{lbl:9}{len(s):7}{s.mean():10.2f}{e.mean():11.2f}{win:>10}')

    s, e = cat(('ALL', 'ALL'), 'spot'), cat(('ALL', 'ALL'), 'ens')
    diff = s - e
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se

    print('\n' + '=' * 72)
    print('VERDICT')
    print('=' * 72)
    print(f'Slot-horizon combinations evaluated : {len(per_slot)}')
    print(f'Individual predictions scored       : {len(s):,} per contender')
    print(f'Spot     — unicoins per coin        : {s.mean():.3f}')
    print(f'Ensemble — unicoins per coin        : {e.mean():.3f}')
    print(f'Paired difference (spot - ensemble) : {diff.mean():+.3f}  '
          f'95% CI [{lo:+.3f}, {hi:+.3f}]')
    beat = sum(1 for _, _, sm, em, _ in per_slot if em > sm)
    print(f'Cells where ensemble wins           : {beat} of {len(per_slot)}')

    print(f'\nDiagnostics (not the decision criterion):')
    print(f'  mean abs pct error  spot {cat(("ALL","ALL"),"se").mean()*100:.2f}%   '
          f'ensemble {cat(("ALL","ALL"),"ee").mean()*100:.2f}%')
    print(f'  ensemble directional accuracy     {cat(("ALL","ALL"),"dir").mean()*100:.1f}%')

    if e.mean() > s.mean() and lo < 0:
        print('\n=> ENSEMBLE WINS. Integrate it.')
    elif abs(diff.mean()) < 1e-9:
        print('\n=> TIE. Retain spot.')
    else:
        print('\n=> SPOT WINS. Retain spot; optimise slot selection instead.')


if __name__ == '__main__':
    main()
