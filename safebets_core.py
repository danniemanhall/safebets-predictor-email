"""
safebets_core.py — headless ranking core for the SafeBets daily email.

Extracted from app.py with no Streamlit import, so it can run under a GitHub
Actions cron. Same EV logic, same thresholds, same payouts as the dashboard.

What this module decides is WHICH slots (asset x timeframe) are worth a
unicoin and in what order. It never decides what price to submit: the
submission value is the live price read off the SafeBets tile at the moment of
entry. Reference prices returned here are for spotting a broken tile, nothing
more.

Caveat that must travel with any number this produces: EV levels are modelled,
not measured. No slot has been validated against resolved records yet, and the
top-of-table figures (~187 u for a 1 u stake) are implausibly high for a
sustainable payout table — there is probably a cap or eligibility rule not
visible in the API. Trust the ORDER, not the levels.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import yfinance as yf

HISTORY_PERIOD = "5y"
MIN_SAMPLES = 100          # below this a slot is dropped rather than guessed at
DEFAULT_TOP_N = 100        # every slot that clears the 1-unicoin stake
DEFAULT_PROBE_N = 0        # redundant at full width: all 24H slots are already in

# --- ASSET MAP ---
asset_map = {
    # Crypto (7)
    "Crypto - BTC": "BTC-USD", "Crypto - ETH": "ETH-USD", "Crypto - SOL": "SOL-USD",
    "Crypto - DOGE": "DOGE-USD", "Crypto - AVAX": "AVAX-USD", "Crypto - LINK": "LINK-USD",
    "Crypto - HYPE": "HYPE32196-USD",

    # Big Tech (9)
    "Tech - NVDA": "NVDA", "Tech - TSLA": "TSLA", "Tech - AAPL": "AAPL", "Tech - MSFT": "MSFT",
    "Tech - AMZN": "AMZN", "Tech - META": "META", "Tech - GOOGL": "GOOGL", "Tech - NFLX": "NFLX",
    "Tech - SPCX": "SPCX",

    # AI Chips (6)
    "Chips - AMD": "AMD", "Chips - MU": "MU", "Chips - SNDK": "SNDK", "Chips - AVGO": "AVGO",
    "Chips - INTC": "INTC", "Chips - ARM": "ARM",

    # Commodities (4)
    # Metals are SPOT, not futures. SafeBets prices these off Twelve Data, whose
    # "Gold"/"Silver" are XAU/USD and XAG/USD spot. COMEX futures (GC=F, SI=F)
    # carry a basis of roughly 0.3-0.4% over 30 days -- about half the 0.75%
    # Bull's Eye band on GOLD -- so the futures contract is the wrong instrument.
    #
    # History note for the ledger: GOLD was quoted as "Gold (PAXG)" / symbol XAU /
    # priceSource coinbase until ~13-14 Aug 2026, then migrated to plain Gold /
    # twelvedata. Predictions from before that window were scored against a
    # different asset and must not be pooled with later ones.
    "Comm - GOLD": "XAUUSD=X", "Comm - SILVER": "XAGUSD=X",
    "Comm - WTI": "CL=F", "Comm - COPPER": "HG=F"
}

# Assets where the Yahoo ticker may not match what SafeBets quotes. SPCX is the
# one to distrust: SpaceX is not publicly traded, so the quote is a private-market
# estimate no Yahoo symbol can match. Reference prices for these are flagged in
# the email; enter them by hand from the tile.
UNVERIFIED_TICKERS = {"SPCX", "HYPE32196-USD", "XAUUSD=X", "XAGUSD=X", "CL=F", "HG=F"}

# Yahoo 404s on XAUUSD=X and XAGUSD=X, so GOLD and SILVER produced no history at
# all. Only the DISTRIBUTION of percentage moves feeds the EV calculation, so the
# ETFs stand in for it: GLD and SLV are fully collateralised spot trackers whose
# returns differ from spot only by expense ratio -- 0.40%/yr on GLD, about 0.03%
# over a 30-day window, against a 0.75% Bull's Eye band. COMEX futures (GC=F,
# SI=F) would be the worse substitute: their basis runs 0.3-0.4% over the same
# 30 days, roughly ten times the error.
HISTORY_PROXY = {
    "XAUUSD=X": "GLD",
    "XAGUSD=X": "SLV",
}

# Tickers whose price level must not be shown as a reference. A proxy's price is
# the wrong number entirely (GLD is ~1/10 of spot gold), and SPCX is a private
# valuation with no public quote. These rows go out price-blank: read the tile.
NO_REFERENCE_PRICE = set(HISTORY_PROXY) | {"SPCX"}

# --- SCORING TABLE ---
#
# Scoring is nested deviation bands with fixed payouts per timeframe. Bull's Eye
# probability is roughly constant across timeframes (the bands are volatility-
# calibrated), while payouts scale ~50x from 24H to 30D. So which WINDOW a
# prediction is spent on matters far more than any forecasting edge.

PAYOUTS = {
    'HOURS_24': {'BULLS_EYE': 20,   'EXCELLENT': 10,  'GREAT': 5,   'GOOD': 1},
    'DAYS_7':   {'BULLS_EYE': 150,  'EXCELLENT': 50,  'GREAT': 20,  'GOOD': 10},
    'DAYS_14':  {'BULLS_EYE': 400,  'EXCELLENT': 150, 'GREAT': 50,  'GOOD': 20},
    'DAYS_30':  {'BULLS_EYE': 1000, 'EXCELLENT': 400, 'GREAT': 150, 'GOOD': 50},
}
PERIOD_DAYS = {'HOURS_24': 1, 'DAYS_7': 7, 'DAYS_14': 14, 'DAYS_30': 30}
TIER_ORDER = ['BULLS_EYE', 'EXCELLENT', 'GREAT', 'GOOD']
PERIOD_LABEL = {'HOURS_24': '24H', 'DAYS_7': '7D', 'DAYS_14': '14D', 'DAYS_30': '30D'}

# All 26 SafeBets symbols, 432 band rows, pulled from the live accuracy-thresholds API.
THRESHOLDS = {
    "AAPL": [
        {"tier": "BULLS_EYE", "deviation": 0.05, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.3, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.35, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.7, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.5, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.7, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 2.5, "periodName": "DAYS_30"},
    ],
    "AMD": [
        {"tier": "BULLS_EYE", "deviation": 0.18, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.42, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.45, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.38, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.9, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.95, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.3, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.52, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.75, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.65, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.9, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.0, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.8, "periodName": "DAYS_30"},
    ],
    "AMZN": [
        {"tier": "BULLS_EYE", "deviation": 0.05, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.85, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 1.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_30"},
    ],
    "ARM": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.48, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.8, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.05, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.3, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.9, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.45, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.25, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.5, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.5, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.65, "periodName": "DAYS_30"},
    ],
    "AVAX": [
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 2.25, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 3.5, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 3.25, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 5.75, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 9.25, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 2.0, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 4.5, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 8.0, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 13.0, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 6.75, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 12.0, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 19.0, "periodName": "DAYS_30"},
    ],
    "AVGO": [
        {"tier": "BULLS_EYE", "deviation": 0.08, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.85, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.48, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.8, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.28, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.68, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.5, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.55, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.44, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.0, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 3.6, "periodName": "DAYS_30"},
    ],
    "BTC": [
        {"tier": "BULLS_EYE", "deviation": 0.34, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 0.69, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 1.43, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 2.54, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.7, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.15, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.6, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.1, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.85, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.9, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.35, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.75, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.35, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.4, "periodName": "DAYS_30"},
    ],
    "COPPER": [
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.65, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.1, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.8, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.6, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.5, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 4.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.9, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.4, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.5, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.2, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 5.1, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 8.2, "periodName": "DAYS_30"},
    ],
    "DOGE": [
        {"tier": "BULLS_EYE", "deviation": 0.62, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 2.59, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 4.61, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.85, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.35, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.5, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.75, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.1, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.15, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.55, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.25, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.65, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 3.3, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 5.2, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.6, "periodName": "DAYS_30"},
    ],
    "ETH": [
        {"tier": "BULLS_EYE", "deviation": 0.38, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 0.77, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 1.59, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 2.84, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.45, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.75, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.75, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 4.2, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.15, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.35, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.85, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.65, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 3.4, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 5.6, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 8.5, "periodName": "DAYS_30"},
    ],
    "GOLD": [
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.75, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.3, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.05, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.85, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.75, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.5, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.7, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.2, "periodName": "DAYS_30"},
    ],
    "GOOGL": [
        {"tier": "BULLS_EYE", "deviation": 0.05, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.35, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.8, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.15, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 1.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_30"},
    ],
    "HYPE": [
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.75, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.8, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 3.05, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 4.7, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.6, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 4.35, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 6.7, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.0, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 4.0, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 6.65, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 9.75, "periodName": "DAYS_30"},
    ],
    "INTC": [
        {"tier": "BULLS_EYE", "deviation": 0.16, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.85, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.4, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.88, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.85, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.15, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.25, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.65, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.45, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.75, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.75, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 3.75, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.3, "periodName": "DAYS_30"},
    ],
    "LINK": [
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 1.15, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 2.1, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 3.3, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 3.0, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 5.5, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 8.75, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.85, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 4.3, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 7.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 12.3, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.7, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 6.25, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 11.25, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 18.0, "periodName": "DAYS_30"},
    ],
    "META": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.75, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.8, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.1, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.4, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.05, "periodName": "DAYS_30"},
    ],
    "MSFT": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.7, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 0.9, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.55, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.3, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.2, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.4, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 1.9, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 3.2, "periodName": "DAYS_30"},
    ],
    "MU": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.95, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.4, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.95, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.05, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.5, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.55, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.35, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 2.9, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.8, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.95, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.15, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.0, "periodName": "DAYS_30"},
    ],
    "NFLX": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.75, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.5, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 1.7, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.7, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.45, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.45, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.4, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 0.95, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 3.55, "periodName": "DAYS_30"},
    ],
    "NVDA": [
        {"tier": "BULLS_EYE", "deviation": 0.12, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.6, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.85, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.15, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.1, "periodName": "DAYS_30"},
    ],
    "SILVER": [
        {"tier": "BULLS_EYE", "deviation": 0.3, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.6, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.35, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.35, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.9, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.1, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.2, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.3, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.8, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.5, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 6.1, "periodName": "DAYS_30"},
    ],
    "SNDK": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.48, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.05, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.75, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.05, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.25, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.8, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.45, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.15, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.35, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.05, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.5, "periodName": "DAYS_30"},
    ],
    "SOL": [
        {"tier": "BULLS_EYE", "deviation": 0.43, "periodName": "MINUTES_2"},
        {"tier": "EXCELLENT", "deviation": 0.86, "periodName": "MINUTES_2"},
        {"tier": "GREAT", "deviation": 1.78, "periodName": "MINUTES_2"},
        {"tier": "GOOD", "deviation": 3.18, "periodName": "MINUTES_2"},
        {"tier": "BULLS_EYE", "deviation": 0.15, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.4, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.8, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.25, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.7, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.3, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.55, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.0, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 2.0, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.35, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 4.95, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 1.45, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.95, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.8, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.1, "periodName": "DAYS_30"},
    ],
    "SPCX": [
        {"tier": "BULLS_EYE", "deviation": 0.2, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.5, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.0, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 1.6, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 1.1, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 2.25, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 3.6, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.65, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 1.6, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 3.15, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 5.0, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.9, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 2.25, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 4.5, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 7.25, "periodName": "DAYS_30"},
    ],
    "TSLA": [
        {"tier": "BULLS_EYE", "deviation": 0.1, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 0.25, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 0.55, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 0.9, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 0.25, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 0.55, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 1.2, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 2.0, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 0.35, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 0.8, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 1.7, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 2.85, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 0.5, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 1.15, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 2.45, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 4.1, "periodName": "DAYS_30"},
    ],
    "WTI": [
        {"tier": "BULLS_EYE", "deviation": 0.45, "periodName": "HOURS_24"},
        {"tier": "EXCELLENT", "deviation": 1.1, "periodName": "HOURS_24"},
        {"tier": "GREAT", "deviation": 1.9, "periodName": "HOURS_24"},
        {"tier": "GOOD", "deviation": 3.0, "periodName": "HOURS_24"},
        {"tier": "BULLS_EYE", "deviation": 1.0, "periodName": "DAYS_7"},
        {"tier": "EXCELLENT", "deviation": 2.4, "periodName": "DAYS_7"},
        {"tier": "GREAT", "deviation": 4.25, "periodName": "DAYS_7"},
        {"tier": "GOOD", "deviation": 6.8, "periodName": "DAYS_7"},
        {"tier": "BULLS_EYE", "deviation": 1.4, "periodName": "DAYS_14"},
        {"tier": "EXCELLENT", "deviation": 3.3, "periodName": "DAYS_14"},
        {"tier": "GREAT", "deviation": 5.8, "periodName": "DAYS_14"},
        {"tier": "GOOD", "deviation": 9.3, "periodName": "DAYS_14"},
        {"tier": "BULLS_EYE", "deviation": 2.1, "periodName": "DAYS_30"},
        {"tier": "EXCELLENT", "deviation": 4.9, "periodName": "DAYS_30"},
        {"tier": "GREAT", "deviation": 8.75, "periodName": "DAYS_30"},
        {"tier": "GOOD", "deviation": 14.0, "periodName": "DAYS_30"},
    ],
}

# app ticker -> SafeBets symbol
SYMBOL_MAP = {v: k.split(" - ")[-1] for k, v in asset_map.items()}

# Different SafeBets services name the same asset differently: the catalog says
# GOLD, the prediction records say XAU. Normalise so a ledger can join them.
SYMBOL_ALIASES = {"XAU": "GOLD", "PAXG": "GOLD"}


# --- HISTORY AND EV (identical logic to app.py, st.cache_data swapped for a
#     plain process-lifetime memo since there is no Streamlit runtime here) ---

_deviation_cache: dict[tuple[str, int], np.ndarray | None] = {}


def empirical_deviations(ticker_symbol, days):
    """
    Actual historical |percentage move| over `days`, from this asset's own
    history. Used instead of a normal approximation because return
    distributions are leptokurtic: more mass near zero AND fatter tails than
    a Gaussian, which shifts tight-band hit rates meaningfully.
    """
    ticker_symbol = HISTORY_PROXY.get(ticker_symbol, ticker_symbol)

    key = (ticker_symbol, days)
    if key in _deviation_cache:
        return _deviation_cache[key]

    try:
        raw = yf.Ticker(ticker_symbol).history(period=HISTORY_PERIOD)
        if raw.empty or len(raw) < days + 60:
            result = None
        else:
            close = raw['Close']
            moves = (close.shift(-days) / close - 1.0).dropna().abs() * 100.0
            result = moves.to_numpy()
    except Exception:
        result = None

    _deviation_cache[key] = result
    return result


def ev_for_symbol(ticker_symbol, sb_symbol):
    """Expected unicoins per prediction, per timeframe, using empirical bands."""
    bands = THRESHOLDS.get(sb_symbol)
    if not bands:
        return None

    by_period = {}
    for row in bands:
        by_period.setdefault(row['periodName'], []).append(
            (float(row['deviation']), row['tier'])
        )

    out = {}
    for period, days in PERIOD_DAYS.items():
        if period not in by_period or period not in PAYOUTS:
            continue
        moves = empirical_deviations(ticker_symbol, days)
        if moves is None or len(moves) < MIN_SAMPLES:
            continue

        ordered = sorted(by_period[period])
        ev, prev_cum, tier_probs = 0.0, 0.0, {}
        for dev, tier in ordered:
            cum = float(np.mean(moves <= dev))
            excl = max(0.0, cum - prev_cum)
            ev += excl * PAYOUTS[period].get(tier, 0)
            tier_probs[tier] = excl
            prev_cum = cum

        out[period] = {
            'ev': ev,
            'p_any': prev_cum,
            'probs': tier_probs,
            'n': len(moves),
        }
    return out


def get_live_price(ticker_symbol):
    """
    Fetch the freshest price available, with fallbacks.

    Reference only. The number that gets submitted is the one on the SafeBets
    tile — this exists so a tile that looks wrong can be spotted.

    Returns (price, source) or (None, reason).
    """
    if ticker_symbol in NO_REFERENCE_PRICE:
        return None, "no public quote"

    try:
        info = yf.Ticker(ticker_symbol).fast_info
        price = info.get("last_price") if hasattr(info, "get") else getattr(info, "last_price", None)
        if price and float(price) > 0:
            return float(price), "live"
    except Exception:
        pass

    try:
        intraday = yf.Ticker(ticker_symbol).history(period="1d", interval="1m")
        if not intraday.empty:
            return float(intraday['Close'].iloc[-1]), "1m bar"
    except Exception:
        pass

    return None, "unavailable"


# --- THE DAILY PAYLOAD ---


def daily_ranking(top_n=DEFAULT_TOP_N, probe_n=DEFAULT_PROBE_N, fetch_prices=True):
    """
    Rank every covered asset x timeframe slot by expected unicoins and return
    the best `top_n`, plus a few 24H probes.

    The probes are the best-EV 24H slots not already in the main list. Their EV
    is poor -- a few unicoins against a 1-unicoin stake -- so they are not there
    to earn. They resolve overnight, which is the only cheap way to find out
    whether the platform's actual reward fields match what THRESHOLDS and
    PAYOUTS predict. Every EV figure in the main list rests on that assumption
    and none of it has been checked against a resolved record.

    Returns a dict:
        {
          "generated_at": datetime (UTC, tz-aware),
          "rows": [ {...}, ... ],          # length <= top_n, best first
          "probes": [ {...}, ... ],        # length <= probe_n, 24H only
          "total_slots": int,              # how many slots were ranked
          "errors": [ "SYMBOL: reason", ... ],
        }

    Each row:
        rank, asset, symbol, ticker, period, period_label,
        ev, p_any, p_bulls_eye, samples,
        ref_price, price_source, unverified
    """
    rows, errors = [], []

    covered = [(name, ticker) for name, ticker in asset_map.items()
               if SYMBOL_MAP.get(ticker) in THRESHOLDS]

    for asset_name, ticker in covered:
        sb_symbol = SYMBOL_MAP.get(ticker)
        try:
            res = ev_for_symbol(ticker, sb_symbol)
        except Exception as exc:
            errors.append(f"{sb_symbol}: {exc}")
            continue
        if not res:
            errors.append(f"{sb_symbol}: no usable history")
            continue

        for period, d in res.items():
            rows.append({
                "asset": asset_name,
                "symbol": sb_symbol,
                "ticker": ticker,
                "period": period,
                "period_label": PERIOD_LABEL.get(period, period),
                "ev": d['ev'],
                "p_any": d['p_any'],
                "p_bulls_eye": d['probs'].get('BULLS_EYE', 0.0),
                "samples": d['n'],
                "ref_price": None,
                "price_source": None,
                "unverified": ticker in UNVERIFIED_TICKERS,
            })

    total_slots = len(rows)
    rows.sort(key=lambda r: r['ev'], reverse=True)

    main = rows[:top_n]
    chosen = {(r['symbol'], r['period']) for r in main}
    probes = [r for r in rows
              if r['period'] == 'HOURS_24' and (r['symbol'], r['period']) not in chosen
              ][:probe_n]

    if fetch_prices:
        prices = {}
        for row in main + probes:
            ticker = row['ticker']
            if ticker not in prices:
                prices[ticker] = get_live_price(ticker)
            row['ref_price'], row['price_source'] = prices[ticker]

    for i, row in enumerate(main, start=1):
        row['rank'] = i
    for i, row in enumerate(probes, start=1):
        row['rank'] = f"P{i}"

    return {
        "generated_at": datetime.now(timezone.utc),
        "rows": main,
        "probes": probes,
        "total_slots": total_slots,
        "errors": errors,
    }


if __name__ == "__main__":
    result = daily_ranking()
    stamp = result['generated_at'].strftime("%Y-%m-%d %H:%M UTC")
    print(f"Generated {stamp} — top {len(result['rows'])} of {result['total_slots']} slots\n")
    print(f"{'#':>2}  {'ASSET':<14} {'TF':<4} {'EV (u)':>7} {'P(BE)':>7} {'REF PRICE':>14}  SOURCE")
    for r in result['rows']:
        price = f"{r['ref_price']:,.4f}" if r['ref_price'] else "—"
        flag = " *" if r['unverified'] else ""
        print(f"{r['rank']:>2}  {r['asset']:<14} {r['period_label']:<4} "
              f"{r['ev']:>7.1f} {r['p_bulls_eye']:>6.1%} {price:>14}  {r['price_source']}{flag}")
    for r in result['probes']:
        price = f"{r['ref_price']:,.4f}" if r['ref_price'] else "—"
        flag = " *" if r['unverified'] else ""
        print(f"{r['rank']:>2}  {r['asset']:<14} {r['period_label']:<4} "
              f"{r['ev']:>7.1f} {r['p_bulls_eye']:>6.1%} {price:>14}  {r['price_source']}{flag}")
    if any(r['unverified'] for r in result['rows']):
        print("\n* reference price may not match the SafeBets tile — trust the tile.")
    if result['errors']:
        print("\nSkipped:", "; ".join(result['errors']))
    print("\nEV levels are modelled, not measured. Trust the order, not the numbers.")
