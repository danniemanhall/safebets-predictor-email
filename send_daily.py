"""
send_daily.py — the whole product.

Computes today's SafeBets slots and emails them, grouped by asset so each tile
is visited once. Daniel opens one email, works down the list, and types the
tile's own current price into every timeframe box shown for that asset.

What the email carries is the SLOT LIST. Prices are reference only: the number
that gets submitted is whatever the tile says at the moment of entry, because an
emailed price starts decaying the second it is sent.

Below the list sits a three-line per-class summary and one flag: what share of
the day's modelled return comes from cells that have never actually resolved.
That flag is the point of the block. The class breakdown itself is descriptive
only — every class clears the 1-coin stake at every horizon, so nothing in it
tells you to skip anything.

The summary reuses the EVs already computed for the ranking rather than calling
class_strategy.class_strategies(), which would re-fetch two years of hourly bars
for 26 assets and risk the Action's timeout. Hourly bars were tested and moved
the 24H estimates by under a percentage point, so nothing is lost by leaving
them out of the daily run.

Configuration is entirely environment variables (GitHub repo secrets):

    SMTP_HOST     e.g. smtp.gmail.com
    SMTP_PORT     587 for STARTTLS (default), 465 for implicit SSL
    SMTP_USER     the sending account
    SMTP_PASS     app password, never the account password
    MAIL_TO       recipient; comma-separate for several
    MAIL_FROM     optional, defaults to SMTP_USER

Local test, sends nothing:

    python send_daily.py --dry-run          # prints the HTML
    python send_daily.py --dry-run --text   # prints the plain-text part

Real send:

    python send_daily.py
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from collections import defaultdict
from email.message import EmailMessage

from safebets_core import daily_ranking, DEFAULT_TOP_N, DEFAULT_PROBE_N
from class_strategy import CLASS_ORDER, asset_class, blended_ev

SUBJECT_PREFIX = "SafeBets — today's slots"

# The ranking rows carry a display label; the strategy tables are keyed by
# period name. One map, so the two never drift apart.
PERIOD_KEY = {
    "24H": "HOURS_24",
    "7D": "DAYS_7",
    "14D": "DAYS_14",
    "30D": "DAYS_30",
}

CLASS_LABEL = {
    "crypto": "Crypto",
    "commodity": "Commodities",
    "equity": "Equities",
}


# --- FORMATTING ---


def format_price(row):
    price = row.get("ref_price")
    if price is None:
        return "—"
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:.6f}"


def _row_html(r, muted=False):
    flag = " *" if (r["unverified"] or r["ref_price"] is None) else ""
    name_color = "#6b7280" if muted else "#111827"
    return f"""
        <tr>
          <td style="padding:9px 6px;border-bottom:1px solid #e5e7eb;color:#9ca3af;font-size:12px;vertical-align:top;">{r['rank']}</td>
          <td style="padding:9px 6px;border-bottom:1px solid #e5e7eb;vertical-align:top;">
            <span style="font-weight:600;color:{name_color};">{r['symbol']}</span>
            <span style="color:#6b7280;">&nbsp;{r['period_label']}</span><br>
            <span style="color:#9ca3af;font-size:11px;">EV {r['ev']:.0f} ú</span>
          </td>
          <td style="padding:9px 6px;border-bottom:1px solid #e5e7eb;text-align:right;font-family:ui-monospace,Menlo,monospace;font-size:13px;vertical-align:top;">{format_price(r)}{flag}</td>
        </tr>"""


GROUP_THRESHOLD = 30       # above this, group by asset so each tile is visited once


def group_rows(rows):
    """
    Group slots by asset, assets ordered by their best slot's rank, timeframes
    within an asset ordered by rank too.

    At fifteen entries a flat EV list is fine. At a hundred it means opening the
    WTI tile four separate times. Grouping means one tile, four boxes, next
    tile. The rank column still shows each slot's true EV position, so stopping
    early still costs you the right things.
    """
    order, buckets = [], {}
    for r in rows:
        if r["symbol"] not in buckets:
            buckets[r["symbol"]] = []
            order.append(r["symbol"])
        buckets[r["symbol"]].append(r)
    return [(sym, buckets[sym]) for sym in order]


# --- CLASS SUMMARY ---


def class_summary(rows):
    """
    Aggregate today's slots into per-class expected returns, blending each
    class-and-horizon cell's modelled EV with Daniel's measured results.

    Returns (per_class, unverified_share, total_ev) where per_class maps a
    class name to {'slots': int, 'ev': float}. unverified_share is the fraction
    of total expected return coming from cells that have never resolved — every
    30D cell, plus equity 14D. That fraction is the number worth watching: it
    says how much of the headline is measurement and how much is the model
    talking to itself.
    """
    groups = defaultdict(list)
    for r in rows:
        period = PERIOD_KEY.get(r["period_label"])
        if period:
            groups[(asset_class(r["symbol"]), period)].append(r)

    per_class = {}
    total_ev = 0.0
    unverified_ev = 0.0

    for (cls, period), members in groups.items():
        model = sum(m["ev"] for m in members) / len(members)
        blend, _measured, _n, days = blended_ev(model, cls, period)
        if blend is None:
            continue
        cell_ev = len(members) * blend

        bucket = per_class.setdefault(cls, {"slots": 0, "ev": 0.0})
        bucket["slots"] += len(members)
        bucket["ev"] += cell_ev

        total_ev += cell_ev
        if days == 0:
            unverified_ev += cell_ev

    share = (unverified_ev / total_ev) if total_ev else 0.0
    return per_class, share, total_ev


def summary_lines(rows):
    """The three class lines, as (label, slots, ev) tuples in a fixed order."""
    per_class, _share, _total = class_summary(rows)
    return [
        (CLASS_LABEL[cls], per_class[cls]["slots"], per_class[cls]["ev"])
        for cls in CLASS_ORDER
        if cls in per_class
    ]


def build_html(result, top_n):
    stamp = result["generated_at"].strftime("%H:%M UTC on %d %b %Y")

    if len(result["rows"]) > GROUP_THRESHOLD:
        chunks = []
        for symbol, group in group_rows(result["rows"]):
            price = format_price(group[0])
            flag = " *" if (group[0]["unverified"] or group[0]["ref_price"] is None) else ""

            cells = []
            for r in group:
                cells.append(
                    f"<td width='25%' style=\"padding:0;text-align:left;\">"
                    f"<span style='font-weight:600;font-size:14px;'>{r['period_label']}</span>"
                    f"<span style='color:#9ca3af;font-size:11px;'>&nbsp;#{r['rank']}</span>"
                    f"</td>"
                )
            while len(cells) < 4:
                cells.append("<td width='25%' style=\"padding:0;\"></td>")

            chunks.append(f"""
        <tr>
          <td style="padding:11px 4px;border-bottom:1px solid #e5e7eb;">
            <table width="100%" style="border-collapse:collapse;">
              <tr>
                <td style="padding:0;font-weight:600;font-size:15px;">{symbol}</td>
                <td style="padding:0;text-align:right;font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#374151;">{price}{flag}</td>
              </tr>
            </table>
            <table width="100%" style="border-collapse:collapse;margin-top:6px;">
              <tr>{''.join(cells)}</tr>
            </table>
          </td>
        </tr>""")
        rows_html = "".join(chunks)
    else:
        rows_html = "".join(_row_html(r) for r in result["rows"])

    probes_html = ""
    if result.get("probes"):
        probe_rows = "".join(_row_html(r, muted=True) for r in result["probes"])
        probes_html = f"""
    <p style="margin:22px 0 6px;font-size:13px;font-weight:600;">Probes — {len(result['probes'])} coins</p>
    <p style="margin:0 0 8px;color:#6b7280;font-size:12px;">
      These resolve tomorrow. They pay badly and are not meant to; they are there
      so the reward fields can be checked against what the model assumes.
    </p>
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
      <tbody>{probe_rows}
      </tbody>
    </table>"""

    # --- class summary block ---
    _per_class, unverified_share, total_ev = class_summary(result["rows"])
    summary_rows = "".join(
        f"""
        <tr>
          <td style="padding:3px 0;font-size:12px;color:#374151;">{label}</td>
          <td style="padding:3px 0;font-size:12px;color:#6b7280;text-align:right;">{slots} slots</td>
          <td style="padding:3px 0;font-size:12px;color:#374151;text-align:right;font-family:ui-monospace,Menlo,monospace;">~{ev:,.0f} ú</td>
        </tr>"""
        for label, slots, ev in summary_lines(result["rows"])
    )

    summary_html = f"""
    <div style="margin:20px 0 0;padding-top:14px;border-top:1px solid #e5e7eb;">
      <p style="margin:0 0 6px;font-size:12px;font-weight:600;color:#111827;">
        Expected return by class — ~{total_ev:,.0f} ú total
      </p>
      <table style="width:100%;border-collapse:collapse;">
        <tbody>{summary_rows}
        </tbody>
      </table>
      <p style="margin:8px 0 0;color:#b45309;font-size:12px;">
        {unverified_share:.0%} of that comes from slots that have never resolved
        for you — every 30D cell, plus equity 14D. Payouts are confirmed at 24H,
        7D and 14D; the 30D tier is still the model's word alone.
      </p>
    </div>"""

    has_flag = any(r["unverified"] or r["ref_price"] is None
                   for r in result["rows"] + result.get("probes", []))
    flag_note = ""
    if has_flag:
        flag_note = (
            "<p style='margin:14px 0 0;color:#b45309;font-size:12px;'>"
            "* Reference price unreliable or unavailable. Read the tile.</p>"
        )

    skipped_note = ""
    if result["errors"]:
        skipped_note = (
            "<p style='margin:6px 0 0;color:#6b7280;font-size:12px;'>"
            f"Not ranked: {'; '.join(result['errors'])}.</p>"
        )

    total_coins = len(result["rows"]) + len(result.get("probes", []))

    return f"""<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:12px;background:#f9fafb;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#111827;">
  <div style="max-width:420px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;">

    <h1 style="margin:0 0 4px;font-size:17px;">Today&rsquo;s slots — {total_coins} coins</h1>
    <p style="margin:0 0 16px;color:#6b7280;font-size:12px;">
      {stamp} · {len(result['rows'])} of {result['total_slots']} slots · one coin each
    </p>

    <table style="width:100%;border-collapse:collapse;font-size:14px;">
      <tbody>{rows_html}
      </tbody>
    </table>
    {probes_html}

    <p style="margin:18px 0 0;font-size:12px;color:#374151;">
      Prices are reference only, as of {stamp} — <strong>type what the tile
      says</strong>. The column is there to catch a tile that looks wrong.
    </p>
    {flag_note}
    {skipped_note}
    {summary_html}

    <p style="margin:16px 0 0;padding-top:12px;border-top:1px solid #e5e7eb;font-size:11px;color:#6b7280;">
      Every slot submits the same value: the tile's current price. EV is modelled
      from historical move distributions and blended with 134 resolved
      predictions, weighted by the number of distinct days those came from.
      <strong>The ordering is more trustworthy than the levels.</strong> Amounts
      are in unicoins, a closed-platform token.
    </p>

  </div>
</body>
</html>"""


def build_text(result, top_n):
    stamp = result["generated_at"].strftime("%H:%M UTC on %d %b %Y")
    lines = [
        f"Today's {top_n} slots",
        f"Generated {stamp}. Best {top_n} of {result['total_slots']} ranked slots.",
        "",
        f"{'#':>2}  {'ASSET':<8}{'WINDOW':<8}{'REF PRICE':>14}{'EV':>7}",
    ]
    for r in result["rows"]:
        flag = " *" if r["unverified"] or r["ref_price"] is None else ""
        lines.append(
            f"{r['rank']:>2}  {r['symbol']:<8}{r['period_label']:<8}"
            f"{format_price(r):>14}{r['ev']:>7.0f}{flag}"
        )
    if result.get("probes"):
        lines += ["", f"PROBES ({len(result['probes'])} coins, resolve tomorrow)"]
        for r in result["probes"]:
            flag = " *" if (r["unverified"] or r["ref_price"] is None) else ""
            lines.append(
                f"{r['rank']:>2}  {r['symbol']:<8}{r['period_label']:<8}"
                f"{format_price(r):>14}{r['ev']:>7.0f}{flag}"
            )

    lines += [
        "",
        f"Reference prices are as of {stamp} — use the tile number if it differs.",
        "What you submit is the live price on the tile.",
    ]
    if any(r["unverified"] or r["ref_price"] is None
           for r in result["rows"] + result.get("probes", [])):
        lines.append("* Reference price unreliable or unavailable. Read the tile.")
    if result["errors"]:
        lines.append(f"Not ranked today: {'; '.join(result['errors'])}.")

    _per_class, unverified_share, total_ev = class_summary(result["rows"])
    lines += ["", f"EXPECTED RETURN BY CLASS — ~{total_ev:,.0f} u total"]
    for label, slots, ev in summary_lines(result["rows"]):
        lines.append(f"  {label:<13}{slots:>3} slots{ev:>12,.0f} u")
    lines.append(
        f"{unverified_share:.0%} of that comes from slots that have never "
        "resolved for you —"
    )
    lines.append(
        "every 30D cell, plus equity 14D. Payouts are confirmed at 24H, 7D "
        "and 14D."
    )

    lines += [
        "",
        "Every slot submits the same value: the tile's current price. EV is",
        "modelled and blended with 134 resolved predictions, weighted by the",
        "number of distinct days those came from. The ordering is more",
        "trustworthy than the levels. Amounts are in unicoins, a closed-platform",
        "token.",
    ]
    return "\n".join(lines)


# --- DELIVERY ---


def send(subject, html, text, cfg):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["mail_from"]
    msg["To"] = ", ".join(cfg["mail_to"])
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    if cfg["port"] == 465:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=30) as s:
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as s:
            s.starttls()
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)


def load_config():
    missing = [k for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "MAIL_TO")
               if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    user = os.environ["SMTP_USER"]
    return {
        "host": os.environ["SMTP_HOST"],
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": user,
        "password": os.environ["SMTP_PASS"],
        "mail_from": os.environ.get("MAIL_FROM", user),
        "mail_to": [a.strip() for a in os.environ["MAIL_TO"].split(",") if a.strip()],
    }


def main():
    parser = argparse.ArgumentParser(description="Email today's SafeBets slot list.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the email instead of sending it")
    parser.add_argument("--text", action="store_true",
                        help="with --dry-run, print the plain-text part")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help=f"how many slots to list (default {DEFAULT_TOP_N})")
    parser.add_argument("--probes", type=int, default=DEFAULT_PROBE_N,
                        help=f"how many 24H probe slots to append (default {DEFAULT_PROBE_N}, 0 to disable)")
    args = parser.parse_args()

    result = daily_ranking(top_n=args.top, probe_n=args.probes)
    if not result["rows"]:
        raise SystemExit("No slots ranked — refusing to send an empty email.")

    subject = f"{SUBJECT_PREFIX} — {result['generated_at'].strftime('%d %b')}"
    html = build_html(result, args.top)
    text = build_text(result, args.top)

    if args.dry_run:
        print(text if args.text else html)
        return

    send(subject, html, text, load_config())
    stamp = result["generated_at"].strftime("%H:%M UTC")
    print(f"Sent {len(result['rows'])} slots + {len(result['probes'])} probes, "
          f"generated {stamp}.", file=sys.stderr)


if __name__ == "__main__":
    main()
