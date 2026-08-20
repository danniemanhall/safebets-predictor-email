"""
ev_table.py — where does the list stop being worth entering?

Prints every ranked slot, best to worst, with the modelled EV per 1-unicoin
stake and a running total. Marks the point where EV drops to the stake itself:
below that line a slot costs more than it returns and belongs off the list.

Everything here is modelled from historical move distributions against the
platform's published bands. Nothing has been checked against a resolved
prediction yet, so treat the cutoff as a working estimate, not a measurement.

    python ev_table.py            # all slots
    python ev_table.py --min 2    # only slots returning at least 2x the stake
"""

import argparse

from safebets_core import daily_ranking

STAKE = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=float, default=0.0,
                        help="hide slots with EV below this (default: show all)")
    args = parser.parse_args()

    result = daily_ranking(top_n=10_000, probe_n=0, fetch_prices=False)
    rows = result["rows"]

    print(f"{len(rows)} slots ranked. Stake is {STAKE:.0f} unicoin each.\n")
    print(f"{'#':>4}  {'ASSET':<8}{'WIN':<6}{'EV':>8}{'P(BE)':>8}{'CUM EV':>10}{'CUM COST':>10}")

    cum_ev = 0.0
    crossed = False
    for r in rows:
        if r["ev"] < args.min:
            continue
        cum_ev += r["ev"]
        if not crossed and r["ev"] < STAKE:
            crossed = True
            print(f"{'':->4}  {'--- below break-even: everything past here costs more than it returns ---'}")
        print(f"{r['rank']:>4}  {r['symbol']:<8}{r['period_label']:<6}"
              f"{r['ev']:>8.2f}{r['p_bulls_eye']:>7.1%}{cum_ev:>10.0f}{r['rank']:>10}")

    worth_it = [r for r in rows if r["ev"] >= STAKE]
    total_ev = sum(r["ev"] for r in worth_it)

    print(f"\nSlots at or above break-even: {len(worth_it)} of {len(rows)}")
    print(f"Entering all {len(worth_it)}: costs {len(worth_it)} coins, "
          f"modelled return {total_ev:,.0f}")

    for n in (5, 10, 15, 20, 30, 50):
        if n <= len(worth_it):
            share = sum(r["ev"] for r in worth_it[:n]) / total_ev
            print(f"  top {n:>3}: {share:5.1%} of the modelled total, for {n} coins")

    print("\nModelled, not measured. The ordering is the reliable part.")


if __name__ == "__main__":
    main()
