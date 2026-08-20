import os
import sys
import argparse
from playwright.sync_api import sync_playwright

GROUPS = {
    "stocks": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "SPCX", "AMD", "MU", "SNDK", "AVGO", "INTC", "ARM"],
    "commodities": ["GOLD", "SILVER", "WTI", "COPPER"],
    "crypto": ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "HYPE"]
}

def run_bot(group_name):
    session_data = os.environ.get("SAFEBETS_SESSION")
    if not session_data:
        print("❌ Error: SAFEBETS_SESSION missing from GitHub Secrets.")
        sys.exit(1)

    # Save session state to temporary file for Playwright
    with open("temp_state.json", "w") as f:
        f.write(session_data)

    print(f"🚀 Starting Autonomous SafeBets Bot [Group: {group_name}]")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state="temp_state.json",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ignore_https_errors=True
        )
        page = context.new_page()

        try:
            print("🔑 Bypassing login and going straight to dashboard...")
            page.goto("https://app.safebets.world", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)

            assets_to_process = GROUPS.get(group_name, [])
            print(f"📋 Processing {len(assets_to_process)} assets in {group_name}...")

            for asset in assets_to_process:
                try:
                    print(f"\n📈 Processing Asset: {asset}")
                    page.click(f'text="{asset}"') 
                    page.wait_for_timeout(2000)

                    spot_price_element = page.locator('.current-price, .spot-price, [data-testid="spot-price"]').first
                    spot_price_text = spot_price_element.inner_text().replace('$', '').replace(',', '').strip()
                    spot_price = float(spot_price_text)
                    print(f"🎯 Extracted Live Spot Price for {asset}: ${spot_price:,.2f}")

                    prediction_inputs = page.locator('input[placeholder*="Target"], input[type="number"]').all()
                    for inp in prediction_inputs:
                        inp.fill(str(spot_price))

                    submit_btn = page.locator('button:has-text("Submit"), button:has-text("Place Bet")').first
                    if submit_btn.is_visible():
                        submit_btn.click()
                        print(f"✅ Submitted bets for {asset}")
                        page.wait_for_timeout(1500)

                except Exception as e:
                    print(f"⚠️ Could not process {asset}: {e}")
                    continue

            print(f"\n🎉 Completed all bets for group: {group_name}")

        except Exception as e:
            print(f"❌ Automation Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["stocks", "commodities", "crypto"], default="crypto")
    args = parser.parse_args()
    
    run_bot(args.group)