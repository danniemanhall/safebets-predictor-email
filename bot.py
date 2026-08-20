import os
import sys
import re
import argparse
from playwright.sync_api import sync_playwright

GROUPS = {
    "crypto": {
        "tabs": ["Crypto"],
        "assets": ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "HYPE"]
    },
    "commodities": {
        "tabs": ["Commodities"],
        "assets": ["GOLD", "SILVER", "WTI", "COPPER"]
    },
    "stocks": {
        "tabs": ["Big Tech", "AI Chips"],
        "assets": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "SPCX", "AMD", "MU", "SNDK", "AVGO", "INTC", "ARM"]
    }
}

TIMEFRAMES = ["1 Day", "7 Days", "14 Days", "30 Days"]

def run_bot(group_name):
    session_data = os.environ.get("SAFEBETS_SESSION")
    if not session_data:
        print("❌ Error: SAFEBETS_SESSION missing from GitHub Secrets.")
        sys.exit(1)

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
            print("🔑 Bypassing login and loading dashboard...")
            page.goto("https://app.safebets.world/dashboard", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)

            # --- DIAGNOSTIC CHECK ---
            current_url = page.url
            page_title = page.title()
            print(f"🌍 Current URL: {current_url}")
            print(f"📄 Page Title: {page_title}")
            
            if "login" in current_url.lower():
                print("❌ REDIRECTED TO LOGIN! SafeBets rejected the session cookie (likely due to IP change).")
                sys.exit(1)
            elif "moment" in page_title.lower() or "cloudflare" in page_title.lower():
                print("🛡️ BLOCKED BY CLOUDFLARE! Bot is stuck on a captcha screen.")
                sys.exit(1)
            # -------------------------

            group_info = GROUPS.get(group_name, {})
            tabs = group_info.get("tabs", [])
            assets = group_info.get("assets", [])

            for tab in tabs:
                try:
                    tab_btn = page.locator(f'text="{tab}"').first
                    if tab_btn.is_visible(timeout=3000):
                        tab_btn.click(timeout=3000)
                        print(f"📂 Clicked Category Tab: {tab}")
                        page.wait_for_timeout(2000)
                except Exception as e:
                    pass

            for asset in assets:
                try:
                    asset_tile = page.locator(f'text="{asset}"').first
                    if not asset_tile.is_visible(timeout=3000):
                        print(f"⚠️ {asset} tile not found on screen.")
                        continue
                    
                    asset_tile.click(timeout=3000)
                    print(f"\n📈 Selected Asset: {asset}")
                    page.wait_for_timeout(1500)

                    current_price_block = page.locator('text="Current price"').locator('..').inner_text(timeout=3000)
                    price_match = re.search(r'([\d,]+\.?\d*)', current_price_block)
                    if not price_match:
                        continue
                    
                    spot_price = float(price_match.group(1).replace(',', ''))
                    print(f"🎯 Extracted Live Spot Price for {asset}: ${spot_price:,.2f}")

                    for tf in TIMEFRAMES:
                        try:
                            tf_btn = page.locator(f'text="{tf}"').first
                            if tf_btn.is_visible(timeout=2000):
                                tf_btn.click(timeout=2000)
                                page.wait_for_timeout(500)

                            input_field = page.locator('input[placeholder*="0.00"]').first
                            input_field.fill(str(spot_price), timeout=2000)

                            submit_btn = page.locator('button:has-text("Submit Prediction")').first
                            if submit_btn.is_visible(timeout=2000):
                                submit_btn.click(timeout=2000)
                                print(f"  ✅ Submitted {tf} prediction (${spot_price:,.2f})")
                                page.wait_for_timeout(1000)
                        except Exception as tf_err:
                            print(f"  ⚠️ Error submitting {tf}: {tf_err}")

                except Exception as e:
                    print(f"⚠️ Could not process asset {asset}: {e}")

            print(f"\n🎉 Completed all predictions for group: {group_name}")

        except Exception as e:
            print(f"❌ Automation Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["stocks", "commodities", "crypto"], default="crypto")
    args = parser.parse_args()
    
    run_bot(args.group)
