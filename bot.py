import os
import sys
import time
import argparse
import imaplib
import email
import re
from email.header import decode_header
from playwright.sync_api import sync_playwright

GROUPS = {
    "stocks": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "SPCX", "AMD", "MU", "SNDK", "AVGO", "INTC", "ARM"],
    "commodities": ["GOLD", "SILVER", "WTI", "COPPER"],
    "crypto": ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "HYPE"]
}

def get_magic_link_from_gmail(user_email, app_password):
    """Connects to Gmail via IMAP to wait for and extract the SafeBets magic link."""
    print("⏳ Connecting to Gmail to intercept Magic Link...")
    
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(user_email, app_password)
        mail.select("inbox")
    except Exception as e:
        print(f"❌ Gmail Login Failed. Check your App Password: {e}")
        return None

    # Poll inbox for up to 60 seconds
    for _ in range(12):
        time.sleep(5)
        print("🔄 Checking inbox for new SafeBets email...")
        
        # Search for recent unread emails
        status, messages = mail.search(None, '(UNSEEN)')
        if status == 'OK' and messages[0]:
            mail_ids = messages[0].split()
            
            for i in reversed(mail_ids):
                status, msg_data = mail.fetch(i, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode(errors="ignore")
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode(errors="ignore")
                        
                        # Extract URL from the email body
                        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', body)
                        for url in urls:
                            # Added safebets.app as a fallback just in case the email link redirects
                            if "safebets.world" in url.lower() or "safebets.app" in url.lower():
                                print("✅ Intercepted Magic Link!")
                                return url
                                
    print("❌ Timed out waiting for SafeBets email.")
    return None

def run_bot(group_name):
    email_address = os.environ.get("SAFEBETS_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not email_address or not app_password:
        print("❌ Error: SAFEBETS_EMAIL or GMAIL_APP_PASSWORD missing from GitHub Secrets.")
        sys.exit(1)

    print(f"🚀 Starting Autonomous SafeBets Bot [Group: {group_name}]")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # THIS IS THE CRITICAL FIX: ignore_https_errors=True
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ignore_https_errors=True
        )
        page = context.new_page()

        try:
            # 1. Trigger Magic Link
            print("🔑 Navigating to login page...")
            page.goto("https://app.safebets.world", wait_until="networkidle")
            
            # Input email and request link
            page.fill('input[type="email"], input[name="email"]', email_address)
            page.click('button[type="submit"], button:has-text("Log In"), button:has-text("Send Link")')
            print("📧 Magic link requested.")

            # 2. Extract Link from Gmail
            magic_link = get_magic_link_from_gmail(email_address, app_password)
            
            if not magic_link:
                browser.close()
                sys.exit(1)

            # 3. Log In via Magic Link
            print("🔗 Navigating to Magic Link URL...")
            page.goto(magic_link, wait_until="networkidle")
            page.wait_for_timeout(5000)

            # 4. Main Betting Loop
            assets_to_process = GROUPS.get(group_name, [])
            print(f"📋 Processing {len(assets_to_process)} assets in {group_name}...")

            for asset in assets_to_process:
                try:
                    print(f"\n📈 Processing Asset: {asset}")
                    page.click(f'text="{asset}"') 
                    page.wait_for_timeout(2000)

                    # Extract current Spot Price from the UI tile
                    spot_price_element = page.locator('.current-price, .spot-price, [data-testid="spot-price"]').first
                    spot_price_text = spot_price_element.inner_text().replace('$', '').replace(',', '').strip()
                    spot_price = float(spot_price_text)
                    print(f"🎯 Extracted Live Spot Price for {asset}: ${spot_price:,.2f}")

                    # Fill inputs for 1D, 7D, 14D, 30D
                    prediction_inputs = page.locator('input[placeholder*="Target"], input[type="number"]').all()
                    for inp in prediction_inputs:
                        inp.fill(str(spot_price))

                    # Click Submit
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