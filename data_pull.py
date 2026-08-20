import yfinance as yf
import pandas as pd

def fetch_commodity_data(ticker_symbol, period="5y"):
    print(f"Fetching historical data for {ticker_symbol}...")
    
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period)
    
    # Keep only the key market data columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# 'GC=F' is Gold Futures on Yahoo Finance
gold_data = fetch_commodity_data("GC=F")

print("\n--- Last 5 Days of Gold Prices ---")
print(gold_data.tail())