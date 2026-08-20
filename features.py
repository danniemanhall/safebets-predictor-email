import yfinance as yf
import pandas as pd
import numpy as np

def create_features(ticker_symbol):
    print(f"Fetching and processing data for {ticker_symbol}...")
    
    # 1. Fetch historical data
    df = yf.Ticker(ticker_symbol).history(period="5y")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 2. Daily Price Return (%)
    df['Return'] = df['Close'].pct_change()
    
    # 3. Simple Moving Averages (10-day and 50-day trends)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 4. Relative Strength Index (RSI - 14-day momentum indicator)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. Target Variable for AI: Did tomorrow close HIGHER (1) or LOWER (0) than today?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop initial rows that have empty numbers due to moving averages calculation
    df = df.dropna()
    
    return df

# Run feature creation for Gold
gold_features = create_features("GC=F")

# Display the last 5 days with indicators
print("\n--- Processed AI Input Features (Last 5 Days) ---")
print(gold_features[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Target']].tail())