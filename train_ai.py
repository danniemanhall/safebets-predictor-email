import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

def create_features(ticker_symbol):
    print(f"Fetching data and calculating advanced features for {ticker_symbol}...")
    df = yf.Ticker(ticker_symbol).history(period="5y")
    
    # Include Volume this time!
    df = df[['Close', 'Volume']].copy()
    
    # 1. Basic Features
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands (20-day)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
    # Give the AI the price's relative position between the bands (0 to 1)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # 4. MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    import numpy as np # Make sure you add this at the very top of your file if it's missing!

    # 5. Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Fix the 'inf' error by replacing infinity with 0, and filling any NaNs with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Target: 1 if tomorrow's price goes UP, 0 if DOWN
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop rows with NaN values created by rolling calculations
    df = df.dropna()
    return df

# 1. Get the data
data = create_features("GC=F")

historical_data = data.iloc[:-1]
todays_data = data.iloc[[-1]]

# Define our EXPANDED feature set
features = [
    'Return', 'SMA_10', 'SMA_50', 'RSI', 
    'BB_Position', 'MACD', 'MACD_Hist', 'Volume_Change'
]

X = historical_data[features]
y = historical_data['Target']

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the XGBoost AI Model
print("Training the upgraded AI model...")
# We add learning_rate and max_depth to prevent the model from memorizing the data (overfitting)
model = xgb.XGBClassifier(
    eval_metric='logloss', 
    random_state=42,
    learning_rate=0.05,
    max_depth=4
)
model.fit(X_train, y_train)

# Test the AI
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- Upgraded Model Backtest Accuracy: {accuracy * 100:.2f}% ---")

# Predict Tomorrow
X_today = todays_data[features]
tomorrow_prediction = model.predict(X_today)[0]
probability = model.predict_proba(X_today)[0]

print("\n--- AI PREDICTION FOR TOMORROW ---")
if tomorrow_prediction == 1:
    print(f"Direction: UP (Confidence: {probability[1]*100:.2f}%)")
else:
    print(f"Direction: DOWN (Confidence: {probability[0]*100:.2f}%)")