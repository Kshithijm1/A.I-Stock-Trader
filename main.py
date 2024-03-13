import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from Stocktrader import Transformer, Time2Vec, SingleAttention, MultiAttention, TransformerEncoder
import matplotlib.pyplot as plt

# Function to add technical indicators to the DataFrame
def add_technical_indicators(df):
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # KDJ Indicator
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # Drop rows with NaN values
    df.dropna(inplace=True)
    return df

# Load and preprocess the dataset with technical indicators
def load_preprocess_data_with_indicators(ticker):
    df = yf.download(ticker, start="1900-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    df = add_technical_indicators(df[['Open', 'High', 'Low', 'Close', 'Volume']])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    seq_length = 50
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 3])  # Using Close price as the target

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

# Specify the stock ticker
ticker = "VLD"

# Load your dataset with technical indicators
X, y, scaler = load_preprocess_data_with_indicators(ticker)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and compile the model with optimized hyperparameters
model = Transformer(time_embedding=True, n_layers=5, d_k=256, d_v=256, n_heads=10, ff_dim=512, feature_size=128, seq_len=50, out_dim=1, dropout=0.4)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Predict and generate signals based on the refined KDJ indicator
def generate_kdj_signals(df):
    signals = []
    for i in range(len(df)):
        if df['K'].iloc[i] > df['D'].iloc[i] and df['J'].iloc[i] > df['K'].iloc[i] and df['K'].iloc[i-1] <= df['D'].iloc[i-1]:
            signals.append('Buy')
        elif df['K'].iloc[i] < df['D'].iloc[i] and df['J'].iloc[i] < df['K'].iloc[i] and df['K'].iloc[i-1] >= df['D'].iloc[i-1]:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

# Download the stock data again to get the dates
df = yf.download(ticker, start="1900-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
df = add_technical_indicators(df[['Open', 'High', 'Low', 'Close', 'Volume']])

# Generate KDJ signals for the test set
test_kdj_signals = generate_kdj_signals(df[-len(X_test):])

# Function to plot the signals
def plot_signals(df, signals):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    buy_signals = df[signals == 'Buy']
    sell_signals = df[signals == 'Sell']
    plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.title(f'Stock Price and Signals for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Plot the last 10 signals
plot_signals(df[-len(test_kdj_signals[-10:]):], pd.Series(test_kdj_signals[-10:], index=df.index[-len(test_kdj_signals[-10:]):]))
