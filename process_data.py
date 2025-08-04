import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Parameters ---
INPUT_FILENAME = "AAPL_data.csv"
LOOK_BACK_PERIOD = 60
TRAIN_SPLIT_RATIO = 0.8

# --- 1. Load and Clean Data ---
print("Loading and selecting data...")
try:
    df = pd.read_csv(INPUT_FILENAME)
    
    # --- ADD THIS LINE TO CLEAN THE DATA ---
    df.dropna(inplace=True) 
    # -----------------------------------------

    close_prices = df['Close'].values.reshape(-1, 1)
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found. Please run 01_fetch_data.py first.")
    exit()

# --- The rest of the script is the same ---
print("Scaling data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)
joblib.dump(scaler, 'price_scaler.pkl')
print("Scaler has been saved to 'price_scaler.pkl'")

print("Creating sequences with a look-back period of 60 days...")
X, y = [], []
for i in range(LOOK_BACK_PERIOD, len(scaled_prices)):
    X.append(scaled_prices[i-LOOK_BACK_PERIOD:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)

print("Splitting data into training and testing sets...")
train_size = int(len(X) * TRAIN_SPLIT_RATIO)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Reshaping data for the LSTM model...")
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("Saving processed data to .npy files...")
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("\n--- Preprocessing Complete ---")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")