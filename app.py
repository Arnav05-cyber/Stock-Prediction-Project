import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import requests  # <-- added for yfinance monkey patch

# --- Monkey patch yfinance with session headers ---
headers = {'User-Agent': 'Mozilla/5.0'}
session = requests.Session()
session.headers.update(headers)

print("Loading trained model, scaler, and ticker data...")

model = tf.keras.models.load_model('stock_forecaster_model.keras')
scaler = joblib.load('price_scaler.pkl')

try:
    tickers_df = pd.read_csv('nasdaqtraded_full.csv', sep='|')
    tickers_df = tickers_df[['Symbol', 'Security Name']].dropna()
except FileNotFoundError:
    print("Warning: Ticker symbol mapping file 'nasdaqtraded_full.csv' not found.")
    tickers_df = None

LOOK_BACK_PERIOD = 60

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
    data = request.get_json()
    if not data or 'company_name' not in data:
        return jsonify({"error": "Missing 'company_name' in request body"}), 400

    company_name = data['company_name']

    if tickers_df is None:
        return jsonify({"error": "Ticker symbol mapping not available."}), 500

    match = tickers_df[tickers_df['Security Name'].str.contains(company_name, case=False)]

    if match.empty:
        return jsonify({"error": f"Company '{company_name}' not found."}), 404

    ticker_symbol = match.iloc[0]['Symbol']

    try:
        print(f"--- DEBUG: yfinance version is {yf.__version__} ---")
        print(f"--- DEBUG: Ticker being fetched: {ticker_symbol} ---")

        # Connectivity check (optional debug)
        try:
            test_resp = requests.get("https://query1.finance.yahoo.com", timeout=5)
            print("DEBUG: Yahoo Finance Connectivity:", test_resp.status_code)
        except Exception as ex:
            print("DEBUG: Failed to reach Yahoo Finance:", ex)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOK_BACK_PERIOD + 30)

        live_data = yf.download(ticker_symbol, start=start_date, end=end_date, session=session)
        print(f"--- DEBUG: Raw data shape from yfinance on Railway: {live_data.shape} ---")

        live_data.dropna(inplace=True)
        print(f"--- DEBUG: Shape after dropna(): {live_data.shape} ---")

        if len(live_data) < LOOK_BACK_PERIOD:
            return jsonify({"error": f"Not enough historical data for {ticker_symbol} to make a prediction."}), 400

        recent_prices = live_data['Close'].values[-LOOK_BACK_PERIOD:].reshape(-1, 1)
        scaled_live_data = scaler.transform(recent_prices)
        X_live = np.reshape(scaled_live_data, (1, LOOK_BACK_PERIOD, 1))
        predicted_scaled_price = model.predict(X_live)
        predicted_price = scaler.inverse_transform(predicted_scaled_price)

        return jsonify({
            "company_name": company_name,
            "ticker_symbol": ticker_symbol,
            "predicted_next_day_close_price": round(float(predicted_price[0][0]), 2)
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
