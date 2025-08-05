import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env if available

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

LOOK_BACK_PERIOD = 60

print("Loading trained model, scaler, and ticker data...")

model = tf.keras.models.load_model('stock_forecaster_model.keras')
scaler = joblib.load('price_scaler.pkl')

try:
    tickers_df = pd.read_csv('nasdaqtraded_full.csv', sep='|')
    tickers_df = tickers_df[['Symbol', 'Security Name']].dropna()
except FileNotFoundError:
    print("Warning: Ticker symbol mapping file 'nasdaqtraded_full.csv' not found.")
    tickers_df = None

app = Flask(__name__)
CORS(app)

# ✅ Function to fetch historical data from Polygon
def fetch_polygon_data(ticker, days=90):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.date()}/{end_date.date()}?adjusted=true&sort=asc&limit=120&apiKey={POLYGON_API_KEY}"

    print(f"Fetching from Polygon: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Polygon API Error: {response.status_code} - {response.text}")
        return None

    data = response.json()
    if "results" not in data or not data["results"]:
        print("No results in Polygon response.")
        return None

    df = pd.DataFrame(data["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"c": "Close"}, inplace=True)
    return df


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
    print(f"Predicting for: {ticker_symbol}")

    try:
        df = fetch_polygon_data(ticker_symbol)
        if df is None or df.shape[0] < LOOK_BACK_PERIOD:
            return jsonify({"error": f"Not enough data for {ticker_symbol}."}), 400

        recent_prices = df['Close'].values[-LOOK_BACK_PERIOD:].reshape(-1, 1)
        scaled_data = scaler.transform(recent_prices)
        X_input = np.reshape(scaled_data, (1, LOOK_BACK_PERIOD, 1))

        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(prediction)

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
