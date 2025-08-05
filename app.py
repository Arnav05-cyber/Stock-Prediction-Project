import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf
from datetime import datetime, timedelta


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

# This is the new function to replace your existing one in app.py
@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict():
    # 1. Get the company name from the request
    data = request.get_json()
    if not data or 'company_name' not in data:
        return jsonify({"error": "Missing 'company_name' in request body"}), 400
    
    company_name = data['company_name']
    
    # 2. Find the ticker symbol for the company name
    if tickers_df is None:
        return jsonify({"error": "Ticker symbol mapping not available."}), 500

    match = tickers_df[tickers_df['Security Name'].str.contains(company_name, case=False)]
    
    if match.empty:
        return jsonify({"error": f"Company '{company_name}' not found."}), 404
    
    ticker_symbol = match.iloc[0]['Symbol']
    
    try:
        # 3. Fetch live data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOK_BACK_PERIOD + 30)
        
        live_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        print("\n--- DEBUG: 1. Initial Data Downloaded (Last 5 rows) ---")
        print(live_data.tail())

        live_data.dropna(inplace=True)
        print("\n--- DEBUG: 2. Data After dropna() (Last 5 rows) ---")
        print(live_data.tail())

        if len(live_data) < LOOK_BACK_PERIOD:
             return jsonify({"error": f"Not enough historical data for {ticker_symbol} to make a prediction."}), 400

        recent_prices = live_data['Close'].values[-LOOK_BACK_PERIOD:].reshape(-1, 1)
        print(f"\n--- DEBUG: 3. Recent {LOOK_BACK_PERIOD} Prices for Scaling (Shape: {recent_prices.shape}) ---")
        # Check for NaNs manually before scaling
        if np.isnan(recent_prices).any():
            print("!!! CRITICAL ERROR: NaN detected in source prices BEFORE scaling !!!")
            
        # 4. Preprocess the live data
        scaled_live_data = scaler.transform(recent_prices)
        print(f"\n--- DEBUG: 4. Scaled Data Sent to Model (Shape: {scaled_live_data.shape}) ---")
        # Check for NaNs after scaling
        if np.isnan(scaled_live_data).any():
            print("!!! CRITICAL ERROR: NaN detected AFTER scaling !!!")
        
        # 5. Reshape the data
        X_live = np.reshape(scaled_live_data, (1, LOOK_BACK_PERIOD, 1))
        
        # 6. Make a prediction
        predicted_scaled_price = model.predict(X_live)
        print(f"\n--- DEBUG: 5. Model Raw Prediction (Scaled) ---")
        print(predicted_scaled_price)
        if np.isnan(predicted_scaled_price).any():
            print("!!! CRITICAL ERROR: NaN predicted by the model !!!")

        # 7. Inverse transform 
        predicted_price = scaler.inverse_transform(predicted_scaled_price)
        print(f"\n--- DEBUG: 6. Final Price (Inverse Transformed) ---")
        print(predicted_price)
        
        # 8. Return the result
        return jsonify({
            "company_name": company_name,
            "ticker_symbol": ticker_symbol,
            "predicted_next_day_close_price": round(float(predicted_price[0][0]), 2)
        })

    except Exception as e:
        # Print the full error to the server console for debugging
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
    
if __name__ == '__main__':
    
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)