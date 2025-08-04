import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --- 1. Load All Artifacts and Test Data ---
print("Loading model, scaler, and test data...")
try:
    # Load the trained model
    model = tf.keras.models.load_model('stock_forecaster_model.keras')
    
    # Load the scaler
    scaler = joblib.load('price_scaler.pkl')
    
    # Load the test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
except FileNotFoundError as e:
    print(f"Error: A required file was not found. {e}")
    print("Please ensure all scripts from 01 to 03 have been run successfully.")
    exit()

print("All files loaded successfully.")


# --- 2. Make Predictions on the Entire Test Set ---
print("Making predictions on the test data...")
predicted_scaled_prices = model.predict(X_test)


# --- 3. Inverse Transform Data to Get Real Dollar Values ---
# We need to un-scale both our predictions and the original data to compare them in real dollars.
print("Converting scaled data back to real dollar values...")
predicted_prices = scaler.inverse_transform(predicted_scaled_prices)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# --- 4. Numerical Evaluation (RMSE) ---
# Calculate the Root Mean Squared Error (RMSE)
# This tells us, on average, how many dollars off our model's prediction is.
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
print("\n--- Model Performance ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print("This means the model's predictions are, on average, off by this amount.")


# --- 5. Visual Evaluation (Plotting) ---
print("\nGenerating plot...")
plt.figure(figsize=(15, 7))
plt.plot(real_prices, color='red', label='Real AAPL Stock Price')
plt.plot(predicted_prices, color='blue', label='Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time (Days in Test Set)')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
print("Displaying plot. Close the plot window to exit the script.")
plt.show()