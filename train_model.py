import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("Loading processed data...")

try: 
  X_train = np.load('X_train.npy')
  y_train = np.load('y_train.npy')
  X_test = np.load('X_test.npy')
  y_test = np.load('y_test.npy')
except FileNotFoundError:
    print("Error: Processed data files not found. Please run process_data.py first.")
    exit()

print("Data loaded successfully.")

print("Building the LSTM model...")

input_shape = (X_train.shape[1], 1)

model = Sequential([
   LSTM(50, return_sequences=True, input_shape=input_shape),
   Dropout(0.2),

   LSTM(50, return_sequences=False),
   Dropout(0.2),

   Dense(units = 25),

   Dense(units = 1) 
])

print("Compiling the model...")

# Create an Adam optimizer with a lower, more stable learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.summary()

print("Training the model...")

history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

print("Model training complete.")

print("Saving the trained model...")
model.save('stock_forecaster_model.keras')
print("Model saved successfully.")