import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Constants
LOOKBACK = 24  # Last 2 hours (5 min intervals)
FUTURE_STEPS = 24  # Predict next 2 hours (5 min intervals)
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def train_lstm_model(train_data):
    """
    Trains the LSTM model on the given training data and saves it.
    """
    global scaler

    # Extract closing prices and scale data
    prices = train_data["close"].values.reshape(-1, 1)
    prices_scaled = scaler.fit_transform(prices)

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)

    # Prepare training dataset
    X, y = [], []
    for i in range(LOOKBACK, len(prices_scaled) - FUTURE_STEPS):
        X.append(prices_scaled[i - LOOKBACK:i])
        y.append(prices_scaled[i:i + FUTURE_STEPS])

    X, y = np.array(X), np.array(y)

    # Define LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(FUTURE_STEPS)
    ])

    model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

def load_trained_model():
    """
    Loads the trained LSTM model from file.
    """
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("No trained model found. Train the model first.")

def predict_next_prices(model, recent_data):
    """
    Predicts the next 2 hours of Bitcoin prices based on recent data.
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler file not found. Train the model first.")

    # Load the scaler
    scaler = joblib.load(SCALER_PATH)

    # Scale recent data
    latest_data = recent_data["close"].values[-LOOKBACK:].reshape(-1, 1)
    latest_scaled = scaler.transform(latest_data).reshape(1, LOOKBACK, 1)

    # Make predictions
    predictions = model.predict(latest_scaled)[0]

    # Reverse Scaling
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions
