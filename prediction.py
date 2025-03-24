# prediction.py
import numpy as np
from model import get_model, SCALER, LOOKBACK, FUTURE_STEPS
import pandas as pd

model = get_model()

def predict_next_prices(df):
    """Predict future prices using the trained LSTM model"""
    # Validate input data
    if len(df) < LOOKBACK:
        raise ValueError(f"Insufficient data: Need at least {LOOKBACK} points, got {len(df)}")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    try:
        # Prepare the most recent data for prediction
        latest_data = df["close"].values[-LOOKBACK:].reshape(-1, 1)
        
        # Scale the data using the pre-fitted scaler
        latest_scaled = SCALER.transform(latest_data)
        
        # Reshape for LSTM input (batch_size, timesteps, features)
        latest_scaled = latest_scaled.reshape(1, LOOKBACK, 1)
        
        # Make prediction
        predicted_scaled = model.predict(latest_scaled, verbose=0)[0]
        
        # Inverse transform to get actual price values
        predictions = SCALER.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
        
        # Generate future timestamps for the predictions
        last_timestamp = df["timestamp"].iloc[-1]
        freq = pd.infer_freq(df["timestamp"].tail(5)) or "5min"  # Default to 5min if can't infer
        future_timestamps = pd.date_range(
            start=last_timestamp,
            periods=FUTURE_STEPS + 1,
            freq=freq
        )[1:]  # Exclude the first which is the last known timestamp
        
        return predictions, future_timestamps
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise ValueError("Failed to generate predictions") from e