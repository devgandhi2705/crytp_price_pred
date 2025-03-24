# data_fetch.py
import requests
import pandas as pd
import ta
from requests.exceptions import RequestException

def fetch_binance_data(limit=120, interval="1m"):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                        "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Calculate indicators only if we have enough data
        if len(df) >= 20:
            try:
                df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
                df["EMA"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
            except Exception as e:
                print(f"Indicator calculation error: {e}")
                df["RSI"] = None
                df["EMA"] = None

        return df[["timestamp", "open", "high", "low", "close", "volume", "RSI", "EMA"]]
    
    except RequestException as e:
        print(f"API request failed: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "RSI", "EMA"])