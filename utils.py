import datetime
import pandas as pd
import os

# Logging file path
LOG_FILE_PATH = "logs/trade_log.csv"

def format_price(price):
    """Formats price to display as currency."""
    return f"${price:,.2f}"

def format_timestamp(timestamp):
    """Formats timestamps for display."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def log_trade(action, price, quantity):
    """
    Logs trades (Buy/Sell) in a CSV file.
    """
    log_entry = {
        "Timestamp": format_timestamp(datetime.datetime.now()),
        "Action": action,
        "Price": price,
        "Quantity": quantity
    }
    
    log_df = pd.DataFrame([log_entry])

    # Check if log file exists
    if not os.path.exists(LOG_FILE_PATH):
        log_df.to_csv(LOG_FILE_PATH, index=False)
    else:
        log_df.to_csv(LOG_FILE_PATH, mode="a", header=False, index=False)

    print(f"Trade logged: {action} {quantity} BTC at {price}")

def load_trade_logs():
    """
    Loads trade history from the log file.
    """
    if os.path.exists(LOG_FILE_PATH):
        return pd.read_csv(LOG_FILE_PATH)
    else:
        return pd.DataFrame(columns=["Timestamp", "Action", "Price", "Quantity"])
