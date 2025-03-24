import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from data_fetch import fetch_crypto_data
from model import predict_next_prices
from trading_bot import execute_trade
from utils import format_price
import os

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App Layout
app.layout = html.Div(
    [
        # Title
        html.H1("Crypto Trading Bot 🚀", className="title"),
        html.Hr(),

        # Live Price & Trading Info
        html.Div(
            [
                html.H3("📈 Live Price:", className="info-title"),
                html.H2(id="live-price", className="price"),
                html.P(id="trade-status", className="trade-status"),
            ],
            className="live-price-container",
        ),

        # Candlestick Chart
        dcc.Graph(id="candlestick-chart"),

        # Prediction Line Chart
        dcc.Graph(id="price-prediction-chart"),

        # Interval Updates (Every Minute)
        dcc.Interval(id="interval-update", interval=60000, n_intervals=0),
    ],
    className="container",
)

# Callbacks for Live Updates
@app.callback(
    [
        Output("live-price", "children"),
        Output("trade-status", "children"),
        Output("candlestick-chart", "figure"),
        Output("price-prediction-chart", "figure"),
    ],
    Input("interval-update", "n_intervals"),
)
def update_dashboard(n_intervals):
    # Fetch latest crypto data
    df = fetch_crypto_data(symbol="BTCUSDT", limit=120)
    latest_price = df["close"].iloc[-1]
    
    # Predict next 2 hours
    predictions = predict_next_prices(df)
    
    # Execute Trading Logic
    trade_message = execute_trade(latest_price, predictions)

    # Create Candlestick Chart
    candlestick_fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    candlestick_fig.update_layout(title="Bitcoin Candlestick Chart", template="plotly_dark")

    # Create Price Prediction Line Chart
    price_pred_fig = go.Figure()
    price_pred_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="Actual Price"))
    future_timestamps = pd.date_range(df["timestamp"].iloc[-1], periods=25, freq="5min")[1:]
    price_pred_fig.add_trace(go.Scatter(x=future_timestamps, y=predictions, mode="lines", name="Predicted Price", line=dict(color="red", dash="dot")))
    price_pred_fig.update_layout(title="Predicted Bitcoin Prices", template="plotly_dark")

    return format_price(latest_price), trade_message, candlestick_fig, price_pred_fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
