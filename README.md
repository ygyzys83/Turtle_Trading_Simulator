# Turtle Trading Simulator

A Streamlit app for exploring a classic turtle-style trend-following strategy with either synthetic price data or live market data from Yahoo Finance.

## Features

- Simulates breakout entries using an N-bar high.
- Uses an SMA trend filter before entering trades.
- Calculates ATR-based stops and position sizing.
- Exits on stop loss or an N-bar low.
- Shows performance metrics, current signal state, live rule values, and an interactive Plotly chart.
- Supports synthetic data runs and optional stock data through `yfinance`.
- Includes a selectable trade log that highlights entries and exits on the chart.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

Start the app with:

```powershell
streamlit run turtle_trading.py
```

The app will open in your browser. Use the sidebar controls to switch between synthetic data and stock data, adjust the strategy windows, risk settings, and ATR stop multiplier.

## Notes

Yahoo Finance intraday data can be delayed and may have period limits depending on the selected interval. This simulator is for research and education only, not financial advice.