"""
Download adjusted close prices for the selected tickers
and save them to a CSV file for the MVO backtest.
"""

import yfinance as yf
import pandas as pd

from config import TICKERS, START_DATE, END_DATE, PRICE_FILE


def download_prices():
    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()

    prices.to_csv(PRICE_FILE)
    print(f"Saved price data to {PRICE_FILE}")
    print(prices.head())
    print(prices.tail())


if __name__ == "__main__":
    download_prices()
