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
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs(TICKERS[0], axis=1, level=1, drop_level=False).copy()
    else:
        prices = data.copy()

    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    prices = prices.ffill()

    missing_counts = prices.isna().sum()
    print("Missing values per ticker before final cleanup:")
    print(missing_counts)

    prices = prices.dropna()

    return prices


if __name__ == "__main__":
    prices = download_prices()
    print(prices.head())
    print(prices.tail())
