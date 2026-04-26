import os
import pandas as pd
import yfinance as yf

from config import DOW_30_TICKERS, TRAIN_START_DATE, TEST_END_DATE


def _flatten_download(df, ticker):
    """
    Convert yfinance output into one clean OHLCV table for one ticker.
    Handles both normal columns and MultiIndex columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If yfinance returns MultiIndex columns, select this ticker level.
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)

    df = df.copy().reset_index()

    # Normalize column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "date": "date",
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "close",
        "volume": "volume",
    }

    df = df.rename(columns=rename_map)

    needed = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}. Columns found: {list(df.columns)}")

    out = df[needed].copy()
    out["tic"] = ticker
    return out


def download_data(tickers=DOW_30_TICKERS, start=TRAIN_START_DATE, end=TEST_END_DATE):
    rows = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            group_by="column"
        )

        if df.empty:
            print(f"Warning: no data for {ticker}, skipping")
            continue

        try:
            rows.append(_flatten_download(df, ticker))
        except Exception as e:
            print(f"Warning: failed to process {ticker}: {e}")

    if not rows:
        raise ValueError("No stock data downloaded. Check internet/yfinance access.")

    return pd.concat(rows, ignore_index=True)


def clean_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df = df.dropna()
    return df


def save_data(df, path="datasets/stock_data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved data to {path}")


def load_data(path="datasets/stock_data.csv"):
    return pd.read_csv(path, parse_dates=["date"])
