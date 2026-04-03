"""
Configuration for the progress-report version of the portfolio project.
This file only includes settings needed for the MVO baseline.
"""

START_DATE = "2011-10-01"
END_DATE = "2021-12-31"

INITIAL_CAPITAL = 100000
LOOKBACK_DAYS = 60
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

TICKERS = [
    "XLB",  # Materials
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "VNQ"   # Real Estate proxy
]

PRICE_FILE = "prices.csv"
