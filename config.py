# config.py

# -----------------------------
# Data settings
# -----------------------------

DOW_30_TICKERS = [
    "AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "UNH",
    "HD", "DIS", "MA", "NVDA", "PYPL", "BAC", "VZ",
    "ADBE", "CMCSA", "NFLX", "KO", "PFE", "PEP",
    "T", "INTC", "CSCO", "ABT", "CRM", "XOM",
    "CVX", "MRK", "NKE"
]

TICKERS = DOW_30_TICKERS

TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2019-12-31"
TEST_START_DATE = "2020-01-01"
TEST_END_DATE = "2021-12-31"

START_DATE = TRAIN_START_DATE
END_DATE = TEST_END_DATE

PRICE_FILE = "prices.csv"

# -----------------------------
# Portfolio settings
# -----------------------------

INITIAL_AMOUNT = 100000
INITIAL_CAPITAL = INITIAL_AMOUNT

LOOKBACK_WINDOW = 252
LOOKBACK_DAYS = 60

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0
TRANSACTION_COST_PCT = 0.001
REWARD_SCALING = 1e-4

# -----------------------------
# Technical indicators
# -----------------------------

TECHNICAL_INDICATORS = [
    "macd",
    "rsi_14",
    "cci_14",
    "adx_14",
    "sma_20",
    "ema_20",
    "bbands_upper",
    "bbands_lower",
    "atr_14",
    "obv",
]

# -----------------------------
# DRL model parameters
# -----------------------------

TIMESTEPS = {
    "a2c": 5000,
    "ppo": 5000,
    "ddpg": 5000,
}

A2C_PARAMS = {
    "n_steps": 5,
    "learning_rate": 0.0007,
    "gamma": 0.99,
}

PPO_PARAMS = {
    "n_steps": 2048,
    "batch_size": 64,
    "learning_rate": 0.00025,
    "gamma": 0.99,
}

DDPG_PARAMS = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "buffer_size": 10000,
}
