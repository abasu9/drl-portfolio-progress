"""
Run yearly MVO backtests for the progress-report repo.
"""

import pandas as pd
import numpy as np

from config import PRICE_FILE, INITIAL_CAPITAL, LOOKBACK_DAYS
from mvo_strategy import generate_mvo_weights
from metrics import summarize_metrics


def load_prices():
    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    return prices.sort_index()


def simulate_portfolio(test_prices, weights_df, initial_capital=INITIAL_CAPITAL):
    portfolio_values = []
    portfolio_value = initial_capital

    common_dates = [d for d in test_prices.index if d in weights_df.index]
    prev_date = None

    for current_date in common_dates:
        if prev_date is None:
            portfolio_values.append((current_date, portfolio_value))
            prev_date = current_date
            continue

        prev_prices = test_prices.loc[prev_date]
        curr_prices = test_prices.loc[current_date]
        weights = weights_df.loc[current_date]

        asset_returns = (curr_prices / prev_prices) - 1
        daily_portfolio_return = np.sum(weights * asset_returns)

        portfolio_value = portfolio_value * (1 + daily_portfolio_return)
        portfolio_values.append((current_date, portfolio_value))
        prev_date = current_date

    return pd.Series(
        [v for _, v in portfolio_values],
        index=[d for d, _ in portfolio_values]
    )


def run_yearly_backtests():
    prices = load_prices()
    results = []

    for year in range(2012, 2022):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        history = prices[prices.index < test_start]
        test_prices = prices[(prices.index >= test_start) & (prices.index <= test_end)]

        if len(history) < LOOKBACK_DAYS or len(test_prices) < 2:
            continue

        combined = pd.concat([history.tail(LOOKBACK_DAYS), test_prices])
        weights_df = generate_mvo_weights(combined)

        weights_df = weights_df[weights_df.index.isin(test_prices.index)]
        portfolio_values = simulate_portfolio(test_prices, weights_df)

        metrics = summarize_metrics(portfolio_values)

        results.append({
            "Year": year,
            "Ann Return": metrics["Annual Return"],
            "Sharpe": metrics["Sharpe"],
            "Max DD": metrics["Max Drawdown"],
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    if not results_df.empty:
        print("\nAggregate (2012-2021):")
        print(f"Ann Return: {results_df['Ann Return'].mean():.4f}")
        print(f"Sharpe: {results_df['Sharpe'].mean():.4f}")
        print(f"Max DD: {results_df['Max DD'].max():.4f}")


if __name__ == "__main__":
    run_yearly_backtests()