"""
Mean-Variance Optimization strategy for the progress-report repo.
This file computes rolling portfolio weights using a max-Sharpe objective.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import LOOKBACK_DAYS, RISK_FREE_RATE


def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_volatility


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    if portfolio_volatility == 0:
        return 1e6
    return -((portfolio_return - risk_free_rate) / portfolio_volatility)


def optimize_weights(window_prices):
    returns = window_prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Handle numerical issues in covariance
    cov_matrix = cov_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    num_assets = len(mean_returns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, RISK_FREE_RATE),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return pd.Series(initial_weights, index=mean_returns.index)

    return pd.Series(result.x, index=mean_returns.index)


def generate_mvo_weights(price_data):
    weights_by_date = {}

    for i in range(LOOKBACK_DAYS, len(price_data)):
        current_date = price_data.index[i]
        window_prices = price_data.iloc[i - LOOKBACK_DAYS:i]
        weights = optimize_weights(window_prices)
        weights_by_date[current_date] = weights

    return pd.DataFrame(weights_by_date).T

