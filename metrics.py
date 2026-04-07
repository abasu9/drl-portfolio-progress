"""
Evaluation metrics for the progress-report MVO baseline.
"""

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


def compute_daily_returns(portfolio_values):
    return portfolio_values.pct_change().dropna()


def annual_return(portfolio_values):
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    num_days = len(portfolio_values)
    if num_days == 0:
        return 0.0
    return (1 + total_return) ** (TRADING_DAYS_PER_YEAR / num_days) - 1


def sharpe_ratio(portfolio_values, risk_free_rate=RISK_FREE_RATE):
    daily_returns = compute_daily_returns(portfolio_values)
    if daily_returns.std() == 0:
        return 0.0
    excess_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    annual_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if annual_vol == 0:
        return 0.0
    return excess_return / annual_vol


def max_drawdown(portfolio_values):
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values / running_max) - 1
    return abs(drawdown.min())


def summarize_metrics(portfolio_values):
    return {
        "Annual Return": annual_return(portfolio_values),
        "Sharpe": sharpe_ratio(portfolio_values),
        "Max Drawdown": max_drawdown(portfolio_values),
    }