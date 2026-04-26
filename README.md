# Deep Reinforcement Learning for Portfolio Allocation

This repository contains our final course project code for portfolio allocation using **Deep Reinforcement Learning (DRL)** and **Mean-Variance Optimization (MVO)**.

The project is inspired by the paper **“Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization.”** The main idea is to check whether RL based portfolio allocation methods can perform competitively against traditional finance baselines.

This is a **course-scale implementation**, not a full exact replication of the original paper. Some parts were adapted because of time, compute, and project constraints.

## Project Idea

Portfolio allocation is about deciding how much capital should be assigned to each stock at a given time. Classical finance methods, like **Mean-Variance Optimization**, try to balance return and risk using historical data.

In this project, we frame portfolio allocation as a reinforcement learning problem. The agent observes market features and portfolio state, then learns how to adjust portfolio weights over time.

We compare RL models with simple and classical baselines to see how useful they are in this setting.

## Methods Compared

| Method | Description |
|---|---|
| **A2C** | Advantage Actor-Critic reinforcement learning model |
| **PPO** | Proximal Policy Optimization reinforcement learning model |
| **DDPG** | Deep Deterministic Policy Gradient reinforcement learning model |
| **Equal Weight** | Assigns the same weight to every stock |
| **Max Sharpe MVO** | Mean-Variance Optimization baseline using maximum Sharpe ratio |

## Data

The project uses historical stock price data downloaded using **yfinance**.

The raw stock data includes:

- date
- open price
- high price
- low price
- close price
- volume
- ticker symbol

The raw downloaded data is included in:

```text
datasets/stock_data.csv
```

The generated feature file is not pushed to GitHub:

```text
datasets/features_data.csv
```

This file is around **792 MB**, which is above GitHub’s normal 100 MB file limit. It is automatically regenerated when running the project, so it does not need to be stored in the repository.

## Feature Engineering

For each stock, we compute several technical indicators:

- MACD
- RSI
- CCI
- ADX
- SMA
- EMA
- Bollinger Bands
- ATR
- OBV

We also compute rolling covariance matrices. This is useful because portfolio allocation depends not only on individual stock behavior, but also on how stocks move together.

## Portfolio Environment

A custom **Gymnasium** environment is used for the portfolio allocation task.

At each step:

1. The agent observes the current market and portfolio state.
2. The agent chooses portfolio allocation weights.
3. The environment updates portfolio value based on stock returns.
4. The agent receives a reward based on portfolio performance.

This setup allows A2C, PPO, and DDPG to learn allocation policies from historical market behavior.

## Final Results

The final experiment was run locally. The table below comes from:

```text
results/performance_summary.csv
```

| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown | Calmar Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2C | 44.56% | 20.32% | 25.75% | 0.8478 | 0.9107 | 30.83% | 0.6592 |
| PPO | 50.11% | 22.62% | 25.53% | 0.9273 | 1.0143 | 29.94% | 0.7554 |
| DDPG | 51.67% | 23.26% | 25.51% | 0.9482 | 1.0420 | 30.48% | 0.7630 |
| Equal Weight | 50.08% | 22.56% | 25.53% | 0.9254 | 1.0118 | 29.98% | 0.7525 |
| Max Sharpe MVO | 64.91% | 28.48% | 33.75% | 0.9190 | 1.0935 | 33.20% | 0.8579 |

## Observations

- **MVO gave the highest cumulative return and annualized return** in this experiment.
- **DDPG had the highest Sharpe ratio among the RL models**.
- **PPO performed close to Equal Weight**, which means it learned a reasonable allocation policy.
- **Equal Weight was a strong baseline**, showing that simple diversification can still work well.
- The RL methods were competitive, but they did not clearly beat MVO in this setup.

## Results and Model Files

The final result files are saved in:

```text
results/
```

Important output files:

```text
results/performance_summary.csv
results/cumulative_returns.png
results/drawdowns.png
results/metrics_comparison.png
results/returns_distribution.png
```

The trained models are saved in:

```text
trained_models/
```

Included trained models:

```text
trained_models/a2c.zip
trained_models/ppo.zip
trained_models/ddpg.zip
```

## Project Structure

```text
.
├── main.py
├── config.py
├── data_downloader.py
├── feature_engineering.py
├── env_portfolio.py
├── models.py
├── benchmarks.py
├── backtest.py
├── visualize.py
├── notebook.ipynb
├── requirements.txt
├── datasets/
│   └── stock_data.csv
├── results/
│   ├── performance_summary.csv
│   ├── cumulative_returns.png
│   ├── drawdowns.png
│   ├── metrics_comparison.png
│   └── returns_distribution.png
└── trained_models/
    ├── a2c.zip
    ├── ppo.zip
    └── ddpg.zip
```

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/abasu9/drl-portfolio-progress.git
cd drl-portfolio-progress
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install ta gymnasium stable-baselines3 pyportfolioopt matplotlib seaborn scikit-learn torch tensorboard
```

Run the full pipeline:

```bash
python main.py
```

This will:

1. Load or download stock data
2. Generate technical indicators and covariance features
3. Train A2C, PPO, and DDPG models
4. Run Equal Weight and MVO baselines
5. Compute performance metrics
6. Save plots and trained models

## Important Note About Large Files

The following file is intentionally not included in GitHub:

```text
datasets/features_data.csv
```

It is around **792 MB**, which is too large for normal GitHub storage. This file is generated by the code, so if it is missing, just run:

```bash
python main.py
```

The project will recreate it automatically.

## Replication Status

This project is a **course-scale replication and extension**.

### Implemented

- Portfolio allocation environment
- A2C, PPO, and DDPG training
- Equal Weight benchmark
- Max Sharpe MVO benchmark
- Technical indicators
- Rolling covariance features
- Backtesting and evaluation
- Result plots
- Saved trained models

### Adapted

- We used a smaller training setup compared to the original paper.
- The stock universe and some implementation details were adapted for this course project.
- The generated feature dataset is not stored in GitHub because of file size limits.

### Not Fully Replicated

- Full-scale training budget from the original paper
- Exact paper data setup
- Large multi-seed experiments

## Main Takeaway

This project shows that DRL models can learn useful portfolio allocation behavior, but classical MVO is still a very strong benchmark. Simple baselines like Equal Weight are also important because they can perform surprisingly well.

The main lesson is that portfolio methods should not be judged only by return. Risk-adjusted metrics like Sharpe ratio, Sortino ratio, drawdown, and volatility are also important.
