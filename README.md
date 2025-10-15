# FE800
FE800 Project Course

# KMRF+MPC Portfolio Optimization - Re-implementation

## Project Overview
This project is a re-implementation of the "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework" model by Gorse et al. The system combines regime prediction with multi-period portfolio optimization to construct long-only portfolios that outperform traditional benchmarks.

## Core Model Components

### KMRF (KAMA+MSR+RF) Regime Prediction Model
- **KAMA+MSR**: Combines Kaufman's Adaptive Moving Average with Markov-Switching Regression
- **Random Forest**: Predicts market regimes (Bullish, Bearish, Other) based on KAMA+MSR labels
- **Regime Classes**: Four base regimes (LV bullish/bearish, HV bullish/bearish) mapped to three prediction classes
- **Trading Strategy**: Contrarian interpretation of signals (Bullish prediction = short signal, Bearish = long signal)

### Model Predictive Control (MPC) Framework
- **Multi-Period Optimization**: Optimizes portfolio weights for horizon H (typically 2 periods)
- **Mean-Variance Approach**: Balances expected returns against risk and transaction costs
- **Long-Only Constraint**: Restricts to positive weights only (wi ≥ 0.01 minimum allocation)
- **Utility Function**: Maximizes returns - risk penalty - trading penalty

### Kalman Filter Enhancement
- **Return Estimation Boost**: Improves accuracy of KMRF-derived return estimates
- **Hidden State Estimation**: Captures difference between actual and estimated returns
- **Multi-Horizon Support**: Extends single-period KMRF predictions to multi-period horizon

## Asset Universe

### Equity Assets (14)
- S&P 500, NASDAQ 100, DAX, FTSE 100, CAC40, Nikkei 225, KOSPI
- S&P/ASX 200, FTSE MIB, Swiss Market, S&P/TSX Composite, TWSE
- MSCI China, NIFTY 50

### Commodity Assets (12)
- Energy: Brent Crude Oil, Natural Gas
- Metals: Gold, Copper, Aluminium, Nickel
- Agriculture: Corn, Soybeans, Wheat, Coffee, Sugar, Live Cattle

### Cash Component
- US 3-Month Treasury Bills (risk-free asset)

## Feature Engineering

### Technical Features
- **TA Library**: Standard technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **tsfresh Package**: Time-series feature extraction (statistical, frequency domain, entropy measures)
- **KAMA Indicators**: Kaufman's Adaptive Moving Average calculations
- **Exponential Weighted Moving Averages**: 10-day EMA for returns and volatility

### Fundamental & Macroeconomic Features
- Economic indicators and market fundamentals
- Cross-asset correlations and volatility measures
- Market regime indicators derived from MSR models

### Feature Selection
- **BorutaShap**: Automated feature selection using shapley values
- **Time-Series Adaptation**: Modified for temporal data with causality preservation
- **Purged Group Time-Series Split (PGTS)**: Prevents data leakage in cross-validation

## Data Pipeline

### Training/Validation Split
- **Training Period**: Data up to 30/03/2018 for KMRF model training
- **Validation Period**: 02/04/2018 - 12/03/2020 for MPC hyperparameter optimization
- **Test Period**: 27/03/2020 - 29/04/2022 for final evaluation
- **Gap Period**: 15 days between validation and test to prevent leakage

### Return Estimation Process
1. Calculate 10-day EMA of historical returns for each asset
2. Transform KMRF regime probabilities into return estimates using contrarian logic
3. Apply Kalman filter to boost accuracy of return estimates
4. Generate multi-horizon estimates using lagged values

### Covariance Matrix Estimation
- **Rolling Window**: 504 trading days (2 years) rolling covariance
- **Lagged Estimation**: Uses historical covariance to predict future risk
- **Asset Risk Integration**: Incorporates regime-based volatility expectations

## Optimization Framework

### MPC Utility Function
```
Maximize: Σ[Expected Returns - γ_sigma * Risk - γ_trade * Trading Costs]
Subject to: 
- wi ≥ 0.01 (minimum 1% allocation)
- Σwi = 1 (fully invested)
- Long-only positions
```

### Key Parameters
- **γ_sigma**: Risk aversion parameter (controls risk-return tradeoff)
- **γ_trade**: Trading penalty (controls portfolio turnover)
- **H**: Investment horizon (typically 2 periods)
- **Bid-Ask Spread**: 20 basis points (0.002) including trading fees

### Transaction Cost Model
- **Linear Component**: Proportional to trade size (bid-ask spread)
- **Market Impact**: Square-root function of trade size relative to volume
- **Volume Adjustment**: Scaled by asset liquidity (EMA of dollar volume)

## Performance Evaluation

### Primary Metrics
- **Annualized Returns**: Risk-adjusted portfolio returns
- **Sortino Ratio**: Downside risk-adjusted performance (optimization target)
- **Sharpe Ratio**: Total risk-adjusted performance
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Information Ratio**: Risk-adjusted excess returns vs benchmarks

### Benchmark Comparisons
- **Buy-and-Hold**: Passive equal-weight portfolio held constant
- **1/N Portfolio**: Daily rebalanced equal-weight portfolio

### Risk Management
- **Dynamic Risk Adjustment**: γ_sigma parameter tuning for different risk tolerances
- **Volatility Monitoring**: Real-time tracking of portfolio volatility
- **Drawdown Control**: Maximum allocation limits and diversification requirements

## Implementation Architecture

### Core Modules
1. **Data Module**: Asset price collection, feature engineering, data preprocessing
2. **KMRF Module**: Regime detection (KAMA+MSR) and prediction (Random Forest)
3. **Kalman Module**: Return estimate enhancement and multi-horizon projection
4. **MPC Module**: Portfolio optimization using CVXPortfolio framework
5. **Evaluation Module**: Performance metrics, backtesting, and visualization

### Key Dependencies
- **CVXPortfolio**: Convex optimization for portfolio construction
- **Optuna**: Hyperparameter optimization for γ_sigma and γ_trade
- **scikit-learn**: Random Forest implementation and model evaluation
- **TA-Lib/ta**: Technical analysis indicators
- **tsfresh**: Automated time-series feature extraction
- **BorutaShap**: Feature selection with shapley value analysis

### Hyperparameter Optimization
- **Objective**: Maximize Sortino ratio on validation set
- **Search Spaces**: 
  - γ_sigma: [0.01, 1000] (log scale)
  - γ_trade: [0.0001, 25.0] (log scale)
- **Method**: Bayesian optimization via Optuna
- **Cross-Validation**: Purged Group Time-Series Split

## Research Objectives

### Primary Goals
1. **Regime Prediction Accuracy**: Improve ex-ante prediction of market regimes
2. **Portfolio Performance**: Achieve superior risk-adjusted returns vs benchmarks
3. **Long-Only Construction**: Eliminate short positions while maintaining performance
4. **Transaction Cost Reality**: Incorporate realistic trading costs and market impact

### Academic Contributions
1. **KMRF Enhancement**: Integration of regime prediction with multi-period optimization
2. **Kalman Filter Innovation**: Novel application for return estimate improvement
3. **Contrarian Strategy Validation**: Empirical evidence for contrarian regime interpretation
4. **Long-Only Optimization**: Practical implementation for institutional constraints

### Expected Outcomes
- **Outperformance**: Superior Sortino and Information ratios vs benchmarks
- **Risk Management**: Lower maximum drawdowns with controlled volatility
- **Scalability**: Framework applicable to different asset classes and time periods
- **Robustness**: Consistent performance across different market regimes

## Project Structure Guidance

This README should help your AI agent understand that the codebase implements a sophisticated quantitative finance system combining machine learning (Random Forest regime prediction), signal processing (Kalman filtering), and convex optimization (MPC portfolio construction) to create a regime-aware portfolio management system. The implementation focuses on practical constraints (long-only, realistic costs) while achieving academic-quality research results through rigorous backtesting and performance evaluation.

The system is designed to predict market regime changes and dynamically adjust portfolio allocations to capitalize on these shifts while maintaining acceptable risk levels for institutional investors. The contrarian trading strategy and multi-period optimization framework represent key innovations over traditional portfolio construction methods.