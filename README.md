# KMRF Portfolio Optimization Methodology
## Complete Mathematical Framework

Based on:
- Pomorski & Gorse (2022) - "Improving on the Markov-Switching Regression Model by the Use of an Adaptive Moving Average"
- Pomorski & Gorse (2023a) - "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes"
- Pomorski & Gorse (2023b) - "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework"

---

## Table of Contents
1. [Overview](#overview)
2. [Stage 1: KAMA+MSR Regime Detection](#stage-1-kamamsr-regime-detection)
3. [Stage 2: KMRF Regime Prediction](#stage-2-kmrf-regime-prediction)
4. [Stage 3: Analytical Portfolio Inputs](#stage-3-analytical-portfolio-inputs)
5. [Stage 4: Portfolio Optimization](#stage-4-portfolio-optimization)
6. [Stage 5: Backtesting Framework](#stage-5-backtesting-framework)
7. [Implementation Reference](#implementation-reference)

---

## Overview

This document describes a complete pipeline for regime-based portfolio optimization, combining:
1. **KAMA+MSR**: Ex-post regime classification (historical labeling)
2. **KMRF**: Ex-ante regime prediction (forward-looking probabilities)
3. **Analytical Inputs**: Closed-form computation of expected returns and covariance
4. **Portfolio Optimization**: Constrained weight optimization
5. **Backtesting**: Historical performance evaluation

### Regime Framework

The market is modeled with **4 distinct regimes** based on volatility and trend:

| Regime | Label | Description |
|--------|-------|-------------|
| 0 | LV_Bull | Low Volatility, Bullish Trend |
| 1 | LV_Bear | Low Volatility, Bearish Trend |
| 2 | HV_Bull | High Volatility, Bullish Trend |
| 3 | HV_Bear | High Volatility, Bearish Trend |

---

## Stage 1: KAMA+MSR Regime Detection

**Purpose:** Classify historical data into regimes (ex-post labeling)

**File:** `kama_msr.py`

**Reference:** Pomorski & Gorse (2022) - "Improving on the Markov-Switching Regression Model by the Use of an Adaptive Moving Average"

### 1.1 KAMA (Kaufman's Adaptive Moving Average)

KAMA (Kaufman, 1995) adapts its smoothing based on market efficiency, distinguishing trending vs. noisy periods. It is used for **trend detection** (bullish vs. bearish).

**Efficiency Ratio (ER):**
$$ER_t = \frac{M_t}{V_t}$$

where:
- **Momentum:** $M_t = P_t - P_{t-n}$ (change in closing price over n-period)
- **Volatility:** $V_t = \sum_{i=1}^{n} |P_{t-i+1} - P_{t-i}|$ (sum of absolute daily price changes)
- $0 \leq ER \leq 1$

**Interpretation:**
- $ER \to 1$: Strong, clearly defined trend (directional movement ≈ total movement)
- $ER \to 0$: Consolidating/directionless market (noisy, range-bound)

**KAMA Update Equation:**
$$KAMA_t = KAMA_{t-1} + C_t (P_t - KAMA_{t-1})$$

**Scaled Smoothing Coefficient:**
$$C_t = [ER_t (k_s - k_l) + k_l]^2$$

**Smoothing Constants:**
$$k_s = \frac{2}{n_s + 1}, \quad k_l = \frac{2}{n_l + 1}$$

where $n_s$ and $n_l$ are shorter and longer time windows respectively (default: $n_s = 2$, $n_l = 30$).

**Filter for Trade Signals:**
$$f_t = \gamma \cdot \sigma(KAMA_t)$$

where:
$$\sigma(KAMA_t) = \text{rolling std of } (KAMA_t - KAMA_{t-1}) \text{ over } n \text{ periods}$$

and $\gamma$ is a control parameter (optimized).

**Trading Signal Rules:**
- **Bullish (buy):** $KAMA_t - KAMA_{low,n} > f_t$ (KAMA advances above its n-day low by more than the filter)
- **Bearish (sell):** $KAMA_{high,n} - KAMA_t > f_t$ (KAMA descends below its n-day high by more than the filter)

where $KAMA_{low,n}$ and $KAMA_{high,n}$ are the rolling minimum and maximum of KAMA over the prior n days.

### 1.2 MSR (Markov-Switching Regression)

MSR (Krolzig, 1997) is a **Markov-Switching Regression** model (not just a regime model) that models volatility states with state-dependent parameters in a regression framework.

**Observation Equation:**
$$\ln r_t = \mu_{S_t} + \beta_{S_t} \cdot \ln r_{t-1} + \sigma_{S_t} \cdot \epsilon_t, \quad \epsilon_t \sim N(0,1)$$

where:
- $\ln r_t$ = log return at time $t$
- $S_t \in \{0, 1\}$ = latent volatility state (0 = low volatility, 1 = high volatility)
- $\mu_{S_t}$ = state-dependent intercept
- $\beta_{S_t}$ = state-dependent coefficient of lagged log returns
- $\sigma_{S_t}$ = state-dependent volatility

**State Equation (Transition Probabilities):**
$$P = \begin{pmatrix} p & 1-p \\ 1-q & q \end{pmatrix}$$

where:
- $p = P(S_t = 0 | S_{t-1} = 0)$ - probability of staying in low volatility
- $q = P(S_t = 1 | S_{t-1} = 1)$ - probability of staying in high volatility

**Full Parameter Vector:**
$$\theta = (p, q, \mu_0, \mu_1, \beta_0, \beta_1, \sigma_0, \sigma_1, \delta)$$

where $\delta = P(S_0 = 0)$ is the initial state distribution parameter.

**Filtered Probabilities:**
$$p_{it} = P(S_t = i | \ln r_{1:t}; \hat{\theta})$$

- $p_{0t}$: probability of low volatility regime at time $t$
- $p_{1t}$: probability of high volatility regime at time $t$

**Estimation via Gibbs Sampling (MCMC):**

1. Initialize regime labels and parameters
2. For each iteration:
   - Sample regime means: $\mu_s | r, S \sim \mathcal{N}(\bar{r}_s, \sigma_s^2/n_s)$
   - Sample regime betas: $\beta_s | r, S$
   - Sample regime variances: $\sigma_s^2 | r, S \sim \text{InvGamma}$
   - Sample transition probabilities: $(p, q) | S \sim \text{Beta}$
   - Sample regime sequence: Forward-filtering backward-sampling (FFBS)
3. Discard burn-in (default: 200 iterations)
4. Use posterior estimates for regime probabilities

**Configuration:**
- Gibbs iterations: 1000
- Burn-in: 200
- Random seed: Configurable for reproducibility

### 1.3 Combined KAMA+MSR Labeling

The 4-regime labels are created by combining MSR volatility states (using 50% probability cutoff) with KAMA trend signals:

| MSR Volatility | KAMA Trend | Final Regime |
|----------------|------------|--------------|
| $P(S_t = 0) > 50\%$ (LV) | Bullish | 0 (LV_Bull) |
| $P(S_t = 0) > 50\%$ (LV) | Bearish | 1 (LV_Bear) |
| $P(S_t = 1) > 50\%$ (HV) | Bullish | 2 (HV_Bull) |
| $P(S_t = 1) > 50\%$ (HV) | Bearish | 3 (HV_Bear) |

**Parameter Optimization:**
The KAMA parameters $(n, n_s, n_l, \gamma)$ are optimized to minimize misclassification score using K-Means clustering on slope-volatility feature space.

**Output:** Time series of regime labels for each asset

---

## Stage 2: KMRF Regime Prediction

**Purpose:** Predict future regime probabilities (ex-ante forecasting)

**File:** `kmrf.py`

**Reference:** Pomorski & Gorse (2023a) - "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes"

**Note:** This implementation uses **XGBoost** (`XGBClassifier`) instead of Random Forest as in the original paper, for improved performance.

### 2.1 Feature Engineering

Features are computed with **1-day lag** to prevent look-ahead bias:

**Price-Derived Features** (all lagged):
- Log returns: $\log(P_t/P_{t-1})$
- Volatility: Rolling std of returns (5, 10, 21, 63 days)
- Momentum: Returns over various windows
- Technical indicators: RSI, MACD, Bollinger Bands, etc.

**Macro Features** (optional):
- VIX, credit spreads, yield curve
- Economic indicators from FRED API

**Feature Selection (BorutaPy/BorutaShap):**
- Uses tree-based model to identify shadow features
- Compares feature importance vs. randomized versions
- Retains only statistically significant features
- 100 max iterations with time-series cross-validation

### 2.2 XGBoost Classifier

**Model:** `xgboost.XGBClassifier`

**Target Variable:** 4-regime labels from KAMA+MSR

**Cross-Validation:** Purged Group Time-Series Split (PGTS) from Lopez de Prado (2018)
- Prevents temporal leakage between train and validation sets
- 15-day gap (purge) between folds

**Training:**
- Hyperparameter tuning on validation set
- Early stopping to prevent overfitting

**Label Adaptation (Optional - 4 regimes → 3 classes):**
Based on contrarian trading logic:
- **Bullish (1):** LV bullish + extension to peak of next HV bullish regime
- **Bearish (-1):** HV bearish + extension to trough of next LV bearish regime
- **Other (0):** Remaining parts of HV bullish and LV bearish regimes

**Output:** Probability distribution over regimes for each forecast horizon

### 2.3 Multi-Horizon Predictions

KMRF generates predictions for multiple horizons (e.g., 1, 5, 10, 21 days ahead):

$$\hat{\pi}_{i,m}^{(h)} = P(S_{t+h}^i = m | \mathcal{F}_t)$$

where:
- $i$ = asset
- $m$ = regime (0-3)
- $h$ = forecast horizon
- $\mathcal{F}_t$ = information available at time $t$

Separate XGBoost models are trained for each prediction horizon.

---

## Stage 3: Analytical Portfolio Inputs

**Purpose:** Compute expected returns (μ) and covariance matrix (Σ)

**File:** `ANALYTICAL_INPUTS.py`

### 3.1 Universe Regime Concept

Instead of tracking $N^4$ regime combinations for $N$ assets, define a single **Universe Regime**:

**Democracy Method** (average across portfolio):
$$\bar{\pi}_m = \frac{1}{N} \sum_{i=1}^{N} \pi_{i,m}$$

**Market Method** (single proxy, e.g., S&P 500):
$$\bar{\pi}_m = \pi_{mkt,m}$$

**Democracy Tiebreaker:** When regimes are tied in majority vote:
- Priority: HV_Bear (3) > HV_Bull (2) > LV_Bear (1) > LV_Bull (0)
- Rationale: High-volatility regimes are more critical for risk management

### 3.2 Expected Returns

Each asset's expected return uses **individual** regime probabilities:

$$\mu_i = \sum_{m=0}^{3} \pi_{i,m} \cdot \mathbb{E}[r_i | \text{regime}=m]$$

where:
- $\pi_{i,m}$ = asset $i$'s forward probability for regime $m$ (from KMRF)
- $\mathbb{E}[r_i | \text{regime}=m]$ = historical mean return in regime $m$

**Annualization:**
$$\mu_i^{ann} = \mu_i^{daily} \times 252$$

### 3.3 Expected Volatilities

$$\sigma_i^2 = \sum_{m=0}^{3} \pi_{i,m} \cdot \text{Var}[r_i | \text{regime}=m]$$

**Annualization:**
$$\sigma_i^{ann} = \sigma_i^{daily} \times \sqrt{252}$$

### 3.4 Blended Correlation Matrix

Correlations blend using **Universe** probabilities:

$$\bar{\rho}_{ij} = \sum_{m=0}^{3} \bar{\pi}_m \cdot \rho_{ij,m}$$

where $\rho_{ij,m}$ is the historical correlation between assets $i$ and $j$ when the market was in regime $m$.

**Rationale:** Correlations represent co-movement driven by overall market environment.

### 3.5 Law of Total Covariance

The final covariance matrix combines two components:

$$\Sigma_{ij} = \underbrace{\sigma_i \cdot \sigma_j \cdot \bar{\rho}_{ij}}_{\text{Within-regime}} + \underbrace{\text{Cov}(\mathbb{E}[r_i|m], \mathbb{E}[r_j|m])}_{\text{Between-regime}}$$

**Covariance of Means (between-regime component):**

$$\text{Cov}(\mathbb{E}[r_i|m], \mathbb{E}[r_j|m]) = \sum_{m=0}^{3} \bar{\pi}_m \cdot (\mu_{i,m} - \mu_i)(\mu_{j,m} - \mu_j)$$

This captures additional covariance from regime transitions—assets that move together during regime switches.

### 3.6 Pipeline Summary

```
Phase 1: Load Data and Compute Historical Statistics
├── load_model_info_objects()      # Load KAMA_MSR and KMRF models
├── compute_regime_statistics()    # μ, σ² for each asset×regime
└── estimate_regime_correlations() # ρ matrix for each market regime

Phase 2: Get Forward-Looking Probabilities  
├── get_forward_regime_probs()     # π_i,m from KMRF predictions
└── compute_universe_probs()       # π̄_m (averaged or market-based)

Phase 3: Compute Expected Returns and Volatilities
├── compute_expected_returns()     # μ = Σ π_i,m × μ_i,m
└── compute_expected_volatilities() # σ = √(Σ π_i,m × σ²_i,m)

Phase 4: Compute Covariance Matrix
├── compute_blended_correlations() # ρ̄ = Σ π̄_m × ρ_m
├── compute_covariance_of_means()  # Between-regime component
└── compute_covariance_matrix()    # Σ = σσ'ρ̄ + Cov(means)
```

---

## Stage 4: Portfolio Optimization

**Purpose:** Compute optimal portfolio weights given μ and Σ

**File:** `PORTFOLIO_OPTIMIZER.py`

### 4.1 Optimization Objectives

| Objective | Formulation |
|-----------|-------------|
| `max_sharpe` | $\max_w \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}$ |
| `min_variance` | $\min_w w^\top \Sigma w$ |
| `mean_variance` | $\max_w w^\top \mu - \gamma \cdot w^\top \Sigma w$ |
| `risk_parity` | $\min_w \sum_i \left(RC_i - \frac{1}{N}\right)^2$ |

where:
- $w$ = weight vector
- $\mu$ = expected returns
- $\Sigma$ = covariance matrix
- $r_f$ = risk-free rate
- $\gamma$ = risk aversion parameter
- $RC_i$ = risk contribution of asset $i$

### 4.2 Constraint Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `allow_short_selling` | Boolean toggle for shorts | `False` |
| `gross_exposure_limit` | Max $\sum_i |w_i|$ | `1.0` |
| `min_weight` | Minimum weight per asset | `0.0` |
| `max_weight` | Maximum weight per asset | `1.0` |
| `max_turnover` | Max $\frac{1}{2}\sum_i |w_i^{new} - w_i^{old}|$ | `None` |

**Short Selling Logic:**
- If `allow_short_selling=False`: Forces `min_weight=0` (long-only)
- If `allow_short_selling=True`: 
  - Respects user-specified `min_weight` (can be negative)
  - Adds gross exposure constraint: $\sum_i |w_i| \leq$ `gross_exposure_limit`

### 4.3 Standard Constraints

**Budget Constraint:**
$$\sum_i w_i = 1$$

**Box Constraints:**
$$w_{min} \leq w_i \leq w_{max} \quad \forall i$$

**Gross Exposure (when shorting):**
$$\sum_i |w_i| \leq L$$

**Turnover Constraint:**
$$\frac{1}{2}\sum_i |w_i^{new} - w_i^{old}| \leq T$$

### 4.4 Solution Methods

- **Max Sharpe:** SLSQP with multiple starting points
- **Min Variance / Mean-Variance:** CVXPY convex solver (ECOS/SCS)
- **Risk Parity:** SLSQP with inverse-volatility initialization

---

## Stage 5: Backtesting Framework

**Purpose:** Evaluate strategy performance on historical data

**File:** `BACKTEST.py`

### 5.1 Backtest Parameters

```python
BACKTEST(
    asset_list: List[str],         # Portfolio assets
    start_date: str,               # First rebalance date (inclusive)
    end_date: str,                 # End date (exclusive)
    rebalance_frequency: int,      # Days between rebalances (1-21)
    objective: str,                # Optimization objective
    allow_short_selling: bool,     # Allow shorts
    gross_exposure_limit: float,   # Max gross exposure
    min_weight: float,             # Min weight per asset
    max_weight: float,             # Max weight per asset  
    max_turnover: float,           # Max turnover per rebalance
    n_days: int,                   # KMRF forecast horizon
    initial_capital: float,        # Starting capital
    transaction_cost_bps: float    # Transaction costs in bps
)
```

### 5.2 Execution Flow

1. **Generate Rebalance Dates:** Every `rebalance_frequency` trading days
2. **For Each Rebalance Date:**
   - Run ANALYTICAL_INPUTS pipeline
   - Compute optimal weights via PORTFOLIO_OPTIMIZER
   - Apply transaction costs
   - Track portfolio value
3. **Between Rebalances:**
   - Hold weights constant
   - Compute daily returns using realized prices

### 5.3 Execution Methods

**Sequential (`run()`):**
- Processes dates one-by-one
- Correctly applies turnover constraints
- Suitable for all configurations

**Parallel (`run_parallel()`):**
- Parallelizes optimizations across dates using joblib
- Significantly faster for many rebalance dates
- Identical results when `max_turnover=None`

### 5.4 Performance Metrics

| Metric | Formula |
|--------|---------|
| Total Return | $(V_T / V_0) - 1$ |
| Annualized Return | $(1 + R_{total})^{252/n} - 1$ |
| Annualized Volatility | $\sigma_{daily} \times \sqrt{252}$ |
| Sharpe Ratio | $(\mu - r_f) / \sigma$ |
| Sortino Ratio | $(\mu - r_f) / \sigma_{downside}$ |
| Max Drawdown | $\min_t \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}$ |
| Calmar Ratio | $\mu_{ann} / |\text{MaxDD}|$ |
| Win Rate | $\#\{r_t > 0\} / n$ |

### 5.5 Benchmark Comparisons

- **S&P 500:** Buy-and-hold SPY
- **Equal Weight:** Equal-weighted portfolio on same rebalance schedule
- **MV Long-Only:** Mean-variance using historical returns only

### 5.6 Visualization

```python
bt.plot_performance()        # Cumulative returns vs benchmarks
bt.plot_detailed_analysis()  # Drawdowns, rolling Sharpe, distribution, weights
bt.plot_weights_heatmap()    # Weight allocation heatmap
```

---

## Implementation Reference

### File Structure

```
├── kama_msr.py              # Stage 1: KAMA, MSR, KAMA_MSR classes
├── kmrf.py                  # Stage 2: KMRF class (XGBoost-based)
├── ANALYTICAL_INPUTS.py     # Stage 3: Analytical μ, Σ computation
├── PORTFOLIO_OPTIMIZER.py   # Stage 4: Weight optimization
├── BACKTEST.py              # Stage 5: Backtesting framework
├── MODEL_INFO.py            # Helper class for model loading
├── derive_features.py       # Feature engineering
└── saved_models/
    ├── KAMA_MSR/            # Fitted regime models
    └── KMRF/                # Fitted prediction models
```

### Example Usage

```python
from ANALYTICAL_INPUTS import ANALYTICAL_INPUTS
from PORTFOLIO_OPTIMIZER import PORTFOLIO_OPTIMIZER
from BACKTEST import BACKTEST

# Single optimization
inputs = ANALYTICAL_INPUTS(
    opt_date='20241001',
    asset_list=['SPDR S&P 500 ETF', 'iShares Russell 2000 ETF'],
    n_days=21,
    annualize=True
)
inputs.run_full_pipeline()

optimizer = PORTFOLIO_OPTIMIZER.from_analytical_inputs(inputs, risk_free_rate=0.05)
weights = optimizer.optimize(
    objective='max_sharpe',
    allow_short_selling=True,
    gross_exposure_limit=1.5
)

# Full backtest
bt = BACKTEST(
    asset_list=['SPDR S&P 500 ETF', 'iShares Russell 2000 ETF'],
    start_date='2022-01-01',
    end_date='2024-01-01',
    rebalance_frequency=21,
    objective='max_sharpe'
)
bt.run_parallel(n_jobs=-1)
bt.print_summary()
bt.plot_performance()
```

---

## References

1. **Kaufman, P. J. (1995).** *Smarter Trading: Improving Performance in Changing Markets.* McGraw-Hill.
2. **Krolzig, H.-M. (1997).** *Markov-Switching Vector Autoregressions.* Springer.
3. **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
4. **Pomorski, D. & Gorse, D. (2022).** "Improving on the Markov-Switching Regression Model by the Use of an Adaptive Moving Average."
5. **Pomorski, D. & Gorse, D. (2023a).** "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes."
6. **Pomorski, D. & Gorse, D. (2023b).** "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework."

---

**Document Version:** 3.1  
**Last Updated:** December 2024  
**Author:** Jesse Goodman
