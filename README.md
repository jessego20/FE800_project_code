# KMRF Portfolio Optimization Methodology
## Complete Mathematical Framework

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

### 1.1 KAMA (Kaufman Adaptive Moving Average)

KAMA adapts its smoothing based on market efficiency, distinguishing trending vs. noisy periods.

**Efficiency Ratio:**
$$ER_t = \frac{|P_t - P_{t-n}|}{\sum_{i=0}^{n-1} |P_{t-i} - P_{t-i-1}|}$$

where:
- $P_t$ = price at time $t$
- $n$ = efficiency ratio period (default: 10)
- Numerator = directional movement
- Denominator = total volatility

**Interpretation:**
- $ER \to 1$: Strong trend (directional movement ≈ total movement)
- $ER \to 0$: Noisy/range-bound (directional movement << total movement)

**Adaptive Smoothing Constant:**
$$SC_t = [ER_t \times (fast - slow) + slow]^2$$

where:
- $fast = 2/(2+1) = 0.667$ (2-period EMA equivalent)
- $slow = 2/(30+1) = 0.0645$ (30-period EMA equivalent)

**KAMA Update:**
$$KAMA_t = KAMA_{t-1} + SC_t \times (P_t - KAMA_{t-1})$$

**Trend Detection:**
- **Bullish:** Price consistently above KAMA
- **Bearish:** Price consistently below KAMA
- **Minimum Duration:** Trends must persist for minimum periods (default: 5)

### 1.2 MSR (Markov-Switching Regime)

MSR models volatility regimes using a hidden Markov model with 2 states (LV/HV).

**Observation Model:**
$$r_t | S_t = s \sim \mathcal{N}(\mu_s, \sigma_s^2)$$

where:
- $r_t$ = return at time $t$
- $S_t \in \{LV, HV\}$ = latent volatility state
- $\mu_s, \sigma_s$ = regime-specific mean and volatility

**Transition Matrix:**
$$\Pi = \begin{bmatrix} \pi_{LV \to LV} & \pi_{LV \to HV} \\ \pi_{HV \to LV} & \pi_{HV \to HV} \end{bmatrix}$$

**Estimation via Gibbs Sampling:**

1. Initialize regime labels randomly
2. For each iteration:
   - Sample regime means: $\mu_s | r, S \sim \mathcal{N}(\bar{r}_s, \sigma_s^2/n_s)$
   - Sample regime variances: $\sigma_s^2 | r, S \sim \text{InvGamma}$
   - Sample transition probabilities: $\pi_{i \to j} | S \sim \text{Dirichlet}$
   - Sample regime sequence: Forward-filtering backward-sampling
3. Discard burn-in (default: 200 iterations)
4. Use posterior mode for final labels

**Configuration:**
- Gibbs iterations: 1000
- Burn-in: 200
- Random seed: Configurable for reproducibility

### 1.3 Combined KAMA+MSR Labeling

The 4-regime labels are created by combining:

| KAMA Trend | MSR Volatility | Final Regime |
|------------|----------------|--------------|
| Bullish | Low Volatility | 0 (LV_Bull) |
| Bearish | Low Volatility | 1 (LV_Bear) |
| Bullish | High Volatility | 2 (HV_Bull) |
| Bearish | High Volatility | 3 (HV_Bear) |

**Output:** Time series of regime labels for each asset

---

## Stage 2: KMRF Regime Prediction

**Purpose:** Predict future regime probabilities (ex-ante forecasting)

**File:** `kmrf.py`

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

**Feature Selection (BorutaPy):**
- Uses Random Forest to identify shadow features
- Compares feature importance vs. randomized versions
- Retains only statistically significant features
- 100 max iterations with time-series cross-validation

### 2.2 Random Forest Classifier

**Model:** `sklearn.ensemble.RandomForestClassifier`

**Target Variable:** 4-regime labels from KAMA+MSR (or adapted 3-class)

**Training:**
- Purged Group Time-Series Split to prevent leakage
- 15-day gap between train/validation sets
- Hyperparameter tuning on validation set

**Output:** Probability distribution over regimes for each forecast horizon

### 2.3 Multi-Horizon Predictions

KMRF generates predictions for multiple horizons (e.g., 1, 5, 10, 21 days ahead):

$$\hat{\pi}_{i,m}^{(h)} = P(S_{t+h}^i = m | \mathcal{F}_t)$$

where:
- $i$ = asset
- $m$ = regime (0-3)
- $h$ = forecast horizon
- $\mathcal{F}_t$ = information available at time $t$

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
├── kmrf.py                  # Stage 2: KMRF class
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

**Document Version:** 3.0  
**Last Updated:** December 2024  
**Author:** Jesse Goodman
