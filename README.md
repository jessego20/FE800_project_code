# Multi-Asset Gaussian Copula Simulation with Bayesian Regime Updates
## Complete Mathematical Methodology

---

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Implementation Details](#implementation-details)
5. [Statistical Properties](#statistical-properties)
6. [Validation Methods](#validation-methods)

---
## Core Model Components

### KMRF (KAMA+MSR+RF) Regime Prediction Model
- **KAMA+MSR**: Combines Kaufman's Adaptive Moving Average with Markov-Switching Regression
- **Random Forest (XGBoost)**: Predicts market regimes based on KAMA+MSR labels
- **Regime Classes**: Four regimes (LV bullish/bearish, HV bullish/bearish)

## Overview

This document provides a complete mathematical explanation of the multi-asset simulation methodology using Gaussian copulas with regime-switching dynamics and Bayesian updates.

### Key Features
- **Regime-dependent correlations**: Different correlation structures across market regimes
- **Bayesian regime evolution**: Regime probabilities update based on observed returns
- **Regime concordance**: Asset regimes conditionally depend on market regime
- **Flexible marginal distributions**: Each asset×regime can have different return distributions
- **Copula framework**: Separates correlation structure from marginal distributions

---

## Mathematical Foundation

### 1. Regime-Switching Framework

We model the market as having **4 distinct regimes** based on S&P 500 volatility and trend:
- **Regime 0**: Low Volatility Bull (LV Bull)
- **Regime 1**: Low Volatility Bear (LV Bear)
- **Regime 2**: High Volatility Bull (HV Bull)
- **Regime 3**: High Volatility Bear (HV Bear)

#### Market Regime Dynamics

Let $S_t^{mkt} \in \{0, 1, 2, 3\}$ denote the market regime at time $t$.

**Transition Probabilities:**
$$P(S_{t+1}^{mkt} = j | S_t^{mkt} = i) = \Pi_{ij}$$

where $\Pi$ is the 4×4 transition matrix estimated from historical S&P 500 data.

**Forward Regime Probabilities (from KMRF):**

Starting from current regime distribution $p_0 = [p_0^0, p_0^1, p_0^2, p_0^3]$:

$$p_t = p_0 \cdot \Pi^t$$

These are **unconditional** forward probabilities that incorporate:
- Historical transition dynamics
- Bayesian decay (reducing influence of current regime over time)
- No information from future returns (ex-ante probabilities)

### 2. Asset Regime Concordance

Each asset $a$ has its own regime $S_t^a \in \{0, 1, 2, 3\}$, but these are **not independent** of the market regime.

**Concordance Matrix:**

Define the conditional probability:
$$C^a_{ij} = P(S_t^a = j | S_t^{mkt} = i)$$

where:
- $i$ = market regime (row)
- $j$ = asset regime (column)
- $C^a$ is a 4×4 stochastic matrix (rows sum to 1)

**Estimation from Historical Data:**

Given historical regime labels $\{S_\tau^{mkt}, S_\tau^a\}_{\tau=1}^T$:

$$\hat{C}^a_{ij} = \frac{\sum_{\tau=1}^T \mathbb{1}\{S_\tau^{mkt}=i, S_\tau^a=j\}}{\sum_{\tau=1}^T \mathbb{1}\{S_\tau^{mkt}=i\}}$$

**Example Concordance Matrix** (for Russell 2000):
```
                  Asset Regime j →
Market Regime i ↓   0      1      2      3
────────────────────────────────────────────
       0 (LV Bull)  0.95   0.04   0.005  0.005
       1 (LV Bear)  0.10   0.85   0.01   0.04
       2 (HV Bull)  0.30   0.16   0.50   0.04
       3 (HV Bear)  0.12   0.30   0.01   0.57
```

**Interpretation:**
- When market is in LV Bull (0), Russell 2000 is 95% likely to also be in LV Bull
- When market is in HV Bear (3), Russell 2000 is 57% likely to be in HV Bear, but 30% chance in LV Bear (flight to quality)

### 3. Regime-Dependent Return Distributions

For each asset $a$ and regime $s \in \{0,1,2,3\}$, returns follow:

$$r_t^a | S_t^a = s \sim F_s^a$$

where $F_s^a$ is the fitted distribution (one of: Normal, Skew-Normal, Student-t, NIG).

**Distribution Parameters:**

For example, if asset $a$ in regime $s$ follows Skew-Normal:
$$F_s^a = \text{SkewNorm}(\mu_s^a, \sigma_s^a, \alpha_s^a)$$

Parameters $\{\mu_s^a, \sigma_s^a, \alpha_s^a\}$ are estimated from historical returns when asset was in regime $s$.

### 4. Regime-Dependent Correlation Structure

Correlations between assets depend on the **market regime**:

$$\Sigma(S_t^{mkt}) = \text{Corr}(r_t^1, r_t^2, ..., r_t^N | S_t^{mkt})$$

**Estimation:**

For market regime $i$, compute sample correlation from historical returns on days when $S_t^{mkt} = i$:

$$\hat{\Sigma}_i = \text{Corr}\left(\{r_\tau : S_\tau^{mkt} = i\}_{\tau=1}^T\right)$$

**Example** (2 assets, market in LV Bull):
```
Σ₀ = [ 1.00  0.84 ]
     [ 0.84  1.00 ]
```

In crisis regimes (HV Bear), correlations tend to be **higher**:
```
Σ₃ = [ 1.00  0.94 ]
     [ 0.94  1.00 ]
```

### 5. Gaussian Copula Framework

**Sklar's Theorem:**

Any multivariate distribution $F(r_1, ..., r_N)$ can be decomposed as:

$$F(r_1, ..., r_N) = C(F_1(r_1), ..., F_N(r_N))$$

where:
- $F_i(r_i)$ = marginal CDF of asset $i$
- $C$ = copula function (captures dependence structure)

**Gaussian Copula:**

$$C(u_1, ..., u_N; \Sigma) = \Phi_\Sigma(\Phi^{-1}(u_1), ..., \Phi^{-1}(u_N))$$

where:
- $\Phi$ = standard normal CDF
- $\Phi^{-1}$ = inverse standard normal CDF (quantile function)
- $\Phi_\Sigma$ = multivariate normal CDF with correlation matrix $\Sigma$
- $u_i \in [0,1]$ = uniform random variables

**Why Gaussian Copula?**
- Tractable: Closed-form transformations
- Flexible: Allows any marginal distributions
- Interpretable: Correlation parameter has clear meaning
- Computationally efficient: No MCMC required

**Limitation:** May underestimate tail dependence compared to t-copula or Clayton copula.

### 6. Bayesian Regime Updates

During simulation, we observe simulated returns and update regime beliefs.

**Bayes' Rule:**

$$P(S_t = s | r_t) = \frac{P(r_t | S_t = s) \cdot P(S_t = s)}{\sum_{s'=0}^3 P(r_t | S_t = s') \cdot P(S_t = s')}$$

where:
- **Likelihood**: $P(r_t | S_t = s) = f_s(r_t)$ (PDF of regime $s$ distribution)
- **Prior**: $P(S_t = s)$ (regime probability before observing $r_t$)
- **Posterior**: $P(S_t = s | r_t)$ (updated regime probability after observing $r_t$)

**Sequential Updating:**

For market regime evolution:

1. **Day $t$ prior** (before observing return):
   $$p_t^{prior} = p_{t-1}^{posterior} \cdot \Pi$$
   
   Or use unconditional forward probability from KMRF as prior:
   $$p_t^{prior} = p_t^{KMRF}$$

2. **Observe simulated return** $r_t^{mkt}$

3. **Compute likelihoods**:
   $$\mathcal{L}_s = f_s^{mkt}(r_t^{mkt})$$

4. **Bayesian update**:
   $$p_t^{posterior}[s] = \frac{\mathcal{L}_s \cdot p_t^{prior}[s]}{\sum_{s'=0}^3 \mathcal{L}_{s'} \cdot p_t^{prior}[s']}$$

5. **Next day**: Use $p_t^{posterior}$ to inform $p_{t+1}^{prior}$

---

## Step-by-Step Workflow

### Initialization Phase

#### Step 0: Data Preparation

For each asset in portfolio:

1. **Load KAMA+MSR model**
   - Historical regime labels: $\{S_\tau^a\}_{\tau=1}^{T_{hist}}$
   - Date range: Asset-specific (e.g., S&P 500 from 1995, Russell 2000 from 2000)

2. **Load KMRF model**
   - Transition matrix: $\Pi^a$ (4×4)
   - Forward regime probabilities: $\{p_t^a\}_{t=0}^{T_{forecast}}$ (unconditional)

3. **Fit regime-specific distributions**
   
   For each regime $s \in \{0,1,2,3\}$:
   - Extract returns: $\{r_\tau^a : S_\tau^a = s\}$
   - Fit 4 candidate distributions:
     - Normal: $\mathcal{N}(\mu, \sigma^2)$
     - Skew-Normal: $\text{SN}(\mu, \sigma, \alpha)$
     - Student-t: $t_\nu(\mu, \sigma)$
     - NIG: $\text{NIG}(\alpha, \beta, \delta, \mu)$
   - Select best via AIC: $\text{AIC} = 2k - 2\ln(\mathcal{L})$
   - Store: $F_s^a$ with parameters

#### Step 1: Estimate Regime Concordance

For each asset $a$ (excluding market asset):

1. **Align data**:
   - Market regimes: $\{S_\tau^{mkt}\}_{\tau \in T_{common}}$
   - Asset regimes: $\{S_\tau^a\}_{\tau \in T_{common}}$
   - $T_{common}$ = intersection of date ranges

2. **Build joint count matrix**:
   $$N_{ij} = \sum_{\tau \in T_{common}} \mathbb{1}\{S_\tau^{mkt}=i, S_\tau^a=j\}$$

3. **Normalize to conditional probabilities**:
   $$C^a_{ij} = \frac{N_{ij}}{\sum_{j'=0}^3 N_{ij'}}$$

4. **Validate**: Each row of $C^a$ sums to 1

5. **Compute concordance statistics**:
   - Overall concordance: $\kappa^a = \frac{1}{4}\sum_{s=0}^3 C^a_{ss}$
   - Per-regime concordance: $\{\kappa^a_s = C^a_{ss}\}_{s=0}^3$

#### Step 2: Estimate Regime-Dependent Correlations

1. **Load market regime labels**: $\{S_\tau^{mkt}\}_{\tau=1}^{T_{hist}}$

2. **Load returns for all assets**: $\{r_\tau^a\}_{a \in \mathcal{A}, \tau=1}^{T_{hist}}$

3. **For each market regime** $i \in \{0,1,2,3\}$:
   
   a. Extract subset: $\mathcal{T}_i = \{\tau : S_\tau^{mkt} = i\}$
   
   b. Build return matrix: $R_i = [r_\tau^a]_{a \in \mathcal{A}, \tau \in \mathcal{T}_i}$
   
   c. Compute correlation: $\Sigma_i = \text{Corr}(R_i)$
   
   d. Validate: 
      - Symmetric
      - Positive semi-definite
      - Diagonal = 1
   
   e. If not PSD, regularize:
      - Eigenvalue decomposition: $\Sigma_i = Q \Lambda Q^T$
      - Clip eigenvalues: $\Lambda' = \max(\Lambda, \epsilon)$
      - Reconstruct: $\Sigma_i' = Q \Lambda' Q^T$

4. **Store**: $\{\Sigma_0, \Sigma_1, \Sigma_2, \Sigma_3\}$

---

### Simulation Phase

For each simulation $k = 1, ..., K$ (e.g., $K = 10,000$):

#### Step 3: Initialize Regime Probabilities (t=0)

**Market regime:**
$$p_0^{mkt} = p_0^{KMRF, mkt}$$
This is the unconditional forward probability from KMRF at $t=0$.

**Asset regimes:**
$$p_0^a = p_0^{KMRF, a} \quad \forall a \in \mathcal{A}$$

#### Step 4: Daily Simulation Loop (t = 0, 1, ..., T-1)

##### Step 4a: Sample Market Regime

**Categorical sampling from current distribution:**
$$S_t^{mkt} \sim \text{Categorical}(p_t^{mkt})$$

**Example:**
If $p_t^{mkt} = [0.90, 0.05, 0.03, 0.02]$, then:
- 90% chance: $S_t^{mkt} = 0$ (LV Bull)
- 5% chance: $S_t^{mkt} = 1$ (LV Bear)
- 3% chance: $S_t^{mkt} = 2$ (HV Bull)
- 2% chance: $S_t^{mkt} = 3$ (HV Bear)

##### Step 4b: Sample Asset Regimes Conditionally

For each asset $a \in \mathcal{A}$:

**Market asset** (e.g., S&P 500):
$$S_t^{mkt\_asset} = S_t^{mkt}$$
(Perfect concordance by definition)

**Other assets**:

Combine two sources of information:

1. **Unconditional prior** (from KMRF): $p_t^a = [p_t^{a,0}, p_t^{a,1}, p_t^{a,2}, p_t^{a,3}]$

2. **Conditional on market regime**: $C^a[S_t^{mkt}, :] = [C^a_{S_t^{mkt}, 0}, ..., C^a_{S_t^{mkt}, 3}]$

**Multiplicative combination:**
$$q_t^a[j] = p_t^a[j] \cdot C^a[S_t^{mkt}, j]$$

**Normalization:**
$$\tilde{q}_t^a = \frac{q_t^a}{\sum_{j=0}^3 q_t^a[j]}$$

**Sample:**
$$S_t^a \sim \text{Categorical}(\tilde{q}_t^a)$$

**Mathematical Interpretation:**

This is an approximation to:
$$P(S_t^a = j | S_t^{mkt}, \text{past data}) \approx P(S_t^a = j | S_t^{mkt}) \cdot P(S_t^a = j | \text{past data})$$

where:
- $P(S_t^a = j | S_t^{mkt})$ is the concordance $C^a[S_t^{mkt}, j]$
- $P(S_t^a = j | \text{past data})$ is the KMRF forward probability $p_t^a[j]$

**Example:**

Suppose for Russell 2000 at day $t=5$:
- Market regime: $S_5^{mkt} = 3$ (HV Bear)
- KMRF forward probs: $p_5^{IWM} = [0.30, 0.40, 0.10, 0.20]$
- Concordance row 3: $C^{IWM}[3,:] = [0.12, 0.30, 0.01, 0.57]$

Combined:
$$q_5^{IWM} = [0.30 \times 0.12, 0.40 \times 0.30, 0.10 \times 0.01, 0.20 \times 0.57]$$
$$= [0.036, 0.120, 0.001, 0.114]$$

Normalized:
$$\tilde{q}_5^{IWM} = [0.133, 0.443, 0.004, 0.421]$$

Result: Russell 2000 has 42% chance of being in HV Bear (down from 57% concordance due to KMRF prior suggesting otherwise).

##### Step 4c: Sample Correlated Returns via Gaussian Copula

**Get correlation matrix for current market regime:**
$$\Sigma_t = \Sigma_{S_t^{mkt}}$$

**Cholesky decomposition** (pre-computed):
$$\Sigma_t = L_t L_t^T$$

where $L_t$ is lower triangular.

**Generate independent standard normals:**
$$z_1, z_2, ..., z_N \sim \mathcal{N}(0, 1) \text{ independently}$$

Let $\mathbf{z} = [z_1, ..., z_N]^T$.

**Induce correlation:**
$$\mathbf{z}_{corr} = L_t \mathbf{z}$$

**Property:** $\mathbf{z}_{corr} \sim \mathcal{N}(\mathbf{0}, \Sigma_t)$

**Proof:**
$$\text{Var}(\mathbf{z}_{corr}) = L_t \text{Var}(\mathbf{z}) L_t^T = L_t I L_t^T = L_t L_t^T = \Sigma_t$$

**Transform to uniform [0,1]:**
$$u_i = \Phi(z_{i,corr})$$

where $\Phi$ is the standard normal CDF.

**Property:** $u_i \in [0,1]$ are uniformly distributed but **correlated** according to the Gaussian copula.

**Clip for numerical stability:**
$$u_i \leftarrow \max(10^{-10}, \min(1 - 10^{-10}, u_i))$$

Prevents issues when $u_i$ is exactly 0 or 1.

**Transform to asset-specific returns (Inverse CDF):**

For each asset $a$:

1. Get asset's regime: $s^a = S_t^a$
2. Get regime distribution: $F_{s^a}^a$
3. Apply inverse CDF:
   $$r_t^a = (F_{s^a}^a)^{-1}(u_a)$$

**Example - Skew-Normal:**
If $F_{s^a}^a = \text{SkewNorm}(\mu, \sigma, \alpha)$:
$$r_t^a = \text{SkewNorm.ppf}(u_a; \mu, \sigma, \alpha)$$

where `.ppf` is the percent-point function (inverse CDF).

**Store simulated return:**
$$R^{(k)}[a, t] = r_t^a$$

##### Step 4d: Bayesian Update of Market Regime

**Observe** the simulated market return: $r_t^{mkt}$

**Compute likelihoods** for each regime $s \in \{0,1,2,3\}$:
$$\mathcal{L}_s = f_s^{mkt}(r_t^{mkt})$$

where $f_s^{mkt}$ is the PDF of the market asset's regime-$s$ distribution.

**Example - Student-t:**
If regime 3 has $F_3^{mkt} = t_5(\mu=-0.01, \sigma=0.03)$:
$$\mathcal{L}_3 = \frac{1}{\sigma \cdot B(\frac{1}{2}, \frac{\nu}{2}) \cdot \sqrt{\nu}} \left(1 + \frac{1}{\nu}\left(\frac{r_t^{mkt} - \mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}$$

**Get prior for next time step:**

Use unconditional forward probability from KMRF:
$$p_{t+1}^{prior} = p_{t+1}^{KMRF, mkt}$$

**Alternative:** Could use transition-based prior:
$$p_{t+1}^{prior} = \Pi^T p_t^{posterior}$$

We use KMRF forward probs because they already incorporate Bayesian decay.

**Bayesian update (element-wise):**
$$p_{t+1}^{posterior}[s] = \frac{\mathcal{L}_s \cdot p_{t+1}^{prior}[s]}{\sum_{s'=0}^3 \mathcal{L}_{s'} \cdot p_{t+1}^{prior}[s']}$$

**Set for next iteration:**
$$p_{t+1}^{mkt} = p_{t+1}^{posterior}$$

**Intuition:**
- If $r_t^{mkt}$ is a large negative return, $\mathcal{L}_3$ (HV Bear) will be high
- Posterior will shift probability mass toward regime 3
- Next day, more likely to sample regime 3
- Creates realistic regime persistence and shock responses

##### Step 4e: Update Asset Regime Probabilities

For non-market assets, simply advance to next day's unconditional forward probability:
$$p_{t+1}^a = p_{t+1}^{KMRF, a}$$

**Note:** We could also do Bayesian updates for each asset based on its observed return, but:
- Adds complexity
- Market regime already provides strong signal
- Asset regimes will adapt via concordance in Step 4b

##### Step 4f: Advance to Next Day

$$t \leftarrow t + 1$$

Repeat Steps 4a-4e until $t = T$.

---

#### Step 5: Output

After $K$ simulations, we have:

$$\mathcal{R} = \{R^{(k)}[a, t]\}_{k=1,...,K; a \in \mathcal{A}; t=0,...,T-1}$$

where $R^{(k)}[a, t]$ is the return of asset $a$ on day $t$ in simulation $k$.

**Structure:**
- Dictionary with keys = asset names
- Values = numpy arrays of shape $(K, T)$

**Example:**
```python
{
    'SPDR S&P 500 ETF': np.array([
        [0.005, -0.012, ..., 0.008],  # Simulation 1
        [0.003,  0.001, ..., -0.002], # Simulation 2
        ...
        [-0.001, 0.015, ..., 0.003]   # Simulation K
    ]),  # Shape: (10000, 21)
    
    'iShares Russell 2000 ETF': np.array([...])  # Shape: (10000, 21)
}
```

---

## Implementation Details

### Numerical Stability

#### 1. Cholesky Decomposition Failure

**Problem:** If $\Sigma$ is not positive definite, Cholesky fails.

**Solution - Eigenvalue Regularization:**
```python
try:
    L = np.linalg.cholesky(Sigma)
except np.linalg.LinAlgError:
    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    eigenvals = np.maximum(eigenvals, 1e-10)  # Floor at small positive
    Sigma_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    L = np.linalg.cholesky(Sigma_reg)
```

**Alternative - SVD-based decomposition:**
```python
eigenvals, eigenvecs = np.linalg.eigh(Sigma)
eigenvals = np.maximum(eigenvals, 1e-10)
L = eigenvecs @ np.diag(np.sqrt(eigenvals))
```

This gives $\Sigma = L L^T$.

#### 2. Extreme Uniform Values

**Problem:** If $u \approx 0$ or $u \approx 1$, inverse CDF can return $\pm \infty$.

**Solution - Clipping:**
```python
u = np.clip(u, 1e-10, 1 - 1e-10)
```

Ensures $u \in [10^{-10}, 1 - 10^{-10}]$.

#### 3. Zero Likelihood

**Problem:** For extreme returns, PDF can be numerically zero.

**Solution - Floor:**
```python
likelihood = max(pdf(return_value), 1e-300)
```

Prevents division by zero in Bayesian update.

#### 4. Underflow in Bayesian Update

**Problem:** If all likelihoods are tiny, product underflows.

**Solution - Log-space computation:**
```python
log_likelihoods = np.log(likelihoods + 1e-300)
log_prior = np.log(prior + 1e-300)
log_posterior = log_likelihoods + log_prior
log_posterior -= log_posterior.max()  # Subtract max for stability
posterior = np.exp(log_posterior)
posterior /= posterior.sum()
```

### Performance Optimizations

#### 1. Pre-compute Cholesky Matrices

**Before simulation:**
```python
cholesky_matrices = {
    regime_id: np.linalg.cholesky(Sigma_regime)
    for regime_id, Sigma_regime in regime_correlations.items()
}
```

**During simulation:**
```python
L = cholesky_matrices[market_regime]  # O(1) lookup, not O(N^3) decomposition
```

**Speedup:** $O(N^3)$ → $O(1)$ per iteration

#### 2. Pre-compute Distribution Objects

**Before simulation:**
```python
dist_info = {
    asset: {
        regime: {
            'distribution': dist_type,
            'params': params,
            'scipy_obj': create_scipy_dist(dist_type, params)
        }
        for regime, params in regime_dists.items()
    }
    for asset, regime_dists in assets_regime_distributions.items()
}
```

**During simulation:**
```python
scipy_dist = dist_info[asset][regime]['scipy_obj']
return_val = scipy_dist.ppf(u)  # Direct call, no dictionary lookups
```

#### 3. Vectorize Where Possible

**Bad - Loop over assets:**
```python
for asset_idx in range(N):
    returns[asset_idx] = inverse_cdf(u[asset_idx], ...)
```

**Good - Vectorized (if distributions are same type):**
```python
returns = scipy_dist.ppf(u)  # Vectorized scipy call
```

However, since each asset can have different distribution types, full vectorization is limited.

---

## Statistical Properties

### Property 1: Correlation Preservation

**Theorem:** The Gaussian copula preserves the target correlation structure.

**Proof Sketch:**

1. Generate $\mathbf{z} \sim \mathcal{N}(0, \Sigma)$
2. Transform $u_i = \Phi(z_i)$
3. For Gaussian marginals $F_i = \mathcal{N}(\mu_i, \sigma_i^2)$:
   $$r_i = \mu_i + \sigma_i \Phi^{-1}(u_i) = \mu_i + \sigma_i z_i$$
4. Correlation:
   $$\rho_{ij} = \text{Corr}(r_i, r_j) = \text{Corr}(\sigma_i z_i, \sigma_j z_j) = \Sigma_{ij}$$

**For non-Gaussian marginals:** Correlation is approximately preserved but not exactly (rank correlation is preserved).

**Spearman's Rho:**
$$\rho_S = \frac{6}{\pi} \arcsin\left(\frac{\rho_{Gaussian}}{2}\right)$$

For $\rho_{Gaussian} = 0.84$, $\rho_S \approx 0.83$.

### Property 2: Marginal Distribution Preservation

**Theorem:** Each asset's returns follow the specified marginal distribution.

**Proof:**

For asset $i$ with CDF $F_i$:
$$P(r_i \leq x) = P(F_i^{-1}(u_i) \leq x) = P(u_i \leq F_i(x)) = F_i(x)$$

Since $u_i \sim \text{Uniform}[0,1]$, we have $P(u_i \leq F_i(x)) = F_i(x)$.

**Empirical Validation:**
- Histogram of simulated returns should match fitted distribution
- Kolmogorov-Smirnov test: $D = \max_x |F_n(x) - F(x)|$
- Should be small relative to critical value

### Property 3: Regime Dynamics

**Expected Regime Occupation:**

Under stationary distribution $\pi$ (eigenvector of $\Pi^T$ with eigenvalue 1):
$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \mathbb{1}\{S_t = s\} \to \pi_s$$

**For S&P 500:**
- LV Bull: $\pi_0 \approx 0.75$ (75% of time)
- LV Bear: $\pi_1 \approx 0.18$
- HV Bull: $\pi_2 \approx 0.01$
- HV Bear: $\pi_3 \approx 0.06$

**Persistence:**

Average duration in regime $s$:
$$D_s = \frac{1}{1 - \Pi_{ss}}$$

Example: If $\Pi_{00} = 0.95$, then $D_0 = 20$ days average in LV Bull.

### Property 4: Tail Dependence

**Gaussian Copula:** Tail dependence coefficient is **zero** (asymptotic independence).

**Formula:**
$$\lambda_L = \lim_{u \to 0} P(U_2 \leq u | U_1 \leq u) = 0$$

**Implication:** Gaussian copula may underestimate joint extreme events (crashes).

**Alternative:** Student-t copula has positive tail dependence:
$$\lambda_L = \lambda_U = 2t_{\nu+1}\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)$$

For $\nu = 5, \rho = 0.84$: $\lambda \approx 0.45$.

**Practical Impact:** In crisis simulations, assets may crash together less often than historical data suggests.

---

## Validation Methods

### 1. Correlation Validation

**Test:** Simulated correlation should match target regime correlations.

**Method:**

For each regime $s$:
1. Filter simulations: $\mathcal{K}_s = \{k : S_t^{mkt,(k)} = s\}$
2. Extract returns: $R_s = \{r_t^{a,(k)} : k \in \mathcal{K}_s, t=0,...,T-1, a \in \mathcal{A}\}$
3. Compute: $\hat{\Sigma}_s = \text{Corr}(R_s)$
4. Compare: $\|\hat{\Sigma}_s - \Sigma_s\|_F < \epsilon$

**Expected:** Should be within sampling error.

**Statistical Test:**

Under null hypothesis $H_0: \hat{\rho} = \rho$:
$$\frac{\sqrt{n-3}}{2} \ln\left(\frac{1+\hat{\rho}}{1-\hat{\rho}}\right) - \frac{\sqrt{n-3}}{2} \ln\left(\frac{1+\rho}{1-\rho}\right) \sim \mathcal{N}(0, 1)$$

### 2. Distribution Validation

**Kolmogorov-Smirnov Test:**

For asset $a$ in regime $s$:
1. Extract returns: $\{r_t^{a,(k)} : S_t^{a,(k)} = s\}$
2. Empirical CDF: $F_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{r_i \leq x\}$
3. Test statistic: $D = \max_x |F_n(x) - F_s^a(x)|$
4. Critical value at $\alpha = 0.05$: $D_{crit} = \frac{1.36}{\sqrt{n}}$
5. Reject if $D > D_{crit}$

**QQ-Plot:**

Plot quantiles of simulated returns vs. theoretical quantiles:
- Should form straight line
- Deviations in tails indicate misfit

### 3. Regime Concordance Validation

**Test:** Observed regime concordance should match historical concordance.

**Method:**

1. For each simulation, record regime pairs: $\{(S_t^{mkt,(k)}, S_t^{a,(k)})\}$
2. Build empirical concordance: $\hat{C}^a_{ij} = \frac{\#\{(i,j)\}}{\#\{i,\cdot\}}$
3. Compare: $\|\hat{C}^a - C^a\|_F < \epsilon$

**Chi-square Test:**

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

where $O_{ij}$ = observed count, $E_{ij}$ = expected count.

Under $H_0$: $\chi^2 \sim \chi^2_{df}$ with $df = (4-1)(4-1) = 9$.

### 4. Bayesian Update Validation

**Test:** Regime probabilities should respond correctly to returns.

**Method:**

1. Simulate large negative return (e.g., -5%)
2. Check if $p_t^{posterior}[3]$ (HV Bear) increases
3. Simulate large positive return (e.g., +3%)
4. Check if $p_t^{posterior}[0]$ (LV Bull) increases

**Quantitative Check:**

$$\frac{p_{t+1}[s]}{p_t[s]} = \frac{\mathcal{L}_s}{\sum_{s'} \mathcal{L}_{s'} p_t[s']} \cdot \frac{p_{t+1}^{prior}[s]}{p_t[s]}$$

For extreme returns, likelihood ratio should dominate.

### 5. Coverage Tests

**Out-of-Sample Backtest:**

1. Simulate returns for period $[T_1, T_2]$
2. Observe actual returns $\{r_t^{actual}\}$
3. For each day, compute percentile of actual return in simulated distribution
4. These percentiles should be $\sim \text{Uniform}[0,1]$

**Kupiec POF Test:**

For $\alpha = 0.05$ VaR:
- Expected violations: $E = \alpha \cdot T$
- Observed violations: $V$
- Test statistic: $LR = -2\ln\left(\frac{(1-\alpha)^{T-V}\alpha^V}{(1-V/T)^{T-V}(V/T)^V}\right)$
- Under $H_0$: $LR \sim \chi^2_1$

---

## Comparison to Alternative Approaches

### 1. Independent Regime Evolution

**Old Approach:**
$$S_t^a \sim \text{Categorical}(\Pi^a S_{t-1}^a)$$

**Problems:**
- Assets can be in contradictory regimes (S&P 500 crashing, bonds rallying)
- Ignores regime correlations
- Unrealistic during crises

**Our Approach:**
$$S_t^a \sim \text{Categorical}(C^a[S_t^{mkt},:] \cdot p_t^a)$$

**Benefits:**
- Assets follow market regime
- Conditional independence given market
- Flight-to-quality captured in concordance matrix

### 2. Fixed Correlation (No Regimes)

**Alternative:**
$$\mathbf{r}_t \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma_{constant})$$

**Problems:**
- Correlation doesn't vary with market conditions
- Underestimates crisis correlations
- Ignores regime-dependent distributions

**Our Approach:**
$$\Sigma_t = \Sigma_{S_t^{mkt}}$$

**Benefits:**
- Higher correlation in crisis regimes
- Better tail risk estimation
- More realistic diversification breakdown

### 3. Historical Simulation

**Alternative:**
$$\mathbf{r}_t \sim \text{Resample from} \{r_\tau\}_{\tau=1}^T$$

**Problems:**
- Limited to historical scenarios
- No forward-looking regime information
- Assumes stationarity

**Our Approach:**
- Generates new scenarios not in history
- Incorporates regime forecasts
- Regime-switching allows non-stationarity

### 4. t-Copula

**Alternative:**
$$C_t(u_1, ..., u_N; \Sigma, \nu) = t_{\Sigma,\nu}(t_\nu^{-1}(u_1), ..., t_\nu^{-1}(u_N))$$

**Advantages over Gaussian:**
- Positive tail dependence
- Better captures joint crashes

**Disadvantages:**
- More complex estimation
- Additional parameter ($\nu$)
- Slower simulation

**When to use t-copula:** If tail dependence is critical (e.g., risk management, stress testing).

---

## Extensions and Future Work

### 1. Asset-Specific Bayesian Updates

Currently, only market regime is updated via Bayes rule. Could extend to:
$$p_{t+1}^a[s] \propto f_s^a(r_t^a) \cdot p_t^a[s]$$

**Trade-off:** More accurate vs. more complex and slower.

### 2. Dynamic Concordance

Currently, $C^a$ is static (estimated from full history). Could allow:
$$C^a_t = f(t, \text{macro variables})$$

**Example:** Concordance increases during recessions.

### 3. Multi-Factor Regimes

Instead of single market regime, use multiple:
- S&P 500 regime (equity market)
- VIX regime (volatility)
- Credit spread regime (credit market)

**Asset regimes depend on multiple factors:**
$$P(S_t^a | S_t^{SPX}, S_t^{VIX}, S_t^{Credit})$$

### 4. Alternative Copulas

- **Student-t**: Better tail dependence
- **Clayton**: Lower tail dependence (crashes)
- **Gumbel**: Upper tail dependence (booms)
- **Vine copulas**: High-dimensional, flexible

### 5. Intraday Regimes

Currently, regimes change daily. Could use:
- Intraday regime switches
- Continuous-time regime processes (Markov jump)

---

## References

### Copula Theory
- Sklar, A. (1959). "Fonctions de répartition à n dimensions et leurs marges"
- Nelsen, R.B. (2006). "An Introduction to Copulas"
- Embrechts, P., Lindskog, F., & McNeil, A. (2003). "Modelling dependence with copulas"

### Regime-Switching Models
- Hamilton, J.D. (1989). "A new approach to the economic analysis of nonstationary time series"
- Ang, A. & Bekaert, G. (2002). "Regime switches in interest rates"
- Guidolin, M. & Timmermann, A. (2008). "International asset allocation under regime switching"

### Bayesian Filtering
- Kalman, R.E. (1960). "A new approach to linear filtering and prediction problems"
- Hamilton, J.D. (1994). "Time Series Analysis"
- Kim, C.J. & Nelson, C.R. (1999). "State-Space Models with Regime Switching"

### Implementation
- SciPy Documentation: scipy.stats
- NumPy Documentation: numpy.linalg
- Pandas Documentation: pandas.DataFrame

---

## Appendix: Code Snippets

### A. Sampling from Categorical Distribution

```python
def sample_categorical(probs: np.ndarray) -> int:
    """
    Sample from categorical distribution.
    
    Parameters:
    -----------
    probs : np.ndarray
        Probability vector (must sum to 1)
    
    Returns:
    --------
    int : Sampled category index
    """
    return np.random.choice(len(probs), p=probs)
```

### B. Inverse CDF Transform

```python
def inverse_cdf(u: float, dist_params: Dict) -> float:
    """
    Transform uniform to distribution-specific value.
    
    Parameters:
    -----------
    u : float
        Uniform random variable in [0, 1]
    dist_params : Dict
        {'distribution': str, 'params': dict}
    
    Returns:
    --------
    float : Transformed value
    """
    dist_type = dist_params['distribution']
    params = dist_params['params']
    
    if dist_type == 'normal':
        return norm.ppf(u, loc=params['loc'], scale=params['scale'])
    elif dist_type == 'skewnorm':
        return skewnorm.ppf(u, a=params['a'], loc=params['loc'], scale=params['scale'])
    elif dist_type == 'student_t':
        return t.ppf(u, df=params['df'], loc=params['loc'], scale=params['scale'])
    elif dist_type == 'nig':
        # Custom implementation or approximation
        return nig_ppf(u, **params)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
```

### C. Bayesian Update

```python
def bayesian_update(prior: np.ndarray, 
                   likelihoods: np.ndarray) -> np.ndarray:
    """
    Perform Bayesian update.
    
    Parameters:
    -----------
    prior : np.ndarray
        Prior probabilities (length 4)
    likelihoods : np.ndarray
        Likelihood for each regime (length 4)
    
    Returns:
    --------
    np.ndarray : Posterior probabilities (normalized)
    """
    # Element-wise product
    unnormalized_posterior = prior * likelihoods
    
    # Normalize
    posterior = unnormalized_posterior / unnormalized_posterior.sum()
    
    return posterior
```

### D. Correlation Matrix Regularization

```python
def regularize_correlation_matrix(Sigma: np.ndarray, 
                                  epsilon: float = 1e-10) -> np.ndarray:
    """
    Ensure correlation matrix is positive definite.
    
    Parameters:
    -----------
    Sigma : np.ndarray
        Correlation matrix (may not be PSD)
    epsilon : float
        Minimum eigenvalue
    
    Returns:
    --------
    np.ndarray : Regularized correlation matrix
    """
    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(Sigma)
    
    # Clip eigenvalues
    eigenvals = np.maximum(eigenvals, epsilon)
    
    # Reconstruct
    Sigma_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Force exact diagonal = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Sigma_reg)))
    Sigma_reg = D_inv_sqrt @ Sigma_reg @ D_inv_sqrt
    
    return Sigma_reg
```

---

**Document Version:** 1.0  
**Last Updated:** November 19, 2024  
**Author:** Jesse Goodman