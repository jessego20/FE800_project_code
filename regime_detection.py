import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict
from scipy.stats import norm, invgamma, multivariate_normal
from scipy.optimize import minimize
from scipy.special import logsumexp
from numba import jit
import pickle
from itertools import product
import random

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KAMA CLASS
# ============================================================================

class KAMA:
    """
    Kaufman's Adaptive Moving Average implementation.
    This class handles the calculation and optimization of KAMA parameters.
    """
    def __init__(self, n: int = 10, n_fast: int = 2, n_slow: int = 30):
        """
        Initialize KAMA parameters
        
        Parameters:
        -----------
        n : int
            Length of the efficiency ratio calculation period
        n_fast : int
            Short-term smoothing constant period (fast)
        n_slow : int
            Long-term smoothing constant period (slow)
        """
        self.n = n
        self.n_fast = n_fast
        self.n_slow = n_slow
        self.k_fast = 2 / (n_fast + 1)
        self.k_slow = 2 / (n_slow + 1)
        
    def calculate_efficiency_ratio(self, prices: pd.Series) -> pd.Series:
        """Calculate the Efficiency Ratio (ER) for a pandas Series."""
        direction = (prices - prices.shift(self.n)).abs()
        volatility = prices.diff().abs().rolling(self.n).sum()
        er = direction / volatility
        er.iloc[:self.n] = 0
        return er
    
    def calculate_smoothing_constant(self, er: pd.Series) -> pd.Series:
        """Calculate the smoothing constant for a pandas Series."""
        return ((er * (self.k_fast - self.k_slow) + self.k_slow) ** 2)
    
    def calculate_kama(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate KAMA values for a pandas Series."""
        er = self.calculate_efficiency_ratio(prices)
        sc = self.calculate_smoothing_constant(er)
        kama = pd.Series(index=prices.index, dtype='float64')
        kama.iloc[0] = prices.iloc[0]
        for t in range(1, self.n):
            kama.iloc[t] = kama.iloc[t-1] + self.k_slow * (prices.iloc[t] - kama.iloc[t-1])
        for t in range(self.n, len(prices)):
            kama.iloc[t] = kama.iloc[t-1] + sc.iloc[t] * (prices.iloc[t] - kama.iloc[t-1])
        return kama.rename('kama'), er.rename('er'), sc.rename('sc')
    
    def optimize_parameters(self, prices: pd.Series, returns: pd.Series, 
                          bounds: Optional[List[Tuple[int, int]]] = None) -> Tuple[int, int, int]:
        """Optimize KAMA parameters using MSE as the objective function"""
        if bounds is None:
            bounds = [(5, 30), (2, 10), (20, 60)]
            
        def objective(params):
            self.n, self.n_fast, self.n_slow = map(int, params)
            self.k_fast = 2 / (self.n_fast + 1)
            self.k_slow = 2 / (self.n_slow + 1)
            
            kama_values, _, _ = self.calculate_kama(prices)
            kama_returns = np.log(kama_values).diff()
            valid_idx = kama_returns.notna()
            mse = ((returns[valid_idx] - kama_returns[valid_idx]) ** 2).mean()
            return mse
        
        result = minimize(
            objective,
            x0=[self.n, self.n_fast, self.n_slow],
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimized_params = tuple(map(int, result.x))
        self.n, self.n_fast, self.n_slow = optimized_params
        self.k_fast = 2 / (self.n_fast + 1)
        self.k_slow = 2 / (self.n_slow + 1)
        
        return optimized_params


# ============================================================================
# MARKOV SWITCHING MODEL CLASS
# ============================================================================

class MarkovSwitchingModel:
    """
    Optimized Markov Switching Regression model using Gibbs sampling.
    Implements: ln r_t = μ_{S_t} + β_{S_t} * ln r_{t-1} + σ_{S_t} * ε_t
    """
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.means = None
        self.betas = None 
        self.stds = None
        self.transition_probs = None
        self.mcmc_samples = None
        self.priors = self._set_default_priors()
        self.fitted_data = None
        
    def _set_default_priors(self) -> Dict:
        """Set default prior distributions"""
        return {
            'mean_prior_mean': 0.0,
            'mean_prior_var': 0.1,
            'beta_prior_mean': 0.0,
            'beta_prior_var': 1.0,
            'sigma_prior_shape': 2.0,
            'sigma_prior_scale': 0.1,
            'transition_prior_alpha': 8.0
        }
    
    def set_priors(self, **kwargs) -> None:
        """Update prior parameters"""
        self.priors.update(kwargs)
    
    def _initialize_from_data(self, returns: np.ndarray) -> None:
        """Initialize model parameters for n_regimes using data quantiles."""
        abs_returns = np.abs(returns - np.mean(returns))
        
        self.means = np.zeros(self.n_regimes)
        self.betas = np.zeros(self.n_regimes)
        self.stds = np.zeros(self.n_regimes)
        
        sorted_indices = np.argsort(abs_returns)
        splits = np.array_split(sorted_indices, self.n_regimes)
        
        for k, regime_indices in enumerate(splits):
            if len(regime_indices) > 2:
                regime_returns = returns[regime_indices]
                self.means[k] = np.mean(regime_returns)
                self.stds[k] = max(np.std(regime_returns), 0.001)
                
                y = regime_returns[1:]
                x = regime_returns[:-1]
                x_std = np.std(x)
                if x_std > 1e-8:
                    corr = np.corrcoef(x, y)[0, 1]
                    self.betas[k] = corr * (np.std(y) / x_std) * 0.1
            else:
                self.means[k] = self.priors['mean_prior_mean']
                self.betas[k] = self.priors['beta_prior_mean']
                self.stds[k] = np.sqrt(self.priors['sigma_prior_scale'] * 
                                      self.priors['sigma_prior_shape'])
        
        self.transition_probs = np.full((self.n_regimes, self.n_regimes), 
                                       0.2 / max(self.n_regimes - 1, 1))
        np.fill_diagonal(self.transition_probs, 0.8)
    
    def _sample_states(self, returns: np.ndarray) -> np.ndarray:
        """Optimized forward-backward sampling using vectorization"""
        T = len(returns)
        n_reg = self.n_regimes
        
        forward = np.zeros((T, n_reg))
        states = np.zeros(T, dtype=np.int32)
        emissions = np.zeros((T, n_reg))
        
        for k in range(n_reg):
            emissions[0, k] = norm.pdf(returns[0], self.means[k], self.stds[k])
        
        for t in range(1, T):
            conditional_means = self.means + self.betas * returns[t-1]
            emissions[t] = norm.pdf(returns[t], conditional_means, self.stds)
        
        forward[0] = emissions[0] / (emissions[0].sum() + 1e-300)
        
        for t in range(1, T):
            forward[t] = emissions[t] * (forward[t-1] @ self.transition_probs)
            forward_sum = forward[t].sum()
            if forward_sum > 1e-300:
                forward[t] /= forward_sum
            else:
                forward[t] = 1.0 / n_reg
        
        states[T-1] = np.random.choice(n_reg, p=forward[T-1])
        
        for t in range(T-2, -1, -1):
            backward_prob = forward[t] * self.transition_probs[:, states[t+1]]
            prob_sum = backward_prob.sum()
            
            if prob_sum > 1e-300:
                backward_prob /= prob_sum
                states[t] = np.random.choice(n_reg, p=backward_prob)
            else:
                states[t] = np.random.choice(n_reg)
                
        return states
    
    def _sample_parameters(self, returns: np.ndarray, states: np.ndarray) -> None:
        """Optimized parameter sampling with efficient indexing"""
        for k in range(self.n_regimes):
            regime_mask = (states == k) & (np.arange(len(states)) > 0)
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) > 0:
                y = returns[regime_indices]
                x = returns[regime_indices - 1]
                
                X = np.column_stack([np.ones(len(y)), x])
                XtX = X.T @ X
                Xty = X.T @ y
                
                V0_inv = np.array([[1/self.priors['mean_prior_var'], 0],
                                  [0, 1/self.priors['beta_prior_var']]])
                mu0 = np.array([self.priors['mean_prior_mean'], 
                               self.priors['beta_prior_mean']])
                
                V_inv = V0_inv + XtX
                V = np.linalg.inv(V_inv)
                mu_post = V @ (V0_inv @ mu0 + Xty)
                
                residuals = y - X @ mu_post
                sse = np.sum(residuals**2)
                
                shape_post = self.priors['sigma_prior_shape'] + len(y) * 0.5
                scale_post = self.priors['sigma_prior_scale'] + 0.5 * sse
                sigma2 = invgamma.rvs(shape_post, scale=scale_post)
                self.stds[k] = np.sqrt(sigma2)
                
                try:
                    coeffs = multivariate_normal.rvs(mu_post, sigma2 * V)
                    self.means[k] = coeffs[0]
                    self.betas[k] = coeffs[1]
                except:
                    self.means[k] = mu_post[0]
                    self.betas[k] = mu_post[1]
            else:
                self.means[k] = norm.rvs(self.priors['mean_prior_mean'], 
                                       np.sqrt(self.priors['mean_prior_var']))
                self.betas[k] = norm.rvs(self.priors['beta_prior_mean'], 
                                       np.sqrt(self.priors['beta_prior_var']))
                self.stds[k] = np.sqrt(invgamma.rvs(self.priors['sigma_prior_shape'], 
                                                  scale=self.priors['sigma_prior_scale']))
    
    def _sample_transitions(self, states: np.ndarray) -> None:
        """Optimized transition probability sampling"""
        state_pairs = np.column_stack([states[:-1], states[1:]])
        counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(self.n_regimes):
            mask = state_pairs[:, 0] == i
            if mask.any():
                next_states = state_pairs[mask, 1]
                counts[i] = np.bincount(next_states, minlength=self.n_regimes)
        
        alpha_prior = self.priors['transition_prior_alpha']
        for i in range(self.n_regimes):
            alpha_post = counts[i] + alpha_prior
            self.transition_probs[i] = np.random.dirichlet(alpha_post)
    
    def fit(self, returns: pd.Series, n_samples: int = 1000, burnin: int = 200, 
            thin: int = 1, verbose: bool = True) -> None:
        """Fit the model using Gibbs sampling"""
        returns_arr = returns.dropna().values.astype(np.float64)
        T = len(returns_arr)
        
        self.fitted_data = returns.dropna()
        self._initialize_from_data(returns_arr)
        
        total_iter = n_samples + burnin
        
        samples = {
            'means': np.zeros((n_samples, self.n_regimes), dtype=np.float64),
            'betas': np.zeros((n_samples, self.n_regimes), dtype=np.float64), 
            'stds': np.zeros((n_samples, self.n_regimes), dtype=np.float64),
            'transition_probs': np.zeros((n_samples, self.n_regimes, self.n_regimes), dtype=np.float64),
            'states': np.zeros((n_samples, T), dtype=np.int32)
        }
        
        if verbose:
            print(f"Running Gibbs sampler: {total_iter} total iterations")
            print(f"Burnin: {burnin}, Samples: {n_samples}, Thin: {thin}")
        
        sample_idx = 0
        
        for iteration in range(total_iter):
            states = self._sample_states(returns_arr)
            self._sample_parameters(returns_arr, states)
            self._sample_transitions(states)
            
            if iteration >= burnin and (iteration - burnin) % thin == 0:
                samples['means'][sample_idx] = self.means
                samples['betas'][sample_idx] = self.betas
                samples['stds'][sample_idx] = self.stds
                samples['transition_probs'][sample_idx] = self.transition_probs
                samples['states'][sample_idx] = states
                sample_idx += 1
                
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{total_iter}")
        
        self.mcmc_samples = samples
        
        self.means = np.mean(samples['means'], axis=0)
        self.betas = np.mean(samples['betas'], axis=0) 
        self.stds = np.mean(samples['stds'], axis=0)
        self.transition_probs = np.mean(samples['transition_probs'], axis=0)
        
        if verbose:
            print("\nGibbs sampling completed!")
            for i in range(self.n_regimes):
                print(f"Regime {i}: μ={self.means[i]:.6f}, β={self.betas[i]:.4f}, σ={self.stds[i]:.6f}")
    
    def get_regime_probabilities(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Get posterior regime probabilities for all regimes"""
        if self.mcmc_samples is None:
            raise ValueError("Model not fitted. Run fit() first.")
        
        if returns is None:
            returns = self.fitted_data
            
        states_samples = self.mcmc_samples['states']
        
        regime_probs = {}
        for k in range(self.n_regimes):
            regime_probs[f'Regime_{k}'] = np.mean(states_samples == k, axis=0)
        
        return pd.DataFrame(regime_probs, index=returns.dropna().index)


# ============================================================================
# KAMA+MSR COMBINED MODEL CLASS
# ============================================================================

class KAMA_MSR:
    """
    Combined KAMA+MSR Model for Four or Six-Regime Classification
    
    Based on: Pomorski & Gorse (2022) "Improving on the Markov-Switching 
    Regression Model by the Use of an Adaptive Moving Average"
    
    This version includes fixes for:
    - Random seed control for reproducibility
    - Proper state reset between fits
    - Correct parameter propagation
    - MSR prior optimization
    """
    
    def __init__(self, 
                 kama_params: Optional[Dict] = None,
                 msr_params: Optional[Dict] = None,
                 filter_params: Optional[Dict] = None,
                 use_three_state_msr: bool = False,
                 random_seed: Optional[int] = None):
        """
        Initialize the combined KAMA+MSR model
        
        Parameters:
        -----------
        kama_params : dict, optional
            KAMA parameters: {'n': 10, 'n_fast': 2, 'n_slow': 30}
        msr_params : dict, optional
            MSR parameters: {'n_regimes': 2}
        filter_params : dict, optional
            Filter parameters: {'n_lookback': 20, 'gamma': 1.0}
        use_three_state_msr : bool, default=False
            Whether to use 3-state MSR (includes medium/extreme volatility)
        random_seed : int, optional
            Random seed for reproducibility (None for different results each time)
        """
        self.random_seed = random_seed
        self._setup_randomness()
        
        # Initialize KAMA
        kama_defaults = {'n': 10, 'n_fast': 2, 'n_slow': 30}
        kama_params = kama_params or kama_defaults
        self.kama = KAMA(**kama_params)
        
        # Initialize MSR
        msr_defaults = {'n_regimes': 2}
        msr_params = msr_params or msr_defaults
        
        self.use_three_state_msr = use_three_state_msr
        if use_three_state_msr:
            msr_params['n_regimes'] = 3
            print("WARNING: Using 3-state MSR. Paper notes this may lead to instability.")
        
        self.msr = MarkovSwitchingModel(**msr_params)
        
        # Filter parameters
        filter_defaults = {'n_lookback': 20, 'gamma': 1.0}
        self.filter_params = filter_params or filter_defaults
        self.n_lookback = self.filter_params['n_lookback']
        self.gamma = self.filter_params['gamma']
        
        self.n_combined_regimes = 6 if use_three_state_msr else 4
        
        # Reset all state
        self._reset_state()
    
    def _setup_randomness(self):
        """Set up random state for reproducibility"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
    
    def _reset_state(self):
        """Reset all model state variables"""
        self.kama_values = None
        self.kama_er = None
        self.kama_sc = None
        self.msr_regime_probs = None
        self.filter_values = None
        self.regime_labels = None
        self.regime_probs = None
        self.prices = None
        self.returns = None
    
    def calculate_kama_filter(self, kama_series: pd.Series) -> pd.Series:
        """Calculate KAMA filter: f_t = γ * σ(KAMA_t)"""
        kama_changes = kama_series.diff()
        kama_std = kama_changes.rolling(window=self.n_lookback).std()
        filter_values = self.gamma * kama_std
        return filter_values
    
    def detect_kama_signals(self, kama_series: pd.Series, 
                           filter_series: pd.Series) -> pd.Series:
        """Detect bullish/bearish signals based on KAMA and filter"""
        signals = pd.Series(0, index=kama_series.index)
        
        kama_low = kama_series.rolling(window=self.n_lookback).min()
        kama_high = kama_series.rolling(window=self.n_lookback).max()
        
        bullish_condition = (kama_series - kama_low) > filter_series
        bearish_condition = (kama_high - kama_series) > filter_series
        
        signals[bullish_condition] = 1
        signals[bearish_condition] = -1
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def fit(self, prices: pd.Series, 
            optimize_kama: bool = False,
            optimize_filter: bool = False,
            msr_verbose: bool = False,
            **msr_fit_kwargs) -> 'KAMA_MSR':
        """
        Fit the combined KAMA+MSR model
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        optimize_kama : bool, default=False
            Whether to optimize KAMA parameters
        optimize_filter : bool, default=False
            Whether to optimize filter parameters (n_lookback, gamma)
        msr_verbose : bool, default=False
            Verbose output for MSR fitting
        **msr_fit_kwargs : additional arguments for MSR.fit()
        """
        # Reset state to ensure clean fit
        self._reset_state()
        self.prices = prices.copy()
        self.returns = np.log(prices).diff().dropna()
        
        # Re-setup randomness for this fit
        self._setup_randomness()
        
        print("=" * 60)
        print("KAMA+MSR COMBINED MODEL FITTING")
        print(f"Mode: {'6-Regime (3-State MSR)' if self.use_three_state_msr else '4-Regime (2-State MSR)'}")
        print("=" * 60)
        
        # Step 1: Fit MSR
        print(f"\n[1/4] Fitting {self.msr.n_regimes}-state MSR model...")
        msr_defaults = {'n_samples': 500, 'burnin': 100, 'thin': 1, 'verbose': msr_verbose}
        msr_defaults.update(msr_fit_kwargs)
        self.msr.fit(self.returns, **msr_defaults)
        self.msr_regime_probs = self.msr.get_regime_probabilities()
        
        # Step 2: Optimize KAMA if requested
        if optimize_kama:
            print("\n[2/4] Optimizing KAMA parameters...")
            optimized_params = self.kama.optimize_parameters(self.prices, self.returns)
            print(f"   Optimized KAMA: n={optimized_params[0]}, "
                  f"n_fast={optimized_params[1]}, n_slow={optimized_params[2]}")
        else:
            print(f"\n[2/4] Using default KAMA: n={self.kama.n}, "
                  f"n_fast={self.kama.n_fast}, n_slow={self.kama.n_slow}")
        
        # Step 3: Calculate KAMA and filter
        print("\n[3/4] Calculating KAMA and filter...")
        self.kama_values, self.kama_er, self.kama_sc = self.kama.calculate_kama(self.prices)
        self.filter_values = self.calculate_kama_filter(self.kama_values)
        
        # Step 4: Classify regimes
        print(f"\n[4/4] Classifying into {self.n_combined_regimes} regimes...")
        kama_signals = self.detect_kama_signals(self.kama_values, self.filter_values)
        self.regime_labels = self._classify_combined_regimes(self.msr_regime_probs, kama_signals)
        self.regime_probs = self._calculate_regime_probabilities()
        
        print("\nModel fitting complete!")
        self._print_regime_summary()
        
        # Step 5: Optimize filter if requested
        if optimize_filter:
            print("\n[Optional] Optimizing filter parameters...")
            self.optimize_filter_params(self.prices, self.returns)
            self.filter_values = self.calculate_kama_filter(self.kama_values)
            kama_signals = self.detect_kama_signals(self.kama_values, self.filter_values)
            self.regime_labels = self._classify_combined_regimes(self.msr_regime_probs, kama_signals)
            self.regime_probs = self._calculate_regime_probabilities()
            print("\nRe-classification with optimized filter complete!")
            self._print_regime_summary()
    
    def _classify_combined_regimes(self, msr_probs: pd.DataFrame, 
                                   kama_signals: pd.Series) -> pd.Series:
        """Classify into 4 or 6 regimes"""
        common_idx = msr_probs.index.intersection(kama_signals.index)
        msr_probs_aligned = msr_probs.loc[common_idx]
        kama_signals_aligned = kama_signals.loc[common_idx]
        
        msr_regime_cols = [col for col in msr_probs_aligned.columns if col.startswith('Regime_')]
        msr_regime = msr_probs_aligned[msr_regime_cols].idxmax(axis=1).str.extract('(\d+)')[0].astype(int)
        
        combined = pd.Series(np.nan, index=common_idx, dtype='Int64')
        
        if self.use_three_state_msr:
            # 6 regimes
            low_var_mask = (msr_regime == 0)
            combined.loc[low_var_mask & (kama_signals_aligned == 1)] = 0
            combined.loc[low_var_mask & (kama_signals_aligned == -1)] = 1
            
            med_var_mask = (msr_regime == 1)
            combined.loc[med_var_mask & (kama_signals_aligned == 1)] = 2
            combined.loc[med_var_mask & (kama_signals_aligned == -1)] = 3
            
            high_var_mask = (msr_regime == 2)
            combined.loc[high_var_mask & (kama_signals_aligned == 1)] = 4
            combined.loc[high_var_mask & (kama_signals_aligned == -1)] = 5
        else:
            # 4 regimes
            low_var_mask = (msr_regime == 0)
            combined.loc[low_var_mask & (kama_signals_aligned == 1)] = 0
            combined.loc[low_var_mask & (kama_signals_aligned == -1)] = 1
            
            high_var_mask = (msr_regime == 1)
            combined.loc[high_var_mask & (kama_signals_aligned == 1)] = 2
            combined.loc[high_var_mask & (kama_signals_aligned == -1)] = 3
        
        combined = combined.ffill()
        return combined
    
    def _calculate_regime_probabilities(self) -> pd.DataFrame:
        """Calculate probabilities for each regime"""
        regime_probs = pd.DataFrame(index=self.regime_labels.index)
        for regime in range(self.n_combined_regimes):
            regime_probs[f'Regime_{regime}'] = (self.regime_labels == regime).astype(float)
        return regime_probs
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """Get regime probabilities"""
        if self.regime_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.regime_probs
    
    def get_trading_signals(self, active_regimes: Optional[List[int]] = None) -> pd.Series:
        """Generate trading signals based on regime classification"""
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if active_regimes is None:
            active_regimes = [0, 5] if self.use_three_state_msr else [0, 3]
        
        signals = pd.Series(0, index=self.regime_labels.index)
        
        for regime in active_regimes:
            if regime >= self.n_combined_regimes:
                raise ValueError(f"Regime {regime} out of range")
            
            if regime % 2 == 0:
                signals[self.regime_labels == regime] = 1
            else:
                signals[self.regime_labels == regime] = -1
        
        return signals
    
    def _print_regime_summary(self):
        """Print summary statistics"""
        if self.regime_labels is None:
            return
        
        regime_names = {
            0: 'Low Variance + Bullish',
            1: 'Low Variance + Bearish',
            2: 'Medium Variance + Bullish' if self.use_three_state_msr else 'High Variance + Bullish',
            3: 'Medium Variance + Bearish' if self.use_three_state_msr else 'High Variance + Bearish',
            4: 'High/Extreme Variance + Bullish',
            5: 'High/Extreme Variance + Bearish'
        }
        
        print("\n" + "=" * 60)
        print("REGIME CLASSIFICATION SUMMARY")
        print("=" * 60)
        
        total_periods = len(self.regime_labels)
        
        for regime in range(self.n_combined_regimes):
            count = (self.regime_labels == regime).sum()
            pct = count / total_periods * 100
            print(f"Regime {regime} ({regime_names[regime]}): {count} periods ({pct:.1f}%)")
        
        changes = (self.regime_labels.diff() != 0).sum()
        print(f"\nTotal regime changes: {changes}")
        print(f"Average regime duration: {total_periods / (changes + 1):.1f} periods")
    
    def plot_regimes(self, figsize=(16, 12)):
        """Plot prices, KAMA, filter, and regime classification"""
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        regime_labels_clean = self.regime_labels.dropna()
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        if self.use_three_state_msr:
            colors = {0: 'green', 1: 'blue', 2: 'yellow',
                         3: 'orange', 4: 'purple', 5: 'red'}
            regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'MV Bull',
                           3: 'MV Bear', 4: 'HV Bull', 5: 'HV Bear'}
        else:
            colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'darkred'}
            regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        # Plot 1: Price and KAMA
        ax1 = axes[0]
        ax1.plot(self.prices.index, self.prices, label='Price', 
                color='black', alpha=0.7, linewidth=1)
        ax1.plot(self.kama_values.index, self.kama_values, 
                label='KAMA', color='blue', linewidth=2)
        
        for regime in range(self.n_combined_regimes):
            mask = (regime_labels_clean == regime)
            if mask.any():
                mask_aligned = pd.Series(False, index=self.prices.index)
                mask_aligned.loc[regime_labels_clean.index[mask]] = True
                
                ax1.fill_between(self.prices.index, self.prices.min(), self.prices.max(),
                               where=mask_aligned.values, alpha=0.2, color=colors[regime],
                               label=regime_names[regime])
        
        ax1.set_title('Price and KAMA with Regime Classification')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: KAMA Filter
        ax2 = axes[1]
        ax2.plot(self.filter_values.index, self.filter_values, 
                label='KAMA Filter', color='purple', linewidth=1.5)
        ax2.set_title('KAMA Filter (γ × σ(KAMA))')
        ax2.set_ylabel('Filter Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MSR Probabilities
        ax3 = axes[2]
        msr_colors = ['blue', 'orange', 'red'] if self.use_three_state_msr else ['blue', 'red']
        msr_labels = ['Low', 'Medium', 'High'] if self.use_three_state_msr else ['Low', 'High']
        
        for i in range(self.msr.n_regimes):
            ax3.plot(self.msr_regime_probs.index, self.msr_regime_probs[f'Regime_{i}'],
                    label=msr_labels[i], color=msr_colors[i], linewidth=1.5)
        
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('MSR Variance Regime Probabilities')
        ax3.set_ylabel('Probability')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regime Classification
        ax4 = axes[3]
        for regime in range(self.n_combined_regimes):
            mask = (regime_labels_clean == regime)
            if mask.any():
                ax4.scatter(regime_labels_clean.index[mask], regime_labels_clean[mask],
                           c=colors[regime], label=regime_names[regime], alpha=0.6, s=10)
        
        ax4.set_title(f'{"Six" if self.use_three_state_msr else "Four"}-Regime Classification')
        ax4.set_ylabel('Regime')
        ax4.set_yticks(list(range(self.n_combined_regimes)))
        ax4.set_yticklabels([regime_names[i] for i in range(self.n_combined_regimes)])
        ax4.legend(loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def optimize_filter_params(self, prices: pd.Series, returns: pd.Series,
                              param_bounds: Optional[Dict] = None) -> Dict:
        """Optimize filter parameters (n_lookback, gamma)"""
        if param_bounds is None:
            param_bounds = {'n_lookback': (10, 50), 'gamma': (0.5, 2.0)}
        
        def objective(params):
            n_lookback, gamma = int(params[0]), params[1]
            self.n_lookback = n_lookback
            self.gamma = gamma
            
            filter_vals = self.calculate_kama_filter(self.kama_values)
            kama_sigs = self.detect_kama_signals(self.kama_values, filter_vals)
            regime_labels = self._classify_combined_regimes(self.msr_regime_probs, kama_sigs)
            
            regime_returns = {}
            for regime in range(self.n_combined_regimes):
                mask = (regime_labels == regime)
                if mask.sum() > 0:
                    regime_returns[regime] = returns.loc[regime_labels.index[mask]]
            
            regime_means = [r.mean() for r in regime_returns.values() if len(r) > 0]
            if len(regime_means) < 2:
                return 1e10
            
            between_var = np.var(regime_means)
            within_var = np.mean([r.var() for r in regime_returns.values() if len(r) > 0])
            
            if within_var == 0:
                return 1e10
            
            return -(between_var / within_var)
        
        print("\nOptimizing filter parameters...")
        result = minimize(
            objective,
            x0=[self.n_lookback, self.gamma],
            bounds=[param_bounds['n_lookback'], param_bounds['gamma']],
            method='L-BFGS-B'
        )
        
        optimized = {'n_lookback': int(result.x[0]), 'gamma': result.x[1]}
        print(f"Optimized: n_lookback={optimized['n_lookback']}, gamma={optimized['gamma']:.3f}")
        
        self.n_lookback = optimized['n_lookback']
        self.gamma = optimized['gamma']
        
        return optimized
    
    def optimize_msr_priors(self, prices: pd.Series, returns: pd.Series,
                           prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                           param_grid: Optional[Dict[str, list]] = None,
                           method: str = 'grid') -> Dict:
        """
        Optimize MarkovSwitchingModel priors to improve regime separation
        """
        
        def evaluate_priors(prior_set: Dict) -> float:
            # Create fresh MSR with new priors
            msr_params = {'n_regimes': self.msr.n_regimes}
            fresh_msr = MarkovSwitchingModel(**msr_params)
            
            if hasattr(fresh_msr, 'set_priors'):
                fresh_msr.set_priors(**prior_set)
            else:
                raise AttributeError("MSR must have set_priors() method")
            
            # Setup randomness for this evaluation
            if self.random_seed is not None:
                seed_offset = abs(hash(str(prior_set))) % 1000
                np.random.seed(self.random_seed + seed_offset)
                random.seed(self.random_seed + seed_offset)
            
            # Fit fresh model
            fresh_msr.fit(returns, n_samples=500, burnin=100, verbose=False)
            msr_probs = fresh_msr.get_regime_probabilities()
            
            if self.kama_values is None:
                self.kama_values, _, _ = self.kama.calculate_kama(prices)
            
            filter_vals = self.calculate_kama_filter(self.kama_values)
            kama_signals = self.detect_kama_signals(self.kama_values, filter_vals)
            new_regime_labels = self._classify_combined_regimes(msr_probs, kama_signals)
            
            regime_returns = {}
            for regime in range(self.n_combined_regimes):
                mask = (new_regime_labels == regime)
                if mask.sum() > 0:
                    regime_returns[regime] = returns.loc[new_regime_labels.index[mask]]
            
            if len(regime_returns) < 2:
                return -np.inf
            
            between_var = np.var([r.mean() for r in regime_returns.values()])
            within_var = np.mean([r.var() for r in regime_returns.values()])
            
            if within_var == 0:
                return -np.inf
            
            return between_var / within_var
        
        best_score = -np.inf
        best_params = None
        
        if method == 'grid':
            if param_grid:
                keys = param_grid.keys()
                for vals in product(*param_grid.values()):
                    candidate = dict(zip(keys, vals))
                    score = evaluate_priors(candidate)
                    if score > best_score:
                        best_score = score
                        best_params = candidate
            elif prior_bounds:
                keys = list(prior_bounds.keys())
                ranges = [np.linspace(b[0], b[1], 3) for b in prior_bounds.values()]
                for vals in product(*ranges):
                    candidate = dict(zip(keys, vals))
                    score = evaluate_priors(candidate)
                    if score > best_score:
                        best_score = score
                        best_params = candidate
        
        elif method == 'random':
            n_samples = 50
            for _ in range(n_samples):
                candidate = {k: random.uniform(low, high) 
                           for k, (low, high) in prior_bounds.items()}
                score = evaluate_priors(candidate)
                if score > best_score:
                    best_score = score
                    best_params = candidate
        
        print(f"\nOptimized priors: {best_params}, Score: {best_score:.4f}")
        
        # Apply best priors
        final_msr = MarkovSwitchingModel(n_regimes=self.msr.n_regimes)
        final_msr.set_priors(**best_params)
        
        # Re-setup randomness for final fit
        self._setup_randomness()
        
        final_msr.fit(returns, n_samples=500, burnin=100, verbose=False)
        self.msr = final_msr
        self.msr_regime_probs = self.msr.get_regime_probabilities()
        
        # Reclassify regimes
        kama_signals = self.detect_kama_signals(self.kama_values, self.filter_values)
        self.regime_labels = self._classify_combined_regimes(self.msr_regime_probs, kama_signals)
        self.regime_probs = self._calculate_regime_probabilities()
        
        return best_params

    def analyze_results(self, data: Optional[pd.DataFrame] = None,
                       data_name: str = "Data") -> pd.DataFrame:
        """Comprehensive analysis of KAMA+MSR model results"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        print(f"\n{'='*60}")
        print(f"KAMA+MSR ANALYSIS RESULTS FOR {data_name.upper()}")
        print(f"{'='*60}")
        
        regime_probs = self.get_regime_probabilities()
        
        # MSR parameters
        print(f"\nMSR Parameters (Underlying {self.msr.n_regimes}-state model):")
        print(f"{'Regime':<8} {'μ (Mean)':<12} {'β (AR)':<10} {'σ (Vol)':<12}")
        print("-" * 45)
        for i in range(self.msr.n_regimes):
            print(f"Regime {i:<2} {self.msr.means[i]:<11.6f} {self.msr.betas[i]:<9.4f} {self.msr.stds[i]:<11.6f}")
        
        # MSR transition probabilities
        print(f"\nMSR Transition Probabilities:")
        header = "From\\To  " + "  ".join([f"Reg {j}" for j in range(self.msr.n_regimes)])
        print(header)
        print("-" * len(header))
        for i in range(self.msr.n_regimes):
            row = f"Regime {i}  " + "  ".join([f"{self.msr.transition_probs[i,j]:.3f}" for j in range(self.msr.n_regimes)])
            print(row)
        
        # KAMA parameters
        print(f"\nKAMA Parameters:")
        print(f"  n (ER period): {self.kama.n}")
        print(f"  n_fast: {self.kama.n_fast}")
        print(f"  n_slow: {self.kama.n_slow}")
        print(f"\nKAMA Filter Parameters:")
        print(f"  n_lookback: {self.n_lookback}")
        print(f"  gamma: {self.gamma}")
        
        # Combined regime classification
        estimated_regimes = self.regime_labels.dropna()
        regime_changes = (estimated_regimes.diff() != 0).sum()
        
        print(f"\nCombined Regime Classification ({self.n_combined_regimes} regimes):")
        for i in range(self.n_combined_regimes):
            count = (estimated_regimes == i).sum()
            pct = count / len(estimated_regimes)
            print(f"Regime {i} periods: {count} days ({pct:.1%})")
        print(f"Number of regime changes: {regime_changes}")
        
        return regime_probs

    def regime_characteristics(self, data: Optional[pd.DataFrame] = None,
                              regime_probs: Optional[pd.DataFrame] = None,
                              data_name: str = "Data") -> pd.Series:
        """Detailed regime characteristics analysis"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities()
        
        print(f"\n{'='*60}")
        print(f"DETAILED REGIME ANALYSIS - {data_name.upper()}")
        print(f"{'='*60}")
        
        regime_classification = self.regime_labels.dropna()
        
        # Define regime names
        if self.use_three_state_msr:
            regime_names = {
                0: "Low Vol Bullish", 1: "Low Vol Bearish",
                2: "Med Vol Bullish", 3: "Med Vol Bearish",
                4: "High Vol Bullish", 5: "High Vol Bearish"
            }
        else:
            regime_names = {
                0: "Low Vol Bullish", 1: "Low Vol Bearish",
                2: "High Vol Bullish", 3: "High Vol Bearish"
            }
        
        for regime in range(self.n_combined_regimes):
            mask = regime_classification == regime
            regime_returns = data['returns'].loc[regime_classification.index[mask]]
            
            if len(regime_returns) > 0:
                regime_name = regime_names[regime]
                
                # Calculate regime duration statistics
                durations = []
                current_duration = 0
                
                for i in range(len(regime_classification)):
                    current_regime = regime_classification.iloc[i]
                    if current_regime == regime:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            durations.append(current_duration)
                            current_duration = 0
                
                if current_duration > 0:
                    durations.append(current_duration)
                
                if durations:
                    avg_duration = np.mean(durations)
                    min_duration = np.min(durations)
                    max_duration = np.max(durations)
                    num_segments = len(durations)
                else:
                    avg_duration = 0
                    min_duration = 0
                    max_duration = 0
                    num_segments = 0
                
                print(f"\n{regime_name} (Regime {regime}):")
                print(f"  Observations: {len(regime_returns)} ({len(regime_returns)/len(data)*100:.1f}%)")
                print(f"  Regime Segments: {num_segments}")
                print(f"  Average Duration: {avg_duration:.1f} periods")
                if min_duration > 0:
                    print(f"  Minimum Duration: {min_duration:.0f} periods")
                if max_duration > 0:
                    print(f"  Maximum Duration: {max_duration:.0f} periods")
                print(f"  Mean Return: {regime_returns.mean():.6f} ({regime_returns.mean()*252:.2%} annualized)")
                print(f"  Volatility: {regime_returns.std():.6f} ({regime_returns.std()*np.sqrt(252):.2%} annualized)")
                print(f"  Sharpe Ratio: {(regime_returns.mean()/regime_returns.std()*np.sqrt(252)):.3f}")
                print(f"  Min Return: {regime_returns.min():.6f}")
                print(f"  Max Return: {regime_returns.max():.6f}")
                print(f"  Skewness: {regime_returns.skew():.3f}")
                print(f"  Excess Kurtosis: {regime_returns.kurtosis():.3f}")
        
        return regime_classification

    def diagnostics(self, data: Optional[pd.DataFrame] = None,
                   regime_probs: Optional[pd.DataFrame] = None,
                   data_name: str = "Data",
                   trace_plots: bool = False):
        """Perform model diagnostics and validation"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities()
        
        print(f"\n{'='*50}")
        print(f"KAMA+MSR MODEL DIAGNOSTICS - {data_name.upper()}")
        print(f"{'='*50}")
        
        returns = data['returns'].dropna().values
        regime_assignment = self.regime_labels.dropna().values
        
        # Map combined regimes back to MSR regimes
        msr_regime_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        
        log_likelihood = 0
        for t in range(1, len(returns)):
            combined_regime = regime_assignment[min(t, len(regime_assignment)-1)]
            msr_regime = msr_regime_mapping.get(combined_regime, 0)
            mean = self.msr.means[msr_regime] + self.msr.betas[msr_regime] * returns[t-1]
            log_likelihood += norm.logpdf(returns[t], mean, self.msr.stds[msr_regime])
        
        print(f"\nCombined Model Fit Statistics:")
        print(f"Log-Likelihood: {log_likelihood:.2f}")
        print(f"Average Log-Likelihood: {(log_likelihood / len(returns)):.2f}")
        
        # KAMA diagnostic
        kama_returns = np.log(self.kama_values).diff().dropna()
        valid_idx = kama_returns.index.intersection(data['returns'].index)
        tracking_error = (data['returns'].loc[valid_idx] - kama_returns.loc[valid_idx]).std()
        print(f"\nKAMA Tracking Error (vs returns): {tracking_error:.6f}")
        print(f"KAMA Efficiency Ratio (mean): {self.kama_er.mean():.4f}")
        
        # Regime stability
        regime_changes = (self.regime_labels.diff() != 0).sum()
        avg_duration = len(self.regime_labels) / (regime_changes + 1)
        print(f"\nRegime Stability:")
        print(f"  Total regime changes: {regime_changes}")
        print(f"  Average regime duration: {avg_duration:.1f} periods")
        print(f"  Change frequency: {regime_changes/len(self.regime_labels):.2%}")
        
        # # Regime separation score
        # print(f"\nRegime Discrimination:")
        
        # regime_returns = {}
        # for regime in range(self.n_combined_regimes):
        #     mask = (self.regime_labels == regime)
        #     if mask.sum() > 0:
        #         regime_returns[regime] = data['returns'].loc[self.regime_labels.index[mask]]
        
        # if len(regime_returns) > 1:
        #     regime_means = [r.mean() for r in regime_returns.values()]
        #     between_var = np.var(regime_means)
        #     within_var = np.mean([r.var() for r in regime_returns.values()])
            
        #     if within_var > 0:
        #         separation_score = between_var / within_var
        #         print(f"  Regime Separation Score: {separation_score:.3f}")
        #         print(f"    - Between-variance: {between_var:.6f}")
        #         print(f"    - Within-variance:  {within_var:.6f}")
                
        #         if separation_score > 5:
        #             print(f"  Interpretation: EXCELLENT regime separation")
        #         elif separation_score > 2:
        #             print(f"  Interpretation: GOOD regime separation")
        #         elif separation_score > 1:
        #             print(f"  Interpretation: MODERATE regime separation")
        #         else:
        #             print(f"  Interpretation: POOR regime separation")
        #     else:
        #         print(f"  Regime Separation Score: Infinite (zero within-variance)")
        # else:
        #     print(f"  Regime Separation Score: Not calculable (fewer than 2 regimes present)")

    def plot_comprehensive_analysis(self, data: Optional[pd.DataFrame] = None,
                                    regime_probs: Optional[pd.DataFrame] = None,
                                    data_name: str = "Data"):
        """Create comprehensive visualization"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities()
        
        fig = plt.figure(figsize=(16, 10))
        
        if self.use_three_state_msr:
            colors_map = {0: 'green', 1: 'blue', 2: 'yellow',
                         3: 'orange', 4: 'purple', 5: 'red'}
            regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'MV Bull',
                           3: 'MV Bear', 4: 'HV Bull', 5: 'HV Bear'}
        else:
            colors_map = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'darkred'}
            regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        regime_assignment = self.regime_labels.dropna()
        
        # Plot 1: Returns
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(data.index, data['returns'], color='black', alpha=0.7, linewidth=0.5)
        for i in range(self.n_combined_regimes):
            mask = regime_assignment == i
            if mask.any():
                mask_aligned = pd.Series(False, index=data.index)
                mask_aligned.loc[regime_assignment.index[mask]] = True
                ax1.fill_between(data.index, data['returns'].min(), data['returns'].max(),
                               where=mask_aligned.values, alpha=0.3,
                               color=colors_map[i], label=regime_names[i])
        ax1.set_title(f'{data_name}: Returns with Combined Regimes')
        ax1.set_ylabel('Daily Log Returns')
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility
        ax2 = plt.subplot(2, 2, 2)
        window = 30
        rolling_vol = data['returns'].rolling(window=window).std() * np.sqrt(252)
        ax2.plot(data.index, rolling_vol, color='black', alpha=0.7,
                label=f'{window}-day Rolling Vol')
        for i in range(self.n_combined_regimes):
            mask = regime_assignment == i
            if mask.any():
                mask_aligned = pd.Series(False, index=data.index)
                mask_aligned.loc[regime_assignment.index[mask]] = True
                ax2.fill_between(data.index, 0, rolling_vol.max(),
                               where=mask_aligned.values, alpha=0.3,
                               color=colors_map[i], label=regime_names[i])
        ax2.set_title('Rolling Volatility vs Regime Detection')
        ax2.set_ylabel('Annualized Volatility')
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Return distributions
        ax3 = plt.subplot(2, 2, 3)
        for i in range(self.n_combined_regimes):
            regime_mask = regime_assignment == i
            regime_returns = data['returns'].loc[regime_assignment.index[regime_mask]]
            if len(regime_returns) > 0:
                ax3.hist(regime_returns, bins=50, alpha=0.6, color=colors_map[i],
                        label=f'{regime_names[i]} (n={len(regime_returns)})', density=True)
        ax3.set_title('Return Distributions by Combined Regime')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Density')
        ax3.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: MSR posteriors
        if self.msr.mcmc_samples is not None:
            ax4 = plt.subplot(2, 2, 4)
            msr_colors = ['blue', 'orange', 'red'][:self.msr.n_regimes]
            for i in range(self.msr.n_regimes):
                ax4.hist(self.msr.mcmc_samples['means'][:, i], bins=30, alpha=0.6,
                        label=f'MSR μ_{i}', color=msr_colors[i], density=True)
            ax4.set_title('MSR Posterior Distributions (Mean Parameters)')
            ax4.set_xlabel('Mean Parameter Value')
            ax4.set_ylabel('Density')
            ax4.legend(loc='upper left', bbox_to_anchor=(0, 1))
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
# ============================================================================
# END OF SCRIPT
# ============================================================================