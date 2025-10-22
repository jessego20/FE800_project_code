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


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import warnings
warnings.filterwarnings('ignore')

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
        """
        Calculate the Efficiency Ratio (ER) for a pandas Series.
        Parameters:
            prices : pd.Series
                Series of price values
        Returns:
            pd.Series : Efficiency ratios
        """
        direction = (prices - prices.shift(self.n)).abs()
        volatility = prices.diff().abs().rolling(self.n).sum()
        er = direction / volatility
        er.iloc[:self.n] = 0
        return er
    
    def calculate_smoothing_constant(self, er: pd.Series) -> pd.Series:
        """
        Calculate the smoothing constant for a pandas Series.
        Parameters:
            er : pd.Series
                Series of efficiency ratios
        Returns:
            pd.Series : Smoothing constants
        """
        return ((er * (self.k_fast - self.k_slow) + self.k_slow) ** 2)
    
    def calculate_kama(self, prices: pd.Series) -> pd.Series:
        """
        Calculate KAMA values for a pandas Series.
        Parameters:
            prices : pd.Series
                Series of price values
        Returns:
            pd.Series : KAMA values
        """
        er = self.calculate_efficiency_ratio(prices)
        sc = self.calculate_smoothing_constant(er)
        kama = pd.Series(index=prices.index, dtype='float64')
        # kama.iloc[:self.n] = np.nan
        # kama.iloc[self.n] = prices.iloc[self.n]
        kama.iloc[0] = prices.iloc[0]
        for t in range(1, self.n):
            kama.iloc[t] = kama.iloc[t-1] + self.k_slow * (prices.iloc[t] - kama.iloc[t-1])
        for t in range(self.n, len(prices)):
            kama.iloc[t] = kama.iloc[t-1] + sc.iloc[t] * (prices.iloc[t] - kama.iloc[t-1])
        return kama.rename('kama'), er.rename('er'), sc.rename('sc')
    
    def optimize_parameters(self, prices: pd.Series, returns: pd.Series, 
                          bounds: Optional[List[Tuple[int, int]]] = None) -> Tuple[int, int, int]:
        """
        Optimize KAMA parameters using MSE as the objective function
        Parameters:
            prices : pd.Series
                Series of price values
            returns : pd.Series
                Series of return values
            bounds : List[Tuple[int, int]], optional
                Bounds for optimization (n, n_fast, n_slow)
        Returns:
            Tuple[int, int, int] : Optimized (n, n_fast, n_slow)
        """
        if bounds is None:
            bounds = [(5, 30), (2, 10), (20, 60)]
            
        def objective(params):
            self.n, self.n_fast, self.n_slow = map(int, params)
            self.k_fast = 2 / (self.n_fast + 1)
            self.k_slow = 2 / (self.n_slow + 1)
            
            kama_values = self.calculate_kama(prices)
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
# End of KAMA class

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
            'mean_prior_var': 1.0,
            'beta_prior_mean': 0.0,
            'beta_prior_var': 1.0,
            'sigma_prior_shape': 2.0,
            'sigma_prior_scale': 0.1,
            'transition_prior_alpha': 2.0
        }
    
    def set_priors(self, **kwargs) -> None:
        """Update prior parameters"""
        self.priors.update(kwargs)
    
    def _initialize_from_data(self, returns: np.ndarray) -> None:
        """Initialize model parameters for n_regimes using data quantiles."""
        abs_returns = np.abs(returns - np.mean(returns))
        
        # Pre-allocate arrays
        self.means = np.zeros(self.n_regimes)
        self.betas = np.zeros(self.n_regimes)
        self.stds = np.zeros(self.n_regimes)
        
        # Sort and split once
        sorted_indices = np.argsort(abs_returns)
        splits = np.array_split(sorted_indices, self.n_regimes)
        
        # Vectorized initialization where possible
        for k, regime_indices in enumerate(splits):
            if len(regime_indices) > 2:
                regime_returns = returns[regime_indices]
                self.means[k] = np.mean(regime_returns)
                self.stds[k] = max(np.std(regime_returns), 0.001)
                
                # Efficient AR coefficient estimation
                y = regime_returns[1:]
                x = regime_returns[:-1]
                x_std = np.std(x)
                if x_std > 1e-8:
                    corr = np.corrcoef(x, y)[0, 1]
                    self.betas[k] = corr * (np.std(y) / x_std) * 0.1
            else:
                # Fallback priors
                self.means[k] = self.priors['mean_prior_mean']
                self.betas[k] = self.priors['beta_prior_mean']
                self.stds[k] = np.sqrt(self.priors['sigma_prior_scale'] * 
                                      self.priors['sigma_prior_shape'])
        
        # Vectorized transition matrix initialization
        self.transition_probs = np.full((self.n_regimes, self.n_regimes), 
                                       0.2 / max(self.n_regimes - 1, 1))
        np.fill_diagonal(self.transition_probs, 0.8)

    def _sample_states(self, returns: np.ndarray) -> np.ndarray:
        """Optimized forward-backward sampling using vectorization"""
        T = len(returns)
        n_reg = self.n_regimes
        
        # Pre-allocate
        forward = np.zeros((T, n_reg))
        states = np.zeros(T, dtype=np.int32)
        
        # Compute all emission probabilities at once (vectorized)
        # Shape: (T, n_regimes)
        emissions = np.zeros((T, n_reg))
        
        # First time step - no AR term
        for k in range(n_reg):
            emissions[0, k] = norm.pdf(returns[0], self.means[k], self.stds[k])
        
        # Remaining time steps with AR term - vectorized across regimes
        for t in range(1, T):
            conditional_means = self.means + self.betas * returns[t-1]
            emissions[t] = norm.pdf(returns[t], conditional_means, self.stds)
        
        # Forward pass - vectorized
        forward[0] = emissions[0] / (emissions[0].sum() + 1e-300)
        
        for t in range(1, T):
            # Vectorized: forward[t] = emission[t] * (forward[t-1] @ transition_probs)
            forward[t] = emissions[t] * (forward[t-1] @ self.transition_probs)
            forward_sum = forward[t].sum()
            if forward_sum > 1e-300:
                forward[t] /= forward_sum
            else:
                forward[t] = 1.0 / n_reg
        
        # Backward sampling
        states[T-1] = np.random.choice(n_reg, p=forward[T-1])
        
        # Vectorized backward pass
        for t in range(T-2, -1, -1):
            # backward_prob = forward[t] * transition_probs[:, states[t+1]]
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
            # Find regime indices (exclude t=0 for AR model)
            regime_mask = (states == k) & (np.arange(len(states)) > 0)
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) > 0:
                y = returns[regime_indices]
                x = returns[regime_indices - 1]
                
                # Stack design matrix once
                X = np.column_stack([np.ones(len(y)), x])
                XtX = X.T @ X
                Xty = X.T @ y
                
                # Prior precision
                V0_inv = np.array([[1/self.priors['mean_prior_var'], 0],
                                  [0, 1/self.priors['beta_prior_var']]])
                mu0 = np.array([self.priors['mean_prior_mean'], 
                               self.priors['beta_prior_mean']])
                
                # Posterior precision and mean
                V_inv = V0_inv + XtX
                V = np.linalg.inv(V_inv)
                mu_post = V @ (V0_inv @ mu0 + Xty)
                
                # Residuals for variance
                residuals = y - X @ mu_post
                sse = np.sum(residuals**2)
                
                # Sample variance
                shape_post = self.priors['sigma_prior_shape'] + len(y) * 0.5
                scale_post = self.priors['sigma_prior_scale'] + 0.5 * sse
                sigma2 = invgamma.rvs(shape_post, scale=scale_post)
                self.stds[k] = np.sqrt(sigma2)
                
                # Sample coefficients
                try:
                    coeffs = multivariate_normal.rvs(mu_post, sigma2 * V)
                    self.means[k] = coeffs[0]
                    self.betas[k] = coeffs[1]
                except:
                    self.means[k] = mu_post[0]
                    self.betas[k] = mu_post[1]
            else:
                # Draw from prior
                self.means[k] = norm.rvs(self.priors['mean_prior_mean'], 
                                       np.sqrt(self.priors['mean_prior_var']))
                self.betas[k] = norm.rvs(self.priors['beta_prior_mean'], 
                                       np.sqrt(self.priors['beta_prior_var']))
                self.stds[k] = np.sqrt(invgamma.rvs(self.priors['sigma_prior_shape'], 
                                                  scale=self.priors['sigma_prior_scale']))
    
    def _sample_transitions(self, states: np.ndarray) -> None:
        """Optimized transition probability sampling"""
        # Vectorized counting
        state_pairs = np.column_stack([states[:-1], states[1:]])
        counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(self.n_regimes):
            mask = state_pairs[:, 0] == i
            if mask.any():
                next_states = state_pairs[mask, 1]
                counts[i] = np.bincount(next_states, minlength=self.n_regimes)
        
        # Sample from Dirichlet
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
        
        # Pre-allocate sample storage
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
        
        # Main MCMC loop
        for iteration in range(total_iter):
            # Gibbs steps
            states = self._sample_states(returns_arr)
            self._sample_parameters(returns_arr, states)
            self._sample_transitions(states)
            
            # Store samples (after burnin, with thinning)
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
        
        # Posterior means
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
        
        # Vectorized probability calculation
        regime_probs = {}
        for k in range(self.n_regimes):
            regime_probs[f'Regime_{k}'] = np.mean(states_samples == k, axis=0)
        
        return pd.DataFrame(regime_probs, index=returns.dropna().index)

    def analyze_results(self, data: Optional[pd.DataFrame] = None, 
                       true_params: Optional[Dict] = None, 
                       data_name: str = "Data") -> pd.DataFrame:
        """Comprehensive analysis of model results"""
        if data is None:
            data = pd.DataFrame({'returns': self.fitted_data})
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULTS FOR {data_name.upper()}")
        print(f"{'='*60}")
        
        regime_probs = self.get_regime_probabilities(data['returns'])
        
        # Parameter estimates
        print(f"\nEstimated Parameters:")
        print(f"{'Regime':<8} {'μ (Mean)':<12} {'β (AR)':<10} {'σ (Vol)':<12}")
        print("-" * 45)
        for i in range(self.n_regimes):
            print(f"Regime {i:<2} {self.means[i]:<11.6f} {self.betas[i]:<9.4f} {self.stds[i]:<11.6f}")
        
        # Transition probabilities
        print(f"\nTransition Probabilities:")
        header = "From\\To  " + "  ".join([f"Reg {j}" for j in range(self.n_regimes)])
        print(header)
        print("-" * len(header))
        for i in range(self.n_regimes):
            row = f"Regime {i}  " + "  ".join([f"{self.transition_probs[i,j]:.3f}" for j in range(self.n_regimes)])
            print(row)
        
        # Expected durations
        print(f"\nExpected Regime Durations:")
        regime_names = {0: 'Low Vol', 1: 'High Vol', 2: 'Extreme Vol'} if self.n_regimes == 3 else {}
        for i in range(self.n_regimes):
            duration = 1 / (1 - self.transition_probs[i,i]) if self.transition_probs[i,i] < 1 else float('inf')
            regime_label = regime_names.get(i, f'Regime {i}')
            print(f"Regime {i} ({regime_label}): {duration:.1f} days")
        
        # True vs estimated comparison
        if true_params is not None:
            print(f"\n{'='*30}")
            print("TRUE vs ESTIMATED COMPARISON")
            print(f"{'='*30}")
            print(f"\n{'Parameter':<15} {'True':<10} {'Estimated':<12} {'Error':<8}")
            print("-" * 50)
            for i in range(self.n_regimes):
                mu_error = abs(true_params['means'][i] - self.means[i])
                beta_error = abs(true_params['betas'][i] - self.betas[i])
                std_error = abs(true_params['stds'][i] - self.stds[i])
                print(f"μ_{i}             {true_params['means'][i]:<10.6f} {self.means[i]:<12.6f} {mu_error:<8.6f}")
                print(f"β_{i}             {true_params['betas'][i]:<10.4f} {self.betas[i]:<12.4f} {beta_error:<8.4f}")
                print(f"σ_{i}             {true_params['stds'][i]:<10.6f} {self.stds[i]:<12.6f} {std_error:<8.6f}")
        
        # Regime classification
        estimated_regimes = regime_probs.idxmax(axis=1).str.replace('Regime_', '').astype(int)
        regime_changes = np.diff(estimated_regimes).astype(bool).sum()
        
        print(f"\nRegime Classification:")
        for i in range(self.n_regimes):
            count = (estimated_regimes == i).sum()
            pct = count / len(estimated_regimes)
            print(f"Estimated Regime {i} periods: {count} days ({pct:.1%})")
        print(f"Number of regime changes: {regime_changes}")
        
        if 'true_regime' in data.columns:
            accuracy = (estimated_regimes.values == data['true_regime'].values).mean()
            print(f"Classification Accuracy: {accuracy:.2%}")
        
        return regime_probs

    def plot_results(self, data: Optional[pd.DataFrame] = None, 
                    regime_probs: Optional[pd.DataFrame] = None,
                    true_params: Optional[Dict] = None, 
                    data_name: str = "Data"):
        """Create comprehensive visualization plots"""
        if data is None:
            data = pd.DataFrame({'returns': self.fitted_data})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities(data['returns'])
        
        fig = plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, self.n_regimes))
        regime_assignment = regime_probs.idxmax(axis=1).str.replace('Regime_', '').astype(int)
        
        # Plot 1: Returns with regime highlighting
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(data.index, data['returns'], color='black', alpha=0.7, linewidth=0.5)
        for i in range(self.n_regimes):
            mask = regime_assignment == i
            ax1.fill_between(data.index, data['returns'].min(), data['returns'].max(),
                            where=mask, alpha=0.3, color=colors[i], label=f'Regime {i}')
        ax1.set_title(f'{data_name}: Returns with Estimated Regimes')
        ax1.set_ylabel('Daily Log Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Rolling volatility
        ax6 = plt.subplot(2, 2, 2)
        window = 30
        rolling_vol = data['returns'].rolling(window=window).std() * np.sqrt(252)
        ax6.plot(data.index, rolling_vol, color='black', alpha=0.7, 
                label=f'{window}-day Rolling Vol')
        for i in range(self.n_regimes):
            mask = regime_assignment == i
            ax6.fill_between(data.index, 0, rolling_vol.max(),
                            where=mask, alpha=0.3, color=colors[i], label=f'Regime {i}')
        ax6.set_title('Rolling Volatility vs Regime Detection')
        ax6.set_ylabel('Annualized Volatility')
        ax6.set_xlabel('Date')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        '''
        # # Plot 2: Regime probabilities
        # ax2 = plt.subplot(3, 2, 2)
        # for i in range(self.n_regimes):
        #     ax2.plot(regime_probs.index, regime_probs[f'Regime_{i}'], 
        #             label=f'Regime {i}', color=colors[i], alpha=0.8)
        # ax2.set_title('Regime Probabilities Over Time')
        # ax2.set_ylabel('Probability')
        # ax2.set_ylim(0, 1)
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)
        
        # # Plot 3: True vs Estimated (if available)
        # if 'true_regime' in data.columns:
        #     ax3 = plt.subplot(3, 2, 3)
        #     ax3.plot(data.index, data['true_regime'], 
        #             label='True Regime', color='green', linewidth=2, alpha=0.8)
        #     ax3.plot(regime_probs.index, regime_assignment, 
        #             label='Estimated Regime', color='orange', linestyle='--', alpha=0.8)
        #     ax3.set_title('True vs Estimated Regime Sequence')
        #     ax3.set_ylabel('Regime')
        #     ax3.legend()
        #     ax3.grid(True, alpha=0.3)
        '''

        # Plot 3: Return distributions
        ax4 = plt.subplot(2, 2, 3)
        for i in range(self.n_regimes):
            regime_mask = regime_assignment == i
            regime_returns = data['returns'][regime_mask]
            if len(regime_returns) > 0:
                ax4.hist(regime_returns, bins=50, alpha=0.6, color=colors[i], 
                        label=f'Regime {i} (n={len(regime_returns)})', density=True)
        ax4.set_title('Return Distributions by Regime')
        ax4.set_xlabel('Daily Returns')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Parameter posteriors
        if self.mcmc_samples is not None:
            ax5 = plt.subplot(2, 2, 4)
            for i in range(self.n_regimes):
                ax5.hist(self.mcmc_samples['means'][:, i], bins=30, alpha=0.6, 
                        label=f'μ_{i}', color=colors[i], density=True)
                if true_params is not None:
                    ax5.axvline(true_params['means'][i], color=colors[i], 
                              linestyle='--', label=f'True μ_{i}')
            ax5.set_title('Posterior Distributions of Mean Parameters')
            ax5.set_xlabel('Mean Parameter Value')
            ax5.set_ylabel('Density')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def regime_characteristics(self, data: Optional[pd.DataFrame] = None,
                              regime_probs: Optional[pd.DataFrame] = None,
                              data_name: str = "Data") -> pd.Series:
        """Detailed regime characteristics analysis"""
        if data is None:
            data = pd.DataFrame({'returns': self.fitted_data})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities(data['returns'])
        
        print(f"\n{'='*50}")
        print(f"DETAILED REGIME ANALYSIS - {data_name.upper()}")
        print(f"{'='*50}")
        
        regime_classification = regime_probs.idxmax(axis=1).str.replace('Regime_', '').astype(int)
        
        # Define regime names
        if self.n_regimes == 3:
            regime_names = {0: "Low Volatility", 1: "High Volatility", 2: "Extreme Volatility"}
        elif self.n_regimes == 2:
            regime_names = {0: "Low Volatility", 1: "High Volatility"}
        else:
            regime_names = {i: f"Regime {i}" for i in range(self.n_regimes)}
        
        for regime in range(self.n_regimes):
            mask = regime_classification == regime
            regime_returns = data['returns'][mask]
            
            if len(regime_returns) > 0:
                regime_name = regime_names[regime]
                print(f"\n{regime_name} (Regime {regime}):")
                print(f"  Observations: {len(regime_returns)} ({len(regime_returns)/len(data)*100:.1f}%)")
                print(f"  Mean Return: {regime_returns.mean():.6f} ({regime_returns.mean()*252:.2%} annualized)")
                print(f"  Volatility: {regime_returns.std():.6f} ({regime_returns.std()*np.sqrt(252):.2%} annualized)")
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
            data = pd.DataFrame({'returns': self.fitted_data})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities(data['returns'])
        
        print(f"\n{'='*50}")
        print(f"MODEL DIAGNOSTICS - {data_name.upper()}")
        print(f"{'='*50}")
        
        # Model fit statistics
        returns = data['returns'].values
        regime_assignment = regime_probs.idxmax(axis=1).str.replace('Regime_', '').astype(int).values
        
        # Vectorized log-likelihood calculation
        log_likelihood = 0
        for t in range(1, len(returns)):
            regime = regime_assignment[t]
            mean = self.means[regime] + self.betas[regime] * returns[t-1]
            log_likelihood += norm.logpdf(returns[t], mean, self.stds[regime])
        
        print(f"\nModel Fit Statistics:")
        print(f"Log-Likelihood: {log_likelihood:.2f}")
        print(f"Average Log-Likelihood: {(log_likelihood / len(returns)):.2f}")
        
        if self.mcmc_samples is not None:
            print(f"\nMCMC Convergence Diagnostics:")
            print(f"Number of samples: {len(self.mcmc_samples['means'])}")
            
            # Parameter stability
            for param_name, param_samples in [('μ', self.mcmc_samples['means']),
                                            ('β', self.mcmc_samples['betas']),
                                            ('σ', self.mcmc_samples['stds'])]:
                for i in range(self.n_regimes):
                    samples = param_samples[:, i]
                    mid_point = len(samples) // 2
                    first_half = np.mean(samples[:mid_point])
                    second_half = np.mean(samples[mid_point:])
                    print(f"{param_name}_{i} stability: First half = {first_half:.6f}, Second half = {second_half:.6f}")
            
            if trace_plots:
                n_params = self.n_regimes
                fig, axes = plt.subplots(3, n_params, figsize=(6*n_params, 8))
                if n_params == 1:
                    axes = axes.reshape(-1, 1)
                
                for i in range(n_params):
                    axes[0, i].plot(self.mcmc_samples['means'][:, i])
                    axes[0, i].set_title(f'Trace Plot: μ_{i}')
                    axes[0, i].grid(True, alpha=0.3)
                    
                    axes[1, i].plot(self.mcmc_samples['betas'][:, i])
                    axes[1, i].set_title(f'Trace Plot: β_{i}')
                    axes[1, i].grid(True, alpha=0.3)
                    
                    axes[2, i].plot(self.mcmc_samples['stds'][:, i])
                    axes[2, i].set_title(f'Trace Plot: σ_{i}')
                    axes[2, i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()

    def save_model(self, filepath: str, burnin: int, n_samples: int, thin : int) -> None:
        """Save model parameters, samples, and fitted data to a file."""
        model_state = {
            'n_regimes': self.n_regimes,
            'means': self.means,
            'betas': self.betas,
            'stds': self.stds,
            'transition_probs': self.transition_probs,
            'mcmc_samples': self.mcmc_samples,
            'priors': self.priors,
            'fitted_data': self.fitted_data,
            'burnin': burnin,
            'n_samples': n_samples,
            'thin': thin,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

    def load_model(self, filepath: str) -> None:
        """Load model parameters, samples, and fitted data from a file."""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        self.n_regimes = model_state['n_regimes']
        self.means = model_state['means']
        self.betas = model_state['betas']
        self.stds = model_state['stds']
        self.transition_probs = model_state['transition_probs']
        self.mcmc_samples = model_state['mcmc_samples']
        self.priors = model_state['priors']
        self.fitted_data = model_state['fitted_data']
# End of MarkovSwitchingModel class

class KAMA_MSR:
    """
    Combined KAMA+MSR Model for Four or Six-Regime Classification
    
    This model combines:
    - Two or three-state MSR for variance regime detection (low/high or low/medium/high variance)
    - KAMA with filter for trend detection (bullish/bearish)
    
    Default: Four regimes (2-state MSR):
    0. Low Variance + Bullish
    1. Low Variance + Bearish
    2. High Variance + Bullish
    3. High Variance + Bearish
    
    Optional: Six regimes (3-state MSR):
    0. Low Variance + Bullish
    1. Low Variance + Bearish
    2. Medium Variance + Bullish
    3. Medium Variance + Bearish
    4. High/Extreme Variance + Bullish
    5. High/Extreme Variance + Bearish
    
    Based on: Pomorski & Gorse (2022) "Improving on the Markov-Switching 
    Regression Model by the Use of an Adaptive Moving Average"
    """
    
    def __init__(self, 
                 kama_params: Optional[Dict] = None,
                 msr_params: Optional[Dict] = None,
                 filter_params: Optional[Dict] = None,
                 use_three_state_msr: bool = False):
        """
        Initialize the combined KAMA+MSR model
        
        Parameters:
        -----------
        kama_params : dict, optional
            KAMA parameters: {'n': 10, 'n_fast': 2, 'n_slow': 30}
        msr_params : dict, optional
            MSR parameters: {'n_regimes': 2, 'n_samples': 1000, 'burnin': 200}
            Note: If use_three_state_msr=True, n_regimes will be overridden to 3
        filter_params : dict, optional
            Filter parameters: {'n_lookback': 20, 'gamma': 1.0}
        use_three_state_msr : bool, default=False
            Whether to use 3-state MSR (low/medium/high variance) instead of 2-state
            Warning: Paper notes 3-state MSR can be unstable and lead to excessive regime switching
        """
        # Initialize KAMA component (assumes KAMA is in the same script)
        kama_defaults = {'n': 10, 'n_fast': 2, 'n_slow': 30}
        kama_params = kama_params or kama_defaults
        self.kama = KAMA(**kama_params)
        
        # Initialize MSR component (assumes MarkovSwitchingModel is in the same script)
        msr_defaults = {'n_regimes': 2}
        msr_params = msr_params or msr_defaults
        
        # Override n_regimes if three-state MSR is requested
        self.use_three_state_msr = use_three_state_msr
        if use_three_state_msr:
            msr_params['n_regimes'] = 3
            print("WARNING: Using 3-state MSR. Paper notes this may lead to instability and excessive switching.")
        
        self.msr = MarkovSwitchingModel(**msr_params)
        
        # Filter parameters
        filter_defaults = {'n_lookback': 20, 'gamma': 1.0}
        self.filter_params = filter_params or filter_defaults
        self.n_lookback = self.filter_params['n_lookback']
        self.gamma = self.filter_params['gamma']
        
        # Number of combined regimes (4 for 2-state MSR, 6 for 3-state MSR)
        self.n_combined_regimes = 6 if use_three_state_msr else 4
        
        # Storage for fitted results
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
        """
        Calculate the KAMA filter as per Equation 9 in the paper:
        f_t = γ * σ(KAMA_t)
        
        where σ(KAMA_t) is the standard deviation of KAMA changes over n days
        
        Parameters:
        -----------
        kama_series : pd.Series
            KAMA values
            
        Returns:
        --------
        pd.Series : Filter values
        """
        # Calculate KAMA changes (x_t = KAMA_t - KAMA_{t-1})
        kama_changes = kama_series.diff()
        
        # Calculate rolling standard deviation over n_lookback period
        kama_std = kama_changes.rolling(window=self.n_lookback).std()
        
        # Apply gamma multiplier
        filter_values = self.gamma * kama_std
        
        return filter_values
    
    def detect_kama_signals(self, kama_series: pd.Series, 
                           filter_series: pd.Series) -> pd.Series:
        """
        Detect bullish/bearish signals based on KAMA and filter
        
        From paper:
        - Bullish: KAMA rises above its low over n days by more than filter
        - Bearish: KAMA falls below its high over n days by more than filter
        
        Parameters:
        -----------
        kama_series : pd.Series
            KAMA values
        filter_series : pd.Series
            Filter values
            
        Returns:
        --------
        pd.Series : Trend signals (1 = bullish, -1 = bearish, 0 = neutral)
        """
        signals = pd.Series(0, index=kama_series.index)
        
        # Calculate rolling min and max over lookback period
        kama_low = kama_series.rolling(window=self.n_lookback).min()
        kama_high = kama_series.rolling(window=self.n_lookback).max()
        
        # Bullish signal: KAMA rises above its low by more than filter
        bullish_condition = (kama_series - kama_low) > filter_series
        
        # Bearish signal: KAMA falls below its high by more than filter  
        bearish_condition = (kama_high - kama_series) > filter_series
        
        signals[bullish_condition] = 1
        signals[bearish_condition] = -1
        
        # Forward fill to maintain signal until next change
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def fit(self, prices: pd.Series, 
            optimize_kama: bool = False,
            msr_verbose: bool = False,
            **msr_fit_kwargs) -> 'KAMA_MSR':
        """
        Fit the combined KAMA+MSR model
        
        Process:
        1. Fit MSR model on log returns to detect variance regimes
        2. Calculate KAMA on prices
        3. Calculate KAMA filter
        4. Detect bullish/bearish trends
        5. Combine into 4 or 6 regimes
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        optimize_kama : bool, default=False
            Whether to optimize KAMA parameters
        msr_verbose : bool, default=False
            Verbose output for MSR fitting
        **msr_fit_kwargs : additional arguments for MSR.fit()
        
        Returns:
        --------
        self
        """
        self.prices = prices.copy()
        self.returns = np.log(prices).diff().dropna()
        
        print("=" * 60)
        print("KAMA+MSR COMBINED MODEL FITTING")
        print(f"Mode: {'6-Regime (3-State MSR)' if self.use_three_state_msr else '4-Regime (2-State MSR)'}")
        print("=" * 60)
        
        # Step 1: Fit MSR model for variance regime detection
        print(f"\n[1/4] Fitting {self.msr.n_regimes}-state MSR model for variance regimes...")
        msr_defaults = {'n_samples': 1000, 'burnin': 200, 'verbose': msr_verbose}
        msr_defaults.update(msr_fit_kwargs)
        self.msr.fit(self.returns, **msr_defaults)
        
        # Get MSR regime probabilities
        self.msr_regime_probs = self.msr.get_regime_probabilities()
        print(f"   MSR regimes: {self.msr.n_regimes} states detected")
        
        # Step 2: Optimize KAMA if requested
        if optimize_kama:
            print("\n[2/4] Optimizing KAMA parameters...")
            optimized_params = self.kama.optimize_parameters(
                prices=self.prices,
                returns=self.returns
            )
            print(f"   Optimized KAMA params: n={optimized_params[0]}, "
                  f"n_fast={optimized_params[1]}, n_slow={optimized_params[2]}")
        else:
            print(f"\n[2/4] Using default KAMA parameters: "
                  f"n={self.kama.n}, n_fast={self.kama.n_fast}, n_slow={self.kama.n_slow}")
        
        # Step 3: Calculate KAMA and filter
        print("\n[3/4] Calculating KAMA and filter...")
        self.kama_values, self.kama_er, self.kama_sc = self.kama.calculate_kama(self.prices)
        self.filter_values = self.calculate_kama_filter(self.kama_values)
        print(f"   Filter lookback: {self.n_lookback} days, gamma: {self.gamma}")
        
        # Step 4: Detect KAMA trend signals
        print(f"\n[4/4] Classifying into {self.n_combined_regimes} regimes...")
        kama_signals = self.detect_kama_signals(self.kama_values, self.filter_values)
        
        # Step 5: Combine MSR variance regimes with KAMA trend signals
        self.regime_labels = self._classify_combined_regimes(
            self.msr_regime_probs, 
            kama_signals
        )
        
        # Calculate regime probabilities
        self.regime_probs = self._calculate_regime_probabilities()
        
        print("\nModel fitting complete!")
        self._print_regime_summary()
        
        return self
    
    def _classify_combined_regimes(self, msr_probs: pd.DataFrame, 
                                   kama_signals: pd.Series) -> pd.Series:
        """
        Classify into 4 or 6 regimes by combining MSR variance and KAMA trend
        
        2-State MSR (4 regimes):
        0: Low Variance + Bullish
        1: Low Variance + Bearish
        2: High Variance + Bullish  
        3: High Variance + Bearish
        
        3-State MSR (6 regimes):
        0: Low Variance + Bullish
        1: Low Variance + Bearish
        2: Medium Variance + Bullish
        3: Medium Variance + Bearish
        4: High/Extreme Variance + Bullish
        5: High/Extreme Variance + Bearish
        
        Parameters:
        -----------
        msr_probs : pd.DataFrame
            MSR regime probabilities
        kama_signals : pd.Series
            KAMA trend signals (1=bull, -1=bear, 0=neutral)
            
        Returns:
        --------
        pd.Series : Combined regime labels
        """
        # Align indices
        common_idx = msr_probs.index.intersection(kama_signals.index)
        msr_probs_aligned = msr_probs.loc[common_idx]
        kama_signals_aligned = kama_signals.loc[common_idx]
        
        # Determine MSR variance regime using argmax (highest probability)
        msr_regime_cols = [col for col in msr_probs_aligned.columns if col.startswith('Regime_')]
        msr_regime = msr_probs_aligned[msr_regime_cols].idxmax(axis=1).str.extract('(\d+)')[0].astype(int)
        
        # Initialize combined regimes
        combined = pd.Series(np.nan, index=common_idx, dtype='Int64')
        
        if self.use_three_state_msr:
            # 3-state MSR: 6 combined regimes
            # Regime 0: Low variance
            low_var_mask = (msr_regime == 0)
            combined.loc[low_var_mask & (kama_signals_aligned == 1)] = 0  # LV Bullish
            combined.loc[low_var_mask & (kama_signals_aligned == -1)] = 1  # LV Bearish
            
            # Regime 1: Medium variance
            med_var_mask = (msr_regime == 1)
            combined.loc[med_var_mask & (kama_signals_aligned == 1)] = 2  # MV Bullish
            combined.loc[med_var_mask & (kama_signals_aligned == -1)] = 3  # MV Bearish
            
            # Regime 2: High/Extreme variance
            high_var_mask = (msr_regime == 2)
            combined.loc[high_var_mask & (kama_signals_aligned == 1)] = 4  # HV Bullish
            combined.loc[high_var_mask & (kama_signals_aligned == -1)] = 5  # HV Bearish
        else:
            # 2-state MSR: 4 combined regimes
            # Regime 0: Low variance
            low_var_mask = (msr_regime == 0)
            combined.loc[low_var_mask & (kama_signals_aligned == 1)] = 0  # LV Bullish
            combined.loc[low_var_mask & (kama_signals_aligned == -1)] = 1  # LV Bearish
            
            # Regime 1: High variance
            high_var_mask = (msr_regime == 1)
            combined.loc[high_var_mask & (kama_signals_aligned == 1)] = 2  # HV Bullish
            combined.loc[high_var_mask & (kama_signals_aligned == -1)] = 3  # HV Bearish
        
        # Handle neutral signals - forward fill the last non-neutral regime
        combined = combined.ffill()
        
        return combined
    
    def _calculate_regime_probabilities(self) -> pd.DataFrame:
        """
        Calculate probabilities for each of the 4 or 6 regimes
        
        Combines MSR probabilities with KAMA signal certainty
        
        Returns:
        --------
        pd.DataFrame : 4 or 6-column dataframe with regime probabilities
        """
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create one-hot encoded regime probabilities
        regime_probs = pd.DataFrame(index=self.regime_labels.index)
        
        for regime in range(self.n_combined_regimes):
            regime_probs[f'Regime_{regime}'] = (self.regime_labels == regime).astype(float)
        
        return regime_probs
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Get the 4 or 6-regime probabilities
        
        Returns:
        --------
        pd.DataFrame : Regime probabilities
        """
        if self.regime_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.regime_probs
    
    def get_trading_signals(self, active_regimes: Optional[List[int]] = None) -> pd.Series:
        """
        Generate trading signals based on regime classification
        
        Paper recommendation (4-regime): Trade only in regimes 0 (LV Bullish) and 3 (HV Bearish)
        
        Parameters:
        -----------
        active_regimes : list, optional
            Which regimes to trade. If None, uses paper defaults:
            - 4-regime mode: [0, 3] (LV Bull, HV Bear)
            - 6-regime mode: [0, 5] (LV Bull, Extreme Vol Bear)
            
            4-regime: 0=LV Bull, 1=LV Bear, 2=HV Bull, 3=HV Bear
            6-regime: 0=LV Bull, 1=LV Bear, 2=MV Bull, 3=MV Bear, 4=HV Bull, 5=HV Bear
            
        Returns:
        --------
        pd.Series : Trading signals (1=long, -1=short, 0=neutral)
        """
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Set default active regimes based on mode
        if active_regimes is None:
            if self.use_three_state_msr:
                active_regimes = [0, 5]  # LV Bullish and Extreme Bearish
            else:
                active_regimes = [0, 3]  # LV Bullish and HV Bearish (paper default)
        
        signals = pd.Series(0, index=self.regime_labels.index)
        
        # Generate signals based on active regimes
        for regime in active_regimes:
            if regime >= self.n_combined_regimes:
                raise ValueError(f"Regime {regime} out of range for {self.n_combined_regimes}-regime model")
            
            # Even regimes are bullish (0, 2, 4), odd are bearish (1, 3, 5)
            if regime % 2 == 0:
                signals[self.regime_labels == regime] = 1  # Long
            else:
                signals[self.regime_labels == regime] = -1  # Short
        
        return signals
    
    def _print_regime_summary(self):
        """Print summary statistics of detected regimes"""
        if self.regime_labels is None:
            return
        
        if self.use_three_state_msr:
            regime_names = {
                0: 'Low Variance + Bullish',
                1: 'Low Variance + Bearish',
                2: 'Medium Variance + Bullish',
                3: 'Medium Variance + Bearish',
                4: 'High/Extreme Variance + Bullish',
                5: 'High/Extreme Variance + Bearish'
            }
        else:
            regime_names = {
                0: 'Low Variance + Bullish',
                1: 'Low Variance + Bearish',
                2: 'High Variance + Bullish',
                3: 'High Variance + Bearish'
            }
        
        print("\n" + "=" * 60)
        print("REGIME CLASSIFICATION SUMMARY")
        print("=" * 60)
        
        total_periods = len(self.regime_labels)
        
        for regime in range(self.n_combined_regimes):
            count = (self.regime_labels == regime).sum()
            pct = count / total_periods * 100
            print(f"Regime {regime} ({regime_names[regime]}): "
                  f"{count} periods ({pct:.1f}%)")
        
        # Regime changes
        changes = (self.regime_labels.diff() != 0).sum()
        print(f"\nTotal regime changes: {changes}")
        print(f"Average regime duration: {total_periods / (changes + 1):.1f} periods")
    
    def plot_regimes(self, figsize=(16, 12)):
        """
        Plot prices, KAMA, filter, and regime classification
        
        Parameters:
        -----------
        figsize : tuple, default=(16, 12)
            Figure size
        """        
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Define colors for regimes
        if self.use_three_state_msr:
            colors = {
                0: 'lightgreen',   # LV Bullish
                1: 'lightcoral',   # LV Bearish
                2: 'green',        # MV Bullish
                3: 'coral',        # MV Bearish
                4: 'darkgreen',    # HV Bullish
                5: 'darkred'       # HV Bearish
            }
            regime_names = {
                0: 'LV Bull', 1: 'LV Bear',
                2: 'MV Bull', 3: 'MV Bear',
                4: 'HV Bull', 5: 'HV Bear'
            }
        else:
            colors = {
                0: 'green',        # LV Bullish
                1: 'lightcoral',   # LV Bearish
                2: 'darkgreen',    # HV Bullish
                3: 'darkred'       # HV Bearish
            }
            regime_names = {
                0: 'LV Bull', 1: 'LV Bear',
                2: 'HV Bull', 3: 'HV Bear'
            }
        
        # Plot 1: Price and KAMA
        ax1 = axes[0]
        ax1.plot(self.prices.index, self.prices, label='Price', 
                color='black', alpha=0.7, linewidth=1)
        ax1.plot(self.kama_values.index, self.kama_values, 
                label='KAMA', color='blue', linewidth=2)
        
        # Color background by regime
        for regime in range(self.n_combined_regimes):
            mask = (self.regime_labels == regime)
            if mask.any():
                ax1.fill_between(self.prices.index, 
                               self.prices.min(), self.prices.max(),
                               where=mask, alpha=0.2, color=colors[regime],
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
        
        # Plot 3: MSR Regime Probabilities
        ax3 = axes[2]
        msr_colors = ['blue', 'orange', 'red'] if self.use_three_state_msr else ['blue', 'red']
        msr_labels = ['Low Variance', 'Medium Variance', 'High Variance'] if self.use_three_state_msr else ['Low Variance', 'High Variance']
        
        for i in range(self.msr.n_regimes):
            ax3.plot(self.msr_regime_probs.index, 
                    self.msr_regime_probs[f'Regime_{i}'],
                    label=msr_labels[i], color=msr_colors[i], linewidth=1.5)
        
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('MSR Variance Regime Probabilities')
        ax3.set_ylabel('Probability')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined Regime Classification
        ax4 = axes[3]
        for regime in range(self.n_combined_regimes):
            mask = (self.regime_labels == regime)
            ax4.scatter(self.regime_labels.index[mask], 
                       self.regime_labels[mask],
                       c=colors[regime], label=regime_names[regime], 
                       alpha=0.6, s=10)
        
        title = f'{"Six" if self.use_three_state_msr else "Four"}-Regime Classification'
        ax4.set_title(title)
        ax4.set_ylabel('Regime')
        ax4.set_yticks(list(range(self.n_combined_regimes)))
        ax4.set_yticklabels([regime_names[i] for i in range(self.n_combined_regimes)])
        ax4.legend(loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def optimize_filter_params(self, prices: pd.Series,
                              returns: pd.Series,
                              param_bounds: Optional[Dict] = None) -> Dict:
        """
        Optimize filter parameters (n_lookback, gamma) using regime separation
        
        Parameters:
        -----------
        prices : pd.Series
            Price series for optimization
        returns : pd.Series
            Return series for evaluation
        param_bounds : dict, optional
            Bounds for parameters: {'n_lookback': (10, 50), 'gamma': (0.5, 2.0)}
            
        Returns:
        --------
        dict : Optimized parameters
        """
        if param_bounds is None:
            param_bounds = {
                'n_lookback': (10, 50),
                'gamma': (0.5, 2.0)
            }
        
        def objective(params):
            n_lookback, gamma = int(params[0]), params[1]
            
            # Update filter params
            self.n_lookback = n_lookback
            self.gamma = gamma
            
            # Recalculate filter and signals
            filter_vals = self.calculate_kama_filter(self.kama_values)
            kama_sigs = self.detect_kama_signals(self.kama_values, filter_vals)
            regime_labels = self._classify_combined_regimes(
                self.msr_regime_probs, kama_sigs
            )
            
            # Calculate regime separation score (negative for minimization)
            # Use variance of returns within regimes vs between regimes
            regime_returns = {}
            for regime in range(self.n_combined_regimes):
                mask = (regime_labels == regime)
                if mask.sum() > 0:
                    regime_returns[regime] = returns.loc[regime_labels.index[mask]]
            
            # Calculate between-regime variance (higher is better)
            regime_means = [r.mean() for r in regime_returns.values() if len(r) > 0]
            if len(regime_means) < 2:
                return 1e10  # Invalid if fewer than 2 regimes
            
            between_var = np.var(regime_means)
            
            # Calculate within-regime variance (lower is better)
            within_var = np.mean([r.var() for r in regime_returns.values() if len(r) > 0])
            
            # Maximize between_var / within_var ratio (minimize negative)
            if within_var == 0:
                return 1e10
            
            score = -(between_var / within_var)
            
            return score
        
        print("\nOptimizing filter parameters...")
        result = minimize(
            objective,
            x0=[self.n_lookback, self.gamma],
            bounds=[param_bounds['n_lookback'], param_bounds['gamma']],
            method='L-BFGS-B'
        )
        
        optimized = {
            'n_lookback': int(result.x[0]),
            'gamma': result.x[1]
        }
        
        print(f"Optimized filter params: n_lookback={optimized['n_lookback']}, "
              f"gamma={optimized['gamma']:.3f}")
        
        # Update model with optimized params
        self.n_lookback = optimized['n_lookback']
        self.gamma = optimized['gamma']
        
        return optimized
# End of KAMA_MSR class