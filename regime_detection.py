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
# End Class

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
# End Class

class KAMA_MSR:

    pass
# End Class