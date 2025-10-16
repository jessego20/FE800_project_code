import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict
from scipy.stats import norm, invgamma, multivariate_normal
from scipy.optimize import minimize

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

class MarkovSwitchingModel:
    """
    Markov Switching Regression model using Gibbs sampling with integrated analysis.
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
        self.fitted_data = None  # Store data for analysis
        
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
        """Simple initialization based on data quantiles"""
        abs_returns = np.abs(returns - np.mean(returns))
        threshold = np.median(abs_returns)
        
        low_vol_mask = abs_returns <= threshold
        high_vol_mask = ~low_vol_mask
        
        self.means = np.zeros(2)
        self.betas = np.zeros(2)
        self.stds = np.zeros(2)
        
        # Regime 0: Low volatility
        if np.sum(low_vol_mask) > 1:
            self.means[0] = np.mean(returns[low_vol_mask])
            self.stds[0] = np.std(returns[low_vol_mask])
            if len(returns[low_vol_mask]) > 1:
                self.betas[0] = np.corrcoef(returns[low_vol_mask][1:], 
                                          returns[low_vol_mask][:-1])[0,1] * 0.1
            else:
                self.betas[0] = 0.0
        else:
            self.means[0] = 0.001
            self.betas[0] = 0.0
            self.stds[0] = 0.01
            
        # Regime 1: High volatility  
        if np.sum(high_vol_mask) > 1:
            self.means[1] = np.mean(returns[high_vol_mask])
            self.stds[1] = np.std(returns[high_vol_mask])
            if len(returns[high_vol_mask]) > 1:
                self.betas[1] = np.corrcoef(returns[high_vol_mask][1:], 
                                          returns[high_vol_mask][:-1])[0,1] * 0.1
            else:
                self.betas[1] = 0.0
        else:
            self.means[1] = 0.001
            self.betas[1] = 0.0
            self.stds[1] = 0.03
        
        self.transition_probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    
    def _sample_states(self, returns: np.ndarray) -> np.ndarray:
        """Sample regime sequence using forward-backward sampling"""
        T = len(returns)
        states = np.zeros(T, dtype=int)
        forward = np.zeros((T, self.n_regimes))
        
        for k in range(self.n_regimes):
            forward[0, k] = 0.5 * norm.pdf(returns[0], self.means[k], self.stds[k])
        forward[0] /= np.sum(forward[0])
        
        for t in range(1, T):
            for j in range(self.n_regimes):
                emission = norm.pdf(returns[t], 
                                  self.means[j] + self.betas[j] * returns[t-1], 
                                  self.stds[j])
                forward[t, j] = emission * np.sum(forward[t-1] * self.transition_probs[:, j])
            
            if np.sum(forward[t]) > 0:
                forward[t] /= np.sum(forward[t])
            else:
                forward[t] = 0.5
        
        states[T-1] = np.random.choice(self.n_regimes, p=forward[T-1])
        
        for t in range(T-2, -1, -1):
            backward_prob = np.zeros(self.n_regimes)
            for i in range(self.n_regimes):
                emission = norm.pdf(returns[t+1], 
                                  self.means[states[t+1]] + self.betas[states[t+1]] * returns[t], 
                                  self.stds[states[t+1]])
                backward_prob[i] = forward[t, i] * self.transition_probs[i, states[t+1]]
            
            if np.sum(backward_prob) > 0:
                backward_prob /= np.sum(backward_prob)
                states[t] = np.random.choice(self.n_regimes, p=backward_prob)
            else:
                states[t] = np.random.choice(self.n_regimes)
                
        return states
    
    def _sample_parameters(self, returns: np.ndarray, states: np.ndarray) -> None:
        """Sample model parameters given states"""
        for k in range(self.n_regimes):
            regime_indices = np.where(states == k)[0]
            regime_indices = regime_indices[regime_indices > 0]
            
            if len(regime_indices) > 0:
                y = returns[regime_indices]
                x = returns[regime_indices - 1]
                X = np.column_stack([np.ones(len(y)), x])
                
                V0_inv = np.diag([1/self.priors['mean_prior_var'], 
                                 1/self.priors['beta_prior_var']])
                mu0 = np.array([self.priors['mean_prior_mean'], 
                               self.priors['beta_prior_mean']])
                
                V_inv = V0_inv + X.T @ X
                V = np.linalg.inv(V_inv)
                mu_post = V @ (V0_inv @ mu0 + X.T @ y)
                
                residuals = y - X @ mu_post
                shape_post = self.priors['sigma_prior_shape'] + len(y)/2
                scale_post = self.priors['sigma_prior_scale'] + 0.5 * np.sum(residuals**2)
                
                sigma2 = invgamma.rvs(shape_post, scale=scale_post)
                self.stds[k] = np.sqrt(sigma2)
                
                coeffs = multivariate_normal.rvs(mu_post, sigma2 * V)
                self.means[k] = coeffs[0]
                self.betas[k] = coeffs[1]
                
            else:
                self.means[k] = norm.rvs(self.priors['mean_prior_mean'], 
                                       np.sqrt(self.priors['mean_prior_var']))
                self.betas[k] = norm.rvs(self.priors['beta_prior_mean'], 
                                       np.sqrt(self.priors['beta_prior_var']))
                self.stds[k] = np.sqrt(invgamma.rvs(self.priors['sigma_prior_shape'], 
                                                  scale=self.priors['sigma_prior_scale']))
    
    def _sample_transitions(self, states: np.ndarray) -> None:
        """Sample transition probabilities"""
        counts = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(len(states) - 1):
            counts[states[t], states[t+1]] += 1
        
        for i in range(self.n_regimes):
            alpha_post = counts[i] + self.priors['transition_prior_alpha']
            self.transition_probs[i] = np.random.dirichlet(alpha_post)
    
    def fit(self, returns: pd.Series, n_samples: int = 1000, burnin: int = 200, 
            thin: int = 1, verbose: bool = True) -> None:
        """Fit the model using Gibbs sampling"""
        returns_arr = returns.dropna().values
        T = len(returns_arr)
        
        # Store data for later analysis
        self.fitted_data = returns.dropna()
        
        self._initialize_from_data(returns_arr)
        
        total_iter = n_samples + burnin
        samples = {
            'means': np.zeros((n_samples, self.n_regimes)),
            'betas': np.zeros((n_samples, self.n_regimes)), 
            'stds': np.zeros((n_samples, self.n_regimes)),
            'transition_probs': np.zeros((n_samples, self.n_regimes, self.n_regimes)),
            'states': np.zeros((n_samples, T), dtype=int)
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
                samples['means'][sample_idx] = self.means.copy()
                samples['betas'][sample_idx] = self.betas.copy()
                samples['stds'][sample_idx] = self.stds.copy()
                samples['transition_probs'][sample_idx] = self.transition_probs.copy()
                samples['states'][sample_idx] = states.copy()
                sample_idx += 1
                
            if verbose and (iteration + 1) % 25 == 0:
                print(f"Iteration {iteration + 1}/{total_iter}")
        
        self.mcmc_samples = samples
        
        self.means = np.mean(samples['means'], axis=0)
        self.betas = np.mean(samples['betas'], axis=0) 
        self.stds = np.mean(samples['stds'], axis=0)
        self.transition_probs = np.mean(samples['transition_probs'], axis=0)
        
        if verbose:
            print("\nGibbs sampling completed!")
            print(f"Regime 0: μ={self.means[0]:.6f}, β={self.betas[0]:.4f}, σ={self.stds[0]:.6f}")
            print(f"Regime 1: μ={self.means[1]:.6f}, β={self.betas[1]:.4f}, σ={self.stds[1]:.6f}")
    
    def get_regime_probabilities(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Get posterior regime probabilities"""
        if self.mcmc_samples is None:
            raise ValueError("Model not fitted. Run fit() first.")
        
        if returns is None:
            returns = self.fitted_data
            
        states_samples = self.mcmc_samples['states']
        regime_probs = np.mean(states_samples, axis=0)
        
        return pd.DataFrame({
            'Regime_0': 1 - regime_probs,
            'Regime_1': regime_probs
        }, index=returns.dropna().index)
    
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
        for i in range(2):
            print(f"Regime {i:<2} {self.means[i]:<11.6f} {self.betas[i]:<9.4f} {self.stds[i]:<11.6f}")
        
        print(f"\nTransition Probabilities:")
        print(f"From\\To   Regime 0   Regime 1")
        print(f"Regime 0    {self.transition_probs[0,0]:.3f}      {self.transition_probs[0,1]:.3f}")
        print(f"Regime 1    {self.transition_probs[1,0]:.3f}      {self.transition_probs[1,1]:.3f}")
        
        # Expected regime durations
        duration_0 = 1 / (1 - self.transition_probs[0,0])
        duration_1 = 1 / (1 - self.transition_probs[1,1])
        print(f"\nExpected Regime Durations:")
        print(f"Regime 0 (Low Vol): {duration_0:.1f} days")
        print(f"Regime 1 (High Vol): {duration_1:.1f} days")
        
        # Compare with true parameters if available
        if true_params is not None:
            print(f"\n{'='*30}")
            print("TRUE vs ESTIMATED COMPARISON")
            print(f"{'='*30}")
            
            print(f"\n{'Parameter':<15} {'True':<10} {'Estimated':<12} {'Error':<8}")
            print("-" * 50)
            for i in range(2):
                mu_error = abs(true_params['means'][i] - self.means[i])
                beta_error = abs(true_params['betas'][i] - self.betas[i])
                std_error = abs(true_params['stds'][i] - self.stds[i])
                
                print(f"μ_{i}             {true_params['means'][i]:<10.6f} {self.means[i]:<12.6f} {mu_error:<8.6f}")
                print(f"β_{i}             {true_params['betas'][i]:<10.4f} {self.betas[i]:<12.4f} {beta_error:<8.4f}")
                print(f"σ_{i}             {true_params['stds'][i]:<10.6f} {self.stds[i]:<12.6f} {std_error:<8.6f}")
        
        # Regime classification
        estimated_regimes = (regime_probs['Regime_1'] > 0.5).astype(int)
        
        print(f"\nRegime Classification:")
        print(f"Estimated Regime 0 periods: {(estimated_regimes == 0).sum()} days ({(estimated_regimes == 0).mean():.1%})")
        print(f"Estimated Regime 1 periods: {(estimated_regimes == 1).sum()} days ({(estimated_regimes == 1).mean():.1%})")
        
        if 'true_regime' in data.columns:
            accuracy = (estimated_regimes.values == data['true_regime'].values).mean()
            print(f"Classification Accuracy: {accuracy:.2%}")
        
        # Volatility analysis
        regime_0_returns = data['returns'][regime_probs['Regime_0'] > 0.5]
        regime_1_returns = data['returns'][regime_probs['Regime_1'] > 0.5]
        
        if len(regime_0_returns) > 0 and len(regime_1_returns) > 0:
            vol_0 = regime_0_returns.std() * np.sqrt(252)
            vol_1 = regime_1_returns.std() * np.sqrt(252)
            
            print(f"\nEmpirical Regime Statistics:")
            print(f"Regime 0 - Mean: {regime_0_returns.mean():.6f}, Vol: {vol_0:.2%}")
            print(f"Regime 1 - Mean: {regime_1_returns.mean():.6f}, Vol: {vol_1:.2%}")
        
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
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Returns with regime highlighting
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(data.index, data['returns'], color='black', alpha=0.7, linewidth=0.5)
        high_vol_mask = regime_probs['Regime_1'] > 0.5
        ax1.fill_between(data.index, data['returns'].min(), data['returns'].max(),
                         where=high_vol_mask, alpha=0.3, color='red', 
                         label='High Vol Regime')
        ax1.set_title(f'{data_name}: Returns with Estimated Regimes')
        ax1.set_ylabel('Daily Log Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime probabilities
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(regime_probs.index, regime_probs['Regime_0'], 
                 label='Low Vol Regime', color='blue', alpha=0.8)
        ax2.plot(regime_probs.index, regime_probs['Regime_1'], 
                 label='High Vol Regime', color='red', alpha=0.8)
        ax2.set_title('Regime Probabilities Over Time')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: True vs Estimated regimes (if available)
        if 'true_regime' in data.columns:
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(data.index, data['true_regime'], 
                     label='True Regime', color='green', linewidth=2, alpha=0.8)
            estimated_regime = (regime_probs['Regime_1'] > 0.5).astype(int)
            ax3.plot(regime_probs.index, estimated_regime, 
                     label='Estimated Regime', color='orange', linestyle='--', alpha=0.8)
            ax3.set_title('True vs Estimated Regime Sequence')
            ax3.set_ylabel('Regime')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Return distributions
        ax4 = plt.subplot(3, 2, 4)
        regime_0_returns = data['returns'][regime_probs['Regime_0'] > 0.5]
        regime_1_returns = data['returns'][regime_probs['Regime_1'] > 0.5]
        
        if len(regime_0_returns) > 0:
            ax4.hist(regime_0_returns, bins=50, alpha=0.6, color='blue', 
                     label=f'Regime 0 (n={len(regime_0_returns)})', density=True)
        if len(regime_1_returns) > 0:
            ax4.hist(regime_1_returns, bins=50, alpha=0.6, color='red', 
                     label=f'Regime 1 (n={len(regime_1_returns)})', density=True)
        
        ax4.set_title('Return Distributions by Regime')
        ax4.set_xlabel('Daily Returns')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Parameter posteriors
        if self.mcmc_samples is not None:
            ax5 = plt.subplot(3, 2, 5)
            ax5.hist(self.mcmc_samples['means'][:, 0], bins=30, alpha=0.6, 
                     label='μ₀', color='blue', density=True)
            ax5.hist(self.mcmc_samples['means'][:, 1], bins=30, alpha=0.6, 
                     label='μ₁', color='red', density=True)
            
            if true_params is not None:
                ax5.axvline(true_params['means'][0], color='blue', linestyle='--', 
                           label='True μ₀')
                ax5.axvline(true_params['means'][1], color='red', linestyle='--', 
                           label='True μ₁')
            
            ax5.set_title('Posterior Distributions of Mean Parameters')
            ax5.set_xlabel('Mean Parameter Value')
            ax5.set_ylabel('Density')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Rolling volatility
        ax6 = plt.subplot(3, 2, 6)
        window = 30
        rolling_vol = data['returns'].rolling(window=window).std() * np.sqrt(252)
        ax6.plot(data.index, rolling_vol, color='black', alpha=0.7, 
                 label=f'{window}-day Rolling Vol')
        ax6.fill_between(data.index, 0, rolling_vol.max(),
                         where=regime_probs['Regime_1'] > 0.5, alpha=0.3, color='red', 
                         label='High Vol Regime')
        ax6.set_title('Rolling Volatility vs Regime Detection')
        ax6.set_ylabel('Annualized Volatility')
        ax6.set_xlabel('Date')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
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
        
        regime_classification = (regime_probs['Regime_1'] > 0.5).astype(int)
        
        for regime in [0, 1]:
            mask = regime_classification == regime
            regime_returns = data['returns'][mask]
            
            if len(regime_returns) > 0:
                regime_name = "Low Volatility" if regime == 0 else "High Volatility"
                print(f"\n{regime_name} Regime (Regime {regime}):")
                print(f"  Observations: {len(regime_returns)} ({len(regime_returns)/len(data)*100:.1f}%)")
                print(f"  Mean Return: {regime_returns.mean():.6f} ({regime_returns.mean()*252:.2%} annualized)")
                print(f"  Volatility: {regime_returns.std():.6f} ({regime_returns.std()*np.sqrt(252):.2%} annualized)")
                print(f"  Min Return: {regime_returns.min():.6f}")
                print(f"  Max Return: {regime_returns.max():.6f}")
                print(f"  Skewness: {regime_returns.skew():.3f}")
                print(f"  Excess Kurtosis: {regime_returns.kurtosis():.3f}")
        
        regime_changes = np.diff(regime_classification).astype(bool).sum()
        print(f"\nRegime Switching Frequency:")
        print(f"  Number of regime changes: {regime_changes}")
        print(f"  Average regime duration: {len(data)/(regime_changes+1):.1f} days")
        
        return regime_classification
    
    def diagnostics(self, data: Optional[pd.DataFrame] = None,
                   regime_probs: Optional[pd.DataFrame] = None,
                   data_name: str = "Data"):
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
        regime_0_mask = regime_probs['Regime_0'] > 0.5
        regime_1_mask = regime_probs['Regime_1'] > 0.5
        
        log_likelihood = 0
        for t in range(1, len(returns)):
            if regime_0_mask.iloc[t]:
                mean = self.means[0] + self.betas[0] * returns[t-1]
                ll = norm.logpdf(returns[t], mean, self.stds[0])
            else:
                mean = self.means[1] + self.betas[1] * returns[t-1]
                ll = norm.logpdf(returns[t], mean, self.stds[1])
            log_likelihood += ll
        
        print(f"\nModel Fit Statistics:")
        print(f"Approximate Log-Likelihood: {log_likelihood:.2f}")
        print(f"Approximate (Average) Log-Likelihood: {(log_likelihood / len(returns)):.2f}")

        
        # n_params = 8
        # aic = 2 * n_params - 2 * log_likelihood
        # print(f"AIC (approximate): {aic:.2f}")
        
        if self.mcmc_samples is not None:
            print(f"\nMCMC Convergence Diagnostics:")
            print(f"Number of samples: {len(self.mcmc_samples['means'])}")
            
            # Parameter stability
            for param_name, param_samples in [('μ', self.mcmc_samples['means']),
                                             ('β', self.mcmc_samples['betas']),
                                             ('σ', self.mcmc_samples['stds'])]:
                for i in range(2):
                    samples = param_samples[:, i]
                    mid_point = len(samples) // 2
                    first_half = np.mean(samples[:mid_point])
                    second_half = np.mean(samples[mid_point:])
                    print(f"{param_name}_{i} stability: First half = {first_half:.6f}, Second half = {second_half:.6f}")
            
            # Trace plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            axes[0, 0].plot(self.mcmc_samples['means'][:, 0])
            axes[0, 0].set_title('Trace Plot: μ₀')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(self.mcmc_samples['means'][:, 1])
            axes[0, 1].set_title('Trace Plot: μ₁')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].plot(self.mcmc_samples['betas'][:, 0])
            axes[0, 2].set_title('Trace Plot: β₀')
            axes[0, 2].grid(True, alpha=0.3)
            
            axes[1, 0].plot(self.mcmc_samples['betas'][:, 1])
            axes[1, 0].set_title('Trace Plot: β₁')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(self.mcmc_samples['stds'][:, 0])
            axes[1, 1].set_title('Trace Plot: σ₀')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(self.mcmc_samples['stds'][:, 1])
            axes[1, 2].set_title('Trace Plot: σ₁')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

