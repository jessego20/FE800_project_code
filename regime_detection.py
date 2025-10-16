import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, invgamma, multivariate_normal
from typing import Tuple, List, Optional, Dict

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
    Markov Switching Regression model using Gibbs sampling as specified in the paper.
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
        
    def _set_default_priors(self) -> Dict:
        """Set default prior distributions"""
        return {
            'mean_prior_mean': 0.0,
            'mean_prior_var': 1.0,
            'beta_prior_mean': 0.0,
            'beta_prior_var': 1.0,
            'sigma_prior_shape': 2.0,
            'sigma_prior_scale': 0.1,
            'transition_prior_alpha': 2.0  # Favors persistence
        }
    
    def set_priors(self, **kwargs) -> None:
        """Update prior parameters"""
        self.priors.update(kwargs)
    
    def _initialize_from_data(self, returns: np.ndarray) -> None:
        """Simple initialization based on data quantiles"""
        # Initialize based on return volatility clustering
        abs_returns = np.abs(returns - np.mean(returns))
        threshold = np.median(abs_returns)
        
        # Low volatility regime (regime 0)
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
        
        # Initialize transition matrix (persistent regimes)
        self.transition_probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    
    def _sample_states(self, returns: np.ndarray) -> np.ndarray:
        """Sample regime sequence using forward-backward sampling"""
        T = len(returns)
        states = np.zeros(T, dtype=int)
        
        # Forward probabilities
        forward = np.zeros((T, self.n_regimes))
        
        # Initialize t=0 (no lagged return available)
        for k in range(self.n_regimes):
            forward[0, k] = 0.5 * norm.pdf(returns[0], self.means[k], self.stds[k])
        forward[0] /= np.sum(forward[0])
        
        # Forward pass t=1 to T-1
        for t in range(1, T):
            for j in range(self.n_regimes):
                emission = norm.pdf(returns[t], 
                                  self.means[j] + self.betas[j] * returns[t-1], 
                                  self.stds[j])
                forward[t, j] = emission * np.sum(forward[t-1] * self.transition_probs[:, j])
            
            if np.sum(forward[t]) > 0:
                forward[t] /= np.sum(forward[t])
            else:
                forward[t] = 0.5  # Uniform fallback
        
        # Backward sampling
        states[T-1] = np.random.choice(self.n_regimes, p=forward[T-1])
        
        for t in range(T-2, -1, -1):
            # Compute filtering probabilities conditional on next state
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
            # Get observations for regime k
            regime_indices = np.where(states == k)[0]
            regime_indices = regime_indices[regime_indices > 0]  # Need lagged observations
            
            if len(regime_indices) > 0:
                y = returns[regime_indices]  # Current returns
                x = returns[regime_indices - 1]  # Lagged returns
                X = np.column_stack([np.ones(len(y)), x])  # Design matrix
                
                # Bayesian linear regression with conjugate priors
                # Prior precision
                V0_inv = np.diag([1/self.priors['mean_prior_var'], 
                                 1/self.priors['beta_prior_var']])
                mu0 = np.array([self.priors['mean_prior_mean'], 
                               self.priors['beta_prior_mean']])
                
                # Posterior parameters
                V_inv = V0_inv + X.T @ X
                V = np.linalg.inv(V_inv)
                mu_post = V @ (V0_inv @ mu0 + X.T @ y)
                
                # Sample variance first (inverse-gamma)
                residuals = y - X @ mu_post
                shape_post = self.priors['sigma_prior_shape'] + len(y)/2
                scale_post = self.priors['sigma_prior_scale'] + 0.5 * np.sum(residuals**2)
                
                sigma2 = invgamma.rvs(shape_post, scale=scale_post)
                self.stds[k] = np.sqrt(sigma2)
                
                # Sample regression coefficients
                coeffs = multivariate_normal.rvs(mu_post, sigma2 * V)
                self.means[k] = coeffs[0]
                self.betas[k] = coeffs[1]
                
            else:
                # Sample from prior if no observations in regime
                self.means[k] = norm.rvs(self.priors['mean_prior_mean'], 
                                       np.sqrt(self.priors['mean_prior_var']))
                self.betas[k] = norm.rvs(self.priors['beta_prior_mean'], 
                                       np.sqrt(self.priors['beta_prior_var']))
                self.stds[k] = np.sqrt(invgamma.rvs(self.priors['sigma_prior_shape'], 
                                                  scale=self.priors['sigma_prior_scale']))
    
    def _sample_transitions(self, states: np.ndarray) -> None:
        """Sample transition probabilities"""
        # Count transitions
        counts = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(len(states) - 1):
            counts[states[t], states[t+1]] += 1
        
        # Sample each row from Dirichlet
        for i in range(self.n_regimes):
            alpha_post = counts[i] + self.priors['transition_prior_alpha']
            self.transition_probs[i] = np.random.dirichlet(alpha_post)
    
    def fit(self, returns: pd.Series, n_samples: int = 1000, burnin: int = 200, 
            thin: int = 1, verbose: bool = True) -> None:
        """
        Fit the model using Gibbs sampling
        
        Parameters:
        -----------
        returns : pd.Series
            Log returns data
        n_samples : int
            Number of post-burnin samples
        burnin : int  
            Number of burnin samples
        thin : int
            Thinning interval
        verbose : bool
            Print progress
        """
        returns_arr = returns.dropna().values
        T = len(returns_arr)
        
        # Initialize parameters
        self._initialize_from_data(returns_arr)
        
        # Storage for samples
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
            # Gibbs sampling steps
            states = self._sample_states(returns_arr)
            self._sample_parameters(returns_arr, states)
            self._sample_transitions(states)
            
            # Store samples after burnin
            if iteration >= burnin and (iteration - burnin) % thin == 0:
                samples['means'][sample_idx] = self.means.copy()
                samples['betas'][sample_idx] = self.betas.copy()
                samples['stds'][sample_idx] = self.stds.copy()
                samples['transition_probs'][sample_idx] = self.transition_probs.copy()
                samples['states'][sample_idx] = states.copy()
                sample_idx += 1
                
            if verbose and (iteration + 1) % 200 == 0:
                print(f"Iteration {iteration + 1}/{total_iter}")
        
        self.mcmc_samples = samples
        
        # Point estimates as posterior means
        self.means = np.mean(samples['means'], axis=0)
        self.betas = np.mean(samples['betas'], axis=0) 
        self.stds = np.mean(samples['stds'], axis=0)
        self.transition_probs = np.mean(samples['transition_probs'], axis=0)
        
        if verbose:
            print("Gibbs sampling completed!")
            print(f"Regime 0: μ={self.means[0]:.4f}, β={self.betas[0]:.4f}, σ={self.stds[0]:.4f}")
            print(f"Regime 1: μ={self.means[1]:.4f}, β={self.betas[1]:.4f}, σ={self.stds[1]:.4f}")
    
    def get_regime_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """Get posterior regime probabilities"""
        if self.mcmc_samples is None:
            raise ValueError("Model not fitted. Run fit() first.")
            
        # Use posterior mean of states across MCMC samples
        states_samples = self.mcmc_samples['states']
        regime_probs = np.mean(states_samples, axis=0)
        
        # Convert to DataFrame
        return pd.DataFrame({
            'Regime_0': 1 - regime_probs,
            'Regime_1': regime_probs
        }, index=returns.dropna().index)

class KAMARegimeSwitchingModel:
    """
    Combined KAMA and Markov Switching Regression model.
    This class integrates both models for regime detection and adaptive trend following.
    """
    def __init__(self, n_regimes: int = 2, n: int = 10, n_fast: int = 2, n_slow: int = 30):
        """
        Initialize the combined model
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes for the MSR model
        n : int
            Length of the efficiency ratio calculation period for KAMA
        n_fast : int
            Fast smoothing constant period for KAMA
        n_slow : int
            Slow smoothing constant period for KAMA
        """
        self.kama = KAMA(n, n_fast, n_slow)
        self.msr = MarkovSwitchingModel(n_regimes)
        
    def fit(self, prices: pd.Series, returns: pd.Series, optimize_kama: bool = True,
            kama_bounds: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Fit both KAMA and MSR models
        Parameters:
            prices : pd.Series
                Series of price values
            returns : pd.Series
                Series of return values
            optimize_kama : bool
                Whether to optimize KAMA parameters
            kama_bounds : List[Tuple[int, int]], optional
                Bounds for KAMA parameter optimization
        """
        if optimize_kama:
            self.kama.optimize_parameters(prices, returns, kama_bounds)
        kama_values = self.kama.calculate_kama(prices)
        kama_returns = np.log(kama_values).diff()
        valid_kama_returns = kama_returns.dropna()
        self.msr.fit(valid_kama_returns)
        
    def predict_regime(self, prices: pd.Series) -> pd.DataFrame:
        """
        Predict regimes using both KAMA and MSR
        Parameters:
            prices : pd.Series
                Series of price values
        Returns:
            pd.DataFrame : DataFrame of regime probabilities
        """
        kama_values = self.kama.calculate_kama(prices)
        kama_returns = np.log(kama_values).diff()
        valid_kama_returns = kama_returns.dropna()
        return self.msr.predict(valid_kama_returns)