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
from sklearn.cluster import KMeans

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UTILITY FUNCTIONS FOR IMPROVEMENTS
# ============================================================================

def enforce_minimum_duration(regime_labels: pd.Series, 
                            min_duration: int = 5,
                            method: str = 'extend') -> pd.Series:
    """
    Enforce minimum duration constraint on regime labels while preserving NA values.
    
    Parameters:
    -----------
    regime_labels : pd.Series
        Original regime classification (may contain NaN/NA values at start)
    min_duration : int, default=5
        Minimum number of periods a regime must persist
    method : str, default='extend'
        - 'extend': Extend each new regime to last at least min_duration periods
        - 'merge': Merge short regimes with adjacent regimes
        - 'majority': Use majority voting in sliding window
    
    Returns:
    --------
    pd.Series : Filtered regime labels with minimum duration enforced, NAs preserved
    """
    if min_duration <= 1:
        return regime_labels.copy()
    
    filtered_labels = regime_labels.copy()
    
    # Find valid (non-NA) portion - use index-based approach
    valid_mask = regime_labels.notna()
    if not valid_mask.any():
        return filtered_labels
    
    # Get numeric positions for valid entries
    valid_positions = np.where(valid_mask)[0]
    first_valid_pos = valid_positions[0]
    
    # Work only on valid subset
    valid_labels = regime_labels.iloc[first_valid_pos:].copy()
    valid_indices = valid_labels.index
    
    if method == 'extend':
        current_regime = None
        regime_start_pos = 0
        
        for i, (idx, val) in enumerate(valid_labels.items()):
            if pd.isna(val):
                continue
            
            if current_regime is None:
                current_regime = val
                regime_start_pos = i
                continue
            
            if val != current_regime:
                periods_in_regime = i - regime_start_pos
                
                if periods_in_regime < min_duration:
                    valid_labels.iloc[i] = current_regime
                else:
                    current_regime = val
                    regime_start_pos = i
    
    elif method == 'merge':
        # Build segments
        segments = []
        current_regime = None
        start_pos = 0
        
        for i, val in enumerate(valid_labels):
            if pd.isna(val):
                continue
                
            if current_regime is None:
                current_regime = val
                start_pos = i
                continue
                
            if val != current_regime:
                segments.append({
                    'regime': current_regime,
                    'start': start_pos,
                    'end': i - 1,
                    'length': i - start_pos
                })
                current_regime = val
                start_pos = i
        
        if current_regime is not None:
            segments.append({
                'regime': current_regime,
                'start': start_pos,
                'end': len(valid_labels) - 1,
                'length': len(valid_labels) - start_pos
            })
        
        # Merge short segments
        for seg in segments:
            if seg['length'] < min_duration:
                # Determine merge target
                if seg is segments[0]:
                    merge_regime = segments[1]['regime'] if len(segments) > 1 else seg['regime']
                elif seg is segments[-1]:
                    merge_regime = segments[-2]['regime']
                else:
                    idx = segments.index(seg)
                    prev_len = segments[idx-1]['length']
                    next_len = segments[idx+1]['length']
                    merge_regime = segments[idx-1]['regime'] if prev_len >= next_len else segments[idx+1]['regime']
                
                # Apply merge
                for pos in range(seg['start'], seg['end'] + 1):
                    if pd.notna(valid_labels.iloc[pos]):
                        valid_labels.iloc[pos] = merge_regime
    
    elif method == 'majority':
        temp_labels = valid_labels.copy()
        for i in range(len(valid_labels)):
            if pd.isna(valid_labels.iloc[i]):
                continue
            
            window_start = max(0, i - min_duration // 2)
            window_end = min(len(valid_labels), i + min_duration // 2 + 1)
            window = valid_labels.iloc[window_start:window_end].dropna()
            
            if len(window) > 0:
                temp_labels.iloc[i] = window.mode().iloc[0]
        
        valid_labels = temp_labels
    
    # Put processed labels back (preserving NAs at beginning)
    filtered_labels.loc[valid_indices] = valid_labels
    
    return filtered_labels

def analyze_regime_durations(regime_labels: pd.Series) -> pd.DataFrame:
    """
    Analyze duration statistics for each regime (NA-safe, index-preserving).
    
    Parameters:
    -----------
    regime_labels : pd.Series
        Regime labels (may contain NAs)
    
    Returns:
    --------
    pd.DataFrame : Duration statistics by regime
    """
    # Work only on valid entries
    labels_clean = regime_labels.dropna()
    
    if len(labels_clean) == 0:
        return pd.DataFrame(columns=['regime', 'count', 'min', 'max', 'mean', 'median', 'std'])
    
    segments = []
    current_regime = labels_clean.iloc[0]
    start_idx = 0
    
    for i in range(1, len(labels_clean)):
        if labels_clean.iloc[i] != current_regime:
            segments.append({
                'regime': current_regime,
                'duration': i - start_idx
            })
            current_regime = labels_clean.iloc[i]
            start_idx = i
    
    # Last segment
    segments.append({
        'regime': current_regime,
        'duration': len(labels_clean) - start_idx
    })
    
    segments_df = pd.DataFrame(segments)
    
    if segments_df.empty:
        return pd.DataFrame(columns=['regime', 'count', 'min', 'max', 'mean', 'median', 'std'])
    
    stats = (
        segments_df.groupby('regime')['duration']
        .agg([
            ('count', 'count'),
            ('min', 'min'),
            ('max', 'max'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std')
        ])
        .reset_index()
    )
    
    return stats

def compute_segment_features(prices: pd.Series, 
                             returns: pd.Series,
                             regime_labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and volatility for each regime segment (NA-safe, index-aligned).
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    returns : pd.Series
        Return series
    regime_labels : pd.Series
        Regime labels (may contain NAs)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (slopes, volatilities) aligned with regime_labels index
    """
    # Initialize with zeros (same length as input)
    slopes = np.zeros(len(regime_labels))
    volatilities = np.zeros(len(regime_labels))
    
    # Work only on valid regime labels
    valid_mask = regime_labels.notna()
    if not valid_mask.any():
        return slopes, volatilities
    
    labels_clean = regime_labels[valid_mask]
    prices_clean = prices.loc[labels_clean.index]
    returns_clean = returns.loc[labels_clean.index]
    
    # Compute features for each segment
    current_regime = labels_clean.iloc[0]
    start_idx = 0
    
    for i in range(1, len(labels_clean) + 1):
        if i == len(labels_clean) or labels_clean.iloc[i] != current_regime:
            end_idx = i
            
            # Get segment data
            segment_prices = prices_clean.iloc[start_idx:end_idx]
            segment_returns = returns_clean.iloc[start_idx:end_idx]
            
            if len(segment_prices) > 1:
                # Compute slope
                log_prices = np.log(segment_prices.values)
                time_idx = np.arange(len(log_prices))
                slope = np.polyfit(time_idx, log_prices, 1)[0]
                
                # Compute volatility
                volatility = segment_returns.std()
            else:
                slope = 0.0
                volatility = 0.0
            
            # Map back to original index positions
            segment_indices = labels_clean.index[start_idx:end_idx]
            for idx in segment_indices:
                pos = regime_labels.index.get_loc(idx)
                slopes[pos] = slope
                volatilities[pos] = volatility
            
            if i < len(labels_clean):
                current_regime = labels_clean.iloc[i]
                start_idx = i
    
    return slopes, volatilities

def calculate_misclassification_score(regime_labels: np.ndarray,
                                      slopes: np.ndarray,
                                      volatilities: np.ndarray,
                                      train_size: float = 0.75,
                                      n_clusters: int = 4,
                                      random_seed: int = 42) -> float:
    """
    Calculate misclassification score using K-Means clustering (NA-safe version).
    
    This is the optimization criterion from the paper (Section 4.2).
    It measures how well the regime labels separate data by slope and volatility.
    
    Parameters:
    -----------
    regime_labels : np.ndarray
        Array of regime labels (0, 1, 2, 3) - may contain NaN values
    slopes : np.ndarray
        Price slopes (trend) for each time period - may contain NaN values
    volatilities : np.ndarray
        Log return volatilities for each time period - may contain NaN values
    train_size : float, default=0.75
        Proportion of data to use for K-Means training
    n_clusters : int, default=4
        Number of clusters (should match number of regimes)
    random_seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    float : Misclassification score (0 = perfect, higher = worse)
    """
    # Convert to pandas for easier NA handling
    regime_series = pd.Series(regime_labels)
    slopes_series = pd.Series(slopes)
    volatilities_series = pd.Series(volatilities)
    
    # Create feature matrix and identify valid entries
    features_df = pd.DataFrame({
        'regime': regime_series,
        'slope': slopes_series,
        'volatility': volatilities_series
    })
    
    # Drop rows with any NaN values
    valid_df = features_df.dropna()
    
    if len(valid_df) < 50:  # Need minimum data for meaningful clustering
        return 1.0
    
    # Extract clean arrays
    regime_labels_clean = valid_df['regime'].values
    X = valid_df[['slope', 'volatility']].values
    
    # Split into train and validation
    split_idx = int(len(X) * train_size)
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    labels_val = regime_labels_clean[split_idx:]
    
    if len(X_train) < 10 or len(X_val) < 10:  # Need minimum samples
        return 1.0
    
    try:
        # Fit K-Means on training data
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(np.unique(regime_labels_clean))),  # Don't exceed unique labels
            n_init=10,
            max_iter=300,
            random_state=random_seed
        )
        kmeans.fit(X_train)
        
        # Predict clusters for validation data
        cluster_predictions = kmeans.predict(X_val)
        
        # Build confusion matrix
        unique_clusters = np.unique(cluster_predictions)
        unique_regimes = np.unique(labels_val)
        
        confusion_matrix = np.zeros((len(unique_clusters), len(unique_regimes)), dtype=int)
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = (cluster_predictions == cluster_id)
            cluster_labels = labels_val[cluster_mask]
            
            for j, regime_id in enumerate(unique_regimes):
                confusion_matrix[i, j] = np.sum(cluster_labels == regime_id)
        
        # Calculate misclassification score
        misclassifications = 0
        
        for i in range(len(unique_clusters)):
            row_total = confusion_matrix[i, :].sum()
            if row_total > 0:
                dominant_count = confusion_matrix[i, :].max()
                misclassifications += (row_total - dominant_count)
        
        # Normalize by total validation samples
        total_samples = len(labels_val)
        misclassification_score = misclassifications / total_samples if total_samples > 0 else 1.0
        
        return misclassification_score
        
    except Exception as e:
        # Return worst score if clustering fails
        return 1.0

# ============================================================================
# KAMA CLASS (WITH IMPROVED OPTIMIZATION)
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
    
    def optimize_parameters(self, 
                          prices: pd.Series,
                          returns: pd.Series,
                          msr_probs: pd.DataFrame,
                          param_grid: Optional[Dict[str, List]] = None,
                          method: str = 'random',
                          n_random_trials: int = 50,
                          n_lookback: int = 20,
                          gamma: float = 1.0,
                          verbose: bool = True) -> Tuple[int, int, int, float]:
        """
        Optimize KAMA and Filter parameters using misclassification score.
        
        This implements the optimization procedure from Section 4.2 of the paper.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series for KAMA calculation
        returns : pd.Series
            Return series for evaluation
        msr_probs : pd.DataFrame
            MSR regime probabilities (from fitted MSR model)
        param_grid : dict, optional
            Custom parameter grid
        method : str, default='random'
            'random': random search with n_random_trials (RECOMMENDED)
            'coarse_to_fine': two-stage grid search
            'grid': exhaustive grid search
        n_random_trials : int, default=50
            Number of random trials
        n_lookback : int, default=20
            Lookback period for filter calculation
        gamma : float, default=1.0
            Current gamma value
        verbose : bool, default=True
            Print progress
        
        Returns:
        --------
        Tuple[int, int, int, float] : Optimized (n, n_fast, n_slow, gamma)
        """
        if param_grid is None:
            param_grid = {
                'n': [5, 7, 10, 12, 15, 20, 25, 30],
                'n_fast': [2, 3, 5, 7, 10],
                'n_slow': [20, 25, 30, 40, 50, 60],
                'gamma': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            }
        
        def objective(params):
            """Objective function: calculate misclassification score"""
            n, n_fast, n_slow, gamma_val = params
            
            try:
                # Calculate KAMA with these parameters
                direction = abs(prices - prices.shift(n))
                volatility = prices.diff().abs().rolling(n).sum()
                er = direction / volatility
                
                fast_alpha = 2 / (n_fast + 1)
                slow_alpha = 2 / (n_slow + 1)
                sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
                
                kama = pd.Series(index=prices.index, dtype=float)
                kama.iloc[n] = prices.iloc[n]
                
                for i in range(n + 1, len(prices)):
                    kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
                
                # Calculate filter
                kama_changes = kama.diff()
                filter_values = gamma_val * kama_changes.rolling(n_lookback).std()
                
                # Generate KAMA signals
                kama_low = kama.rolling(window=n_lookback).min()
                kama_high = kama.rolling(window=n_lookback).max()
                
                signals = pd.Series(0, index=kama.index)
                bullish_condition = (kama - kama_low) > filter_values
                bearish_condition = (kama_high - kama) > filter_values
                signals[bullish_condition] = 1
                signals[bearish_condition] = -1
                signals = signals.replace(0, np.nan).ffill().fillna(0)
                
                # Generate combined regimes
                common_idx = msr_probs.index.intersection(signals.index)
                msr_probs_aligned = msr_probs.loc[common_idx]
                signals_aligned = signals.loc[common_idx]
                
                msr_regime_cols = [col for col in msr_probs_aligned.columns if col.startswith('Regime_')]
                msr_regime = msr_probs_aligned[msr_regime_cols].idxmax(axis=1).str.extract('(\\d+)')[0].astype(int)
                
                n_regimes = len(msr_regime_cols)
                combined = pd.Series(np.nan, index=common_idx, dtype='Int64')
                
                for msr_state in range(n_regimes):
                    mask = (msr_regime == msr_state)
                    combined.loc[mask & (signals_aligned == 1)] = 2 * msr_state
                    combined.loc[mask & (signals_aligned == -1)] = 2 * msr_state + 1
                
                combined = combined.ffill()
                
                # Compute segment features
                slopes, volatilities = compute_segment_features(
                    prices, returns, combined
                )
                
                # Calculate misclassification score
                score = calculate_misclassification_score(
                    combined.values,
                    slopes,
                    volatilities,
                    n_clusters=n_regimes * 2
                )
                
                return score
                
            except Exception as e:
                if verbose:
                    print(f"Error with params {params}: {e}")
                return 1.0
        
        best_score = np.inf
        best_params = (self.n, self.n_fast, self.n_slow, gamma)
        
        if method == 'random':
            if verbose:
                print(f"Running random search with {n_random_trials} trials...")
            
            for trial in range(n_random_trials):
                params = (
                    random.choice(param_grid['n']),
                    random.choice(param_grid['n_fast']),
                    random.choice(param_grid['n_slow']),
                    random.choice(param_grid['gamma'])
                )
                
                score = objective(params)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    if verbose:
                        print(f"Trial {trial+1}/{n_random_trials}: "
                              f"n={params[0]}, n_fast={params[1]}, n_slow={params[2]}, "
                              f"gamma={params[3]:.2f}, score={score:.4f} **BEST**")
        
        elif method == 'coarse_to_fine':
            if verbose:
                print("Stage 1: Coarse grid search...")
            
            coarse_grid = {
                'n': [5, 10, 15, 20, 25, 30],
                'n_fast': [2, 5, 10],
                'n_slow': [20, 30, 40, 50, 60],
                'gamma': [0.5, 1.0, 1.5, 2.0]
            }
            
            coarse_candidates = list(product(
                coarse_grid['n'],
                coarse_grid['n_fast'],
                coarse_grid['n_slow'],
                coarse_grid['gamma']
            ))
            
            for i, params in enumerate(coarse_candidates):
                score = objective(params)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    if verbose:
                        print(f"  [{i+1}/{len(coarse_candidates)}] "
                              f"params={params}, score={score:.4f} **BEST**")
            
            if verbose:
                print(f"\nStage 2: Fine search around {best_params}...")
            
            n_best, nf_best, ns_best, g_best = best_params
            
            fine_grid = {
                'n': list(range(max(5, n_best-3), min(31, n_best+4))),
                'n_fast': list(range(max(2, nf_best-1), min(11, nf_best+2))),
                'n_slow': list(range(max(20, ns_best-10), min(61, ns_best+11), 5)),
                'gamma': [max(0.5, g_best-0.25), g_best, min(2.0, g_best+0.25)]
            }
            
            fine_candidates = list(product(
                fine_grid['n'],
                fine_grid['n_fast'],
                fine_grid['n_slow'],
                fine_grid['gamma']
            ))
            
            for i, params in enumerate(fine_candidates):
                if params == best_params:
                    continue
                
                score = objective(params)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    if verbose:
                        print(f"  [{i+1}/{len(fine_candidates)}] "
                              f"params={params}, score={score:.4f} **BEST**")
        
        if verbose:
            print(f"\n{'='*80}")
            print("OPTIMIZATION COMPLETE!")
            print(f"{'='*80}")
            print(f"Best parameters:")
            print(f"  n         = {best_params[0]}")
            print(f"  n_fast    = {best_params[1]}")
            print(f"  n_slow    = {best_params[2]}")
            print(f"  gamma     = {best_params[3]:.2f}")
            print(f"Best misclassification score: {best_score:.4f}")
            print(f"{'='*80}\n")
        
        # Update instance parameters
        self.n = best_params[0]
        self.n_fast = best_params[1]
        self.n_slow = best_params[2]
        self.k_fast = 2 / (self.n_fast + 1)
        self.k_slow = 2 / (self.n_slow + 1)
        
        return best_params


# ============================================================================
# MARKOV SWITCHING MODEL CLASS (UNCHANGED)
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
# KAMA+MSR COMBINED MODEL CLASS (WITH IMPROVEMENTS INTEGRATED)
# ============================================================================

class KAMA_MSR:
    """
    Combined KAMA+MSR Model for Four or Six-Regime Classification
    
    Based on: Pomorski & Gorse (2022) "Improving on the Markov-Switching 
    Regression Model by the Use of an Adaptive Moving Average"
    
    IMPROVEMENTS INCLUDED:
    - Minimum regime duration enforcement
    - Fixed KAMA parameter optimization using misclassification score
    - Duration analysis tools
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
            Whether to use 3-state MSR
        random_seed : int, optional
            Random seed for reproducibility
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
        
        # NEW: Minimum duration parameters
        self.min_regime_duration = None
        self.duration_method = 'extend'
        
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
    
    def set_minimum_duration(self, min_duration: int, method: str = 'extend'):
        """
        Set minimum regime duration constraint and apply to existing labels.
        
        Parameters:
        -----------
        min_duration : int
            Minimum number of periods a regime must persist
        method : str, default='extend'
            Enforcement method: 'extend', 'merge', or 'majority'
        """
        self.min_regime_duration = min_duration
        self.duration_method = method
        
        # Re-filter existing regime labels if already fitted
        if hasattr(self, 'regime_labels') and self.regime_labels is not None:
            self.regime_labels = enforce_minimum_duration(
                self.regime_labels, min_duration, method
            )
            self.regime_probs = self._calculate_regime_probabilities()
    
    def analyze_regime_durations(self) -> pd.DataFrame:
        """
        Analyze duration statistics for each regime.
        
        Returns:
        --------
        pd.DataFrame : Statistics with min, max, mean, median durations per regime
        """
        if not hasattr(self, 'regime_labels') or self.regime_labels is None:
            raise ValueError("Model must be fitted before analyzing durations")
        
        return analyze_regime_durations(self.regime_labels)

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

        # Use only historical data (exclude current bar) by shifting
        kama_low = kama_series.rolling(window=self.n_lookback).min().shift(1)
        kama_high = kama_series.rolling(window=self.n_lookback).max().shift(1)

        bullish_condition = (kama_series - kama_low) > filter_series
        bearish_condition = (kama_high - kama_series) > filter_series

        signals[bullish_condition] = 1
        signals[bearish_condition] = -1

        # Forward fill to maintain regime until signal change
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        # Apply 1-day lag so signal at close of day X triggers regime change on X+1
        # This shift preserves the previous regime on day X
        signals = signals.shift(1).ffill().fillna(0)

        return signals
    
    def fit(self, asset_name: str, prices: pd.Series, 
            optimize_kama: bool = True,
            kama_optimization_method: str = 'random',
            n_random_trials: int = 50,
            min_regime_duration: Optional[int] = None,
            duration_enforcement_method: str = 'extend',
            msr_verbose: bool = False,
            **msr_fit_kwargs) -> 'KAMA_MSR':
        """
        Fit the combined KAMA+MSR model
        
        Parameters:
        -----------
        asset_name : str
            Name of the asset
        prices : pd.Series
            Price series
        optimize_kama : bool, default=False
            Whether to optimize KAMA parameters using misclassification score
        kama_optimization_method : str, default='random'
            Method: 'random' (fast) or 'coarse_to_fine' (thorough)
        n_random_trials : int, default=50
            Number of random trials if using random search
        optimize_filter : bool, default=False
            Whether to optimize filter parameters
        min_regime_duration : int, optional
            If set, enforce minimum duration on regimes
        duration_enforcement_method : str, default='extend'
            Method for duration enforcement: 'extend', 'merge', or 'majority'
        msr_verbose : bool, default=False
            Verbose output for MSR fitting
        **msr_fit_kwargs : additional arguments for MSR.fit()
        """
        # Reset state to ensure clean fit
        self._reset_state()
        self.prices = prices.copy()
        self.returns = np.log(prices).diff().dropna()
        
        # Set minimum duration if requested
        if min_regime_duration is not None:
            self.min_regime_duration = min_regime_duration
            self.duration_method = duration_enforcement_method
        
        # Re-setup randomness for this fit
        self._setup_randomness()
        
        print()
        print("=" * 80)
        print(f"KAMA+MSR COMBINED MODEL FITTING for {asset_name}")
        print(f"Mode: {'6-Regime (3-State MSR)' if self.use_three_state_msr else '4-Regime (2-State MSR)'}")
        print("=" * 80)

        # Step 1: Fit MSR
        print(f"\n[1/5] Fitting {self.msr.n_regimes}-state MSR model...")
        msr_defaults = {'n_samples': 500, 'burnin': 100, 'thin': 1, 'verbose': msr_verbose}
        msr_defaults.update(msr_fit_kwargs)
        self.msr.fit(self.returns, **msr_defaults)
        self.msr_regime_probs = self.msr.get_regime_probabilities()
        
        # Step 2: Optimize KAMA if requested
        if optimize_kama:
            print(f"\n[2/5] Optimizing KAMA parameters using {kama_optimization_method} method...")
            optimized_params = self.kama.optimize_parameters(
                prices=self.prices,
                returns=self.returns,
                msr_probs=self.msr_regime_probs,
                method=kama_optimization_method,
                n_random_trials=n_random_trials,
                n_lookback=self.n_lookback,
                gamma=self.gamma,
                verbose=True
            )
            # Extract gamma from optimization
            self.gamma = optimized_params[3]
            print(f"   Updated gamma: {self.gamma:.3f}")
        else:
            print(f"\n[2/5] Using default KAMA: n={self.kama.n}, "
                  f"n_fast={self.kama.n_fast}, n_slow={self.kama.n_slow}")
        
        # Step 3: Calculate KAMA and filter
        print("\n[3/5] Calculating KAMA and filter...")
        self.kama_values, self.kama_er, self.kama_sc = self.kama.calculate_kama(self.prices)
        self.filter_values = self.calculate_kama_filter(self.kama_values)
        
        # Step 4: Classify regimes
        print(f"\n[4/5] Classifying into {self.n_combined_regimes} regimes...")
        kama_signals = self.detect_kama_signals(self.kama_values, self.filter_values)
        self.regime_labels = self._classify_combined_regimes(self.msr_regime_probs, kama_signals)
        
        # Step 5: Apply minimum duration if requested
        if self.min_regime_duration is not None:
            print(f"\n[5/5] Applying minimum duration ({self.min_regime_duration} periods, method='{self.duration_method}')...")
            
            # Show before stats
            changes_before = (self.regime_labels.diff() != 0).sum()
            print(f"   Before filtering: {changes_before} regime changes")
            
            self.regime_labels = enforce_minimum_duration(
                self.regime_labels,
                self.min_regime_duration,
                self.duration_method
            )
            
            # Show after stats
            changes_after = (self.regime_labels.diff() != 0).sum()
            print(f"   After filtering:  {changes_after} regime changes")
            print(f"   Reduction: {100*(1 - changes_after/changes_before):.1f}%")
            
            # Print duration analysis
            print("\n   Regime duration statistics:")
            duration_stats = self.analyze_regime_durations()
            print(duration_stats.to_string(index=False))
        else:
            print("\n[5/5] No minimum duration enforcement")
        
        self.regime_probs = self._calculate_regime_probabilities()
        
        print("\nModel fitting complete!")
        self._print_regime_summary()
        
        return self
    
    def _classify_combined_regimes(self, msr_probs: pd.DataFrame, 
                                   kama_signals: pd.Series) -> pd.Series:
        """Classify into 4 or 6 regimes"""
        common_idx = msr_probs.index.intersection(kama_signals.index)
        msr_probs_aligned = msr_probs.loc[common_idx]
        kama_signals_aligned = kama_signals.loc[common_idx]
        
        msr_regime_cols = [col for col in msr_probs_aligned.columns if col.startswith('Regime_')]
        msr_regime = msr_probs_aligned[msr_regime_cols].idxmax(axis=1).str.extract('(\\d+)')[0].astype(int)
        
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
        
        print("\n" + "=" * 80)
        print("REGIME CLASSIFICATION SUMMARY")
        print("=" * 80)
        
        total_periods = len(self.regime_labels)
        
        for regime in range(self.n_combined_regimes):
            count = (self.regime_labels == regime).sum()
            pct = count / total_periods * 100
            print(f"Regime {regime} ({regime_names[regime]}): {count} periods ({pct:.1f}%)")
        
        changes = (self.regime_labels.diff() != 0).sum()
        print(f"\nTotal regime changes: {changes}")
        print(f"Average regime duration: {total_periods / (changes + 1):.1f} periods")
    
    def plot_regimes(self, figsize=(16, 8), data_name: str = "Asset"):
        """Plot prices, KAMA, filter, and regime classification"""
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        regime_labels_clean = self.regime_labels.dropna()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
        
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

        ax1.set_title(f'{data_name}: Price and KAMA with Regime Classification')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime Classification
        ax4 = axes[1]
        for regime in range(self.n_combined_regimes):
            mask = (regime_labels_clean == regime)
            if mask.any():
                ax4.scatter(regime_labels_clean.index[mask], regime_labels_clean[mask],
                           c=colors[regime], label=regime_names[regime], alpha=0.6, s=10)
        
        ax4.set_title(f'{data_name}: {"Six" if self.use_three_state_msr else "Four"}-Regime Classification')
        ax4.set_ylabel('Regime')
        ax4.set_yticks(list(range(self.n_combined_regimes)))
        ax4.set_yticklabels([regime_names[i] for i in range(self.n_combined_regimes)])
        ax4.legend(loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)

        # # Plot 3: KAMA Filter
        # ax2 = axes[2]
        # ax2.plot(self.filter_values.index, self.filter_values, 
        #         label='KAMA Filter', color='purple', linewidth=1.5)
        # ax2.set_title('KAMA Filter (γ × σ(KAMA))')
        # ax2.set_ylabel('Filter Value')
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)
        
        # # Plot 4: MSR Probabilities
        # ax3 = axes[3]
        # msr_colors = ['blue', 'orange', 'red'] if self.use_three_state_msr else ['blue', 'red']
        # msr_labels = ['Low', 'Medium', 'High'] if self.use_three_state_msr else ['Low', 'High']
        
        # for i in range(self.msr.n_regimes):
        #     ax3.plot(self.msr_regime_probs.index, self.msr_regime_probs[f'Regime_{i}'],
        #             label=msr_labels[i], color=msr_colors[i], linewidth=1.5)
        
        # ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        # ax3.set_title('MSR Variance Regime Probabilities')
        # ax3.set_ylabel('Probability')
        # ax3.set_ylim(0, 1)
        # ax3.legend()
        # ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def analyze_results(self, data: Optional[pd.DataFrame] = None,
                       data_name: str = "Asset") -> pd.DataFrame:
        """Comprehensive analysis of KAMA+MSR model results"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        print(f"\n{'='*80}")
        print(f"KAMA+MSR ANALYSIS RESULTS FOR {data_name.upper()}")
        print(f"{'='*80}")
        
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
                              data_name: str = "Asset") -> pd.Series:
        """Detailed regime characteristics analysis"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities()
        
        print(f"\n{'='*80}")
        print(f"DETAILED REGIME ANALYSIS - {data_name.upper()}")
        print(f"{'='*80}")
        
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
                   data_name: str = "Asset",
                   trace_plots: bool = False):
        """Perform model diagnostics and validation"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities()
        
        print(f"\n{'='*80}")
        print(f"KAMA+MSR MODEL DIAGNOSTICS - {data_name.upper()}")
        print(f"{'='*80}")
        
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
                                    data_name: str = "Asset",
                                    figsize=(16, 12)):
        """Create comprehensive visualization"""
        if data is None:
            data = pd.DataFrame({'returns': self.returns, 'prices': self.prices})
        
        if regime_probs is None:
            regime_probs = self.get_regime_probabilities()
        
        fig = plt.figure(figsize=figsize)
        
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