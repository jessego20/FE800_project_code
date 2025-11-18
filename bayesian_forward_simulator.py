"""
Bayesian Forward Simulator for Regime-Conditional Return Forecasting

This module implements a Bayesian approach to simulate forward return paths by:
1. Computing forward regime probabilities using HMM transitions and KMRF predictions
2. Fitting regime-specific return distributions with statistical significance testing
3. Running Monte Carlo simulations with adaptive Bayesian regime updates
4. Analyzing and visualizing simulated paths against realized returns

Author: Jesse Goodman
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from scipy import stats as scipy_stats
from scipy.stats import skewnorm, t, norminvgauss
from scipy.optimize import minimize, curve_fit
import warnings
import contextlib
import io

from kama_msr import KAMA_MSR
from kmrf import KMRF

warnings.filterwarnings('ignore')


class BayesianForwardSimulator:
    """
    Simulate forward return paths using Bayesian regime updates and fitted distributions.
    
    This class combines:
    - HMM-based regime transition dynamics
    - KMRF machine learning predictions
    - Statistical distribution fitting with significance testing
    - Monte Carlo simulation with path-dependent regime updates
    
    Parameters
    ----------
    kama_msr : KAMA_MSR
        Fitted KAMA+MSR model up to day t
    kmrf : KMRF
        Statically trained KMRF model for regime prediction
    n_days : int, default=21
        Number of days to simulate forward
    alpha_confidence : float, default=0.75
        Confidence weight on KMRF likelihood vs HMM prior (0=HMM only, 1=KMRF only)
    significance_level : float, default=0.05
        Significance level for distribution selection hypothesis tests
    
    Attributes
    ----------
    forward_probs : pd.DataFrame
        Forward regime probabilities (n_days × 4)
    regime_distributions : dict
        Fitted distribution parameters for each regime
    transition_matrix : pd.DataFrame
        Laplace-smoothed transition matrix (4 × 4)
    simulated_returns : pd.DataFrame
        Simulated daily returns (n_days × n_simulations)
    """
    
    def __init__(
        self,
        kama_msr: KAMA_MSR,
        kmrf: KMRF,
        n_days: int = 21,
        alpha_confidence: float = 1.0,
        significance_level: float = 0.05
    ):
        self.kama_msr = kama_msr
        self.kmrf = kmrf
        self.n_days = n_days
        self.alpha = alpha_confidence
        self.sig_level = significance_level
        
        # Storage for predictions and diagnostics
        self.all_oos_predictions = None
        self.prediction_date = None
        
        # These will be populated by compute methods
        self.forward_probs: Optional[pd.DataFrame] = None
        self.regime_distributions: Optional[Dict[int, Dict]] = None
        self.transition_matrix: Optional[pd.DataFrame] = None
        self.simulated_returns: Optional[pd.DataFrame] = None
        self.validation_df: Optional[pd.DataFrame] = None
        
    def compute_forward_regime_probs(self) -> pd.DataFrame:
        """
        Compute forward regime probabilities using Bayesian updates.
        
        Combines:
        1. HMM transition dynamics with exponential certainty decay
        2. KMRF likelihood at t+1
        3. Steady-state convergence for long horizons
        
        Returns
        -------
        pd.DataFrame
            Forward probabilities (n_days × 4) with uncertainty factors
        """
        with contextlib.redirect_stdout(io.StringIO()):
            # Fit exponential certainty decay to regime autocorrelations
            regime_autocorrs = {}
            for k in range(1, self.n_days + 1):
                regime_autocorrs[k] = self.kama_msr.regime_labels.autocorr(lag=k)
            
            regime_autocorrs = pd.Series(regime_autocorrs)
            
            def exp_decay(delta, a, lambda_decay):
                return a * np.exp(-lambda_decay * delta)
            
            popt, _ = curve_fit(
                exp_decay, 
                regime_autocorrs.index, 
                regime_autocorrs.values, 
                p0=(0.5, 0.05)
            )
            a_fit, lambda_decay_fit = popt
            
            # Get transition matrix with Laplace smoothing
            transition_results = self.kama_msr.regime_transition_analysis(in_depth=False)
            P = transition_results[1]['transition_matrix'] + 1
            P = P.apply(lambda row: row / row.sum(), axis=1)
            self.transition_matrix = P
            
            # Compute steady-state distribution
            eigvals, eigvecs = np.linalg.eig(P.T)
            steady_state = eigvecs[:, np.isclose(eigvals, 1)]
            steady_state = steady_state[:, 0].real
            steady_state /= steady_state.sum()
            
            # Get KMRF likelihood at t+1
            # Get the current date from KAMA_MSR and find the next trading day
            current_date = self.kama_msr.regime_labels.index[-1]
            
            # Get all out-of-sample predictions from KMRF
            all_oos_predictions = self.kmrf.predict_all_oos()
            
            # Find the next trading day after current_date in the predictions
            future_dates = all_oos_predictions.index[all_oos_predictions.index > current_date]
            if len(future_dates) == 0:
                raise ValueError(
                    f"No out-of-sample predictions available after {current_date}. "
                    f"KMRF predictions range: {all_oos_predictions.index[0]} to {all_oos_predictions.index[-1]}"
                )
            
            next_trading_day = future_dates[0]
            kmrf_likelihood_t1 = all_oos_predictions.loc[next_trading_day]\
                .reset_index(drop=True).rename('')
            
            # HMM prior: P(regime_t+1 | regime_t)
            prior_t1 = P.loc[self.kama_msr.regime_labels.iloc[-1]]\
                .reset_index(drop=True).rename('')
            
            # Bayesian posterior at t+1
            unnormalized = prior_t1.pow(1 - self.alpha).multiply(
                kmrf_likelihood_t1.pow(self.alpha)
            )
            posterior_t1 = unnormalized / unnormalized.sum()
            
            # Forward propagation with uncertainty decay
            forward_probs = np.zeros((self.n_days, 4))
            uncertainty_factor = 1 - a_fit * np.exp(-lambda_decay_fit * 1)
            forward_probs[0] = (1 - uncertainty_factor) * posterior_t1 + \
                               uncertainty_factor * steady_state
            
            for i in range(1, self.n_days):
                delta = i + 1
                uncertainty_factor = 1 - a_fit * np.exp(-lambda_decay_fit * delta)
                probs_raw_delta = forward_probs[i - 1] @ P.values
                forward_probs[i] = (1 - uncertainty_factor) * probs_raw_delta + \
                                   uncertainty_factor * steady_state
        
        # Create DataFrame with proper index
        day_t = self.kama_msr.regime_labels.index[-1].strftime('%Y-%m-%d')
        new_index = self.kmrf.raw_ohlc.loc[day_t:][1:self.n_days + 1].index
        
        df = pd.DataFrame(
            forward_probs, 
            index=new_index, 
            columns=range(0, 4)
        ).rename_axis('regime', axis=1)
        df['uncertainty_factor'] = [
            1 - a_fit * np.exp(-lambda_decay_fit * delta) 
            for delta in range(0, self.n_days)
        ]
        
        self.forward_probs = df
        return df
    
    def test_significance(self, returns: pd.Series) -> Dict:
        """
        Test statistical significance of skewness and excess kurtosis.
        
        H0: skewness = 0 (symmetric distribution)
        H0: excess kurtosis = 0 (normal tail behavior)
        
        Uses Z-test for large samples (n≥150) or bootstrap for small samples.
        
        Parameters
        ----------
        returns : pd.Series
            Return data
            
        Returns
        -------
        dict
            Test results with p-values and significance flags
        """
        n = len(returns)
        skew_val = returns.skew()
        kurt_val = returns.kurtosis()
        
        if n >= 150:
            # Analytical Z-test for large samples
            se_skew = np.sqrt(6 / n)
            se_kurt = np.sqrt(24 / n)
            
            z_skew = skew_val / se_skew
            z_kurt = kurt_val / se_kurt
            
            # Two-tailed p-value
            p_skew = 2 * (1 - scipy_stats.norm.cdf(abs(z_skew)))
            p_kurt = 2 * (1 - scipy_stats.norm.cdf(abs(z_kurt)))
            
            skew_significant = p_skew < self.sig_level
            kurt_significant = p_kurt < self.sig_level
            
        else:
            # Bootstrap hypothesis test for small samples
            n_boot = 2000
            boot_skews = []
            boot_kurts = []
            
            np.random.seed(1010)
            for _ in range(n_boot):
                boot_sample = returns.sample(n=n, replace=True)
                boot_skews.append(boot_sample.skew())
                boot_kurts.append(boot_sample.kurtosis())
            
            boot_skews = np.array(boot_skews)
            boot_kurts = np.array(boot_kurts)
            
            # Center at 0 under H0
            boot_skews_centered = boot_skews - np.mean(boot_skews)
            boot_kurts_centered = boot_kurts - np.mean(boot_kurts)
            
            # P-value = proportion more extreme than observed
            p_skew = np.mean(np.abs(boot_skews_centered) >= abs(skew_val))
            p_kurt = np.mean(np.abs(boot_kurts_centered) >= abs(kurt_val))
            
            skew_significant = p_skew < self.sig_level
            kurt_significant = p_kurt < self.sig_level
        
        return {
            'skew_significant': skew_significant,
            'kurt_significant': kurt_significant,
            'skew': skew_val,
            'kurt': kurt_val,
            'p_skew': p_skew,
            'p_kurt': p_kurt,
            'n': n
        }
    
    def fit_distribution(self, returns: pd.Series, verbose: bool = True) -> Dict:
        """
        Fit appropriate distribution based on statistical significance tests.
        
        Decision tree:
        - Normal: Neither skewness nor kurtosis significant
        - Skew-Normal: Only skewness significant
        - Student-t: Only kurtosis significant
        - Normal Inverse Gaussian: Both significant
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns for a regime
        verbose : bool, default=True
            Print fitting diagnostics
            
        Returns
        -------
        dict
            Distribution parameters and metadata
        """
        sig_test = self.test_significance(returns)
        
        mu = returns.mean()
        sigma = returns.std()
        skew_val = sig_test['skew']
        kurt_val = sig_test['kurt']
        p_skew = sig_test['p_skew']
        p_kurt = sig_test['p_kurt']
        n = sig_test['n']
        
        skew_sig = sig_test['skew_significant']
        kurt_sig = sig_test['kurt_significant']
        
        if verbose:
            print(f"  N={n}, μ={mu:.5f}, σ={sigma:.5f}")
            print(f"  skew={skew_val:.3f} (p={p_skew:.4f})"
                  f"{'***' if p_skew < 0.01 else '**' if p_skew < 0.05 else ''}")
            print(f"  kurt={kurt_val:.3f} (p={p_kurt:.4f})"
                  f"{'***' if p_kurt < 0.01 else '**' if p_kurt < 0.05 else ''}")
        
        # Case 1: Normal
        if not skew_sig and not kurt_sig:
            if verbose:
                print("  → Normal distribution")
            return {
                'distribution': 'normal',
                'params': (mu, sigma),
                'loc': mu,
                'scale': sigma,
                'method': 'analytical',
                'empirical': {'mean': mu, 'std': sigma, 'skew': skew_val, 
                            'kurt': kurt_val, 'n': n}
            }
        
        # Case 2: Skew-Normal
        elif skew_sig and not kurt_sig:
            def skew_objective(a):
                delta = a / np.sqrt(1 + a**2)
                theoretical_skew = ((4 - np.pi) / 2) * (delta**3) / \
                                  (1 - 2 * delta**2 / np.pi)**(3/2)
                return (theoretical_skew - skew_val)**2
            
            result = minimize(
                skew_objective, 
                x0=np.sign(skew_val) * 2, 
                bounds=[(-20, 20)], 
                method='L-BFGS-B'
            )
            a = result.x[0]
            
            delta = a / np.sqrt(1 + a**2)
            omega = sigma / np.sqrt(1 - 2 * delta**2 / np.pi)
            xi = mu - omega * delta * np.sqrt(2 / np.pi)
            
            if verbose:
                print(f"  → Skew-Normal: a={a:.4f}")
            return {
                'distribution': 'skewnorm',
                'params': (a, xi, omega),
                'a': a,
                'loc': xi,
                'scale': omega,
                'method': 'moments',
                'empirical': {'mean': mu, 'std': sigma, 'skew': skew_val,
                            'kurt': kurt_val, 'n': n}
            }
        
        # Case 3: Student-t
        elif not skew_sig and kurt_sig:
            if kurt_val > 0.1:
                df = max(2.5, min(30, 6 / kurt_val + 4))
            else:
                df = 30
            
            if df > 2:
                scale_t = sigma * np.sqrt((df - 2) / df)
            else:
                scale_t = sigma
            
            if verbose:
                print(f"  → Student-t: df={df:.2f}")
            return {
                'distribution': 'student_t',
                'params': (df, mu, scale_t),
                'df': df,
                'loc': mu,
                'scale': scale_t,
                'method': 'moments',
                'empirical': {'mean': mu, 'std': sigma, 'skew': skew_val,
                            'kurt': kurt_val, 'n': n}
            }
        
        # Case 4: Normal Inverse Gaussian
        else:
            if verbose:
                print("  → Normal Inverse Gaussian (both skew and kurt significant)")
            
            try:
                a, b, loc, scale = norminvgauss.fit(returns.values)
                
                if a <= 0 or abs(b) >= a or scale <= 0 or not np.isfinite(loc):
                    raise ValueError(
                        f"Invalid NIG params: a={a:.4f}, b={b:.4f}, "
                        f"loc={loc:.5f}, scale={scale:.5f}"
                    )
                
                if verbose:
                    print(f"     MLE converged: a={a:.4f}, b={b:.4f}, "
                          f"loc={loc:.5f}, scale={scale:.5f}")
                
                return {
                    'distribution': 'norminvgauss',
                    'params': (a, b, loc, scale),
                    'a': a,
                    'b': b,
                    'loc': loc,
                    'scale': scale,
                    'method': 'MLE',
                    'empirical': {'mean': mu, 'std': sigma, 'skew': skew_val,
                                'kurt': kurt_val, 'n': n}
                }
                
            except Exception as e:
                if verbose:
                    print(f"     NIG MLE failed ({e}), falling back to Student-t")
                
                df = max(3.0, min(15, 6 / kurt_val + 4) if kurt_val > 0 else 10)
                scale_t = sigma * np.sqrt((df - 2) / df)
                
                if verbose:
                    print(f"     Using Student-t: df={df:.2f}, scale={scale_t:.5f}")
                
                return {
                    'distribution': 'student_t',
                    'params': (df, mu, scale_t),
                    'df': df,
                    'loc': mu,
                    'scale': scale_t,
                    'method': 'moments_fallback',
                    'empirical': {'mean': mu, 'std': sigma, 'skew': skew_val,
                                'kurt': kurt_val, 'n': n}
                }
    
    def fit_regime_distributions(self, verbose: bool = True) -> Dict[int, Dict]:
        """
        Fit distributions for each regime with statistical significance testing.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print fitting diagnostics
            
        Returns
        -------
        dict
            {regime_id: distribution_params}
        """
        df = self.kama_msr.regime_labels.to_frame('regime')\
            .join(self.kama_msr.returns.rename('return')).dropna()
        
        distributions = {}
        
        if verbose:
            print("\n" + "="*80)
            print("FITTING DISTRIBUTIONS BY REGIME (WITH SIGNIFICANCE TESTING)")
            print("="*80)
        
        for regime in sorted(df['regime'].unique()):
            if verbose:
                print(f"\nRegime {regime}:")
            regime_returns = df[df['regime'] == regime]['return']
            distributions[regime] = self.fit_distribution(regime_returns, verbose)
        
        if verbose:
            print("\n" + "="*80)
        
        self.regime_distributions = distributions
        return distributions
    
    def validate_distributions(self) -> pd.DataFrame:
        """
        Validate and summarize fitted distribution parameters.
        
        Returns
        -------
        pd.DataFrame
            Validation summary showing parameter validity for each regime
        """
        if self.regime_distributions is None:
            raise ValueError("Must call fit_regime_distributions() first")
        
        validation = []
        
        for regime, params in self.regime_distributions.items():
            dist_type = params['distribution']
            
            if dist_type == 'normal':
                loc, scale = params['params']
                valid = scale > 0 and np.isfinite(loc) and np.isfinite(scale)
                
            elif dist_type == 'skewnorm':
                a, loc, scale = params['params']
                valid = scale > 0 and all(np.isfinite([a, loc, scale]))
                
            elif dist_type == 'student_t':
                df, loc, scale = params['params']
                valid = df > 0 and scale > 0 and all(np.isfinite([df, loc, scale]))
            
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params['params']
                valid = (a > 0 and abs(b) < a and scale > 0 and 
                        all(np.isfinite([a, b, loc, scale])))
                
            else:
                valid = False
            
            validation.append({
                'regime': regime,
                'distribution': dist_type,
                'valid': valid,
                'loc': params.get('loc'),
                'scale': params.get('scale'),
                'df': params.get('df', np.nan),
                'a': params.get('a', np.nan),
                'b': params.get('b', np.nan),
                'n_obs': params['empirical']['n']
            })
        
        df = pd.DataFrame(validation)
        
        print("\n" + "="*80)
        print("DISTRIBUTION PARAMETER VALIDATION")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        if not df['valid'].all():
            invalid = df[~df['valid']]
            print(f"\n⚠️  WARNING: {len(invalid)} regime(s) have invalid parameters!")
            print(invalid.to_string(index=False))
        else:
            print("\n✓ All distributions have valid parameters")
        
        self.validation_df = df
        return df
    
    def sample_from_distribution(
        self, 
        dist_params: Dict, 
        size: int = 1, 
        max_return: float = 0.5
    ) -> np.ndarray:
        """
        Sample from fitted distribution with safeguards.
        
        Parameters
        ----------
        dist_params : dict
            Distribution parameters
        size : int
            Number of samples
        max_return : float
            Maximum absolute daily return (clipping bound)
            
        Returns
        -------
        np.ndarray
            Random samples
        """
        dist_type = dist_params['distribution']
        params = dist_params['params']
        
        if dist_type == 'normal':
            loc, scale = params
            if not (np.isfinite(loc) and np.isfinite(scale) and scale > 0):
                raise ValueError(f"Invalid normal params: loc={loc}, scale={scale}")
            samples = np.random.normal(loc, scale, size=size)
        
        elif dist_type == 'skewnorm':
            a, loc, scale = params
            if not (np.isfinite(a) and np.isfinite(loc) and 
                   np.isfinite(scale) and scale > 0):
                raise ValueError(
                    f"Invalid skewnorm params: a={a}, loc={loc}, scale={scale}"
                )
            samples = skewnorm.rvs(a, loc=loc, scale=scale, size=size)
        
        elif dist_type == 'student_t':
            df, loc, scale = params
            if not (np.isfinite(df) and df > 0 and np.isfinite(loc) and 
                   np.isfinite(scale) and scale > 0):
                raise ValueError(
                    f"Invalid student_t params: df={df}, loc={loc}, scale={scale}"
                )
            samples = t.rvs(df, loc=loc, scale=scale, size=size)
        
        elif dist_type == 'norminvgauss':
            a, b, loc, scale = params
            
            if not (np.isfinite(a) and a > 0):
                raise ValueError(f"Invalid a={a}, must be > 0 and finite")
            if not (np.isfinite(b) and abs(b) < a):
                raise ValueError(f"Invalid b={b}, must satisfy |b| < a={a}")
            if not (np.isfinite(scale) and scale > 0):
                raise ValueError(f"Invalid scale={scale}, must be > 0 and finite")
            if not np.isfinite(loc):
                raise ValueError(f"Invalid loc={loc}, must be finite")
            
            try:
                samples = norminvgauss.rvs(a, b, loc=loc, scale=scale, size=size)
            except Exception as e:
                raise ValueError(
                    f"norminvgauss.rvs failed with params a={a}, b={b}, "
                    f"loc={loc}, scale={scale}: {e}"
                )
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # Clip extreme values
        samples = np.clip(samples, -max_return, max_return)
        
        if not np.all(np.isfinite(samples)):
            n_invalid = np.sum(~np.isfinite(samples))
            raise ValueError(
                f"Generated {n_invalid} non-finite samples from {dist_type}"
            )
        
        return samples
    
    def simulate(
        self, 
        n_simulations: int = 10000, 
        random_seed: int = 1010
    ) -> pd.DataFrame:
        """
        Simulate forward return paths with Bayesian regime updates.
        
        Parameters
        ----------
        n_simulations : int, default=10000
            Number of Monte Carlo paths
        random_seed : int, default=1010
            Random seed for reproducibility
            
        Returns
        -------
        pd.DataFrame
            Simulated daily returns (n_days × n_simulations)
        """
        if self.forward_probs is None:
            raise ValueError("Must call compute_forward_regime_probs() first")
        if self.regime_distributions is None:
            raise ValueError("Must call fit_regime_distributions() first")
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not computed")
        
        np.random.seed(random_seed)
        
        P = self.transition_matrix.values
        forward_probs_array = self.forward_probs.iloc[:, :4].values
        
        simulated_returns = np.zeros((n_simulations, self.n_days))
        
        for sim in range(n_simulations):
            # Sample initial regime from Bayesian posterior at t+1
            current_probs = forward_probs_array[0].copy()
            current_regime = np.random.choice(4, p=current_probs)
            
            for day in range(self.n_days):
                # Draw return from current regime's distribution
                dist_params = self.regime_distributions[current_regime]
                daily_return = self.sample_from_distribution(dist_params, size=1)[0]
                simulated_returns[sim, day] = daily_return
                
                # Bayesian update for next day
                if day < self.n_days - 1:
                    hmm_prior = forward_probs_array[day + 1]
                    transition_likelihood = P[current_regime]
                    
                    unnormalized = hmm_prior * transition_likelihood
                    current_probs = unnormalized / unnormalized.sum()
                    
                    current_regime = np.random.choice(4, p=current_probs)
        
        # Create DataFrame
        df = pd.DataFrame(simulated_returns).T
        df.rename(index=lambda d: f'Day t+{d+1}', inplace=True)
        df.rename(columns=lambda i: f'Sim {i+1}', inplace=True)
        
        self.simulated_returns = df
        return df
    
    def get_true_path(self) -> pd.DataFrame:
        """
        Extract true realized return path for comparison.
        
        Returns
        -------
        pd.DataFrame
            True cumulative returns over the forecast horizon
        """
        true_path = self.kmrf.raw_ohlc.droplevel(0, axis=1)['close']\
            .loc[self.kama_msr.returns.index[-1]:][:self.n_days + 1]\
            .pct_change().dropna()\
            .add(1).cumprod().sub(1)\
            .to_frame().rename(columns={'close': 'True Path'})
        
        return true_path
    
    def plot_simulated_paths(
        self, 
        n_paths_to_plot: int = 1000,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot simulated cumulative return paths vs true path.
        
        Parameters
        ----------
        n_paths_to_plot : int, default=1000
            Number of simulated paths to display (randomly sampled)
        figsize : tuple, default=(12, 6)
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.simulated_returns is None:
            raise ValueError("Must call simulate() first")
        
        # Get true path
        true_path = self.get_true_path()
        
        # Compute cumulative returns
        sim_paths = self.simulated_returns.add(1).cumprod(axis=0).subtract(1)
        sim_paths.index = true_path.index
        
        # Sample paths to plot
        if n_paths_to_plot < sim_paths.shape[1]:
            cols_to_plot = np.random.choice(
                sim_paths.columns, 
                size=n_paths_to_plot, 
                replace=False
            )
            sim_paths_plot = sim_paths[cols_to_plot]
        else:
            sim_paths_plot = sim_paths
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for col in sim_paths_plot.columns:
            ax.plot(
                sim_paths_plot.index, 
                sim_paths_plot[col], 
                color='lightblue', 
                alpha=0.3,
                linewidth=0.5
            )
        
        ax.plot(
            true_path.index, 
            true_path['True Path'], 
            color='red', 
            linewidth=2.5, 
            label='True Path',
            zorder=100
        )
        
        ax.set_title(
            f'Simulated {self.n_days}-Day Paths vs True Path\n'
            f'({n_paths_to_plot:,} paths shown)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_terminal_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 100,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot histogram of terminal (day n_days) cumulative returns.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size
        bins : int, default=100
            Number of histogram bins
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.simulated_returns is None:
            raise ValueError("Must call simulate() first")
        
        # Get terminal cumulative returns
        sim_paths = self.simulated_returns.add(1).cumprod(axis=0).subtract(1)
        terminal_returns = sim_paths.iloc[-1]
        
        # Get true terminal return
        true_path = self.get_true_path()
        true_terminal = true_path.iloc[-1, 0]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(
            terminal_returns, 
            bins=bins, 
            alpha=0.7, 
            color='steelblue',
            edgecolor='black',
            linewidth=0.5
        )
        
        ax.axvline(
            true_terminal, 
            color='red', 
            linewidth=2.5, 
            label=f'True Return: {true_terminal:.2%}',
            linestyle='--'
        )
        
        ax.axvline(
            terminal_returns.median(), 
            color='green', 
            linewidth=2, 
            label=f'Median Simulated: {terminal_returns.median():.2%}',
            linestyle=':'
        )
        
        ax.set_title(
            f'Distribution of Terminal ({self.n_days}-Day) Cumulative Returns',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Cumulative Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for simulated terminal returns.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if self.simulated_returns is None:
            raise ValueError("Must call simulate() first")
        
        sim_paths = self.simulated_returns.add(1).cumprod(axis=0).subtract(1)
        terminal_returns = sim_paths.iloc[-1]
        
        true_path = self.get_true_path()
        true_terminal = true_path.iloc[-1, 0]
        
        stats = terminal_returns.describe()
        stats['true_return'] = true_terminal
        stats['percentile_of_true'] = (terminal_returns <= true_terminal).mean() * 100
        
        print("\n" + "="*80)
        print(f"TERMINAL ({self.n_days}-DAY) RETURN STATISTICS")
        print("="*80)
        print(stats.to_string())
        print("="*80)
        print(f"\nTrue return was at the {stats['percentile_of_true']:.1f}th percentile "
              f"of simulated distribution")
        
        return stats
    
    def run_full_analysis(
        self,
        n_simulations: int = 10000,
        random_seed: int = 1010,
        plot_paths: bool = True,
        plot_terminal: bool = True,
        n_paths_to_plot: int = 1000
    ) -> Dict:
        """
        Run complete analysis pipeline.
        
        Parameters
        ----------
        n_simulations : int, default=10000
            Number of Monte Carlo simulations
        random_seed : int, default=1010
            Random seed
        plot_paths : bool, default=True
            Whether to plot cumulative paths
        plot_terminal : bool, default=True
            Whether to plot terminal distribution
        n_paths_to_plot : int, default=1000
            Number of paths to show in plot
            
        Returns
        -------
        dict
            Dictionary containing all results
        """
        print("\n" + "="*80)
        print("BAYESIAN FORWARD SIMULATION - FULL ANALYSIS")
        print("="*80)
        
        # Step 1: Forward regime probabilities
        print("\n[1/5] Computing forward regime probabilities...")
        self.compute_forward_regime_probs()
        print(f"✓ Computed {self.n_days}-day forward probabilities")
        
        # Step 2: Fit distributions
        print("\n[2/5] Fitting regime-specific distributions...")
        self.fit_regime_distributions(verbose=True)
        
        # Step 3: Validate
        print("\n[3/5] Validating distribution parameters...")
        self.validate_distributions()
        
        # Step 4: Simulate
        print(f"\n[4/5] Running {n_simulations:,} Monte Carlo simulations...")
        self.simulate(n_simulations=n_simulations, random_seed=random_seed)
        print(f"✓ Simulated {n_simulations:,} paths")
        
        # Step 5: Analysis
        print("\n[5/5] Computing summary statistics...")
        stats = self.summary_statistics()
        
        # Plotting
        if plot_paths:
            print("\nGenerating path visualization...")
            self.plot_simulated_paths(n_paths_to_plot=n_paths_to_plot)
            plt.show()
        
        if plot_terminal:
            print("Generating terminal distribution...")
            self.plot_terminal_distribution()
            plt.show()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'forward_probs': self.forward_probs,
            'regime_distributions': self.regime_distributions,
            'validation': self.validation_df,
            'simulated_returns': self.simulated_returns,
            'summary_stats': stats
        }
