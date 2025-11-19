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
from typing import Dict, Tuple, Optional, Union
from scipy import stats as scipy_stats
from scipy.stats import skewnorm, t, norminvgauss, norm
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
        # Handle case where fewer than n_days are available
        day_t = self.kama_msr.regime_labels.index[-1].strftime('%Y-%m-%d')
        available_dates = self.kmrf.raw_ohlc.loc[day_t:][1:].index
        actual_days = min(self.n_days, len(available_dates))
        
        if actual_days < self.n_days:
            print(f"WARNING: Only {actual_days} days available after {day_t}, " 
                  f"requested {self.n_days}. Using available days.")
            forward_probs = forward_probs[:actual_days]
        
        new_index = available_dates[:actual_days]
        
        df = pd.DataFrame(
            forward_probs, 
            index=new_index, 
            columns=range(0, 4)
        ).rename_axis('regime', axis=1)
        df['uncertainty_factor'] = [
            1 - a_fit * np.exp(-lambda_decay_fit * delta) 
            for delta in range(0, actual_days)
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
    
    def validate_distributions(self, verbose: bool = True) -> pd.DataFrame:
        """
        Validate fitted distributions with goodness-of-fit tests.
        
        Performs:
        - Parameter validity checks
        - Kolmogorov-Smirnov test
        - Anderson-Darling test
        - Chi-square test
        - Q-Q plot correlation
        
        Parameters
        ----------
        verbose : bool, default=True
            Print detailed validation results
        
        Returns
        -------
        pd.DataFrame
            Validation summary with goodness-of-fit statistics
        """
        if self.regime_distributions is None:
            raise ValueError("Must call fit_regime_distributions() first")
        
        from scipy import stats
        
        validation = []
        
        for regime, params in self.regime_distributions.items():
            dist_type = params['distribution']
            returns = self.kama_msr.returns[self.kama_msr.regime_labels == regime]
            
            # Parameter validity
            if dist_type == 'normal':
                loc, scale = params['params']
                valid = scale > 0 and np.isfinite(loc) and np.isfinite(scale)
                dist_obj = stats.norm(loc=loc, scale=scale)
                
            elif dist_type == 'skewnorm':
                a, loc, scale = params['params']
                valid = scale > 0 and all(np.isfinite([a, loc, scale]))
                dist_obj = stats.skewnorm(a, loc=loc, scale=scale)
                
            elif dist_type == 'student_t':
                df, loc, scale = params['params']
                valid = df > 0 and scale > 0 and all(np.isfinite([df, loc, scale]))
                dist_obj = stats.t(df, loc=loc, scale=scale)
            
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params['params']
                valid = (a > 0 and abs(b) < a and scale > 0 and 
                        all(np.isfinite([a, b, loc, scale])))
                dist_obj = stats.norminvgauss(a, b, loc=loc, scale=scale)
                
            else:
                valid = False
                dist_obj = None
            
            # Goodness-of-fit tests
            if valid and len(returns) > 5:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = stats.kstest(returns, dist_obj.cdf)
                
                # Anderson-Darling test (for normal only)
                if dist_type == 'normal':
                    ad_result = stats.anderson(returns, dist='norm')
                    ad_stat = ad_result.statistic
                    # Use 5% critical value
                    ad_pval = 0.05 if ad_stat < ad_result.critical_values[2] else 0.01
                else:
                    ad_stat = np.nan
                    ad_pval = np.nan
                
                # Chi-square test
                observed, bin_edges = np.histogram(returns, bins=min(10, len(returns)//5))
                expected = np.diff(dist_obj.cdf(bin_edges)) * len(returns)
                # Remove bins with expected < 5
                mask = expected >= 5
                if mask.sum() > 1:
                    # Normalize to ensure sums match (handle numerical precision)
                    obs_masked = observed[mask]
                    exp_masked = expected[mask]
                    exp_masked = exp_masked * (obs_masked.sum() / exp_masked.sum())  # Rescale
                    chi2_stat, chi2_pval = stats.chisquare(obs_masked, exp_masked)
                else:
                    chi2_stat, chi2_pval = np.nan, np.nan
                
                # Q-Q plot correlation
                theoretical_quantiles = dist_obj.ppf(np.linspace(0.01, 0.99, len(returns)))
                empirical_quantiles = np.sort(returns)
                qq_corr = np.corrcoef(theoretical_quantiles, empirical_quantiles)[0, 1]
                
            else:
                ks_stat, ks_pval = np.nan, np.nan
                ad_stat, ad_pval = np.nan, np.nan
                chi2_stat, chi2_pval = np.nan, np.nan
                qq_corr = np.nan
            
            validation.append({
                'regime': regime,
                'distribution': dist_type,
                'n_obs': len(returns),
                'valid': valid,
                'KS_stat': ks_stat,
                'KS_pval': ks_pval,
                'AD_stat': ad_stat,
                'AD_pval': ad_pval,
                'Chi2_stat': chi2_stat,
                'Chi2_pval': chi2_pval,
                'QQ_corr': qq_corr,
                'loc': params.get('loc'),
                'scale': params.get('scale'),
                'AIC': params.get('aic', np.nan)
            })
        
        df = pd.DataFrame(validation)
        
        if verbose:
            print("\n" + "="*80)
            print("DISTRIBUTION GOODNESS-OF-FIT VALIDATION")
            print("="*80)
            print("\nParameter Validity & Sample Size:")
            print(df[['regime', 'distribution', 'n_obs', 'valid']].to_string(index=False))
            
            print("\nGoodness-of-Fit Tests (p-values):")
            print(df[['regime', 'KS_pval', 'AD_pval', 'Chi2_pval', 'QQ_corr']].to_string(index=False))
            
            print("\nInterpretation:")
            print("  - p-value > 0.05: Distribution fits well (cannot reject)")
            print("  - p-value < 0.05: Distribution may not fit well (reject)")
            print("  - QQ_corr > 0.95: Excellent fit")
            print("  - QQ_corr > 0.90: Good fit")
            
            # Flag poor fits
            poor_fits = df[(df['KS_pval'] < 0.05) | (df['QQ_corr'] < 0.90)]
            if len(poor_fits) > 0:
                print(f"\n⚠️  WARNING: {len(poor_fits)} regime(s) have poor fit:")
                print(poor_fits[['regime', 'distribution', 'KS_pval', 'QQ_corr']].to_string(index=False))
            else:
                print("\n✓ All distributions pass goodness-of-fit tests")
            
            print("="*80)
        
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
        
        # Use actual number of days available (may be < n_days)
        actual_days = len(forward_probs_array)
        simulated_returns = np.zeros((n_simulations, actual_days))
        
        for sim in range(n_simulations):
            # Sample initial regime from Bayesian posterior at t+1
            current_probs = forward_probs_array[0].copy()
            current_regime = np.random.choice(4, p=current_probs)
            
            for day in range(actual_days):
                # Draw return from current regime's distribution
                dist_params = self.regime_distributions[current_regime]
                daily_return = self.sample_from_distribution(dist_params, size=1)[0]
                simulated_returns[sim, day] = daily_return
                
                # Bayesian update for next day
                if day < actual_days - 1:
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
        # Use actual available days (stored in forward_probs)
        actual_days = len(self.forward_probs) if self.forward_probs is not None else self.n_days
        
        true_path = self.kmrf.raw_ohlc.droplevel(0, axis=1)['close']\
            .loc[self.kama_msr.returns.index[-1]:][:actual_days + 1]\
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
    
    def plot_distribution_qq(
        self,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Q-Q plots for all regime distributions.
        
        Quantile-Quantile plots compare empirical vs. theoretical quantiles
        to visually assess goodness of fit.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 10)
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure with 4 subplots (one per regime)
        """
        if self.regime_distributions is None:
            raise ValueError("Must call fit_regime_distributions() first")
        
        from scipy import stats
        
        regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for regime in range(4):
            ax = axes[regime]
            
            # Get data and distribution
            returns = self.kama_msr.returns[self.kama_msr.regime_labels == regime]
            params = self.regime_distributions[regime]
            dist_type = params['distribution']
            
            if len(returns) < 5:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(returns)})',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
                ax.set_title(f'Regime {regime} ({regime_names[regime]})',
                           fontweight='bold')
                continue
            
            # Create theoretical distribution
            if dist_type == 'normal':
                loc, scale = params['params']
                dist_obj = stats.norm(loc=loc, scale=scale)
            elif dist_type == 'skewnorm':
                a, loc, scale = params['params']
                dist_obj = stats.skewnorm(a, loc=loc, scale=scale)
            elif dist_type == 'student_t':
                df, loc, scale = params['params']
                dist_obj = stats.t(df, loc=loc, scale=scale)
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params['params']
                dist_obj = stats.norminvgauss(a, b, loc=loc, scale=scale)
            
            # Generate Q-Q plot data
            n = len(returns)
            probabilities = np.linspace(1/(n+1), n/(n+1), n)
            theoretical_quantiles = dist_obj.ppf(probabilities)
            empirical_quantiles = np.sort(returns)
            
            # Plot
            ax.scatter(theoretical_quantiles, empirical_quantiles,
                      alpha=0.6, s=20, edgecolors='none')
            
            # Add diagonal line (perfect fit)
            min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
            max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, label='Perfect Fit')
            
            # Compute correlation
            corr = np.corrcoef(theoretical_quantiles, empirical_quantiles)[0, 1]
            
            # Format
            ax.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax.set_ylabel('Empirical Quantiles', fontsize=10)
            ax.set_title(
                f'Regime {regime} ({regime_names[regime]})\n'
                f'{dist_type}, n={len(returns)}, corr={corr:.4f}',
                fontweight='bold', fontsize=11
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.suptitle(
            'Q-Q Plots: Empirical vs. Theoretical Quantiles',
            fontsize=14, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_distribution_fit(
        self,
        figsize: Tuple[int, int] = (14, 10),
        bins: int = 50,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot histograms with fitted distribution overlays.
        
        Parameters
        ----------
        figsize : tuple, default=(14, 10)
            Figure size
        bins : int, default=50
            Number of histogram bins
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure with 4 subplots
        """
        if self.regime_distributions is None:
            raise ValueError("Must call fit_regime_distributions() first")
        
        from scipy import stats
        
        regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for regime in range(4):
            ax = axes[regime]
            
            # Get data and distribution
            returns = self.kama_msr.returns[self.kama_msr.regime_labels == regime]
            params = self.regime_distributions[regime]
            dist_type = params['distribution']
            
            if len(returns) < 5:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(returns)})',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
                ax.set_title(f'Regime {regime} ({regime_names[regime]})',
                           fontweight='bold')
                continue
            
            # Plot histogram
            ax.hist(returns, bins=bins, density=True, alpha=0.6,
                   color='steelblue', edgecolor='black', linewidth=0.5,
                   label='Empirical')
            
            # Create theoretical distribution
            if dist_type == 'normal':
                loc, scale = params['params']
                dist_obj = stats.norm(loc=loc, scale=scale)
            elif dist_type == 'skewnorm':
                a, loc, scale = params['params']
                dist_obj = stats.skewnorm(a, loc=loc, scale=scale)
            elif dist_type == 'student_t':
                df, loc, scale = params['params']
                dist_obj = stats.t(df, loc=loc, scale=scale)
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params['params']
                dist_obj = stats.norminvgauss(a, b, loc=loc, scale=scale)
            
            # Plot fitted distribution
            x = np.linspace(returns.min(), returns.max(), 200)
            ax.plot(x, dist_obj.pdf(x), 'r-', linewidth=2, label='Fitted')
            
            # Statistics
            mean_emp = returns.mean()
            std_emp = returns.std()
            skew_emp = stats.skew(returns)
            kurt_emp = stats.kurtosis(returns)
            
            # Format
            ax.set_xlabel('Return', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(
                f'Regime {regime} ({regime_names[regime]})\n'
                f'{dist_type}, n={len(returns)}',
                fontweight='bold', fontsize=11
            )
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            textstr = f'μ={mean_emp:.4f}\nσ={std_emp:.4f}\nskew={skew_emp:.2f}\nkurt={kurt_emp:.2f}'
            ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(
            'Distribution Fit: Empirical vs. Fitted',
            fontsize=14, fontweight='bold', y=0.995
        )
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
    
    def validate_simulations(
        self, 
        simulated_returns: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Validate simulated returns against fitted distributions.
        
        Checks:
        - Simulated vs. theoretical moments (mean, std, skew, kurtosis)
        - Distribution coverage tests
        - Serial correlation tests
        - Regime frequency validation
        
        Parameters
        ----------
        simulated_returns : pd.DataFrame or np.ndarray, optional
            Simulated returns to validate. If None, uses self.simulated_returns.
            Can be from copula simulation or individual simulation.
            Shape: (n_days, n_simulations) for DataFrame or any shape for ndarray
        verbose : bool, default=True
            Print validation results
            
        Returns
        -------
        dict
            Validation statistics and test results
        """
        # Use provided simulated_returns or fall back to self.simulated_returns
        if simulated_returns is None:
            if self.simulated_returns is None:
                raise ValueError("Must provide simulated_returns or call simulate() first")
            returns_to_validate = self.simulated_returns
        else:
            returns_to_validate = simulated_returns
        
        from scipy import stats
        
        results = {}
        
        # Extract all simulated returns
        if hasattr(returns_to_validate, 'values'):
            all_returns = returns_to_validate.values.flatten()
        else:
            all_returns = returns_to_validate.flatten()
        
        # Compute theoretical moments from regime distributions
        regime_probs = self.forward_probs.iloc[:, :4].mean(axis=0).values
        theoretical_mean = 0
        theoretical_var = 0
        
        for regime, prob in enumerate(regime_probs):
            dist_params = self.regime_distributions[regime]
            if dist_params['distribution'] == 'normal':
                loc, scale = dist_params['params']
                theoretical_mean += prob * loc
                theoretical_var += prob * (scale**2 + loc**2)
            elif dist_params['distribution'] == 'skewnorm':
                a, loc, scale = dist_params['params']
                theoretical_mean += prob * loc
                theoretical_var += prob * (scale**2 + loc**2)
            elif dist_params['distribution'] == 'student_t':
                df, loc, scale = dist_params['params']
                theoretical_mean += prob * loc
                if df > 2:
                    theoretical_var += prob * (scale**2 * df/(df-2) + loc**2)
        
        theoretical_var -= theoretical_mean**2
        theoretical_std = np.sqrt(theoretical_var)
        
        # Compute simulated moments
        simulated_mean = np.mean(all_returns)
        simulated_std = np.std(all_returns)
        simulated_skew = stats.skew(all_returns)
        simulated_kurt = stats.kurtosis(all_returns)
        
        # Moment tests
        results['moments'] = {
            'mean_theoretical': theoretical_mean,
            'mean_simulated': simulated_mean,
            'mean_diff': simulated_mean - theoretical_mean,
            'std_theoretical': theoretical_std,
            'std_simulated': simulated_std,
            'std_ratio': simulated_std / theoretical_std if theoretical_std > 0 else np.nan,
            'skewness': simulated_skew,
            'kurtosis': simulated_kurt
        }
        
        # Coverage test: Are simulated returns within expected range?
        # Use combined distribution (mixture)
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        coverage = {}
        for p in percentiles:
            empirical = np.percentile(all_returns, p)
            coverage[f'p{p}'] = empirical
        results['percentiles'] = coverage
        
        # Serial correlation test (Ljung-Box)
        if len(all_returns) > 20:
            lb_stat, lb_pval = stats.acorr_ljungbox(all_returns[:min(1000, len(all_returns))], 
                                                    lags=[10], return_df=False)
            results['serial_correlation'] = {
                'ljung_box_stat': lb_stat[0],
                'ljung_box_pval': lb_pval[0],
                'significant': lb_pval[0] < 0.05
            }
        
        # Normality test (on daily returns)
        if len(all_returns) > 50:
            jb_stat, jb_pval = stats.jarque_bera(all_returns[:min(5000, len(all_returns))])
            results['normality_test'] = {
                'jarque_bera_stat': jb_stat,
                'jarque_bera_pval': jb_pval,
                'is_normal': jb_pval > 0.05
            }
        
        if verbose:
            print("\n" + "="*80)
            print("SIMULATION VALIDATION")
            print("="*80)
            
            print("\nMoment Comparison:")
            print(f"  Mean:     Theoretical={theoretical_mean:.6f}, Simulated={simulated_mean:.6f}, Diff={results['moments']['mean_diff']:.6f}")
            print(f"  Std Dev:  Theoretical={theoretical_std:.6f}, Simulated={simulated_std:.6f}, Ratio={results['moments']['std_ratio']:.3f}")
            print(f"  Skewness: {simulated_skew:.4f}")
            print(f"  Kurtosis: {simulated_kurt:.4f}")
            
            print("\nDistribution Percentiles:")
            for p in [1, 5, 25, 50, 75, 95, 99]:
                print(f"  {p:>2}th: {coverage[f'p{p}']:>8.4f}")
            
            if 'serial_correlation' in results:
                print("\nSerial Correlation (Ljung-Box Test):")
                print(f"  Statistic: {results['serial_correlation']['ljung_box_stat']:.4f}")
                print(f"  p-value:   {results['serial_correlation']['ljung_box_pval']:.4f}")
                if results['serial_correlation']['significant']:
                    print("  ⚠️  WARNING: Significant serial correlation detected")
                else:
                    print("  ✓ No significant serial correlation")
            
            if 'normality_test' in results:
                print("\nNormality Test (Jarque-Bera):")
                print(f"  Statistic: {results['normality_test']['jarque_bera_stat']:.4f}")
                print(f"  p-value:   {results['normality_test']['jarque_bera_pval']:.4f}")
                if results['normality_test']['is_normal']:
                    print("  ✓ Cannot reject normality")
                else:
                    print("  ⚠️  Non-normal distribution (expected for regime-switching)")
            
            print("="*80)
        
        return results
    
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
    
    # ========================================================================
    # GAUSSIAN COPULA METHODS (Phase 3)
    # ========================================================================
    
    @staticmethod
    def simulate_multiasset_copula(
        assets_forward_probs: Dict[str, pd.DataFrame],
        assets_regime_distributions: Dict[str, Dict[int, Dict]],
        regime_correlations: Dict[int, pd.DataFrame],
        market_regime_probs: pd.DataFrame,
        regime_concordance: Dict[str, pd.DataFrame],
        market_asset: str = 'SPDR S&P 500 ETF',
        n_simulations: int = 10000,
        random_seed: int = 1010,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Simulate correlated multi-asset return paths using Gaussian Copula with Bayesian regime updates.
        
        Workflow (Mathematically Sound):
        --------------------------------
        1. Start with unconditional forward regime probabilities for all assets
           - These come from KMRF with Bayesian decay over time
           - P_t(regime) at each day t from forward probability computation
        
        2. For each simulation path:
           a. Initialize regime probabilities from day 0 unconditional priors
           b. For each day:
              - Sample market regime from current market regime distribution
              - Sample asset regimes conditionally: P(asset_regime | market_regime, asset_prior)
              - Sample correlated returns from regime-dependent copula
              - Perform Bayesian update on market regime based on observed return
              - Update asset regime priors for next day
        
        Mathematical Foundation:
        -----------------------
        Market Regime Evolution:
            P(M_t | r_{0:t-1}) ∝ P(r_{t-1} | M_{t-1}) × P(M_t | M_{t-1})
            where P(M_t | M_{t-1}) comes from KMRF forward probabilities
        
        Asset Regime Conditioning:
            P(A_t | M_t, prior) = P(A_t | M_t) × P(A_t) / Z
            where P(A_t | M_t) is the concordance matrix
                  P(A_t) is the unconditional KMRF forward probability
                  Z is normalization constant
        
        Return Generation:
            r_t ~ Copula(F_1(r_1 | A_1,t), ..., F_n(r_n | A_n,t); Σ_{M_t})
            where Σ_{M_t} is the correlation matrix for market regime M_t
        
        Parameters
        ----------
        assets_forward_probs : Dict[str, pd.DataFrame]
            Unconditional forward regime probabilities for each asset (n_days × 4)
            From KMRF with Bayesian decay - these are time-varying priors
        
        assets_regime_distributions : Dict[str, Dict[int, Dict]]
            Distribution parameters for each asset and regime
            Format: {asset: {regime_id: {'distribution': str, 'params': dict}}}
        
        regime_correlations : Dict[int, pd.DataFrame]
            Correlation matrices conditional on market regime
            Format: {regime_id: DataFrame with correlation matrix}
        
        market_regime_probs : pd.DataFrame
            Market asset's unconditional forward regime probabilities (n_days × 4)
        
        regime_concordance : Dict[str, pd.DataFrame]
            Conditional regime probabilities P(asset_regime | market_regime)
            Format: {asset: DataFrame where [i,j] = P(asset_regime=j | market_regime=i)}
        
        market_asset : str, default='SPDR S&P 500 ETF'
            Asset defining the market regime
        
        n_simulations : int, default=10000
            Number of Monte Carlo simulation paths
        
        random_seed : int, default=1010
            Random seed for reproducibility
        
        verbose : bool, default=True
            Print progress and diagnostic information
            
        Returns
        -------
        Dict[str, np.ndarray]
            Simulated daily returns for each asset
            Shape: {asset_name: (n_simulations, n_days)}
        
        Notes
        -----
        - Market asset must be in assets_forward_probs
        - All assets must have same number of days
        - Regime concordance is required (not optional in corrected version)
        - Each simulation path has independent regime evolution with Bayesian updates
        """
        np.random.seed(random_seed)
        
        # Setup
        asset_names = list(assets_forward_probs.keys())
        n_assets = len(asset_names)
        n_days = len(market_regime_probs)
        market_asset_idx = asset_names.index(market_asset)
        
        # Validate
        for asset in asset_names:
            if len(assets_forward_probs[asset]) != n_days:
                raise ValueError(f"Asset {asset} has {len(assets_forward_probs[asset])} days, expected {n_days}")
        
        for regime_id, corr_matrix in regime_correlations.items():
            if list(corr_matrix.index) != asset_names:
                raise ValueError(f"Regime {regime_id} correlation matrix doesn't match assets")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"GAUSSIAN COPULA MULTI-ASSET SIMULATION WITH BAYESIAN UPDATES")
            print(f"{'='*80}")
            print(f"Assets: {n_assets} | Days: {n_days} | Simulations: {n_simulations:,}")
            print(f"Market asset: {market_asset}")
            print(f"Regime concordance: ENABLED")
        
        # Initialize output
        simulated_returns = {asset: np.zeros((n_simulations, n_days)) for asset in asset_names}
        
        # Pre-compute Cholesky decompositions for correlation matrices
        cholesky_matrices = {}
        for regime_id, corr_matrix in regime_correlations.items():
            # Handle both DataFrame and ndarray inputs
            corr_array = corr_matrix.values if hasattr(corr_matrix, 'values') else corr_matrix
            
            try:
                cholesky_matrices[regime_id] = np.linalg.cholesky(corr_array)
            except np.linalg.LinAlgError:
                eigenvals, eigenvecs = np.linalg.eigh(corr_array)
                eigenvals = np.maximum(eigenvals, 1e-10)
                cholesky_matrices[regime_id] = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Pre-compute distribution parameters
        dist_info = {
            asset: {regime_id: assets_regime_distributions[asset][regime_id] for regime_id in range(4)}
            for asset in asset_names
        }
        
        # Extract arrays for faster access
        market_probs_array = market_regime_probs.iloc[:, :4].values
        assets_probs_arrays = {
            asset: probs.iloc[:, :4].values
            for asset, probs in assets_forward_probs.items()
        }
        concordance_arrays = {
            asset: (regime_concordance[asset].values if hasattr(regime_concordance[asset], 'values') 
                   else regime_concordance[asset])
            for asset in asset_names if asset != market_asset
        }
        
        if verbose:
            print(f"\nRunning simulations...")
        
        # Main simulation loop with Bayesian updates
        for sim_idx in range(n_simulations):
            # Initialize regime probabilities from unconditional forward probs (day 0)
            market_regime_probs_current = market_probs_array[0].copy()
            asset_regime_probs_current = {
                asset: assets_probs_arrays[asset][0].copy()
                for asset in asset_names if asset != market_asset
            }
            
            for day in range(n_days):
                # ================================================================
                # STEP 1: Sample market regime from current posterior
                # ================================================================
                market_regime = np.random.choice(4, p=market_regime_probs_current)
                
                # ================================================================
                # STEP 2: Sample asset regimes conditionally on market regime
                # ================================================================
                asset_regimes = np.zeros(n_assets, dtype=int)
                asset_regimes[market_asset_idx] = market_regime
                
                for asset_idx, asset in enumerate(asset_names):
                    if asset != market_asset:
                        # Combine unconditional prior with market regime conditioning
                        unconditional_prior = asset_regime_probs_current[asset]
                        conditional_on_market = concordance_arrays[asset][market_regime, :]
                        
                        # P(asset_regime | market_regime, prior) ∝ P(asset | market) × P(asset)
                        combined_probs = conditional_on_market * unconditional_prior
                        combined_probs = combined_probs / combined_probs.sum()
                        
                        asset_regimes[asset_idx] = np.random.choice(4, p=combined_probs)
                
                # ================================================================
                # STEP 3: Sample correlated returns using Gaussian copula
                # ================================================================
                # Get Cholesky decomposition for current market regime
                cholesky = cholesky_matrices[market_regime]
                
                # Generate independent standard normals
                z_independent = np.random.standard_normal(n_assets)
                
                # Induce correlation via Cholesky: z_correlated = L @ z_independent
                z_correlated = cholesky @ z_independent
                
                # Transform to uniform [0,1] via standard normal CDF
                u = norm.cdf(z_correlated)
                u = np.clip(u, 1e-10, 1 - 1e-10)  # Avoid numerical issues
                
                # Transform to returns using asset-specific regime distributions
                returns = np.zeros(n_assets)
                for asset_idx, asset in enumerate(asset_names):
                    regime = asset_regimes[asset_idx]
                    returns[asset_idx] = BayesianForwardSimulator._inverse_cdf_copula(
                        u[asset_idx],
                        dist_info[asset][regime]
                    )
                
                # Store returns
                for asset_idx, asset in enumerate(asset_names):
                    simulated_returns[asset][sim_idx, day] = returns[asset_idx]
                
                # ================================================================
                # STEP 4: Bayesian update for next day
                # ================================================================
                if day < n_days - 1:
                    # Update market regime probabilities based on observed return
                    market_return = returns[market_asset_idx]
                    
                    # Compute likelihoods P(return | regime) for all regimes
                    likelihoods = np.array([
                        BayesianForwardSimulator._compute_likelihood(
                            market_return,
                            dist_info[market_asset][regime_id]
                        )
                        for regime_id in range(4)
                    ])
                    
                    # Prior from KMRF forward probabilities (next day)
                    prior = market_probs_array[day + 1]
                    
                    # Bayesian update: P(regime | return) ∝ P(return | regime) × P(regime)
                    posterior = likelihoods * prior
                    posterior = posterior / posterior.sum()
                    
                    # Update for next iteration
                    market_regime_probs_current = posterior
                    
                    # Update asset regime priors for next day
                    for asset in asset_names:
                        if asset != market_asset:
                            asset_regime_probs_current[asset] = assets_probs_arrays[asset][day + 1]
            
            # Progress update
            if verbose and (sim_idx + 1) % 2000 == 0:
                print(f"  Completed {sim_idx + 1:,} / {n_simulations:,} simulations")
        
        if verbose:
            print(f"✓ Completed all {n_simulations:,} simulations")
            print(f"\nSimulation output shapes:")
            for asset in asset_names:
                print(f"  {asset}: {simulated_returns[asset].shape}")
        
        return simulated_returns
    
    @staticmethod
    def _compute_likelihood(return_value: float, dist_params: Dict) -> float:
        """
        Compute likelihood P(return | regime) using PDF of fitted distribution.
        
        This is used for Bayesian updates to the market regime probabilities after
        observing each day's simulated return.
        
        Parameters
        ----------
        return_value : float
            Observed return value
        dist_params : dict
            Distribution parameters with keys 'distribution' and 'params'
        
        Returns
        -------
        float
            Probability density at return_value, with numerical safeguards
        
        Notes
        -----
        - Returns small epsilon (1e-100) instead of exactly zero to avoid numerical issues
        - Handles all distribution types: normal, skewnorm, student_t, norminvgauss
        """
        epsilon = 1e-100
        dist_type = dist_params['distribution']
        params = dist_params['params']
        
        try:
            if dist_type == 'normal':
                loc, scale = params
                pdf = norm.pdf(return_value, loc=loc, scale=scale)
            
            elif dist_type == 'skewnorm':
                a, loc, scale = params
                pdf = skewnorm.pdf(return_value, a, loc=loc, scale=scale)
            
            elif dist_type == 'student_t':
                df, loc, scale = params
                pdf = t.pdf(return_value, df, loc=loc, scale=scale)
            
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params
                pdf = norminvgauss.pdf(return_value, a, b, loc=loc, scale=scale)
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
            
            # Safeguard against numerical zeros
            return max(pdf, epsilon)
        
        except Exception as e:
            # If PDF computation fails, return small value to avoid breaking update
            return epsilon

    
    @staticmethod
    def _inverse_cdf_copula(u: float, dist_params: Dict) -> float:
        """
        Transform uniform [0,1] to distribution-specific sample via inverse CDF.
        
        This is the key step in Gaussian Copula that preserves marginal distributions
        while using the copula for correlation structure.
        
        Parameters
        ----------
        u : float
            Uniform random variable in [0, 1]
        dist_params : dict
            Distribution parameters with keys 'distribution' and 'params'
            
        Returns
        -------
        float
            Sample from the specified distribution
            
        Notes
        -----
        - Uses scipy's .ppf() (percent point function = inverse CDF)
        - Handles edge cases (u=0, u=1) with small epsilon
        - Supports: normal, skewnorm, student_t, norminvgauss
        """
        # Handle edge cases
        epsilon = 1e-10
        u = np.clip(u, epsilon, 1 - epsilon)
        
        dist_type = dist_params['distribution']
        params = dist_params['params']
        
        try:
            if dist_type == 'normal':
                loc, scale = params
                return norm.ppf(u, loc=loc, scale=scale)
            
            elif dist_type == 'skewnorm':
                a, loc, scale = params
                return skewnorm.ppf(u, a, loc=loc, scale=scale)
            
            elif dist_type == 'student_t':
                df, loc, scale = params
                return t.ppf(u, df, loc=loc, scale=scale)
            
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params
                return norminvgauss.ppf(u, a, b, loc=loc, scale=scale)
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        except Exception as e:
            raise ValueError(
                f"Failed to compute inverse CDF for {dist_type} with u={u}: {e}"
            )

