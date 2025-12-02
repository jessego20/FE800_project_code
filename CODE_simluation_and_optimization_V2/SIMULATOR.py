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
from pathlib import Path

from kama_msr import KAMA_MSR
from kmrf import KMRF

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

TRADING_DAYS = CustomBusinessDay(calendar=USFederalHolidayCalendar())

warnings.filterwarnings('ignore')

KM_MODEL_BASE_PATH = Path('saved_models/KAMA_MSR/us_equity')
KMRF_PREDICTIONS_BASE_PATH = Path('data/multi_horizon_predictions')

# Using saved KAMA+MSR models
def get_KM_model_dates(KM_MODEL_BASE_PATH = Path('saved_models/KAMA_MSR/us_equity')) -> pd.Series:
    return pd.Series([f.stem for f in list(KM_MODEL_BASE_PATH.glob('*'))]).sort_values().iloc[1:].reset_index(drop=True)

def get_KM_model_paths(MODEL_DATE: str, KM_MODEL_BASE_PATH = Path('saved_models/KAMA_MSR/us_equity')) ->  pd.Series:
    return pd.Series(list((KM_MODEL_BASE_PATH / MODEL_DATE).glob('*'))).sort_values().reset_index(drop=True)

# Using saved KMRF predictions
def get_asset_names(KMRF_PREDICTIONS_BASE_PATH = Path('data/multi_horizon_predictions')) -> pd.Series:
    kmrf_preds_paths = list(KMRF_PREDICTIONS_BASE_PATH.glob('*'))
    return pd.Series([f.stem.split('multi')[0][:-1].replace('_', ' ') for f in kmrf_preds_paths]).sort_values().reset_index(drop=True)

def get_KMRF_prediction_paths(KMRF_PREDICTIONS_BASE_PATH = Path('data/multi_horizon_predictions')) -> pd.Series:
    return pd.Series(list(KMRF_PREDICTIONS_BASE_PATH.glob('*'))).sort_values().reset_index(drop=True)

class SIMULATOR:
    def __init__(self, km_model_path: str = None, kmrf_preds_path: str = None):
        if km_model_path is None or kmrf_preds_path is None:
            # TODO: Fit KAMA_MSR model and generate KMRF predictions
            raise notImplementedError("Automatic model fitting and prediction generation not implemented yet. Please provide paths.")
        
        self.regime_to_int = {'P(LV_Bull)': 0, 'P(LV_Bear)': 1, 'P(HV_Bull)': 2, 'P(HV_Bear)': 3}
        self.int_to_regime = {v: k for k, v in self.regime_to_int.items()}
        
        # Paths
        self.km_model_path = km_model_path
        self.kmrf_preds_path = kmrf_preds_path

        # KAMA MSR Model info
        self.km_model = pd.read_pickle(km_model_path)
        self.km_model_date = Path(km_model_path).parent.stem

        # General Info
        self.asset_name = self.km_model.prices.name

        # KMRF Predictions & Regime Info
        self.kmrf_predictions = pd.read_pickle(kmrf_preds_path)
        self.kmrf_predictions.index = pd.MultiIndex.from_arrays([
            self.kmrf_predictions.index.get_level_values(0),
            self.kmrf_predictions.index.get_level_values(1),
            self.kmrf_predictions.index.map(lambda x: x[0] + (x[1]-1) * TRADING_DAYS)
            ],
            names=['date', 'horizon', 'prediction_date']
        )
        self.kmrf_predictions_int = self.kmrf_predictions.rename(columns=self.regime_to_int)
        self.regime_pred_rankings = self.kmrf_predictions.drop(columns=['model_end_date']).apply(lambda row: list(row.sort_values(ascending=False).index), axis=1).to_frame(name='regime_ranking')
        self.regime_pred_rankings_int = self.regime_pred_rankings.apply(lambda row: [self.regime_to_int[regime] for regime in row['regime_ranking']], axis=1).to_frame(name='regime_ranking')

    def show_class_variables(self):
        for var_name, var_value in self.__dict__.items():
            print(f"{var_name}: {type(var_value)}")

    def test_skew_and_kurt_significance(self):
        returns = self.km_model.returns.dropna().to_frame(name='returns')
        returns = returns.join(self.km_model.regime_labels.rename("regime")).dropna()

        # Test skewness and kurtosis for each regim
        significance = {0: {'skew': False, 'kurt': False}, 
                        1: {'skew': False, 'kurt': False}, 
                        2: {'skew': False, 'kurt': False}, 
                        3: {'skew': False, 'kurt': False}}
        regime_samples = {0: 0, 1: 0, 2: 0, 3: 0}
        for regime in range(4):
            regime_returns = returns[returns['regime'] == regime]['returns']
            n = len(regime_returns)
            regime_samples[regime] = n
            if n < 20:
                continue
            
            skewness = scipy_stats.skew(regime_returns)
            kurtosis = scipy_stats.kurtosis(regime_returns, fisher=True)

            skew_se = np.sqrt(6/n)
            kurt_se = np.sqrt(24/n)

            skew_z = skewness / skew_se
            kurt_z = kurtosis / kurt_se

            significance_level = 0.05
            z_critical = scipy_stats.norm.ppf(1 - significance_level/2)

            significance[regime]['skew'] = bool(abs(skew_z) > z_critical)
            significance[regime]['kurt'] = bool(abs(kurt_z) > z_critical)

        self.significance_test_results = significance
        self.regime_sample_sizes = regime_samples

    def fit_regime_return_distributions(self):

        if not hasattr(self, 'significance_test_results'):
            raise ValueError("Must call test_skew_and_kurt_significance() before fitting distributions")
        
        dists = {0: None, 1: None, 2: None, 3: None}
        for regime in self.significance_test_results.keys():
            
            regime_returns = self.km_model.returns[self.km_model.regime_labels == regime]

            if not (self.significance_test_results[regime]['skew'] or self.significance_test_results[regime]['kurt']):
                # fitting normal distribution
                mean = regime_returns.mean()
                std = regime_returns.std()
                dists[regime] = {'type': 'normal',
                                 'params': (mean, std),
                                 'loc': mean,
                                 'scale': std**2,
                                 'method': 'moments',
                                 'dist_obj': norm(mean, std)
                                }
            elif self.significance_test_results[regime]['skew'] and not self.significance_test_results[regime]['kurt']:
                # fitting skew normal distribution
                # Using MLE to estimate parameters
                a, loc, scale = skewnorm.fit(regime_returns)
                dists[regime] = {'type': 'skew_normal',
                                 'params': (a, loc, scale),
                                 'a': a,
                                 'loc': loc,
                                 'scale': scale,
                                 'method': 'MLE',
                                 'dist_obj': skewnorm(a, loc, scale)
                                }
            elif not self.significance_test_results[regime]['skew'] and self.significance_test_results[regime]['kurt']:
                # fitting t-distribution
                # Using method of moments to estimate parameters
                mean = regime_returns.mean()
                std = regime_returns.std()
                kurt = scipy_stats.kurtosis(regime_returns, fisher=True)
                # Estimate degrees of freedom from kurtosis
                df = 4 + (6 / kurt)  # Approximation
                dists[regime] = {'type': 't',
                                 'params': (df, mean, std),
                                 'df': df,
                                 'loc': mean,
                                 'scale': std,
                                 'method': 'moments',
                                 'dist_obj': t(df=df, loc=mean, scale=std)
                                }
            else:  
                # fitting NIG distribution
                # Using MLE to estimate parameters
                a, b, loc, scale = norminvgauss.fit(regime_returns)
                dists[regime] = {'type': 'NIG',
                                 'params': (a, b, loc, scale),
                                 'a': a,
                                 'b': b,
                                 'loc': loc,
                                 'scale': scale,
                                 'method': 'MLE',
                                 'dist_obj': norminvgauss(a, b, loc, scale)
                                }
        self.fitted_distributions = dists

    def goodness_of_fit_tests(self, verbose: bool = True) -> pd.DataFrame:
        """
        Validate fitted distributions with goodness-of-fit tests.
        
        Performs:
        - Parameter validity checks
        - Kolmogorov-Smirnov test
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
        if not hasattr(self, 'fitted_distributions') or self.fitted_distributions is None:
            raise ValueError("Must call fit_regime_return_distributions() first")
        
        validation = []
        
        for regime, params in self.fitted_distributions.items():
            dist_type = params['type']
            returns = self.km_model.returns[self.km_model.regime_labels == regime].dropna()
            
            # Parameter validity check based on distribution type
            if dist_type == 'normal':
                loc, scale = params['params']
                valid = scale > 0 and np.isfinite(loc) and np.isfinite(scale)
                dist_obj = params['dist_obj']
                
            elif dist_type == 'skew_normal':
                a, loc, scale = params['params']
                valid = scale > 0 and all(np.isfinite([a, loc, scale]))
                dist_obj = params['dist_obj']
                
            elif dist_type == 't':
                df, loc, scale = params['params']
                valid = df > 0 and scale > 0 and all(np.isfinite([df, loc, scale]))
                dist_obj = params['dist_obj']
            
            elif dist_type == 'NIG':
                a, b, loc, scale = params['params']
                valid = (a > 0 and abs(b) < a and scale > 0 and 
                        all(np.isfinite([a, b, loc, scale])))
                dist_obj = params['dist_obj']
                
            else:
                valid = False
                dist_obj = None
            
            # Goodness-of-fit tests
            if valid and len(returns) > 5:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = scipy_stats.kstest(returns, dist_obj.cdf)
                
                # Q-Q plot correlation
                theoretical_quantiles = dist_obj.ppf(np.linspace(0.01, 0.99, len(returns)))
                empirical_quantiles = np.sort(returns)
                qq_corr = np.corrcoef(theoretical_quantiles, empirical_quantiles)[0, 1]
                
            else:
                ks_stat, ks_pval = np.nan, np.nan
                qq_corr = np.nan
            
            validation.append({
                'regime': regime,
                'regime_name': self.int_to_regime[regime],
                'distribution': dist_type,
                'n_obs': len(returns),
                'valid': valid,
                'KS_stat': ks_stat,
                'KS_pval': ks_pval,
                'QQ_corr': qq_corr,
                'loc': params.get('loc'),
                'scale': params.get('scale'),
                'method': params.get('method')
            })
        
        df = pd.DataFrame(validation)
        
        if verbose:
            print("\n" + "="*80)
            print("DISTRIBUTION GOODNESS-OF-FIT VALIDATION")
            print("="*80)
            print(f"\nAsset: {self.asset_name}")
            print(f"Model Date: {self.km_model_date}")
            print("\nParameter Validity & Sample Size:")
            print(df[['regime', 'regime_name', 'distribution', 'n_obs', 'valid', 'method']].to_string(index=False))
            
            print("\nGoodness-of-Fit Tests:")
            print(df[['regime', 'regime_name', 'KS_stat', 'KS_pval', 'QQ_corr']].to_string(index=False))
            
            print("\nInterpretation:")
            print("  - KS p-value > 0.05: Distribution fits well (cannot reject)")
            print("  - KS p-value < 0.05: Distribution may not fit well (reject)")
            print("  - QQ_corr > 0.95: Excellent fit")
            print("  - QQ_corr > 0.90: Good fit")
            
            # Flag poor fits
            poor_fits = df[(df['KS_pval'] < 0.05) | (df['QQ_corr'] < 0.90)]
            if len(poor_fits) > 0:
                print(f"\n⚠️  WARNING: {len(poor_fits)} regime(s) have poor fit:")
                print(poor_fits[['regime', 'regime_name', 'distribution', 'KS_pval', 'QQ_corr']].to_string(index=False))
            else:
                print("\n✓ All distributions pass goodness-of-fit tests")
            
            print("="*80)
        
        self.gof_results = df
        return df

    def get_forward_regime_probs(self, prediction_date: str | pd.Timestamp, n_days: int = 21) -> pd.DataFrame:
        """
        Get forward regime probabilities from KMRF predictions for a given date.
        
        This extracts the multi-horizon KMRF predictions starting from the given date,
        providing regime probabilities for each day in the forecast horizon.
        
        Parameters
        ----------
        prediction_date : str or pd.Timestamp
            The date from which to start forward predictions (YYYYMMDD format)
        n_days : int, default=21
            Number of days of forward predictions to retrieve
            
        Returns
        -------
        pd.DataFrame
            Forward regime probabilities with shape (n_days, 4)
            Columns are regime probabilities: P(LV_Bull), P(LV_Bear), P(HV_Bull), P(HV_Bear)
            Index is prediction dates (actual calendar dates)
        """
        prediction_date = pd.Timestamp(prediction_date)
        
        # Filter KMRF predictions for the given starting date and up to n_days horizons
        # KMRF predictions have MultiIndex: (date, horizon, prediction_date)
        # where 'date' is the model training end date, 'horizon' is 1,2,...,21
        # and 'prediction_date' is the actual future date being predicted
        
        # Find predictions where the base date matches
        available_dates = self.kmrf_predictions.index.get_level_values('date').unique()
        
        # Find the closest available date <= prediction_date
        valid_dates = available_dates[available_dates <= prediction_date]
        if len(valid_dates) == 0:
            raise ValueError(f"No KMRF predictions available on or before {prediction_date}")
        
        base_date = valid_dates.max()
        
        # Extract predictions for this base date across all horizons
        mask = self.kmrf_predictions.index.get_level_values('date') == base_date
        forward_preds = self.kmrf_predictions[mask].copy()
        
        # Filter to only the requested number of days
        horizons = forward_preds.index.get_level_values('horizon')
        mask_horizon = horizons <= n_days
        forward_preds = forward_preds[mask_horizon]
        
        # Drop the model_end_date column if it exists, keep only probability columns
        prob_cols = ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']
        forward_preds = forward_preds[prob_cols]
        
        # Re-index by prediction_date for clarity
        forward_preds = forward_preds.droplevel(['date', 'horizon'])
        forward_preds.index.name = 'prediction_date'
        
        return forward_preds

    def get_transition_matrix(self, laplace_smoothing: bool = True) -> pd.DataFrame:
        """
        Compute regime transition matrix from historical regime labels.
        
        Parameters
        ----------
        laplace_smoothing : bool, default=True
            Apply Laplace smoothing to avoid zero probabilities
            
        Returns
        -------
        pd.DataFrame
            4x4 transition matrix where P[i,j] = P(regime_j | regime_i)
        """
        regime_labels = self.km_model.regime_labels.dropna()
        
        # Count transitions
        transitions = np.zeros((4, 4))
        for i in range(len(regime_labels) - 1):
            from_regime = int(regime_labels.iloc[i])
            to_regime = int(regime_labels.iloc[i + 1])
            transitions[from_regime, to_regime] += 1
        
        # Apply Laplace smoothing if requested
        if laplace_smoothing:
            transitions += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / row_sums
        
        regime_names = ['LV_Bull', 'LV_Bear', 'HV_Bull', 'HV_Bear']
        self.transition_matrix = pd.DataFrame(
            transition_matrix,
            index=regime_names,
            columns=regime_names
        )
        return self.transition_matrix

    def sample_from_distribution(self, regime: int, size: int = 1, max_return: float = 0.5) -> np.ndarray:
        """
        Sample from the fitted distribution for a given regime.
        
        Parameters
        ----------
        regime : int
            Regime index (0-3)
        size : int, default=1
            Number of samples to draw
        max_return : float, default=0.5
            Maximum absolute daily return (clipping bound)
            
        Returns
        -------
        np.ndarray
            Random samples from the regime-specific distribution
        """
        if not hasattr(self, 'fitted_distributions'):
            raise ValueError("Must call fit_regime_return_distributions() first")
        
        dist_params = self.fitted_distributions[regime]
        dist_obj = dist_params['dist_obj']
        
        # Sample from the distribution
        samples = dist_obj.rvs(size=size)
        
        # Clip extreme values
        samples = np.clip(samples, -max_return, max_return)
        
        return samples

    def inverse_cdf(self, u: float, regime: int) -> float:
        """
        Transform uniform [0,1] to distribution-specific sample via inverse CDF.
        
        This is used in Gaussian copula to preserve marginal distributions
        while maintaining correlation structure.
        
        Parameters
        ----------
        u : float
            Uniform random variable in [0, 1]
        regime : int
            Regime index (0-3) to determine which distribution to use
            
        Returns
        -------
        float
            Sample from the regime-specific distribution
        """
        if not hasattr(self, 'fitted_distributions'):
            raise ValueError("Must call fit_regime_return_distributions() first")
        
        # Handle edge cases
        epsilon = 1e-10
        u = np.clip(u, epsilon, 1 - epsilon)
        
        dist_params = self.fitted_distributions[regime]
        dist_obj = dist_params['dist_obj']
        
        return dist_obj.ppf(u)
    
    def prepare_for_simulation(self):
        """
        Run all preparatory steps needed for simulation.
        
        This is a convenience method that runs:
        1. test_skew_and_kurt_significance()
        2. fit_regime_return_distributions()
        3. get_transition_matrix()
        """
        self.test_skew_and_kurt_significance()
        self.fit_regime_return_distributions()
        self.get_transition_matrix()