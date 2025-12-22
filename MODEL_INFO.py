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

class MODEL_INFO:
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
        preds = pd.read_pickle(kmrf_preds_path).reset_index()
        preds['date'] = preds['date'] - TRADING_DAYS
        self.kmrf_predictions = preds.set_index(['date', 'horizon'])
        self.kmrf_predictions.index = pd.MultiIndex.from_arrays([
            self.kmrf_predictions.index.get_level_values(0),
            self.kmrf_predictions.index.get_level_values(1),
            self.kmrf_predictions.index.map(lambda x: x[0] + (x[1]) * TRADING_DAYS)
            ],
            names=['date', 'horizon', 'prediction_date']
        )
        self.kmrf_predictions_int = self.kmrf_predictions.rename(columns=self.regime_to_int)
        self.regime_pred_rankings = self.kmrf_predictions.drop(columns=['model_end_date']).apply(lambda row: list(row.sort_values(ascending=False).index), axis=1).to_frame(name='regime_ranking')
        self.regime_pred_rankings_int = self.regime_pred_rankings.apply(lambda row: [self.regime_to_int[regime] for regime in row['regime_ranking']], axis=1).to_frame(name='regime_ranking')

    def show_class_variables(self):
        for var_name, var_value in self.__dict__.items():
            print(f"{var_name}: {type(var_value)}")

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

