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