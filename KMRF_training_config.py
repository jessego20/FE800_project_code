"""
Configuration file for KMRF model training and optimization.

This file defines all configuration parameters for:
- Asset universe and cross-asset feature groups
- Train/validation/test date splits
- XGBoost hyperparameters (paper defaults and search ranges)
- Transaction cost assumptions
- Optimization settings

Usage:
    from config import Config
    config = Config()
    assets = config.ASSETS
    xgb_params = config.DEFAULT_XGB_PARAMS
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd


class KMRF_Training_Config:
    """
    Configuration for KMRF training and optimization for a specific asset.
    
    This class provides all configuration parameters needed to train and optimize
    a KMRF model for a single asset, including:
    - Asset-specific parameters (asset class, cross-asset features, transaction costs)
    - Date ranges for train/val/test splits
    - XGBoost hyperparameters (defaults and search spaces)
    - Feature engineering settings
    - Strategy and optimization parameters
    
    Args:
        asset_name: Name of the asset (must be in ASSETS dict)
        classification_type: 'original' (4-regime) or 'adapted' (3-class)
        use_data_type: 'master' or 'individual' (data source)
        feature_window_size: Size of feature window (default=1)
    
    Example:
        config = KMRF_Training_Config(
            asset_name='SPDR S&P 500 ETF',
            classification_type='original',
            use_data_type='master',
            feature_window_size=1
        )
        xgb_params = config.get_xgb_params()
        cross_features = config.get_cross_asset_features()
    """
    
    # =========================================================================
    # CLASS-LEVEL CONSTANTS (SHARED ACROSS ALL INSTANCES)
    # =========================================================================
    
    # Asset classes available in master_df.csv
    ASSET_CLASSES = ['us_equity', 'commodity', 'int_equity']
    
    # Valid asset classes for feature_asset_classes parameter
    # (includes 'universe' in addition to standard asset classes)
    VALID_FEATURE_ASSET_CLASSES = ['us_equity', 'commodity', 'int_equity', 'universe']
    
    # Specific assets to train (full names as they appear in master_df.csv)
    # Organized by asset class for clarity
    ASSETS = {
        'us_equity': [
            'SPDR S&P 500 ETF', 'Invesco QQQ Trust', 'iShares Russell 2000 ETF',
            'SPDR Dow Jones Industrial Average ETF', 'Energy Select Sector SPDR',
            'Financial Select Sector SPDR', 'Utilities Select Sector SPDR',
            'Industrial Select Sector SPDR', 'Health Care Select Sector SPDR',
            'Technology Select Sector SPDR', 'Materials Select Sector SPDR',
            'Consumer Discretionary Select Sector SPDR',
            'Consumer Staples Select Sector SPDR', 'iShares S&P 500 Growth ETF',
            'iShares S&P 500 Value ETF', 'iShares Russell 2000 Growth ETF',
            'iShares Russell 2000 Value ETF', 'iShares Russell Mid-Cap ETF',
            'iShares Micro-Cap ETF'
        ],
        'commodity': [
            'Gold Futures', 'Wheat Futures', 'Corn Futures', 'Copper', 'Sugar',
            'Silver Futures', 'US Dollar', 'Soybean Futures', 'Lumber Futures',
            'Live Cattle Futures', 'Natural Gas', 'Coffee', 'Brent Crude Oil',
            'Heating Oil'
        ],
        'int_equity': [
            'Vanguard Total International Stock ETF',
            'Vanguard FTSE Developed Markets ETF',
            'Vanguard FTSE Emerging Markets ETF', 'Vanguard FTSE Europe ETF',
            'Vanguard FTSE Pacific ETF', 'iShares China Large-Cap ETF',
            'iShares MSCI Japan ETF', 'iShares MSCI India ETF'
        ],
        'universe': [
            'IVV - iShares Core S&P 500 ETF',
            'IJH - iShares Core S&P Mid-Cap ETF',
            'IWM - iShares Russell 2000 ETF',
            'EFA - iShares MSCI EAFE ETF',
            'EEM - iShares MSCI Emerging Markets ETF',
            'AGG - iShares Core U.S. Aggregate Bond ETF',
            'SPTL - SPDR Portfolio Long Term Treasury ETF',
            'HYG - iShares iBoxx $ High Yield Corporate Bond ETF',
            'SPBO - SPDR Portfolio Corporate Bond ETF',
            'IYR - iShares U.S. Real Estate ETF',
            'DBC - Invesco DB Commodity Index Tracking Fund',
            'GLD - SPDR Gold Shares'
        ]
    }
    
    # Flatten asset list for easy iteration
    ALL_ASSETS = [asset for assets in ASSETS.values() for asset in assets]
    
    # =========================================================================
    # CROSS-ASSET FEATURE GROUPS
    # =========================================================================
    
    # Define which cross-asset features to use for each asset
    # Key: asset name, Value: list of asset classes to include as features
    # Note: The asset's own class is always included automatically
    
    # Build CROSS_ASSET_FEATURES programmatically based on asset class
    _CROSS_ASSET_FEATURES_MAP = {}
    
    # US Equity: Add commodity features only
    for _asset in ASSETS['us_equity']:
        _CROSS_ASSET_FEATURES_MAP[_asset] = ['commodity']
    
    # Commodity: Add US equity and international equity features
    for _asset in ASSETS['commodity']:
        _CROSS_ASSET_FEATURES_MAP[_asset] = ['us_equity', 'int_equity']
    
    # International Equity: Add US equity and commodity features
    for _asset in ASSETS['int_equity']:
        _CROSS_ASSET_FEATURES_MAP[_asset] = ['us_equity', 'commodity']

    # Universe ETFs: use all universe etfs as cross-asset features
    for _asset in ASSETS['universe']:
        _CROSS_ASSET_FEATURES_MAP[_asset] = ['universe']

    # =========================================================================
    # DATE RANGES
    # =========================================================================
    
    # KAMA+MSR model fitting end date (to avoid lookahead bias)
    END_DATE = '20181231'  # Dec 31, 2018
    
    # Train/Validation/Test split dates
    TRAIN_END = pd.to_datetime('2019-01-01')
    VALIDATION_START = pd.to_datetime('2019-02-01')
    VALIDATION_END = pd.to_datetime('2021-12-30')
    TEST_START = pd.to_datetime('2022-02-01')
    
    # Alternative split for walk-forward analysis
    WALK_FORWARD_SPLITS = [
        {
            'train_start': '1995-01-01',
            'train_end': '2015-12-31',
            'val_start': '2016-01-01',
            'val_end': '2017-12-31',
            'test_start': '2018-01-01',
            'test_end': '2018-12-31'
        },
        {
            'train_start': '1995-01-01',
            'train_end': '2016-12-31',
            'val_start': '2017-01-01',
            'val_end': '2018-12-31',
            'test_start': '2019-01-01',
            'test_end': '2019-12-31'
        },
        {
            'train_start': '1995-01-01',
            'train_end': '2017-12-31',
            'val_start': '2018-01-01',
            'val_end': '2019-12-31',
            'test_start': '2020-01-01',
            'test_end': '2020-12-31'
        },
    ]
    
    # =========================================================================
    # XGBOOST HYPERPARAMETERS
    # =========================================================================
    
    # Default XGBoost hyperparameters (converted from RF paper defaults)
    EQUITY_DEFAULT_XGB_PARAMS = {
        'n_estimators': 220,
        'max_depth': 13,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.25,
        'min_child_weight': 95,
        'gamma': 0.045,
        'random_state': 1010,
        'n_jobs': -1,
        'tree_method': 'hist',
        'enable_categorical': False
    }

    COMMODITY_DEFAULT_XGB_PARAMS = {
        'n_estimators': 280,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.4,
        'min_child_weight': 95,
        'gamma': 0.02,
        'random_state': 1010,
        'n_jobs': -1,
        'tree_method': 'hist',
        'enable_categorical': False
    }

    # Hyperparameter search ranges for Optuna optimization
    XGB_SEARCH_SPACE = {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 300, 'step': 10},
        'max_depth': {'type': 'int', 'low': 1, 'high': 20},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.1, 'high': 1.0},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 100, 'step': 5},
        'gamma': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
        'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
    }
    
    # =========================================================================
    # FEATURE ENGINEERING PARAMETERS
    # =========================================================================
    
    # Feature window search range
    FEATURE_WINDOW_SEARCH_SPACE = {
        'feature_window_size': {'type': 'int', 'low': 1, 'high': 10}
    }
    
    # Consensus feature selection parameters
    DEFAULT_CONSENSUS_PARAMS = {
        'min_votes': 2,  # Minimum methods that must select a feature
        'variance_threshold': 0.01,  # Variance threshold for low-variance filter
        'cumulative_importance_threshold': 0.95,  # RF importance cumulative threshold
        'mi_top_pct': 0.3  # Top X% by mutual information
    }
    
    # Consensus feature selection search space
    CONSENSUS_SEARCH_SPACE = {
        'min_votes': {'type': 'int', 'low': 2, 'high': 3},
        'variance_threshold': {'type': 'float', 'low': 0.001, 'high': 0.05, 'log': True},
        'cumulative_importance_threshold': {'type': 'float', 'low': 0.85, 'high': 0.98, 'step': 0.01},
        'mi_top_pct': {'type': 'float', 'low': 0.2, 'high': 0.5, 'step': 0.05}
    }
    
    # =========================================================================
    # TRANSACTION COSTS
    # =========================================================================
    
    # Transaction cost assumptions by asset class (in basis points)
    TRANSACTION_COSTS = { # units are decimals
        'us_equity': 0.004,
        'commodity': 0.0027,
        'int_equity': 0.004,
        'universe': 0.004
    }
    
    # Slippage factor (multiplier on volatility)
    SLIPPAGE_FACTOR = 0.0  # 10% of daily volatility
    
    # =========================================================================
    # SINGLE-ASSET STRATEGY PARAMETERS
    # =========================================================================
    
    # Default strategy configuration
    DEFAULT_STRATEGY_PARAMS = {
        'signal_strategy': 'threshold',  # 'threshold', 'proportional', or 'regime_specific'
        'bull_threshold': 0.6,  # P(Bullish) > 0.6 → Long
        'bear_threshold': 0.6,  # P(Bearish) > 0.6 → Reduce/Cash
        'position_scaling': 1.0,  # Scaling factor for proportional strategy
        'max_position_size': 1.0,  # Maximum allocation (100%)
        'min_position_size': 0.0,  # Minimum allocation when signal is active
        'rebalance_frequency': 1,  # Days between rebalances (1 = daily)
        'stop_loss_pct': 0.0,  # Stop loss (0 = disabled)
        'take_profit_pct': 0.0,  # Take profit (0 = disabled)
    }
    
    # Strategy parameter search space
    STRATEGY_SEARCH_SPACE = {
        'signal_strategy': {'type': 'categorical', 'choices': ['threshold', 'proportional', 'regime_specific']},
        'bull_threshold': {'type': 'float', 'low': 0.5, 'high': 0.8, 'step': 0.05},
        'bear_threshold': {'type': 'float', 'low': 0.5, 'high': 0.8, 'step': 0.05},
        'position_scaling': {'type': 'float', 'low': 0.5, 'high': 1.5, 'step': 0.1},
        'max_position_size': {'type': 'float', 'low': 0.5, 'high': 1.0, 'step': 0.1},
        'min_position_size': {'type': 'float', 'low': 0.0, 'high': 0.2, 'step': 0.05},
        'rebalance_frequency': {'type': 'int', 'low': 1, 'high': 20},
        'stop_loss_pct': {'type': 'float', 'low': 0.0, 'high': 0.1, 'step': 0.01},
        'take_profit_pct': {'type': 'float', 'low': 0.0, 'high': 0.2, 'step': 0.02},
    }
    
    # =========================================================================
    # OPTIMIZATION SETTINGS
    # =========================================================================
    
    # Optuna study configuration
    OPTUNA_CONFIG = {
        'n_trials': 200,  # Number of trials per asset (full optimization)
        'n_trials_quick': 50,  # Quick test trials
        'timeout': None,  # No timeout (can set to seconds if needed)
        'n_jobs': -1,  # Parallel trials (1 = sequential, -1 = all cores)
        'direction': 'maximize',  # Maximize Sortino ratio
        'sampler': 'TPE',  # Tree-structured Parzen Estimator
        'pruner': 'MedianPruner',  # Prune unpromising trials
    }
    
    # Early stopping criteria
    EARLY_STOPPING = {
        'enabled': True,
        'patience': 20,  # Stop if no improvement for N trials
        'min_trials': 50,  # Minimum trials before early stopping
        'min_sortino': 0.5,  # Stop if validation Sortino < threshold after min_trials
    }
    
    # Optimization objective
    OPTIMIZATION_OBJECTIVE = 'sortino_ratio'  # Can also be 'sharpe_ratio', 'calmar_ratio', etc.
    
    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================
    
    # Directory for saving optimized models
    MODEL_SAVE_DIR = 'saved_models/optimized_KMRF'
    
    # Directory for saving optimization results
    RESULTS_SAVE_DIR = 'optimization_results'
    
    # File naming convention
    MODEL_NAME_TEMPLATE = '{asset_class}/{asset_name}/optimized_KMRF_model_{date}.pkl'
    RESULTS_NAME_TEMPLATE = '{asset_class}/{asset_name}/optimization_results_{date}.json'
    FEATURES_NAME_TEMPLATE = '{asset_class}/{asset_name}/selected_features_{date}.json'
    
    # =========================================================================
    # CLASSIFICATION TYPE
    # =========================================================================
    
    # Use 'original' for 4-regime KAMA+MSR labels (0,1,2,3)
    # Use 'adapted' for 3-class KMRF labels (-1,0,1)
    CLASSIFICATION_TYPE = 'original'
    
    # =========================================================================
    # REGIME LABEL MAPPING
    # =========================================================================
    
    REGIME_NAMES = {
        'original': {
            0: 'LV Bullish',
            1: 'LV Bearish',
            2: 'HV Bullish',
            3: 'HV Bearish'
        },
        'adapted': {
            -1: 'Bearish',
            0: 'Other',
            1: 'Bullish'
        }
    }
    
    # =========================================================================
    # INSTANCE INITIALIZATION
    # =========================================================================
    
    def __init__(
        self,
        asset_name: str,
        classification_type: str = 'original',
        use_data_type: str = 'master',
        end_date: str = '20181231',
        feature_window_size: int = 1,
        feature_asset_classes: Optional[List[str]] = None,
        cross_asset_specific: Optional[List[str]] = None,
        use_boruta_selection: bool = False,
        use_consensus_selection: bool = False
    ):
        """
        Initialize configuration for a specific asset.
        
        Args:
            asset_name: Name of the asset (must be in ASSETS dict)
            classification_type: 'original' (4-regime) or 'adapted' (3-class)
            use_data_type: 'master' or 'individual' (data source)
            feature_window_size: Size of feature window (default=1)
            feature_asset_classes: List of asset class names to use as cross-asset features
                                  (default=None, uses preset defaults based on asset)
            cross_asset_specific: List of specific cross-asset tickers to include (default=None, uses all)
            use_boruta_selection: Whether to use Boruta feature selection (default=False)
            use_consensus_selection: Whether to use consensus feature selection (default=False)
        
        Raises:
            ValueError: If asset_name is not found in ASSETS configuration
        """
        # Validate asset name
        if not self._is_valid_asset(asset_name):
            raise ValueError(
                f"Asset '{asset_name}' not found in ASSETS configuration. "
                f"Available assets: {self.get_all_assets()}"
            )
        
        # Store instance variables
        self.asset_name = asset_name
        self.classification_type = classification_type
        self.use_data_type = use_data_type
        self.feature_window_size = feature_window_size
        self.END_DATE = end_date
        
        # Validate and store cross_asset_specific
        if cross_asset_specific is not None:
            # Validate that all provided asset names are valid
            all_valid_assets = self.get_all_assets()
            invalid_assets = [asset for asset in cross_asset_specific 
                            if asset not in all_valid_assets]
            if invalid_assets:
                raise ValueError(
                    f"Invalid asset names in cross_asset_specific: {invalid_assets}. "
                    f"Must be valid asset names from ASSETS configuration. "
                    f"Use get_all_assets() to see available assets."
                )
            self.cross_asset_specific = cross_asset_specific
        else:
            self.cross_asset_specific = []
            
        self.use_boruta_selection = use_boruta_selection
        self.use_consensus_selection = use_consensus_selection
        
        # Derive asset class
        self.asset_class = self._get_asset_class_for_name(asset_name)
        
        # Set feature_asset_classes: use provided value, or fall back to preset defaults
        if feature_asset_classes is not None:
            # Validate that all provided classes are valid
            invalid_classes = [cls for cls in feature_asset_classes 
                             if cls not in self.VALID_FEATURE_ASSET_CLASSES]
            if invalid_classes:
                raise ValueError(
                    f"Invalid asset classes in feature_asset_classes: {invalid_classes}. "
                    f"Valid classes are: {self.VALID_FEATURE_ASSET_CLASSES}"
                )
            self._feature_asset_classes = feature_asset_classes
        else:
            # Use preset defaults from the map if not explicitly provided
            self._feature_asset_classes = self._CROSS_ASSET_FEATURES_MAP.get(asset_name, [])
        
        # Set transaction costs based on asset class
        self._transaction_costs = self.TRANSACTION_COSTS.get(
            self.asset_class, 
            0
        )
        
        # Set XGB parameters based on asset class
        if self.asset_class == 'us_equity':
            self._default_xgb_params = self.EQUITY_DEFAULT_XGB_PARAMS.copy()
        elif self.asset_class == 'commodity':
            self._default_xgb_params = self.COMMODITY_DEFAULT_XGB_PARAMS.copy()
        elif self.asset_class == 'int_equity':
            # International equity uses equity defaults
            self._default_xgb_params = self.EQUITY_DEFAULT_XGB_PARAMS.copy()
        elif self.asset_class == 'universe':
            # Universe uses equity defaults
            self._default_xgb_params = self.EQUITY_DEFAULT_XGB_PARAMS.copy()
        else:
            # Fallback to equity defaults
            self._default_xgb_params = self.EQUITY_DEFAULT_XGB_PARAMS.copy()
    
    # =========================================================================
    # CLASS METHODS (STATIC ASSET INFORMATION)
    # =========================================================================
    
    @classmethod
    def _is_valid_asset(cls, asset_name: str) -> bool:
        """Check if asset name exists in ASSETS configuration."""
        for assets in cls.ASSETS.values():
            if asset_name in assets:
                return True
        return False
    
    @classmethod
    def _get_asset_class_for_name(cls, asset_name: str) -> str:
        """Get asset class for a given asset name."""
        for asset_class, assets in cls.ASSETS.items():
            if asset_name in assets:
                return asset_class
        raise ValueError(f"Asset '{asset_name}' not found in ASSETS configuration")
    
    @classmethod
    def get_all_assets(cls) -> List[str]:
        """Get all assets across all asset classes."""
        return [asset for assets in cls.ASSETS.values() for asset in assets]
    
    @classmethod
    def get_assets_by_class(cls, asset_class: str) -> List[str]:
        """Get all assets for a specific asset class."""
        return cls.ASSETS.get(asset_class, [])
    
    # =========================================================================
    # INSTANCE METHODS (ASSET-SPECIFIC GETTERS)
    # =========================================================================
    
    def get_asset_name(self) -> str:
        """Get the asset name."""
        return self.asset_name
    
    def get_asset_class(self) -> str:
        """Get the asset class."""
        return self.asset_class
    
    def get_classification_type(self) -> str:
        """Get the classification type."""
        return self.classification_type
    
    def get_use_data_type(self) -> str:
        """Get the data source type."""
        return self.use_data_type
    
    def get_feature_window_size(self) -> int:
        """Get the feature window size."""
        return self.feature_window_size
    
    def get_feature_asset_classes(self) -> List[str]:
        """Get cross-asset feature classes for this asset."""
        return self._feature_asset_classes.copy()
    
    # Backwards compatibility alias
    def get_cross_asset_features(self) -> List[str]:
        """Get cross-asset feature classes for this asset (alias for get_feature_asset_classes)."""
        return self.get_feature_asset_classes()
    
    def get_cross_asset_specific(self) -> List[str]:
        """Get specific cross-asset tickers to include."""
        return self.cross_asset_specific.copy()
    
    def get_use_boruta_selection(self) -> bool:
        """Get whether to use Boruta feature selection."""
        return self.use_boruta_selection
    
    def get_use_consensus_selection(self) -> bool:
        """Get whether to use consensus feature selection."""
        return self.use_consensus_selection
    
    def get_transaction_costs(self) -> float:
        """Get transaction cost parameters for this asset."""
        return self._transaction_costs
    
    def get_xgb_params(self) -> Dict:
        """Get default XGB hyperparameters for this asset."""
        return self._default_xgb_params.copy()
    
    def get_xgb_search_space(self) -> Dict:
        """Get XGB hyperparameter search space."""
        return self.XGB_SEARCH_SPACE.copy()
    
    def get_consensus_params(self) -> Dict:
        """Get consensus feature selection parameters."""
        return self.DEFAULT_CONSENSUS_PARAMS.copy()
    
    def get_consensus_search_space(self) -> Dict:
        """Get consensus feature selection search space."""
        return self.CONSENSUS_SEARCH_SPACE.copy()
    
    def get_strategy_params(self) -> Dict:
        """Get default strategy parameters."""
        return self.DEFAULT_STRATEGY_PARAMS.copy()
    
    def get_strategy_search_space(self) -> Dict:
        """Get strategy parameter search space."""
        return self.STRATEGY_SEARCH_SPACE.copy()
    
    def get_optuna_config(self) -> Dict:
        """Get Optuna optimization configuration."""
        return self.OPTUNA_CONFIG.copy()
    
    def get_early_stopping_config(self) -> Dict:
        """Get early stopping configuration."""
        return self.EARLY_STOPPING.copy()
    
    def get_optimization_objective(self) -> str:
        """Get optimization objective."""
        return self.OPTIMIZATION_OBJECTIVE
    
    def get_date_ranges(self) -> Dict:
        """Get train/validation/test date ranges."""
        return {
            'end_date': self.END_DATE,
            'train_end': self.TRAIN_END,
            'validation_start': self.VALIDATION_START,
            'validation_end': self.VALIDATION_END,
            'test_start': self.TEST_START
        }
    
    def get_regime_names(self) -> Dict:
        """Get regime label names for the current classification type."""
        return self.REGIME_NAMES.get(self.classification_type, {})
    
    def get_model_save_path(self, date: str = None) -> str:
        """Generate model save path for this asset."""
        from pathlib import Path
        import datetime as dt
        
        if date is None:
            date = dt.datetime.now().strftime('%Y%m%d')
        
        # Replace spaces and special characters in asset name
        safe_asset_name = self.asset_name.replace(' ', '_').replace('/', '_')
        
        path = self.MODEL_NAME_TEMPLATE.format(
            asset_class=self.asset_class,
            asset_name=safe_asset_name,
            date=date
        )
        
        return str(Path(self.MODEL_SAVE_DIR) / path)
    
    def get_results_save_path(self, date: str = None) -> str:
        """Generate results save path for this asset."""
        from pathlib import Path
        import datetime as dt
        
        if date is None:
            date = dt.datetime.now().strftime('%Y%m%d')
        
        safe_asset_name = self.asset_name.replace(' ', '_').replace('/', '_')
        
        path = self.RESULTS_NAME_TEMPLATE.format(
            asset_class=self.asset_class,
            asset_name=safe_asset_name,
            date=date
        )
        
        return str(Path(self.RESULTS_SAVE_DIR) / path)
    
    def get_features_save_path(self, date: str = None) -> str:
        """Generate features save path for this asset."""
        from pathlib import Path
        import datetime as dt
        
        if date is None:
            date = dt.datetime.now().strftime('%Y%m%d')
        
        safe_asset_name = self.asset_name.replace(' ', '_').replace('/', '_')
        
        path = self.FEATURES_NAME_TEMPLATE.format(
            asset_class=self.asset_class,
            asset_name=safe_asset_name,
            date=date
        )
        
        return str(Path(self.RESULTS_SAVE_DIR) / path)
    
    def get_all_config(self) -> Dict:
        """
        Get all configuration as a dictionary.
        Useful for logging and saving complete configuration.
        """
        return {
            'asset_name': self.asset_name,
            'asset_class': self.asset_class,
            'classification_type': self.classification_type,
            'use_data_type': self.use_data_type,
            'feature_window_size': self.feature_window_size,
            'feature_asset_classes': self._feature_asset_classes,
            'cross_asset_specific': self.cross_asset_specific,
            'use_boruta_selection': self.use_boruta_selection,
            'use_consensus_selection': self.use_consensus_selection,
            'transaction_costs': self._transaction_costs,
            'xgb_params': self._default_xgb_params,
            'date_ranges': self.get_date_ranges(),
            'regime_names': self.get_regime_names(),
            'consensus_params': self.get_consensus_params(),
            'strategy_params': self.get_strategy_params(),
            'optuna_config': self.get_optuna_config(),
            'optimization_objective': self.OPTIMIZATION_OBJECTIVE
        }
    
    def print_summary(self):
        """Print a summary of this asset's configuration."""
        print("="*80)
        print(f"KMRF CONFIGURATION FOR: {self.asset_name}")
        print("="*80)
        
        print(f"\n📊 ASSET INFORMATION:")
        print(f"  Asset Name: {self.asset_name}")
        print(f"  Asset Class: {self.asset_class}")
        print(f"  Classification Type: {self.classification_type}")
        print(f"  Data Source: {self.use_data_type}")
        
        print(f"\n🔧 FEATURE SETTINGS:")
        print(f"  Feature Window Size: {self.feature_window_size}")
        print(f"  Feature Asset Classes: {self._feature_asset_classes}")
        if self.cross_asset_specific:
            print(f"  Specific Cross-Assets: {self.cross_asset_specific}")
        print(f"  Boruta Selection: {self.use_boruta_selection}")
        print(f"  Consensus Selection: {self.use_consensus_selection}")
        
        print(f"\n📅 DATE RANGES:")
        print(f"  KAMA+MSR End Date: {self.END_DATE}")
        print(f"  Training: up to {self.TRAIN_END.date()}")
        print(f"  Validation: {self.VALIDATION_START.date()} to {self.VALIDATION_END.date()}")
        print(f"  Test: {self.TEST_START.date()} onwards")
        
        print(f"\n🚀 XGB HYPERPARAMETERS:")
        for key, value in self._default_xgb_params.items():
            if key not in ['random_state', 'n_jobs', 'tree_method', 'enable_categorical']:
                print(f"  {key}: {value}")
        
        print(f"\n💰 TRANSACTION COSTS:")
        print(f"  Cost: {self._transaction_costs*100:.4f}%")
        
        print(f"\n⚙️ OPTIMIZATION SETTINGS:")
        print(f"  Trials (full): {self.OPTUNA_CONFIG['n_trials']}")
        print(f"  Objective: {self.OPTIMIZATION_OBJECTIVE}")
        print(f"  Early stopping: {self.EARLY_STOPPING['enabled']}")
        
        print("="*80)
    
    def __repr__(self) -> str:
        """String representation of the config."""
        return (
            f"KMRF_Training_Config(asset_name='{self.asset_name}', "
            f"asset_class='{self.asset_class}', "
            f"classification_type='{self.classification_type}', "
            f"feature_window_size={self.feature_window_size})"
        )


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

def load_config(
    asset_name: str,
    classification_type: str = 'original',
    use_data_type: str = 'master',
    feature_window_size: int = 1
) -> KMRF_Training_Config:
    """
    Load configuration object for a specific asset.
    
    Args:
        asset_name: Name of the asset
        classification_type: 'original' or 'adapted'
        use_data_type: 'master' or 'individual'
        feature_window_size: Feature window size
    
    Returns:
        KMRF_Training_Config instance
    """
    return KMRF_Training_Config(
        asset_name=asset_name,
        classification_type=classification_type,
        use_data_type=use_data_type,
        feature_window_size=feature_window_size
    )


def get_assets_by_class(asset_class: str) -> List[str]:
    """Get all assets for a specific asset class."""
    return KMRF_Training_Config.get_assets_by_class(asset_class)


def get_all_assets() -> List[str]:
    """Get all assets across all asset classes."""
    return KMRF_Training_Config.get_all_assets()


def print_config_summary(asset_name: str = None):
    """
    Print a summary of the configuration.
    
    Args:
        asset_name: If provided, prints asset-specific config.
                   If None, prints overview of all assets.
    """
    if asset_name:
        # Print asset-specific configuration
        config = KMRF_Training_Config(asset_name)
        config.print_summary()
    else:
        # Print overview of all assets
        print("="*80)
        print("KMRF OPTIMIZATION CONFIGURATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 ASSET UNIVERSE:")
        for asset_class in KMRF_Training_Config.ASSET_CLASSES:
            assets = KMRF_Training_Config.ASSETS.get(asset_class, [])
            print(f"  {asset_class}: {len(assets)} assets")
        print(f"  Total: {len(KMRF_Training_Config.get_all_assets())} assets")
        
        print(f"\n📅 DATE RANGES:")
        print(f"  KAMA+MSR End Date: {KMRF_Training_Config.END_DATE}")
        print(f"  Training: up to {KMRF_Training_Config.TRAIN_END.date()}")
        print(f"  Validation: {KMRF_Training_Config.VALIDATION_START.date()} to {KMRF_Training_Config.VALIDATION_END.date()}")
        print(f"  Test: {KMRF_Training_Config.TEST_START.date()} onwards")
        
        print(f"\n🌲 EQUITY RF HYPERPARAMETERS (Paper Defaults):")
        for key, value in KMRF_Training_Config.EQUITY_DEFAULT_RF_PARAMS.items():
            if key not in ['random_state', 'n_jobs', 'class_weight']:
                print(f"  {key}: {value}")
        
        print(f"\n🌲 COMMODITY RF HYPERPARAMETERS (Paper Defaults):")
        for key, value in KMRF_Training_Config.COMMODITY_DEFAULT_RF_PARAMS.items():
            if key not in ['random_state', 'n_jobs', 'class_weight']:
                print(f"  {key}: {value}")

        print(f"\n💰 TRANSACTION COSTS:")
        for asset_class, costs in KMRF_Training_Config.TRANSACTION_COSTS.items():
            print(f"  {asset_class}: {costs*100:.4f}%")
        
        print(f"\n⚙️ OPTIMIZATION SETTINGS:")
        print(f"  Trials (full): {KMRF_Training_Config.OPTUNA_CONFIG['n_trials']}")
        print(f"  Trials (quick): {KMRF_Training_Config.OPTUNA_CONFIG['n_trials_quick']}")
        print(f"  Objective: {KMRF_Training_Config.OPTIMIZATION_OBJECTIVE}")
        print(f"  Early stopping: {KMRF_Training_Config.EARLY_STOPPING['enabled']}")
        
        print(f"\n🎯 FEATURE SETTINGS:")
        print(f"  Consensus min votes: {KMRF_Training_Config.DEFAULT_CONSENSUS_PARAMS['min_votes']}")
        
        print("="*80)


# Test the configuration
if __name__ == "__main__":
    # Print overview
    print_config_summary()
    
    # Test instance creation and methods
    print("\n\n🧪 TESTING INSTANCE METHODS:")
    print("="*80)
    
    test_asset = 'SPDR S&P 500 ETF'
    print(f"\nCreating config for: {test_asset}")
    config = KMRF_Training_Config(
        asset_name=test_asset,
        classification_type='original',
        use_data_type='master',
        feature_window_size=1
    )
    
    print(f"\n{config}")
    print(f"\nAsset Class: {config.get_asset_class()}")
    print(f"Cross-Asset Features: {config.get_cross_asset_features()}")
    print(f"Transaction Costs: {config.get_transaction_costs()*100:.4f}%")
    print(f"Feature Window Size: {config.get_feature_window_size()}")
    
    print(f"\n📁 Save Paths:")
    print(f"  Model: {config.get_model_save_path()}")
    print(f"  Results: {config.get_results_save_path()}")
    print(f"  Features: {config.get_features_save_path()}")
    
    print(f"\n🌲 RF Parameters (first 5):")
    rf_params = config.get_rf_params()
    for i, (key, value) in enumerate(rf_params.items()):
        if i < 5:
            print(f"  {key}: {value}")
    
    print("\n\n" + "="*80)
    print("DETAILED ASSET CONFIGURATION:")
    print("="*80)
    config.print_summary()
