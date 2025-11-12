"""
KMRF (KAMA+MSR+XGB) - Regime Prediction Model

This module implements the KMRF regime prediction model, which combines:
- KAMA (Kaufman's Adaptive Moving Average) for trend detection
- MSR (Markov-Switching Regression) for volatility regime detection
- XGB (XGBoost) for ex-ante regime prediction

Based on the papers by Pomorski & Gorse:
- "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes"
- "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework"

Modified to use XGBoost instead of Random Forest for improved performance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import pickle
from glob import glob
import matplotlib.pyplot as plt

# Import the TimeSeriesDerivedFields class for feature engineering
# import derive_features as dd # feature engineering is completed in feature_engineering.ipynb and saved in 'ready' data 

# Import for feature selection
try:
    from boruta import BorutaPy
except ImportError:
    BorutaPy = None
    warnings.warn("boruta not installed. Feature selection will be skipped. Install with: pip install boruta")

warnings.filterwarnings('ignore')

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.resolve()


class KMRF:
    """
    KMRF: KAMA+MSR+XGB Regime Prediction Model
    
    This class handles the complete pipeline for regime prediction including:
    - Loading multi-asset data from CSV files
    - Loading macroeconomic data and aligning it
    - Computing/loading technical features
    - Loading KAMA+MSR regime labels from saved models
    - Optionally adapting 4-regime labels to 3-class KMRF labels
    - Feature selection using Boruta algorithm
    - XGBoost training for regime prediction
    - Performance evaluation
    
    Classification Types:
    - Adapted (default): 3-class KMRF labels:
      * Bullish (1): LV bullish + extension to peak of next HV bullish
      * Bearish (-1): HV bearish + extension to trough of next LV bearish
      * Other (0): Remaining periods
    - Original: 4-regime KAMA+MSR labels:
      * LV Bullish (0), LV Bearish (1), HV Bullish (2), HV Bearish (3)
    
    The classification type is determined during initialization via classification_type parameter.
    """
    
    def __init__(
        self,
        asset_name: str,
        asset_class: str = 'us_equity',
        data_path: Optional[Union[str, Path]] = None,
        kama_msr_model_dir: Optional[Union[str, Path]] = None,
        end_date: str = '20190101',
        use_data_type: str = 'master',
        validation_start: str = '2019-02-01',
        validation_end: str = '2021-12-30',
        test_start: str = '2022-02-01',
        random_seed: int = 1010,
        classification_type: str = 'adapted',
        feature_window_size: int = 1,
        feature_asset_classes: Optional[List[str]] = None,
        cross_asset_specific: Optional[List[str]] = None,
        xgb_params: Optional[Dict] = None,
        use_boruta_selection: bool = False,
        use_consensus_selection: bool = False
    ):
        """
        Initialize the KMRF model for a specific asset.
        
        Parameters
        ----------
        asset_name : str
            Name of the specific asset (e.g., 'SPDR S&P 500 ETF')
        asset_class : str, default='us_equity'
            Asset class category ('us_equity', 'commodity', 'int_equity')
        data_path : str or Path, optional
            Path to the data file
        kama_msr_model_dir : str or Path, optional
            Path to the KAMA+MSR model directory
        end_date : str, default='20190101'
            End date for KAMA+MSR model selection
        use_data_type : str, default='ready'
            Type of data to use for loading features:
            - 'master': Load from master_df.csv (consolidated file with all assets)
            - 'ready': Load from pre-computed features in 'ready' folder
            - 'raw': Compute features from raw OHLC data (not yet implemented)
        validation_start : str, default='2019-04-01'
            Start date for validation period
        validation_end : str, default='2019-09-30'
            End date for validation period
        test_start : str, default='2020-01-01'
            Start date for test period (extends to end of available data)
        random_seed : int, default=1010
            Random seed for reproducibility
        classification_type : str, default='adapted'
            Type of regime classification to use:
            - 'adapted': 3-class KMRF labels (Bullish=1, Bearish=-1, Other=0)
            - 'original': 4-regime KAMA+MSR labels (0=LV Bullish, 1=LV Bearish, 2=HV Bullish, 3=HV Bearish)
        feature_window_size : int, default=1
            Number of time steps (days) to include as features for each prediction.
            If 1: Use only features from day t-1 to predict regime at day t (standard approach)
            If >1: Stack features from the last N days (t-N+1 to t-1) to predict regime at day t
            Example: feature_window_size=5 creates 5x more features (days t-5 through t-1)
        feature_asset_classes : List[str], optional
            List of asset class names to use as input features.
            If None, uses only the target asset's own features.
            Example: ['us_equity', 'int_equity', 'commodity']
        xgb_params : dict, optional
            XGBoost hyperparameters to override defaults.
            Example: {'n_estimators': 200, 'max_depth': 10}
        use_boruta_selection : bool, default=False
            If True, use Boruta algorithm for feature selection.
            If both use_boruta_selection and use_consensus_selection are True,
            consensus selection will be used (takes precedence).
        use_consensus_selection : bool, default=False
            If True, use consensus feature selection (combining multiple methods).
            This takes precedence over Boruta selection if both are True.
        """
        self.asset_name = asset_name
        self.asset_class = asset_class
        self.classification_type = classification_type
        self.end_date = end_date
        self.random_seed = random_seed
        self.use_data_type = use_data_type
        
        # Feature selection flags
        self.use_boruta_selection = use_boruta_selection
        self.use_consensus_selection = use_consensus_selection
        self.use_data_type = use_data_type
        
        # Validate use_data_type
        if use_data_type not in ['master', 'ready', 'raw']:
            raise ValueError(
                f"use_data_type must be 'master', 'ready', or 'raw', got '{use_data_type}'"
            )
        
        # Feature window parameters
        self.feature_window_size = feature_window_size
        self.feature_asset_classes = feature_asset_classes if feature_asset_classes is not None else []
        self.cross_asset_specific = cross_asset_specific if cross_asset_specific is not None else []
        self.xgb_params = xgb_params if xgb_params is not None else {}

        # Data split dates
        self.validation_start = pd.to_datetime(validation_start)
        self.validation_end = pd.to_datetime(validation_end)
        self.test_start = pd.to_datetime(test_start)
        
        # Validate classification type
        if classification_type not in ['adapted', 'original']:
            raise ValueError(
                f"classification_type must be 'adapted' or 'original', got '{classification_type}'"
            )
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Set default paths
        if data_path is None:
            if use_data_type == 'ready':
                data_path_map = {
                    'us_equity': BASE_DIR / 'data/ready/us_equity.csv',
                    'commodity': BASE_DIR / 'data/ready/commodity.csv',
                    'int_equity': BASE_DIR / 'data/ready/int_equity.csv',
                    'us_treasury': BASE_DIR / 'data/ready/us_treasury.csv'
                }
                self.data_path = data_path_map.get(asset_class, BASE_DIR / 'data')
            elif use_data_type == 'master':
                # For master type, data_path points to master_df.csv
                self.data_path = BASE_DIR / 'data/master_df.csv'
            elif use_data_type == 'raw':
                # TODO: implement using raw data instead of 'ready' data
                raise NotImplementedError("Raw data loading not yet implemented. Use 'ready' or 'master'.")
        else:
            self.data_path = Path(data_path)
        
        if kama_msr_model_dir is None:
            self.kama_msr_model_dir = BASE_DIR / f'saved_models/KAMA_MSR/{asset_class}/{end_date}/'
        else:
            self.kama_msr_model_dir = Path(kama_msr_model_dir)
        
        # Initialize data containers
        self.raw_ohlc: Optional[pd.DataFrame] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None  # Original KAMA+MSR 4-regime labels
        self.adapted_labels: Optional[pd.Series] = None  # Adapted 3-class KMRF labels
        self.macro_data: Optional[pd.DataFrame] = None
        self.kama_msr_model: Optional[object] = None
        self.selected_features: Optional[List[str]] = None
        
        # Cross-asset data containers
        self.cross_asset_features: Optional[pd.DataFrame] = None
        
        # Data splits
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val_proba: Optional[pd.Series] = None  # None for val/test (no labels available)
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test_proba: Optional[pd.Series] = None  # None for val/test (no labels available)

        # Model components
        self.feature_selector = None
        self.xgb_model = None
        self.scaler = None
        self.performance_metrics: Dict = {}
        
        # Label mapping for XGBoost (adapted classification uses -1, 0, 1 but XGBoost needs 0, 1, 2)
        self._label_mapping: Optional[Dict] = None
        self._inverse_label_mapping: Optional[Dict] = None
        
        print(f"KMRF model initialized")
        print(f"  Asset: {self.asset_name}")
        print(f"  Asset class: {self.asset_class}")
        print(f"  Classification type: {self.classification_type}")
        print(f"  End date: {self.end_date}")
        print(f"  Data type: {self.use_data_type}")
        print(f"  Data path: {self.data_path}")
        print(f"  KAMA+MSR model directory: {self.kama_msr_model_dir}")
        print(f"  Validation period: {validation_start} to {validation_end}")
        print(f"  Test start: {test_start}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Feature window size: {self.feature_window_size} days")
        print(f"  Feature asset classes: {self.feature_asset_classes}")
        print(f"  Feature selection: Boruta={self.use_boruta_selection}, Consensus={self.use_consensus_selection}")
        if self.xgb_params:
            print(f"  Custom XGB parameters: {self.xgb_params}")

    def set_raw_ohlc(self, data: Optional[pd.DataFrame] = None, use_master_df: bool = False):
        if data is not None:
            self.raw_ohlc = data
        elif use_master_df:
            # Load OHLC from master_df.csv
            master_df_path = BASE_DIR / 'data/master_df.csv'
            if master_df_path.exists():
                # Load only OHLC columns for this asset
                # We'll load the full data and extract OHLC
                full_data = pd.read_csv(
                    master_df_path,
                    index_col=0,
                    header=[0, 1, 2],
                    parse_dates=True
                )
                full_data.index = pd.to_datetime(full_data.index)
                
                # Extract this asset's OHLC data
                # Look for open, high, low, close columns
                try:
                    ohlc_data = full_data.loc[:, (slice(None), self.asset_name, ['open', 'high', 'low', 'close'])]
                    # Flatten to 2-level (asset_name, ohlc_field)
                    ohlc_data.columns = ohlc_data.columns.droplevel(0)
                    # Keep only the OHLC level
                    ohlc_data.columns = [(self.asset_name, col) for col in ohlc_data.columns.get_level_values(1)]
                    self.raw_ohlc = ohlc_data.dropna(how='all')
                except:
                    # If extraction fails, set to None
                    print(f"  Warning: Could not extract OHLC data from master_df for {self.asset_name}")
                    self.raw_ohlc = None
            else:
                self.raw_ohlc = None
        else:
            us_equity_symbol_names = {
                # BOND ETFS
                'BIL': 'SPDR Bloomberg 1-3 Month T-Bill ETF',
                'SHY': 'iShares 1-3 Year Treasury Bond ETF',
                'IEF': 'iShares 7-10 Year Treasury Bond ETF',
                # MAJOR INDICES
                '^GSPC': 'S&P 500',
                '^IXIC': 'Nasdaq Composite',
                '^NDX': 'Nasdaq 100',
                '^RUT': 'Russell 2000',
                '^DJI': 'Dow Jones Industrial Average',
                '^RUI': 'Russell 1000',
                '^RUA': 'Russell 3000',
                
                # MAIN BROAD MARKET ETFS
                'SPY': 'SPDR S&P 500 ETF',
                'VOO': 'Vanguard S&P 500 ETF',
                'RSP': 'Invesco S&P 500 Equal Weight ETF',
                'IVV': 'iShares Core S&P 500 ETF',
                'QQQ': 'Invesco QQQ Trust',
                'QQQM': 'Invesco Nasdaq 100 ETF',
                'ONEQ': 'Fidelity Nasdaq Composite Index ETF',
                'IWM': 'iShares Russell 2000 ETF',
                'IWB': 'iShares Russell 1000 ETF',
                'IWV': 'iShares Russell 3000 ETF',
                'DIA': 'SPDR Dow Jones Industrial Average ETF',
                'VTI': 'Vanguard Total Stock Market ETF',
                
                # S&P 500 SECTOR ETFS (SELECT SECTOR SPDRS)
                'XLE': 'Energy Select Sector SPDR',
                'XLF': 'Financial Select Sector SPDR',
                'XLU': 'Utilities Select Sector SPDR',
                'XLI': 'Industrial Select Sector SPDR',
                'XLV': 'Health Care Select Sector SPDR',
                'XLK': 'Technology Select Sector SPDR',
                'XLB': 'Materials Select Sector SPDR',
                'XLY': 'Consumer Discretionary Select Sector SPDR',
                'XLP': 'Consumer Staples Select Sector SPDR',
                'XLRE': 'Real Estate Select Sector SPDR',
                'XLC': 'Communication Services Select Sector SPDR',
                
                # GROWTH ETFs
                'IVW': 'iShares S&P 500 Growth ETF',
                'VONG': 'Vanguard Russell 1000 Growth ETF',
                'IWF': 'iShares Russell 1000 Growth ETF',
                'IWO': 'iShares Russell 2000 Growth ETF',
                'VUG': 'Vanguard Growth ETF',
                'SPYG': 'SPDR Portfolio S&P 500 Growth ETF',
                
                # VALUE ETFs
                'IVE': 'iShares S&P 500 Value ETF',
                'VONV': 'Vanguard Russell 1000 Value ETF',
                'IWD': 'iShares Russell 1000 Value ETF',
                'IWN': 'iShares Russell 2000 Value ETF',
                'VTV': 'Vanguard Value ETF',
                'SPYV': 'SPDR Portfolio S&P 500 Value ETF',
                
                # SIZE ETFs
                'IWR': 'iShares Russell Mid-Cap ETF',
                'IWC': 'iShares Micro-Cap ETF',
                'IJH': 'iShares Core S&P Mid-Cap ETF',
                'IJR': 'iShares Core S&P Small-Cap ETF',
                'MDY': 'SPDR S&P MidCap 400 ETF',
                'SLY': 'SPDR S&P 600 Small Cap ETF',
                'VO': 'Vanguard Mid-Cap ETF',
                'VB': 'Vanguard Small-Cap ETF',
                'SCHA': 'Schwab U.S. Small-Cap ETF',
                'SCHM': 'Schwab U.S. Mid-Cap ETF',
                'VTWO': 'Vanguard Russell 2000 ETF',
                'VTHR': 'Vanguard Russell 3000 ETF',
                'THRK': 'iShares Russell 3000 ETF',
                'SPSM': 'SPDR Portfolio S&P 600 Small Cap ETF',
                'SMLF': 'iShares Small-Cap US Equity Factor ETF',
                
                # NASDAQ SPECIFIC
                'QTEC': 'First Trust Nasdaq-100 Technology Sector Index Fund',
                'QQEW': 'First Trust Nasdaq-100 Equal Weighted Index Fund',
                'QQQG': 'Pacer Nasdaq 100 Top 50 Cash Cows Dividend Growth ETF',
                'QQQV': 'Pacer Nasdaq 100 Top 50 Value ETF',
                
                # DIVIDEND/QUALITY
                'SCHD': 'Schwab U.S. Dividend Equity ETF',
                'VYM': 'Vanguard High Dividend Yield ETF',
                'DVY': 'iShares Select Dividend ETF',
                'QUAL': 'iShares MSCI USA Quality Factor ETF',
                'USMV': 'iShares MSCI USA Min Vol Factor ETF',
                
                # EQUAL WEIGHT
                'EWSC': 'Invesco S&P SmallCap 600 Equal Weight ETF',
                'EWMC': 'Invesco S&P MidCap 400 Equal Weight ETF',
            }
            int_equity_symbol_names = {
                'VXUS': 'Vanguard Total International Stock ETF',
                'VEA': 'Vanguard FTSE Developed Markets ETF',
                'VWO': 'Vanguard FTSE Emerging Markets ETF',
                'VGK': 'Vanguard FTSE Europe ETF',
                'VPL': 'Vanguard FTSE Pacific ETF',
                'FXI': 'iShares China Large-Cap ETF',
                'EWJ': 'iShares MSCI Japan ETF',
                'INDA': 'iShares MSCI India ETF',
            }
            etf_symbol_name_dict = {**us_equity_symbol_names, **int_equity_symbol_names}

            fmp_comm = pd.read_csv(BASE_DIR / 'data/inputs/fmp_commodity_list.csv')
            comm_symbol_name_dict = fmp_comm.set_index('symbol')['name'].to_dict()
            comm_symbol_name_dict.update({'Nickel': 'Nickel'})

            universe_symbol_name_dict = {
                'IVV': 'IVV - iShares Core S&P 500 ETF',
                'IJH': 'IJH - iShares Core S&P Mid-Cap ETF',
                'IWM': 'IWM - iShares Russell 2000 ETF',
                'EFA': 'EFA - iShares MSCI EAFE ETF',
                'EEM': 'EEM - iShares MSCI Emerging Markets ETF',
                'AGG': 'AGG - iShares Core U.S. Aggregate Bond ETF',
                'SPTL': 'SPTL - SPDR Portfolio Long Term Treasury ETF',
                'HYG': 'HYG - iShares iBoxx $ High Yield Corporate Bond ETF',
                'SPBO': 'SPBO - SPDR Portfolio Corporate Bond ETF',
                'IYR': 'IYR - iShares U.S. Real Estate ETF',
                'DBC': 'DBC - Invesco DB Commodity Index Tracking Fund',
                'GLD': 'GLD - SPDR Gold Shares',
            }
            if self.asset_class == 'universe':
                raw_price_data = pd.read_csv(BASE_DIR / 'data/processed/universe_etfs.csv', index_col=0, header=[0, 1], parse_dates=True)
                raw_price_data.index = pd.to_datetime(raw_price_data.index)
                raw_price_data.rename(columns=universe_symbol_name_dict, level=0, inplace=True)
            elif self.asset_class != 'commodity':
                raw_price_data = pd.read_csv(BASE_DIR / 'data/processed/all_etf_data.csv', index_col=0, header=[0, 1], parse_dates=True)
                raw_price_data.index = pd.to_datetime(raw_price_data.index)
                raw_price_data.rename(columns=etf_symbol_name_dict, level=0, inplace=True)
            else:
                raw_price_data = pd.read_csv(BASE_DIR / 'data/processed/commodity_data.csv', index_col=0, header=[0, 1], parse_dates=True)
                raw_price_data.index = pd.to_datetime(raw_price_data.index)
                raw_price_data.rename(columns=comm_symbol_name_dict, level=0, inplace=True)

            self.raw_ohlc = raw_price_data[[(self.asset_name, 'open'), 
                                            (self.asset_name, 'high'), 
                                            (self.asset_name, 'low'), 
                                            (self.asset_name, 'close')]].dropna(how='all')

    def load_data(self, rename_map: Optional[Dict] = None, use_master_df: bool = True) -> pd.DataFrame:
        """
        Load data for the specific asset.
        
        Parameters
        ----------
        rename_map : dict, optional
            Dictionary to rename columns
        use_master_df : bool, default=True
            If True, loads from master_df.csv (3-level columns: asset_class, asset_name, feature)
            If False, loads from legacy ready data files (2-level columns: asset_name, feature)
        
        Returns
        -------
        pd.DataFrame
            Feature data for the specific asset
        """
        if use_master_df:
            # Load from master_df.csv
            master_df_path = BASE_DIR / 'data/master_df.csv'
            if not master_df_path.exists():
                raise FileNotFoundError(f"Master data file not found: {master_df_path}")
            
            print(f"\nLoading data from: {master_df_path}")
            
            self.set_raw_ohlc(use_master_df=False)
            
            # Load full master dataset with 3-level columns
            full_data = pd.read_csv(
                master_df_path,
                index_col=0,
                header=[0, 1, 2],
                parse_dates=True
            )
            full_data.index = pd.to_datetime(full_data.index)
            
            if rename_map:
                full_data.rename(columns=rename_map, level=1, inplace=True)
            
            # Extract only this asset's data
            # Check if asset exists
            available_assets = full_data.columns.get_level_values(1).unique().tolist()
            
            if self.asset_name not in available_assets:
                raise ValueError(
                    f"Asset '{self.asset_name}' not found in data. "
                    f"Available assets: {', '.join(available_assets[:10])}..."
                )
            
            # Extract asset data: select by asset_name (level 1), keep all features (level 2)
            # Result will be single-level columns (just features)
            self.raw_data = full_data.xs(self.asset_name, level=1, axis=1).dropna(how='all')
            
            # The result has multi-index columns (asset_class, feature), flatten to just feature names
            if isinstance(self.raw_data.columns, pd.MultiIndex):
                # Keep only the feature level (level 1 after xs)
                self.raw_data.columns = self.raw_data.columns.get_level_values(-1)
            
            print(f"Loaded data for: {self.asset_name}")
            print(f"  Rows: {self.raw_data.shape[0]}")
            print(f"  Columns: {self.raw_data.shape[1]}")
            print(f"  Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
            
        else:
            # Legacy loading from ready data files
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            print(f"\nLoading data from: {self.data_path}")

            self.set_raw_ohlc()
            
            # Load full dataset
            full_data = pd.read_csv(
                self.data_path,
                index_col=0,
                header=[0, 1],
                parse_dates=True
            )
            full_data.index = pd.to_datetime(full_data.index)
            
            if rename_map:
                full_data.rename(columns=rename_map, level=0, inplace=True)
            
            # Extract only this asset's data
            available_assets = full_data.columns.get_level_values(0).unique().tolist()
            
            if self.asset_name not in available_assets:
                raise ValueError(
                    f"Asset '{self.asset_name}' not found in data. "
                    f"Available assets: {', '.join(available_assets[:10])}..."
                )
            
            self.raw_data = full_data.xs(self.asset_name, level=0, axis=1).dropna(how='all')
            
            print(f"Loaded data for: {self.asset_name}")
            print(f"  Rows: {self.raw_data.shape[0]}")
            print(f"  Columns: {self.raw_data.shape[1]}")
            print(f"  Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        
        return self.raw_data
    
    def get_features(self) -> pd.DataFrame:
        """Get features for the asset."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_data_type in ['ready', 'master']:
            print(f"\nExtracting pre-computed features for {self.asset_name}...")
            self.features = self.raw_data.copy()
            print(f"  Features shape: {self.features.shape}")
        elif self.use_data_type == 'raw':
            # TODO: implement using raw data instead of 'ready' data
            raise NotImplementedError("Feature computation from raw data not yet implemented. Use 'ready' or 'master'.")
        else:
            raise ValueError(f"Invalid use_data_type: {self.use_data_type}")
        
        self.features = self.features.dropna(how='all')
        return self.features
    
    def load_kama_msr_labels(self, use_master_label_df: bool = True) -> pd.Series:
        """
        Load KAMA+MSR regime labels from saved model or master_label_df for this asset.
        
        Parameters
        ----------
        use_master_label_df : bool, default=True
            If True, loads labels from master_label_df.csv
            If False, loads labels from saved KAMA+MSR model files (legacy)
        
        Returns
        -------
        pd.Series
            Regime labels for the asset
        """
        if use_master_label_df:
            # Load from master_label_df.csv
            master_label_path = BASE_DIR / 'data/master_label_df.csv'
            if not master_label_path.exists():
                raise FileNotFoundError(f"Master label file not found: {master_label_path}")
            
            print(f"\n{'='*80}")
            print(f"LOADING KAMA+MSR LABELS FOR {self.asset_name}")
            print(f"{'='*80}")
            print(f"Loading from: {master_label_path}")
            
            # Load full master label dataset with 2-level columns (asset_class, asset_name)
            # The third row is just "date" repeated, so we use header=[0,1]
            full_labels = pd.read_csv(
                master_label_path,
                index_col=0,
                header=[0, 1],
                parse_dates=True
            )
            full_labels.index = pd.to_datetime(full_labels.index)
            
            # Check if asset exists
            available_assets = full_labels.columns.get_level_values(1).unique().tolist()
            
            if self.asset_name not in available_assets:
                raise ValueError(
                    f"Asset '{self.asset_name}' not found in label data. "
                    f"Available assets: {', '.join(available_assets[:10])}..."
                )
            
            # Extract labels for this asset
            # Select by asset_name (level 1), should give us a Series
            asset_labels = full_labels.xs(self.asset_name, level=1, axis=1)
            
            # If multi-index result, take the first (should only be one column per asset)
            if isinstance(asset_labels, pd.DataFrame):
                asset_labels = asset_labels.iloc[:, 0]
            
            self.labels = asset_labels.dropna()
            
            print(f"✓ Loaded labels for: {self.asset_name}")
            print(f"  Label date range: {self.labels.index[0]} to {self.labels.index[-1]}")
            print(f"  Total periods: {len(self.labels)}")
            
        else:
            # Legacy loading from saved KAMA+MSR model files
            if not self.kama_msr_model_dir.exists():
                raise FileNotFoundError(f"KAMA+MSR model directory not found: {self.kama_msr_model_dir}")
            
            print(f"\n{'='*80}")
            print(f"LOADING KAMA+MSR LABELS FOR {self.asset_name}")
            print(f"{'='*80}")
            print(f"Model directory: {self.kama_msr_model_dir}")
            
            # Try to find model file
            model_pattern = f"{self.asset_name}_KAMA-MSR_4-regimes.pkl"
            model_files = list(self.kama_msr_model_dir.glob(model_pattern))
            
            if not model_files:
                asset_safe = self.asset_name.replace(' ', '_')
                model_pattern = f"{asset_safe}_KAMA-MSR_4-regimes.pkl"
                model_files = list(self.kama_msr_model_dir.glob(model_pattern))
            
            if not model_files:
                raise FileNotFoundError(f"Model not found for: {self.asset_name}")
            
            model_file = model_files[0]
            
            print(f"Loading from: {model_file.name}")
            
            with open(model_file, 'rb') as f:
                self.kama_msr_model = pickle.load(f)
            
            if not hasattr(self.kama_msr_model, 'regime_labels'):
                raise ValueError(f"No regime_labels attribute in model for: {self.asset_name}")
            
            self.labels = self.kama_msr_model.regime_labels.copy()
            
            print(f"✓ Loaded labels for: {self.asset_name}")
            print(f"  Label date range: {self.labels.index[0]} to {self.labels.index[-1]}")
            print(f"  Total periods: {len(self.labels)}")
        
        # Print distribution
        print(f"\n  Original 4-regime distribution:")
        regime_names = {0: 'LV Bullish', 1: 'LV Bearish', 2: 'HV Bullish', 3: 'HV Bearish'}
        for regime in [0, 1, 2, 3]:
            count = (self.labels == regime).sum()
            pct = (count / len(self.labels.dropna())) * 100
            print(f"    {regime} - {regime_names[regime]:>12}: {count:>5} ({pct:>5.1f}%)")
        
        print(f"{'='*80}")
        
        return self.labels
    
    def load_macro_data(self, macro_data_path: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
        """
        Load macroeconomic data and align it with the main dataset.
        
        The macro data is typically at different frequencies (daily, monthly, quarterly).
        This method forward-fills the data to match the daily frequency of asset data.
        """
        if macro_data_path is None:
            macro_data_path = BASE_DIR / 'data/ready/macro_data_daily.csv'
        else:
            macro_data_path = Path(macro_data_path)
        
        if not macro_data_path.exists():
            print(f"Warning: Macro data file not found at {macro_data_path}")
            print("Continuing without macroeconomic features...")
            return None
        
        print(f"\nLoading macroeconomic data from: {macro_data_path}")
        
        macro_df = pd.read_csv(macro_data_path, index_col=0, header=0, parse_dates=True)
        macro_df.index = pd.to_datetime(macro_df.index)
        
        if self.features is not None:
            macro_df = macro_df.reindex(self.features.index).ffill()
            print(f"Aligned macro data to features index")
        
        self.macro_data = macro_df.dropna(how='all')
        
        print(f"Macro data shape: {macro_df.shape}")
        print(f"Macro indicators: {len(macro_df.columns.get_level_values(0).unique())}")
        
        return self.macro_data
    
    def load_cross_asset_features(self, use_master_df: bool = True) -> pd.DataFrame:
        """
        Load features from multiple asset classes for cross-asset predictions.
        
        This method loads feature data from all asset classes specified in 
        self.feature_asset_classes and combines them into a single DataFrame.
        
        Parameters
        ----------
        use_master_df : bool, default=True
            If True, loads from master_df.csv (3-level columns)
            If False, loads from legacy ready data files (2-level columns)
        
        Returns
        -------
        pd.DataFrame
            Combined features from all specified asset classes with multi-level columns
            where level 0 is asset name and level 1 is feature name.
        """
        if not self.feature_asset_classes:
            print("No feature asset classes specified. Using only target asset features.")
            return self.features
        
        print(f"\n{'='*80}")
        print(f"LOADING CROSS-ASSET FEATURES")
        print(f"{'='*80}")
        print(f"Target asset: {self.asset_name} ({self.asset_class})")
        print(f"Feature asset classes: {self.feature_asset_classes}")
        
        if use_master_df:
            # Load from master_df.csv
            master_df_path = BASE_DIR / 'data/master_df.csv'
            if not master_df_path.exists():
                raise FileNotFoundError(f"Master data file not found: {master_df_path}")
            
            print(f"\n  Loading all features from: {master_df_path}")
            
            # Load full master dataset with 3-level columns
            full_data = pd.read_csv(
                master_df_path,
                index_col=0,
                header=[0, 1, 2],
                parse_dates=True
            )
            full_data.index = pd.to_datetime(full_data.index)
            
            print(f"    Loaded full shape: {full_data.shape}")
            
            # Filter to only the specified asset classes
            # Level 0 is asset_class
            asset_class_mask = full_data.columns.get_level_values(0).isin(self.feature_asset_classes)
            filtered_data = full_data.loc[:, asset_class_mask]
            
            print(f"    Filtered to {self.feature_asset_classes}: {filtered_data.shape}")
            print(f"    Date range: {filtered_data.index[0]} to {filtered_data.index[-1]}")
            
            # For compatibility, we need to convert 3-level (asset_class, asset_name, feature)
            # to 2-level (asset_name, feature) to match existing code expectations
            # Drop the asset_class level
            filtered_data.columns = filtered_data.columns.droplevel(0)
            
            combined_features = filtered_data.dropna(how='all')
            
            # Align with target asset's index if features exist
            if self.features is not None:
                combined_features = combined_features.reindex(self.features.index).ffill()
            
            self.cross_asset_features = combined_features
            
            print(f"\n  Combined cross-asset features:")
            print(f"    Total shape: {combined_features.shape}")
            print(f"    Total assets: {len(combined_features.columns.get_level_values(0).unique())}")
            print(f"    Features per asset: ~{combined_features.shape[1] // len(combined_features.columns.get_level_values(0).unique())}")
            print(f"    Date range: {combined_features.index[0]} to {combined_features.index[-1]}")
            print(f"{'='*80}")
            
        else:
            # Legacy loading from ready data files
            all_features = []
            
            for asset_cls in self.feature_asset_classes:
                # Determine data path for this asset class
                if self.use_data_type == 'ready':
                    data_path_map = {
                        'us_equity': BASE_DIR / 'data/ready/us_equity.csv',
                        'commodity': BASE_DIR / 'data/ready/commodity.csv',
                        'int_equity': BASE_DIR / 'data/ready/int_equity.csv',
                        'us_treasury': BASE_DIR / 'data/ready/us_treasury.csv'
                    }
                    asset_data_path = data_path_map.get(asset_cls, BASE_DIR / 'data')
                elif self.use_data_type == 'raw':
                    raise NotImplementedError("Raw data loading not yet implemented. Use 'ready' or 'master'.")
                else:
                    raise ValueError(f"Invalid use_data_type for legacy loading: {self.use_data_type}")
                
                if not asset_data_path.exists():
                    print(f"  WARNING: Data file not found for {asset_cls}: {asset_data_path}")
                    continue
                
                print(f"\n  Loading {asset_cls} features from: {asset_data_path}")
                
                # Load full dataset for this asset class
                asset_data = pd.read_csv(
                    asset_data_path,
                    index_col=0,
                    header=[0, 1],
                    parse_dates=True
                )
                asset_data.index = pd.to_datetime(asset_data.index)
                
                print(f"    Loaded shape: {asset_data.shape}")
                print(f"    Assets in {asset_cls}: {len(asset_data.columns.get_level_values(0).unique())}")
                print(f"    Date range: {asset_data.index[0]} to {asset_data.index[-1]}")
                
                all_features.append(asset_data)
            
            if not all_features:
                raise ValueError("No feature data loaded from any asset class")
            
            # Combine all features
            combined_features = pd.concat(all_features, axis=1)
            combined_features = combined_features.dropna(how='all')
            
            # Align with target asset's index if features exist
            if self.features is not None:
                combined_features = combined_features.reindex(self.features.index).ffill()
            
            self.cross_asset_features = combined_features
            
            print(f"\n  Combined cross-asset features:")
            print(f"    Total shape: {combined_features.shape}")
            print(f"    Total assets: {len(combined_features.columns.get_level_values(0).unique())}")
            print(f"    Features per asset: ~{combined_features.shape[1] // len(combined_features.columns.get_level_values(0).unique())}")
            print(f"    Date range: {combined_features.index[0]} to {combined_features.index[-1]}")
            print(f"{'='*80}")
        
        return self.cross_asset_features
    
    def select_features_boruta(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_iter: int = 100,
        percentile: int = 100,
        pvalue: float = 0.01,
        verbose: int = 2
    ) -> List[str]:
        """
        Use Boruta algorithm for feature selection.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels
        max_iter : int, default=100
            Maximum number of iterations
        percentile : int, default=100
            Percentile to use for determining importance threshold
        pvalue : float, default=0.05
            P-value threshold for feature selection
        verbose : int, default=2
            Verbosity level
            
        Returns:
        --------
        List[str] : Names of selected features
        """
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestClassifier
        
        print(f"\n{'='*80}")
        print("BORUTA FEATURE SELECTION")
        print(f"{'='*80}")
        print(f"Initial features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Max iterations: {max_iter}")
        
        # Handle missing values
        print("\nHandling missing values...")
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[valid_mask].copy()
        y_clean = y.loc[valid_mask].copy()
        
        print(f"  Removed {(~valid_mask).sum()} samples with missing values")
        print(f"  Clean dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        # Initialize Random Forest for Boruta
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            n_jobs=-1,
            random_state=self.random_seed,
            class_weight='balanced'
        )
        
        # Initialize Boruta
        selector = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            max_iter=max_iter,
            perc=percentile,
            alpha=pvalue,
            verbose=verbose,
            random_state=self.random_seed
        )
        
        print("Fitting Boruta...")
        # Convert to numpy arrays - handle nullable integer types
        X_array = X_clean.values
        # Convert y_clean to numpy array, handling nullable integer types
        if hasattr(y_clean, 'to_numpy'):
            y_array = y_clean.to_numpy(dtype='int64', na_value=-999)
        else:
            y_array = y_clean.values.astype('int64')
        
        # Fit Boruta (standard API: fit(X, y))
        selector.fit(X_array, y_array)
        
        # Get selected features using support_ attribute
        self.selected_features = X_clean.columns[selector.support_].tolist()
        
        print(f"\n{'='*80}")
        print("BORUTA RESULTS")
        print(f"{'='*80}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Rejected features: {(~selector.support_).sum()}")
        print(f"Tentative features: {selector.support_weak_.sum()}")
        
        if len(self.selected_features) > 0:
            print(f"\nTop 20 selected features:")
            # Get feature importances from the fitted estimator
            feature_importances = selector.estimator.feature_importances_
            selected_importances = [(feat, imp) for feat, imp, selected in 
                                   zip(X_clean.columns, feature_importances, selector.support_) 
                                   if selected]
            selected_importances.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feat, imp) in enumerate(selected_importances[:20], 1):
                # Convert feature name to string in case it's a tuple
                feat_str = str(feat) if not isinstance(feat, str) else feat
                print(f"  {i:2d}. {feat_str:<50s} (importance: {imp:.6f})")
        else:
            print("\nWARNING: No features selected by Boruta!")
            print("Using all features instead.")
            self.selected_features = X_clean.columns.tolist()
        
        return self.selected_features
    
    def consensus_feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        min_votes: int = 2,
        variance_threshold: float = 0.01,
        cumulative_importance_threshold: float = 0.95,
        mi_top_pct: float = 0.3
    ) -> Tuple[List[str], pd.DataFrame, Dict[str, set]]:
        """
        Multi-method consensus feature selection.
        
        Uses multiple fast feature selection methods and keeps features selected by 
        at least `min_votes` methods. This provides robust feature selection without
        the computational cost of Boruta.
        
        Methods used:
        1. Built-in RF importance (instant) - cumulative importance threshold
        2. Variance threshold (instant) - removes low-variance features
        3. Mutual Information (1-2 min) - information-theoretic relevance
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        min_votes : int, default=2
            Minimum number of methods that must select a feature to be included
        variance_threshold : float, default=0.01
            Variance threshold for removing low-variance features
        cumulative_importance_threshold : float, default=0.95
            Keep features that account for this fraction of total RF importance
        mi_top_pct : float, default=0.3
            Keep top X% of features by mutual information score
            
        Returns
        -------
        consensus_features : List[str]
            Features selected by >= min_votes methods
        vote_df : pd.DataFrame
            DataFrame with vote counts and method selections for each feature
        all_methods : Dict[str, set]
            Dictionary mapping method names to their selected feature sets
            
        Examples
        --------
        >>> selected, votes_df, methods = model.consensus_feature_selection(X_train, y_train)
        >>> print(f"Selected {len(selected)} features from {len(X_train.columns)}")
        >>> print(votes_df[votes_df['votes'] >= 2].head())
        """
        from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
        
        print(f"\n{'='*80}")
        print("CONSENSUS FEATURE SELECTION")
        print(f"{'='*80}")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Minimum votes required: {min_votes}")
        
        # Flatten column names if they're tuples (from multi-index)
        original_columns = X_train.columns
        if isinstance(X_train.columns[0], tuple):
            print(f"\nFlattening multi-index column names...")
            X_train_flat = X_train.copy()
            X_train_flat.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) 
                                    for col in X_train.columns]
            # Create mapping from flattened to original
            col_mapping = dict(zip(X_train_flat.columns, original_columns))
        else:
            X_train_flat = X_train
            col_mapping = {col: col for col in X_train.columns}
        
        all_methods = {}
        
        # Method 1: Built-in XGB importance (instant)
        print(f"\nMethod 1: XGB Feature Importance (Cumulative {cumulative_importance_threshold*100:.0f}%)")
        if self.xgb_model is None:
            raise ValueError("Model must be trained first. Call fit() before consensus_feature_selection()")
        
        imp_df = pd.DataFrame({
            'feature': X_train_flat.columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        imp_df['cumulative'] = imp_df['importance'].cumsum() / imp_df['importance'].sum()
        selected_imp_flat = set(imp_df[imp_df['cumulative'] <= cumulative_importance_threshold]['feature'])
        # Map back to original column names
        selected_imp = {col_mapping[f] for f in selected_imp_flat}
        all_methods['builtin_importance'] = selected_imp
        print(f"  Selected: {len(selected_imp)} features")
        
        # Method 2: Variance threshold (instant)
        print(f"\nMethod 2: Variance Threshold (>{variance_threshold})")
        var_selector = VarianceThreshold(threshold=variance_threshold)
        var_selector.fit(X_train_flat.values)  # Use .values to avoid column name issues
        selected_var_flat = set(X_train_flat.columns[var_selector.get_support()])
        # Map back to original column names
        selected_var = {col_mapping[f] for f in selected_var_flat}
        all_methods['variance'] = selected_var
        print(f"  Selected: {len(selected_var)} features")

        # Method 3: Mutual Information (fast - 1-2 min)
        print(f"\nMethod 3: Mutual Information (Top {mi_top_pct*100:.0f}%)")
        mi_scores = mutual_info_classif(X_train_flat.values, y_train, random_state=self.random_seed)
        mi_df = pd.DataFrame({'feature': X_train_flat.columns, 'mi': mi_scores})
        mi_df = mi_df.sort_values('mi', ascending=False)
        n_top_mi = int(len(X_train_flat.columns) * mi_top_pct)
        selected_mi_flat = set(mi_df.head(n_top_mi)['feature'])
        # Map back to original column names
        selected_mi = {col_mapping[f] for f in selected_mi_flat}
        all_methods['mutual_info'] = selected_mi
        print(f"  Selected: {len(selected_mi)} features")
        
        # Count votes for each feature (using original column names)
        print(f"\nCounting votes...")
        feature_votes = {}
        for feature in original_columns:
            votes = sum(1 for method_features in all_methods.values() if feature in method_features)
            feature_votes[feature] = votes
        
        # Keep features with >= min_votes
        consensus_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
        
        # Create summary DataFrame (convert tuples to strings for display)
        feature_names_display = [str(f) if isinstance(f, tuple) else f for f in feature_votes.keys()]
        vote_df = pd.DataFrame({
            'feature': feature_names_display,
            'feature_original': list(feature_votes.keys()),  # Keep original for indexing
            'votes': list(feature_votes.values()),
            'builtin': [f in all_methods['builtin_importance'] for f in feature_votes.keys()],
            'variance': [f in all_methods['variance'] for f in feature_votes.keys()],
            'mutual_info': [f in all_methods['mutual_info'] for f in feature_votes.keys()],
        }).sort_values('votes', ascending=False)
        
        # Store selected features
        self.selected_features = consensus_features
        
        # Print results
        print(f"\n{'='*80}")
        print("CONSENSUS RESULTS")
        print(f"{'='*80}")
        print(f"Selected features: {len(consensus_features)} (min_votes >= {min_votes})")
        print(f"Reduction: {100 * (1 - len(consensus_features)/len(original_columns)):.1f}%")
        
        print(f"\nVote distribution:")
        vote_counts = vote_df['votes'].value_counts().sort_index(ascending=False)
        for votes, count in vote_counts.items():
            pct = 100 * count / len(original_columns)
            print(f"  {votes} votes: {count:4d} features ({pct:5.1f}%)")
        
        # Show top features with maximum votes
        max_votes = vote_df['votes'].max()
        if max_votes >= min_votes:
            top_voted = vote_df[vote_df['votes'] == max_votes]
            print(f"\nTop features ({max_votes} votes): {len(top_voted)} features")
            if len(top_voted) <= 20:
                for idx, row in top_voted.iterrows():
                    # Use the display name (string version)
                    print(f"  ✓ {row['feature']}")
            else:
                print(f"  (showing first 20)")
                for idx, row in top_voted.head(20).iterrows():
                    print(f"  ✓ {row['feature']}")
        
        return consensus_features, vote_df, all_methods
    
    def expand_feature_window(
        self,
        X: pd.DataFrame,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Expand features to include multiple time steps for each prediction.
        
        Instead of using features from only day t-1 to predict regime at day t,
        this method stacks features from days t-window_size through t-1.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features with datetime index
        window_size : int, optional
            Number of time steps to include. If None, uses self.feature_window_size
        
        Returns
        -------
        pd.DataFrame
            Expanded features with columns named: original_feature_lag1, original_feature_lag2, etc.
            Shape will be (n_samples - window_size + 1, n_features * window_size)
        
        Examples
        --------
        If window_size=3 and input has features ['close', 'volume']:
        Output will have: ['close_lag1', 'volume_lag1', 'close_lag2', 'volume_lag2', 'close_lag3', 'volume_lag3']
        
        Row at date t will contain:
        - close_lag1: close from t-1
        - close_lag2: close from t-2
        - close_lag3: close from t-3
        """
        if window_size is None:
            window_size = self.feature_window_size
        
        # If window_size is 1, return original features (no expansion needed)
        if window_size == 1:
            return X
        
        print(f"\n  Expanding features to {window_size}-day window:")
        print(f"    Original features: {X.shape[1]}")
        
        # Create lagged features for each time step in the window
        expanded_dfs = []
        for lag in range(1, window_size + 1):
            # Shift features by 'lag' days and rename columns
            lagged = X.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in X.columns]
            expanded_dfs.append(lagged)
        
        # Concatenate all lagged features
        X_expanded = pd.concat(expanded_dfs, axis=1)
        
        # Drop rows with NaN (first window_size-1 rows will have NaN)
        X_expanded = X_expanded.dropna()
        
        print(f"    Expanded features: {X_expanded.shape[1]} ({X.shape[1]} × {window_size})")
        print(f"    Samples after window: {X_expanded.shape[0]} (dropped {len(X) - len(X_expanded)} initial rows)")
        
        return X_expanded
    
    def adapt_regime_labels(
        self,
        price_data: Optional[pd.Series] = None,
        labels: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Adapt 4-regime KAMA+MSR labels to 3-class KMRF labels for this asset.
        
        Implements the label transformation from the paper:
        1. Bullish (1): LV bullish + extension to peak of next HV bullish
        2. Bearish (-1): HV bearish + extension to trough of next LV bearish
        3. Other (0): Remaining periods (including post-peak HV bullish and post-trough LV bearish)
        """
        if self.classification_type != 'adapted':
            raise ValueError("Classification type is not 'adapted'. Cannot adapt labels.")
        
        if labels is None:
            if self.labels is None:
                raise ValueError("No labels available. Load KAMA+MSR labels first.")
            labels = self.labels
        
        if price_data is None:
            if self.raw_data is None:
                raise ValueError("No price data available. Load data first.")
            # Get close price from raw data
            if 'close' in self.raw_data.columns:
                price_data = self.raw_data['close']
            else:
                numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
                price_data = self.raw_data[numeric_cols[0]] if len(numeric_cols) > 0 else None
        
        print(f"\n{'='*80}")
        print(f"ADAPTING REGIME LABELS FOR {self.asset_name}")
        print(f"{'='*80}")
        
        # Initialize as Other (0)
        adapted = pd.Series(0, index=labels.index, dtype=int)
        
        # Process regimes
        i = 0
        while i < len(labels):
            current_regime = labels.iloc[i]
            
            if pd.isna(current_regime):
                i += 1
                continue
            
            current_regime = int(current_regime)
            
            # Find regime end
            j = i + 1
            while j < len(labels) and labels.iloc[j] == current_regime:
                j += 1
            
            regime_start = i
            regime_end = j
            
            # LV Bullish (0) → Extend to peak of HV Bullish (2)
            if current_regime == 0:
                extension_end = regime_end
                
                k = regime_end
                while k < len(labels):
                    next_regime = labels.iloc[k]
                    if pd.isna(next_regime):
                        k += 1
                        continue
                    
                    next_regime = int(next_regime)
                    
                    if next_regime == 2:  # HV Bullish
                        hv_start = k
                        while k < len(labels) and int(labels.iloc[k]) == 2:
                            k += 1
                        hv_end = k
                        
                        if price_data is not None:
                            hv_indices = labels.index[hv_start:hv_end]
                            hv_prices = price_data.loc[hv_indices]
                            if len(hv_prices) > 0:
                                peak_idx = hv_prices.idxmax()
                                peak_pos = labels.index.get_loc(peak_idx)
                                extension_end = peak_pos + 1
                                
                                # Mark remaining HV Bullish after peak as Other
                                adapted.iloc[extension_end:hv_end] = 0
                        break
                    k += 1
                
                # Mark as Bullish up to extension
                adapted.iloc[regime_start:extension_end] = 1
            
            # HV Bearish (3) → Extend to trough of LV Bearish (1)
            elif current_regime == 3:
                extension_end = regime_end
                
                k = regime_end
                while k < len(labels):
                    next_regime = labels.iloc[k]
                    if pd.isna(next_regime):
                        k += 1
                        continue
                    
                    next_regime = int(next_regime)
                    
                    if next_regime == 1:  # LV Bearish
                        lv_start = k
                        while k < len(labels) and int(labels.iloc[k]) == 1:
                            k += 1
                        lv_end = k
                        
                        if price_data is not None:
                            lv_indices = labels.index[lv_start:lv_end]
                            lv_prices = price_data.loc[lv_indices]
                            if len(lv_prices) > 0:
                                trough_idx = lv_prices.idxmin()
                                trough_pos = labels.index.get_loc(trough_idx)
                                extension_end = trough_pos + 1
                                
                                # Mark remaining LV Bearish after trough as Other
                                adapted.iloc[extension_end:lv_end] = 0
                        break
                    k += 1
                
                # Mark as Bearish up to extension
                adapted.iloc[regime_start:extension_end] = -1
            
            i = regime_end
        
        # Print distribution
        print(f"\n  Adapted 3-class distribution:")
        label_map = {-1: 'Bearish', 0: 'Other', 1: 'Bullish'}
        dist = adapted.value_counts().sort_index()
        for label_val in [1, 0, -1]:
            if label_val in dist:
                count = dist[label_val]
                pct = (count / len(adapted)) * 100
                print(f"    {label_map[label_val]:>8} ({label_val:>2}): {count:>5} ({pct:>5.1f}%)")
        
        print(f"{'='*80}")
        
        # Store adapted labels in member variable
        self.adapted_labels = adapted
        
        return adapted
    
    def prepare_training_data(
        self,
        include_macro: bool = True,
        use_cross_asset_features: bool = True,
        use_master_df: bool = True,
        select_features: bool = False,
        boruta_params: Optional[Dict] = None,
        split_data: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for model training.
        
        This comprehensive method:
        1. Optionally loads and combines cross-asset features
        2. Optionally loads and combines macroeconomic data
        3. Uses the classification type specified during initialization
        4. Aligns feature and label indices
        5. Optionally applies Boruta feature selection (on training data only)
        6. Optionally splits data into train/validation/test sets
        
        Parameters
        ----------
        include_macro : bool, default=True
            Load and include macroeconomic features
        use_cross_asset_features : bool, default=True
            Load and use features from multiple asset classes specified in feature_asset_classes
        use_master_df : bool, default=True
            Whether to use master_df.csv for cross-asset features
        select_features : bool, default=False
            Apply Boruta feature selection (on training data only)
        boruta_params : dict, optional
            Parameters for Boruta (max_iter, percentile, pvalue, verbose)
        split_data : bool, default=True
            Split data into train/validation/test sets
            
        Returns
        -------
        tuple of (pd.DataFrame, pd.Series)
            (features, labels) ready for training
            If split_data=True, sets self.X_train, self.y_train, etc.
            
        Notes
        -----
        The classification type is determined during initialization:
        - If classification_type='adapted': Uses 3-class KMRF labels
        - If classification_type='original': Uses 4-regime KAMA+MSR labels
        
        Feature selection (if select_features=True):
        - Uses standard Boruta algorithm
        - Applied ONLY on training data to avoid data leakage
        - Same features used for validation and test sets
        """
        if self.features is None or self.labels is None:
            raise ValueError(
                "Features and labels must be available first. "
                "Call get_features() and load_kama_msr_labels()"
            )
        
        print(f"\n{'='*80}")
        print(f"PREPARING TRAINING DATA FOR {self.asset_name}")
        print(f"{'='*80}")
        
        # Step 1: Start with target asset features
        print(f"\nStep 1: Loading Target Asset Features")
        X = self.features.copy()
        print(f"  Target asset features shape: {X.shape}")
        print(f"  Asset: {self.asset_name}")
        
        # Step 2: Add cross-asset features if requested
        if use_cross_asset_features:
            print(f"\nStep 2: Adding Cross-Asset Features")
            print(f"  Feature asset classes: {self.feature_asset_classes}")
            
            if self.cross_asset_features is None:
                self.load_cross_asset_features(use_master_df=use_master_df)
            
            # Check if target asset features are already in cross_asset_features
            # cross_asset_features has multi-level columns: (asset_name, feature)
            if isinstance(self.cross_asset_features.columns, pd.MultiIndex):
                # Get unique asset names in cross_asset_features
                cross_asset_names = self.cross_asset_features.columns.get_level_values(0).unique().tolist()
                
                # If target asset is in cross_asset_features, remove it to avoid duplication
                if self.asset_name in cross_asset_names:
                    print(f"  Removing target asset from cross-asset features to avoid duplication")
                    # Keep only assets that are NOT the target asset
                    other_assets = [a for a in cross_asset_names if a != self.asset_name]
                    cross_features_to_add = self.cross_asset_features[other_assets].copy()
                else:
                    cross_features_to_add = self.cross_asset_features.copy()
            else:
                cross_features_to_add = self.cross_asset_features.copy()

            if self.cross_asset_specific and len(self.cross_asset_specific) > 0:
                print(f"  Selecting specific assets for cross-asset features: {self.cross_asset_specific}")
                col_mask = cross_features_to_add.columns.get_level_values(0).isin(self.cross_asset_specific)
                cross_features_to_add = cross_features_to_add.loc[:, col_mask]

            print(f"  Cross-asset features to add: {cross_features_to_add.shape}")
            print(f"  Assets included ({cross_features_to_add.columns.get_level_values(0).nunique()}): {cross_features_to_add.columns.get_level_values(0).unique().tolist()}")
            
            # Align cross-asset features to target asset index
            cross_features_aligned = cross_features_to_add.reindex(X.index).ffill()
            
            # Combine target asset features with cross-asset features
            X = pd.concat([X, cross_features_aligned], axis=1)
            print(f"  Combined features shape: {X.shape}")
            print(f"  = Target ({self.features.shape[1]}) + Cross-asset ({cross_features_aligned.shape[1]})")
        else:
            print(f"\nStep 2: Skipping cross-asset features")
        
        # Step 3: Load and combine macro data if requested
        if include_macro:
            print(f"\nStep 3: Loading Macroeconomic Data")
            
            # Check if we're using master_df (which includes macro data)
            if self.use_data_type == 'master':
                # Extract macro data from master_df
                if use_master_df:
                    # Load master_df to extract macro data
                    master_df_path = BASE_DIR / 'data' / 'master_df.csv'
                    master_df = pd.read_csv(master_df_path, header=[0, 1, 2], index_col=0, parse_dates=True)
                    
                    # Check if macro_daily exists in master_df
                    if 'macro_daily' in master_df.columns.get_level_values(0):
                        macro_data = master_df['macro_daily'].copy()
                        print(f"  Extracted macro data from master_df.csv")
                        print(f"  Macro features shape: {macro_data.shape}")
                        
                        # Flatten macro data columns (it has 2 levels after selecting 'macro_daily')
                        # Format: (asset_name, feature) -> we want just the feature names
                        if isinstance(macro_data.columns, pd.MultiIndex):
                            # Create new column names: asset_name + '_' + feature
                            new_cols = ['_'.join(col).strip() for col in macro_data.columns.values]
                            macro_data.columns = new_cols
                        
                        # Align macro data to feature index
                        macro_aligned = macro_data.reindex(X.index).ffill()
                        X = pd.concat([X, macro_aligned], axis=1)
                        print(f"  Combined shape: {X.shape}")
                        print(f"  = Previous ({X.shape[1] - macro_aligned.shape[1]}) + Macro ({macro_aligned.shape[1]})")
                    else:
                        print(f"  Warning: 'macro_daily' not found in master_df.csv")
                else:
                    print(f"  Macro data already in features (use_master_df=True expected)")
            else:
                # Load macro data from separate file for 'ready' or 'raw' modes
                if self.macro_data is None:
                    self.load_macro_data()
                
                if self.macro_data is not None:
                    # Align macro data to feature index
                    macro_aligned = self.macro_data.reindex(X.index).ffill()
                    X = pd.concat([X, macro_aligned], axis=1)
                    print(f"  Combined shape: {X.shape}")
                else:
                    print(f"  Macro data not available, skipping...")
        else:
            print(f"\nStep 3: Skipping macroeconomic data")
        
        # Step 4: Get labels based on classification type
        if self.classification_type == 'adapted':
            if self.adapted_labels is not None:
                # Use pre-adapted labels
                print(f"\nStep 4: Using Pre-Adapted Labels")
                print(f"  (3-class labels from previous adapt_regime_labels() call)")
                y = self.adapted_labels.copy()
            else:
                # Adapt now
                print(f"\nStep 4: Adapting Labels")
                print("  4-regime → 3-class (Bullish=1, Bearish=-1, Other=0)")
                y = self.adapt_regime_labels(price_data=None, labels=self.labels)
        else:  # original
            print(f"\nStep 4: Using Original 4-Regime Labels")
            print(f"  (LV Bullish=0, LV Bearish=1, HV Bullish=2, HV Bearish=3)")
            y = self.labels.copy()
        
        print(f"  Labels Shape: ({len(y)},)")
        
        # Step 5: Clean features (but keep all dates for validation/test)
        print(f"\nStep 5: Cleaning Features")
        print(f"  Features date range: {X.index[0]} to {X.index[-1]}")
        print(f"  Labels date range: {y.index[0]} to {y.index[-1]}")
        print(f"  Note: Keeping all feature dates (including validation/test periods)")
        
        # Identify columns with too many NaN values across ALL data
        nan_threshold = 0.5
        nan_counts = X.isna().sum()
        total_rows = len(X)
        bad_cols = nan_counts[nan_counts > total_rows * nan_threshold].index.tolist()
        
        if bad_cols:
            print(f"  Dropping {len(bad_cols)} features with >{nan_threshold*100:.0f}% NaN values")
            X = X.drop(columns=bad_cols)
        
        # Forward-fill and backward-fill NaN values
        X = X.ffill().bfill()
        
        print(f"  Features cleaned and aligned")
        
        # Step 6: Expand feature window if requested
        if self.feature_window_size > 1:
            print(f"\nStep 6: Expanding Feature Window")
            X = self.expand_feature_window(X, window_size=self.feature_window_size)
            print(f"  Expanded features shape: {X.shape}")
            print(f"  Features date range: {X.index[0]} to {X.index[-1]}")
        else:
            print(f"\nStep 6: Using standard single-step features (feature_window_size=1)")
        
        # Step 7: Split data if requested
        if split_data:
            print(f"\nStep 7: Splitting Data")
            self.split_train_val_test(X, y, select_features=select_features, boruta_params=boruta_params)
            return self.X_train, self.y_train
        else:
            print(f"\nStep 7: No data splitting")
            
            # Step 8: Feature selection on full dataset
            if select_features:
                print(f"\nStep 8: Feature Selection (Boruta)")
                if boruta_params is None:
                    boruta_params = {'max_iter': 100}
                
                selected_features = self.select_features_boruta(X, y, **boruta_params)
                X = X[selected_features]
            else:
                print(f"\nStep 8: Skipping feature selection")
            
            return X, y
    
    def split_train_val_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        select_features: bool = False,
        boruta_params: Optional[Dict] = None
    ) -> None:
        """
        Split data into train, validation, and test sets.
        
        Training: Where KAMA+MSR labels exist (features + labels)
        Validation: validation_start to validation_end (features only, no labels)
        Test: test_start onwards (features only, no labels)
        
        Note: KAMA+MSR labels only exist for training period to avoid lookahead bias.
        Validation and test sets have features but no labels - they're used for
        generating predictions from the trained model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (extends beyond label period)
        y : pd.Series
            Labels (only available for training period)
        select_features : bool, default=False
            Apply Boruta feature selection on training data
        boruta_params : dict, optional
            Parameters for Boruta
        """
        print(f"  Train/Validation/Test Split:")
        print(f"    Training   : Before {self.validation_start.date()}")
        print(f"    Validation : {self.validation_start.date()} to {self.validation_end.date()}")
        print(f"    Test       : {self.test_start.date()} to End")
        
        # Split features by dates FIRST (features have full date range)
        train_dates_mask = X.index < self.validation_start
        val_mask = (X.index >= self.validation_start) & (X.index <= self.validation_end)
        test_mask = X.index >= self.test_start
        
        # Get training features where dates < validation_start
        X_train_dates = X[train_dates_mask].copy()
        
        # Intersect training features with label dates (filter to where labels exist)
        common_train_idx = X_train_dates.index.intersection(y.index)
        self.X_train = X_train_dates.loc[common_train_idx]
        self.y_train = y.loc[common_train_idx]
        # push train start date forward 252d to make sure all features are valid - i.e. start at 253rd day
        self.X_train = self.X_train.iloc[252:]
        self.y_train = self.y_train.loc[self.X_train.index]

        # Validation: Features only (from full feature set X)
        self.X_val = X[val_mask].copy()
        self.y_val = None
        
        # Test: Features only (from full feature set X)
        self.X_test = X[test_mask].copy()
        self.y_test = None
        
        print(f"\n  Split sizes:")
        if len(self.X_train) > 0:
            print(f"    Training   : {len(self.X_train)} samples with labels ({self.X_train.index[0].date()} to {self.X_train.index[-1].date()})")
        else:
            print(f"    Training   : 0 samples (EMPTY)")
        
        if len(self.X_val) > 0:
            print(f"    Validation : {len(self.X_val)} samples (features only, no labels) ({self.X_val.index[0].date()} to {self.X_val.index[-1].date()})")
        else:
            print(f"    Validation : 0 samples (EMPTY)")
        
        if len(self.X_test) > 0:
            print(f"    Test       : {len(self.X_test)} samples (features only, no labels) ({self.X_test.index[0].date()} to {self.X_test.index[-1].date()})")
        else:
            print(f"    Test       : 0 samples (EMPTY)")
        
        # Feature selection on training data only
        if select_features:
            print(f"\n  Applying Boruta feature selection on training data...")
            if boruta_params is None:
                boruta_params = {'max_iter': 100}
            
            selected_features = self.select_features_boruta(self.X_train, self.y_train, **boruta_params)
            
            # Apply same features to validation and test
            self.X_train = self.X_train[selected_features]
            if len(self.X_val) > 0:
                self.X_val = self.X_val[selected_features]
            if len(self.X_test) > 0:
                self.X_test = self.X_test[selected_features]
            
            print(f"  Feature selection applied to all splits")
        
        # Print label distributions (only for training)
        print(f"\n  Label distribution:")
        print(f"    Training (with labels):")
        if len(self.y_train) > 0:
            dist = self.y_train.value_counts().sort_index()
            
            if self.classification_type == 'adapted':
                label_map = {-1: 'Bearish', 0: 'Other', 1: 'Bullish'}
                for label_val in [1, 0, -1]:
                    if label_val in dist:
                        count = dist[label_val]
                        pct = (count / len(self.y_train)) * 100
                        print(f"      {label_map[label_val]:>8} ({label_val:>2}): {count:>5} ({pct:>5.1f}%)")
            else:  # original
                label_map = {0: 'LV_Bull', 1: 'LV_Bear', 2: 'HV_Bull', 3: 'HV_Bear'}
                for label_val in [0, 1, 2, 3]:
                    if label_val in dist:
                        count = dist[label_val]
                        pct = (count / len(self.y_train)) * 100
                        print(f"      {label_map[label_val]:>8} ({label_val:>2}): {count:>5} ({pct:>5.1f}%)")
        else:
            print(f"      EMPTY")
        
        print(f"    Validation: No labels (prediction only)")
        print(f"    Test: No labels (prediction only)")
        
        print(f"\n{'='*80}")
        print(f"DATA PREPARATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nNote: Validation and test sets have features only.")
        print(f"      Use trained model to generate predictions on these sets.")

    def adjust_train_val_test_dates(self, 
                              val_start: pd.Timestamp | str, 
                              val_end: pd.Timestamp | str, 
                              test_start: pd.Timestamp | str, 
                              test_end: pd.Timestamp | str) -> None:
        if self.X_train is None:
            raise ValueError("Training data not available. Cannot adjust validation/test dates.")

        # print(self.X_train.first_valid_index() - self.raw_data.first_valid_index())
        if self.X_train.first_valid_index() - self.raw_data.first_valid_index() < pd.Timedelta(days=360):
            # Training data does not have sufficient lookback period for complete feature computation
            self.X_train = self.X_train.iloc[252:]
            self.y_train = self.y_train.loc[self.X_train.index]
            print(f"\nADJUSTED TRAINING DATA FORWARD 1 YEAR TO ENSURE COMPLETE FEATURE COMPUTATION\nMUST RETRAIN MODEL")

        X_val_plus_test = pd.concat([self.X_val, self.X_test], axis=0)

        new_X_val = X_val_plus_test.loc[val_start:val_end]
        new_X_test = X_val_plus_test.loc[test_start:test_end]
        self.X_val = new_X_val
        self.X_test = new_X_test
        self.validation_start = pd.to_datetime(val_start)
        self.validation_end = pd.to_datetime(val_end)
        self.test_start = pd.to_datetime(test_start)

    def __repr__(self) -> str:
        """String representation of the KMRF model."""
        status = [f"KMRF('{self.asset_class}, {self.asset_name}')"]
        
        if self.raw_ohlc is not None:
            status.append(f"Raw OHLC Data {self.raw_ohlc.shape}: {self.raw_ohlc.index[0].date()} to {self.raw_ohlc.index[-1].date()}")
        
        if self.features is not None:
            status.append(f"Technical Features {self.features.shape}: {self.features.index[0].date()} to {self.features.index[-1].date()}")

        if self.macro_data is not None:
            status.append(f"Macro Features {self.macro_data.shape}: {self.macro_data.index[0].date()} to {self.macro_data.index[-1].date()}")

        if self.labels is not None:
            status.append(f"4-Regime Labels ({len(self.labels)}): {self.labels.index[0].date()} to {self.labels.index[-1].date()}")
        
        if self.adapted_labels is not None:
            status.append(f"3-Class Labels ({len(self.adapted_labels)}): {self.adapted_labels.index[0].date()} to {self.adapted_labels.index[-1].date()}")
        
        if self.X_train is not None:
            status.append(f"Train {self.X_train.shape}: {self.X_train.index[0].date()} to {self.X_train.index[-1].date()}")
        
        if self.X_val is not None:
            status.append(f"Val {self.X_val.shape}: {self.X_val.index[0].date()} to {self.X_val.index[-1].date()}")
        
        if self.X_test is not None:
            status.append(f"Test {self.X_test.shape}: {self.X_test.index[0].date()} to {self.X_test.index[-1].date()}")
        
        return "\n".join(status)
    
    def get_split_info(self) -> dict:
        """
        Get information about the data splits.
        
        Returns
        -------
        dict
            Dictionary containing split information
        """
        info = {
            'train': {
                'size': len(self.X_train) if self.X_train is not None else 0,
                'features': self.X_train.shape[1] if self.X_train is not None else 0,
                'has_labels': self.y_train is not None,
                'date_range': (
                    f"{self.X_train.index[0].date()} to {self.X_train.index[-1].date()}"
                    if self.X_train is not None and len(self.X_train) > 0 else "N/A"
                )
            },
            'validation': {
                'size': len(self.X_val) if self.X_val is not None else 0,
                'features': self.X_val.shape[1] if self.X_val is not None else 0,
                'has_labels': False,
                'date_range': (
                    f"{self.X_val.index[0].date()} to {self.X_val.index[-1].date()}"
                    if self.X_val is not None and len(self.X_val) > 0 else "N/A"
                )
            },
            'test': {
                'size': len(self.X_test) if self.X_test is not None else 0,
                'features': self.X_test.shape[1] if self.X_test is not None else 0,
                'has_labels': False,
                'date_range': (
                    f"{self.X_test.index[0].date()} to {self.X_test.index[-1].date()}"
                    if self.X_test is not None and len(self.X_test) > 0 else "N/A"
                )
            }
        }
        return info
    
    def fit(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.DataFrame] = None,
        xgb_params: Optional[Dict] = None
    ) -> 'KMRF':
        """
        Fit XGBoost classifier for regime prediction.
        
        This method trains the XGB component of the KMRF model using prepared features
        and adapted labels. The trained model can then generate ex-ante regime predictions.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix. If None, uses output from prepare_training_data()
        y : pd.DataFrame, optional
            Target labels. If None, uses output from prepare_training_data()
        xgb_params : dict, optional
            XGBoost hyperparameters. Defaults based on paper Table 2:
            - n_estimators: 100-300
            - max_depth: 1-20
            - learning_rate: 0.01-0.3
            - subsample: 0.5-1.0
            If provided, overrides instance-level xgb_params.
            
        Returns
        -------
        KMRF
            Self (for method chaining)
            
        Notes
        -----
        Per the paper, hyperparameters should be optimized using Optuna to maximize
        the Sortino ratio. This implementation uses reasonable defaults.
        
        Examples
        --------
        >>> kmrf.fit()  # Uses all training data
        >>> kmrf.fit(xgb_params={'n_estimators': 200})  # Override XGB params
        """
        from xgboost import XGBClassifier
        
        if X is None or y is None:
            X = self.X_train
            y = self.y_train
        
        print(f"\n{'='*80}")
        print(f"TRAINING XGBOOST CLASSIFIER")
        print(f"{'='*80}")
        
        # Merge instance-level and call-level XGB parameters
        # Call-level params override instance-level params
        merged_xgb_params = self.xgb_params.copy() if self.xgb_params else {}
        if xgb_params:
            merged_xgb_params.update(xgb_params)
        
        # Default XGB parameters (converting RF-based values to XGB equivalents)
        if self.asset_class == 'commodity':
            default_params = {
                'n_estimators': 280,  # Same as paper's RF n_estimators
                'max_depth': 3,
                'learning_rate': 0.1,  # XGB default
                'subsample': 0.8,  # XGB equivalent of max_samples
                'colsample_bytree': 0.4,  # XGB equivalent of max_features
                'min_child_weight': 95,  # XGB equivalent of min_samples_leaf
                'gamma': 0.02,  # XGB regularization
                'random_state': self.random_seed,
                'n_jobs': -1,
                'tree_method': 'hist',
                'enable_categorical': False
            }
        else:    
            default_params = {
                'n_estimators': 220,  # Same as paper's RF n_estimators
                'max_depth': 13,
                'learning_rate': 0.1,  # XGB default
                'subsample': 0.8,  # XGB equivalent of max_samples
                'colsample_bytree': 0.25,  # XGB equivalent of max_features
                'min_child_weight': 95,  # XGB equivalent of min_samples_leaf
                'gamma': 0.045,  # XGB regularization
                'random_state': self.random_seed,
                'n_jobs': -1,
                'tree_method': 'hist',
                'enable_categorical': False
            }
        
        # Update defaults with merged params
        default_params.update(merged_xgb_params)
        
        print(f"XGB Parameters:")
        for key, val in default_params.items():
            print(f"  {key}: {val}")
        
        # Flatten multi-index for sklearn
        X_flat = X.copy()
        X_flat.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) 
                          for col in X.columns]
        
        y_flat = y.iloc[:, 0] if len(y.shape) > 1 else y
        # y_shifted = y.shift(-1)  # Shift labels to predict next period
        
        # Remove NaN
        valid_idx = ~(X_flat.isna().any(axis=1) | y_flat.isna())
        X_clean = X_flat[valid_idx]
        y_clean = y_flat[valid_idx]
        
        print(f"\nTraining samples: {len(X_clean)}")
        print(f"Features: {X_clean.shape[1]}")
        print(f"Date range: {X_clean.index[0]} to {X_clean.index[-1]}")
        print(f"\nClass distribution:")
        print(y_clean.value_counts().sort_index())
        
        # XGBoost requires labels to start from 0 and be consecutive integers
        
        if self.classification_type == 'adapted':
            # Adapted: {-1, 0, 1} -> {0, 1, 2}
            # But if not all regimes are present, we need to handle it similarly to 'original'
            unique_regimes = sorted(y_clean.unique())
            
            # Check if all 3 regimes are present
            expected_regimes = [-1, 0, 1]
            missing_regimes = [r for r in expected_regimes if r not in unique_regimes]
            
            if missing_regimes:
                print(f"\n⚠️  WARNING: Not all regimes present in training data!")
                print(f"  Expected regimes: {expected_regimes}")
                print(f"  Present regimes:  {unique_regimes}")
                print(f"  Missing regimes:  {missing_regimes}")
                print(f"  Missing regime predictions will be set to 0.0 probability")
            
            # Always use the full mapping even if some regimes are missing
            self._label_mapping = {-1: 0, 0: 1, 1: 2}
            self._inverse_label_mapping = {0: -1, 1: 0, 2: 1}
            y_clean_mapped = y_clean.map(self._label_mapping).astype(int)
            
            print(f"\nRemapping labels for XGBoost (adapted classification):")
            print(f"  Original labels: {unique_regimes}")
            print(f"  Mapped labels:   {sorted(y_clean_mapped.unique())}")
            print(f"  Mapping: {self._label_mapping}")
        else:
            # Original: Labels are [0, 1, 2, 3] representing [LV_Bull, LV_Bear, HV_Bull, HV_Bear]
            # But if not all regimes are present in training, we need to map them to consecutive integers [0, 1, 2, ...]
            unique_regimes = sorted(y_clean.unique())
            
            # Check if all 4 regimes are present
            expected_regimes = [0, 1, 2, 3]
            missing_regimes = [r for r in expected_regimes if r not in unique_regimes]
            
            if missing_regimes:
                print(f"\n⚠️  WARNING: Not all regimes present in training data!")
                print(f"  Expected regimes: {expected_regimes}")
                print(f"  Present regimes:  {unique_regimes}")
                print(f"  Missing regimes:  {missing_regimes}")
                print(f"  Missing regime predictions will be set to 0.0 probability")
            
            # Create mapping from original regime number to consecutive integers
            self._label_mapping = {int(regime): idx for idx, regime in enumerate(unique_regimes)}
            # Create inverse mapping from XGBoost's consecutive integers back to original regime numbers
            self._inverse_label_mapping = {idx: int(regime) for regime, idx in self._label_mapping.items()}
            
            y_clean_mapped = y_clean.map(self._label_mapping).astype(int)
            
            print(f"\nRemapping labels for XGBoost (original classification):")
            print(f"  Original regime labels: {unique_regimes}")
            print(f"  Mapped to XGBoost:      {sorted(y_clean_mapped.unique())}")
            print(f"  Mapping: {self._label_mapping}")
            print(f"  Inverse mapping: {self._inverse_label_mapping}")
        
        # Train XGBoost
        print(f"\nFitting XGBoost...")
        # Delete any existing model to avoid class mismatch issues
        # This is important when fit() is called multiple times (e.g., during feature selection)
        if hasattr(self, 'xgb_model') and self.xgb_model is not None:
            del self.xgb_model
        self.xgb_model = XGBClassifier(**default_params)
        self.xgb_model.fit(X_clean.values, y_clean_mapped.values)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Model ready for prediction")
        
        return self
    
    def predict(
        self,
        X: Optional[pd.DataFrame] = None,
        test_or_val: str = 'val'
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Generate regime predictions for new data.
        
        This method produces ex-ante regime predictions for the next trading day.
        Returns probabilities for each of the regime classes per asset.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for prediction. If None, uses X_test from train/val/test split
        test_or_val : str, default='test'
            Whether to use test or validation data if X is None
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for probability of each regime class
            
        Notes
        -----
        Predictions should be interpreted with CONTRARIAN strategy (per paper):
        - High P(Bull) → SHORT signal (market overbought)
        - High P(Bear) → LONG signal (market oversold)  
        - High P(Other) → CLOSE position
        
        Examples
        --------
        >>> predictions = kmrf.predict()  # Get probabilities for X_test
        >>> predictions = kmrf.predict(X_test)  # Explicit
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call fit() first or load a saved model.")
        
        if X is None:
            if self.X_test is None and self.X_val is None:
                raise ValueError("No test or validation data available. Either provide X or run prepare_training_data() with split_data=True.")
            X = self.X_test if test_or_val.lower() == 'test' else self.X_val

        print(f"\nGenerating predictions...")
        print(f"  Input shape: {X.shape}")
        
        # Apply feature selection if model was trained with selected features
        if self.selected_features is not None and len(self.selected_features) > 0:
            # Check if we need to select features or if they're already selected
            missing_features = set(self.selected_features) - set(X.columns)
            if missing_features:
                # X has more features than selected - need to filter
                print(f"  Applying feature selection: {len(self.selected_features)} features")
                X = X[self.selected_features]
            else:
                # X already has only the selected features
                print(f"  Using pre-selected features: {len(X.columns)}")
        
        # Flatten multi-index columns if present
        X_flat = X.copy()
        X_flat.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) 
                          for col in X.columns]
        
        # Handle NaN values
        X_clean = X_flat.fillna(method='ffill').fillna(method='bfill')
        
        # Get probabilities for each class
        proba = self.xgb_model.predict_proba(X_clean.values)
        
        # Create DataFrame with meaningful column names
        # XGBoost only outputs probabilities for classes it was trained on
        
        if self.classification_type == 'adapted':
            # For adapted classification, XGBoost outputs probabilities for the classes it learned
            # We need to map them back to the original regime numbers using inverse_label_mapping
            regime_names = {-1: 'P(Bear)', 0: 'P(Other)', 1: 'P(Bull)'}
            
            # XGBoost's classes_ gives us the consecutive integers [0, 1, 2, ...]
            # Map them back to original regime numbers, then to regime names
            columns = []
            for xgb_class in self.xgb_model.classes_:
                original_regime = self._inverse_label_mapping[int(xgb_class)]
                columns.append(regime_names[original_regime])
            
            all_columns = [regime_names[i] for i in [-1, 0, 1]]  # All 3 regimes for final output
            
        else:  # original
            # For original classification, XGBoost outputs probabilities for the classes it learned
            # We need to map them back to the original regime numbers using inverse_label_mapping
            regime_names = {0: 'P(LV_Bull)', 1: 'P(LV_Bear)', 2: 'P(HV_Bull)', 3: 'P(HV_Bear)'}
            
            # XGBoost's classes_ gives us the consecutive integers [0, 1, 2, ...]
            # Map them back to original regime numbers, then to regime names
            columns = []
            for xgb_class in self.xgb_model.classes_:
                original_regime = self._inverse_label_mapping[int(xgb_class)]
                columns.append(regime_names[original_regime])
            
            all_columns = [regime_names[i] for i in range(4)]  # All 4 regimes for final output
        
        result = pd.DataFrame(
            proba,
            index=X.index,
            columns=columns
        )

        if self.classification_type == 'original':
            # Ensure all 4 regime columns are present, filling missing ones with 0.0
            for col in all_columns:
                if col not in result.columns:
                    result[col] = 0.0  # Fill missing columns with zeros
            result = result[all_columns]
        else:  # adapted
            # Ensure all 3 regime columns are present, filling missing ones with 0.0
            for col in all_columns:
                if col not in result.columns:
                    result[col] = 0.0  # Fill missing columns with zeros
            result = result[all_columns]

        print(f"✓ Generated probability predictions: {result.shape}")
        if test_or_val.lower() == 'test':
            self.y_test_proba = result
        else:
            self.y_val_proba = result
        return result
    
    def viz_predictions(self, test_or_val: str = 'test') -> None:
        if test_or_val.lower() == 'test':
            if self.y_test_proba is None:
                raise ValueError("No test predictions available. Generate predictions first.")
                # pred_proba = self.predict(test_or_val='test')
            else:
                pred_proba = self.y_test_proba
        else:
            if self.y_val_proba is None:
                raise ValueError("No validation predictions available. Generate predictions first.")
                # pred_proba = self.predict(test_or_val='val')
            else:
                pred_proba = self.y_val_proba

        if self.classification_type == 'original':
            for col in ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']:
                if col not in pred_proba.columns:
                    pred_proba[col] = 0.0  # Fill missing columns with zeros
            pred_proba = pred_proba[['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']]
            pred = pred_proba.apply(lambda row: np.argmax(row), axis=1)
        else:
            for col in ['P(Bull)', 'P(Other)', 'P(Bear)']:
                if col not in pred_proba.columns:
                    pred_proba[col] = 0.0  # Fill missing columns with zeros
            pred_proba = pred_proba[['P(Bear)', 'P(Other)', 'P(Bull)']]
            pred = pred_proba.apply(lambda row: np.argmax(row)-1, axis=1)

        # Get raw price data for test period - aligned with prediction dates
        asset_price = self.raw_ohlc[[(self.asset_name, 'open'), (self.asset_name, 'close')]]
        asset_price = asset_price.loc[pred_proba.index]
        asset_price.columns = ['open', 'close']

        # Create comprehensive visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])

        # === Price Plot with Regime Overlays ===
        # Plot asset price
        ax1.plot(asset_price.index, asset_price['close'], color='black', linewidth=1.5, 
                    alpha=0.9, label='Asset Price', zorder=3)

        # Define colors and labels for regimes based on classification type
        if self.classification_type == 'original':
            colors = {0: 'green', 1: 'blue', 2: 'yellow', 3: 'darkred'}
            labels_dict = {0: 'LV_Bull', 1: 'LV_Bear', 2: 'HV_Bull', 3: 'HV_Bear'}
            prob_cols = {0: 'P(LV_Bull)', 1: 'P(LV_Bear)', 2: 'P(HV_Bull)', 3: 'P(HV_Bear)'}
        else:
            colors = {1: 'green', 0: 'grey', -1: 'darkred'}
            labels_dict = {1: 'Bullish', 0: 'Other', -1: 'Bearish'}
            prob_cols = {1: 'P(Bull)', 0: 'P(Other)', -1: 'P(Bear)'}


        # Shade regime periods
        for regime in labels_dict:
            mask = pred == regime
            if mask.any():
                ax1.fill_between(
                    asset_price.index,
                    asset_price['close'].min() * 0.95,
                    asset_price['close'].max() * 1.05,
                    where=mask,
                    alpha=0.4,
                    color=colors[regime],
                    label=f'Predicted {labels_dict[regime]}',
                    zorder=1
                )

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{self.asset_name} with Predicted Regimes - {test_or_val.capitalize()} Period ({pred_proba.index[0].date()} to {pred_proba.index[-1].date()})', 
                        fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3, zorder=2)

        # === Regime Probabilities ===
        # Plot regime probabilities
        # Handle different regime types based on classification
        if self.classification_type == 'original':
            # Original 4-regime system
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(LV_Bull)'],
                            alpha=0.7, color='green', label='P(LV Bullish)')
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(LV_Bear)'],
                            alpha=0.7, color='blue', label='P(LV Bearish)')
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(HV_Bull)'],
                            alpha=0.7, color='yellow', label='P(HV Bullish)')
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(HV_Bear)'],
                            alpha=0.7, color='darkred', label='P(HV Bearish)')
        else:
            # Adapted 3-regime system
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(Bull)'],
                            alpha=0.7, color='green', label='P(Bull)')
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(Other)'],
                            alpha=0.7, color='gray', label='P(Other)')
            ax2.fill_between(pred_proba.index, 0, pred_proba['P(Bear)'],
                            alpha=0.7, color='darkred', label='P(Bear)')

        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Regime Prediction Probabilities', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # === Summary Statistics ===
        print(f"\n" + "="*60)
        print(f"PREDICTION SUMMARY FOR {self.asset_name}")
        print(f"="*60)
        print(f"Test Period: {pred_proba.index[0].date()} to {pred_proba.index[-1].date()}")
        print(f"Total Trading Days: {len(pred_proba)}")

        # Price performance during test period
        price_start = asset_price['open'].iloc[0]
        price_end = asset_price['close'].iloc[-1]
        total_return = (price_end - price_start) / price_start * 100

        print(f"\nPrice Performance:")
        print(f"  Start Price (open on {asset_price.index[0].date()}): ${price_start:.2f}")
        print(f"  End Price (close on {asset_price.index[-1].date()}): ${price_end:.2f}")
        print(f"  Buy and Hold Return: {total_return:.2f}%")

        # Regime statistics
        if self.classification_type == 'original':
            regime_names = {
                0: 'LV Bullish',
                1: 'LV Bearish',
                2: 'HV Bullish',
                3: 'HV Bearish'
            }
        else: # adapted
            regime_names = {
                -1: 'Bearish',
                0: 'Other',
                1: 'Bullish'
            }
        print(f"\nRegime Prediction Statistics:")
        for regime, name in regime_names.items():
            count = (pred == regime).sum()
            pct = (count / len(pred)) * 100
            avg_prob = pred_proba[prob_cols[regime]].mean()
            print(f"  {name:>8}: {count:>3} days ({pct:>5.1f}%) | Avg Prob: {avg_prob:.3f}")

        # High confidence predictions
        high_confidence = pred_proba.max(axis=1) > 0.7
        print(f"\nHigh Confidence Predictions (>70%): {high_confidence.sum()} days ({high_confidence.mean()*100:.1f}%)")

        plt.tight_layout()
        plt.show()

    def viz_training_labels(self) -> None:
        # TODO: implement
        pass

    def save_model(self, model_path: Union[str, Path]) -> Path:
        """
        Save trained KMRF model to disk.
        
        Saves the complete model including:
        - XGBoost classifier
        - Selected features (if feature selection was used)
        - Train/val/test splits (X_train, y_train, X_val, X_test)
        - All model data (features, labels, cross-asset features, etc.)
        - Model metadata (asset class, end_date, classification_type, etc.)
        
        This allows you to load the model later without retraining or re-running feature selection.
        
        Parameters
        ----------
        model_path : str or Path
            Full path where the model should be saved (including filename and .pkl extension)
            Parent directories will be created if they don't exist
            
        Returns
        -------
        Path
            Full path to saved model file
            
        Examples
        --------
        >>> # Save model with explicit path
        >>> kmrf.fit()
        >>> model_path = kmrf.save_model('saved_models/my_kmrf_model.pkl')
        >>> 
        >>> # Load later
        >>> kmrf = KMRF.load_model('saved_models/my_kmrf_model.pkl')
        """
        if self.xgb_model is None:
            raise ValueError("No model to save. Train model first with fit().")
        
        model_path = Path(model_path)
        
        # Create parent directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Package model data - save everything needed to skip retraining
        model_data = {
            'xgb_model': self.xgb_model,
            'selected_features': self.selected_features,
            'asset_class': self.asset_class,
            'asset_name': self.asset_name,
            'end_date': self.end_date,
            'classification_type': self.classification_type,
            'random_seed': self.random_seed,
            'model_params': self.xgb_model.get_params(),
            'validation_start': self.validation_start,
            'validation_end': self.validation_end,
            'test_start': self.test_start,
            # Feature engineering parameters
            'feature_window_size': self.feature_window_size,
            'feature_asset_classes': self.feature_asset_classes,
            'cross_asset_specific': self.cross_asset_specific,
            'xgb_params': self.xgb_params,
            'use_boruta_selection': self.use_boruta_selection,
            'use_consensus_selection': self.use_consensus_selection,
            # Label mapping for adapted classification
            'label_mapping': self._label_mapping,
            'inverse_label_mapping': self._inverse_label_mapping,
            # Save all data
            'raw_data': self.raw_data,
            'ohlc_data': self.raw_ohlc,
            'features': self.features,
            'labels': self.labels,
            'adapted_labels': self.adapted_labels,
            'macro_data': self.macro_data,
            'cross_asset_features': self.cross_asset_features,
            # Save train/val/test splits
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_val_proba': self.y_val_proba,
            'y_test_proba': self.y_test_proba,
        }
        
        # Save
        print(f"\nSaving model to: {model_path}")
        print(f"  Asset: {self.asset_name}")
        print(f"  Asset class: {self.asset_class}")
        print(f"  Classification type: {self.classification_type}")
        print(f"  Features: {len(self.selected_features) if self.selected_features else 'all'}")
        print(f"  Training samples: {len(self.X_train) if self.X_train is not None else 0}")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved successfully")
        
        return model_path

    def pipeline(
        self, 
        optimize: bool = False
    ) -> None:
        """
        Run the full KMRF pipeline: load data, prepare features, train model.
        
        This method executes the complete training workflow using parameters
        specified during model initialization (in __init__). It:
        1. Loads data (raw/ready/master based on use_data_type)
        2. Computes/loads features for target asset
        3. Loads cross-asset features (if feature_asset_classes specified)
        4. Loads macroeconomic data
        5. Loads KAMA+MSR regime labels
        6. Adapts labels if classification_type='adapted'
        7. Prepares training data with feature engineering
        8. Applies feature selection (Boruta, Consensus, or None based on flags)
        9. Splits into train/val/test sets
        10. Trains XGBoost classifier
        
        All configuration is read from instance variables set during initialization:
        - self.use_data_type: Determines data source ('master', 'ready', 'raw')
        - self.feature_asset_classes: List of asset classes for cross-asset features
        - self.classification_type: 'adapted' (3-class) or 'original' (4-regime)
        - self.feature_window_size: Number of time steps to include as features
        - self.use_boruta_selection: If True, use Boruta feature selection
        - self.use_consensus_selection: If True, use consensus feature selection (overrides Boruta)
        - self.xgb_params: XGBoost hyperparameters
        
        Parameters
        ----------
        optimize : bool, default=False
            Whether to run hyperparameter optimization
            If False, runs the full pipeline with parameters as set by initialization
            If True, runs Optuna optimization to find best hyperparameters (TODO: implement)
        
        Notes
        -----
        Feature selection behavior:
        - If use_consensus_selection=True: Uses consensus method (RF importance + variance + MI)
        - Elif use_boruta_selection=True: Uses Boruta algorithm
        - Else: Uses all features (no selection)
        
        Consensus selection is recommended for faster, more robust feature selection.
        
        Examples
        --------
        >>> # Initialize with configuration
        >>> kmrf = KMRF(
        ...     asset_name='SPDR S&P 500 ETF',
        ...     asset_class='us_equity',
        ...     use_data_type='master',
        ...     feature_asset_classes=['us_equity', 'commodity'],
        ...     classification_type='adapted',
        ...     feature_window_size=1,
        ...     use_consensus_selection=True
        ... )
        >>> 
        >>> # Run complete pipeline
        >>> kmrf.pipeline()
        >>> 
        >>> # Model is now trained and ready for predictions
        >>> predictions = kmrf.predict()
        """
        if optimize:
            # TODO: Implement hyperparameter optimization workflow
            # This should:
            # 1. Set up Optuna study with appropriate search spaces
            # 2. Define objective function (e.g., Sortino ratio on validation set)
            # 3. Run optimization trials
            # 4. Update model with best parameters
            # 5. Retrain on full training set with optimal hyperparameters
            raise NotImplementedError(
                "Hyperparameter optimization (optimize=True) is not yet implemented. "
                "This will be added in a future update to support Optuna-based "
                "optimization of RF parameters, feature selection parameters, and "
                "strategy parameters."
            )
        
        print(f"\n{'='*80}")
        print(f"KMRF PIPELINE FOR {self.asset_name}")
        print(f"{'='*80}")
        print(f"Asset Class: {self.asset_class}")
        print(f"Classification Type: {self.classification_type}")
        print(f"Data Type: {self.use_data_type}")
        print(f"Feature Asset Classes: {self.feature_asset_classes}")
        print(f"Cross-Asset Specifics: {self.cross_asset_specific}")
        print(f"    ^ if empty, uses all assets in 'Feature Asset Classes'")
        print(f"Feature Window Size: {self.feature_window_size}")
        print(f"Use Boruta Selection: {self.use_boruta_selection}")
        print(f"Use Consensus Selection: {self.use_consensus_selection}")
        print(f"{'='*80}")
        
        # Step 1: Load data based on use_data_type
        print(f"\n[Step 1/10] Loading Data...")
        use_master_files = (self.use_data_type == 'master')
        raw_or_ready_data = self.load_data(use_master_df=use_master_files)
        
        # Step 2: Get/compute features
        print(f"\n[Step 2/10] Computing Features...")
        features = self.get_features()
        
        # Step 3: Load KAMA+MSR labels
        print(f"\n[Step 3/10] Loading KAMA+MSR Labels...")
        kama_msr_labels = self.load_kama_msr_labels(use_master_label_df=use_master_files)
        
        # Step 4: Adapt labels if needed
        if self.classification_type == 'adapted':
            print(f"\n[Step 4/10] Adapting Labels (4-regime → 3-class)...")
            adapted_labels = self.adapt_regime_labels(kama_msr_labels)
        else:
            print(f"\n[Step 4/10] Using Original 4-Regime Labels...")
        
        # Step 5-9: Prepare training data (includes cross-asset, macro, feature selection, split)
        print(f"\n[Step 5-9/10] Preparing Training Data...")
        print(f"  - Loading cross-asset features: {len(self.feature_asset_classes) > 0}")
        print(f"  - Including macroeconomic data: True")
        
        # Determine feature selection method based on flags
        if self.use_consensus_selection:
            feature_selection_method = "Consensus (RF importance + variance + MI)"
        elif self.use_boruta_selection:
            feature_selection_method = "Boruta"
        else:
            feature_selection_method = "None (using all features)"
        print(f"  - Feature selection method: {feature_selection_method}")
        print(f"  - Train/val/test split: Yes")
        
        # Determine if cross-asset features should be used
        use_cross_asset = len(self.feature_asset_classes) > 0
        
        X_train, y_train = self.prepare_training_data(
            include_macro=True,
            use_cross_asset_features=use_cross_asset,
            use_master_df=use_master_files,
            select_features=False,  # We'll do feature selection separately based on flags
            split_data=True
        )
        
        # Apply feature selection based on initialization flags
        if self.use_consensus_selection:
            # Consensus selection takes precedence
            print(f"\n[Consensus Feature Selection]")
            self.fit()  # Need to fit model first for feature importances
            selected_features, vote_df, all_methods = self.consensus_feature_selection(
                X_train=self.X_train,
                y_train=self.y_train,
                min_votes=2,
                variance_threshold=0.01,
                cumulative_importance_threshold=0.95,
                mi_top_pct=0.3
            )
            
            # Apply selected features to all splits
            print(f"\nApplying selected features to all data splits...")
            self.X_train = self.X_train[selected_features]
            if self.X_val is not None and len(self.X_val) > 0:
                self.X_val = self.X_val[selected_features]
            if self.X_test is not None and len(self.X_test) > 0:
                self.X_test = self.X_test[selected_features]
            
            print(f"  ✓ Training: {self.X_train.shape}")
            if self.X_val is not None:
                print(f"  ✓ Validation: {self.X_val.shape}")
            if self.X_test is not None:
                print(f"  ✓ Test: {self.X_test.shape}")
                
        elif self.use_boruta_selection:
            # Use Boruta feature selection
            print(f"\n[Boruta Feature Selection]")
            if BorutaPy is None:
                print("  ⚠️  WARNING: Boruta not installed. Skipping feature selection.")
                print("  Install with: pip install boruta")
            else:
                selected_features = self.select_features_boruta(
                    X=self.X_train,
                    y=self.y_train,
                    max_iter=100,
                    percentile=100,
                    pvalue=0.01,
                    verbose=2
                )
                
                # Apply selected features to all splits
                print(f"\nApplying selected features to all data splits...")
                self.X_train = self.X_train[selected_features]
                if self.X_val is not None and len(self.X_val) > 0:
                    self.X_val = self.X_val[selected_features]
                if self.X_test is not None and len(self.X_test) > 0:
                    self.X_test = self.X_test[selected_features]
                
                print(f"  ✓ Training: {self.X_train.shape}")
                if self.X_val is not None:
                    print(f"  ✓ Validation: {self.X_val.shape}")
                if self.X_test is not None:
                    print(f"  ✓ Test: {self.X_test.shape}")
        else:
            # No feature selection - use all features
            print(f"\n[No Feature Selection - Using All Features]")
            print(f"  Training: {self.X_train.shape}")
            if self.X_val is not None:
                print(f"  Validation: {self.X_val.shape}")
            if self.X_test is not None:
                print(f"  Test: {self.X_test.shape}")
        
        # Step 10: Train XGBoost Classifier
        print(f"\n[Step 10/10] Training XGBoost Classifier...")
        self.fit()
        
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Model trained and ready for predictions")
        print(f"✓ Use predict() to generate regime probabilities")
        print(f"✓ Use save_model() to persist trained model")
        print(f"{'='*80}\n")

    @classmethod
    def load_model(cls, model_path: Union[str, Path], verbose: bool = False) -> 'KMRF':
        """
        Load a saved KMRF model.
        
        This loads a complete trained model, including:
        - Trained XGBoost classifier
        - Selected features (if feature selection was used)
        - Train/val/test data splits
        - All model data (features, labels, cross-asset features, etc.)
        - All model metadata
        
        After loading, you can immediately use predict() without retraining.
        
        Parameters
        ----------
        model_path : str or Path
            Path to saved model file (.pkl)
        verbose : bool, default=False
            If True, prints detailed information about the loaded model
            
        Returns
        -------
        KMRF
            Loaded model instance with all state restored
            
        Examples
        --------
        >>> # Load a previously trained model
        >>> kmrf = KMRF.load_model('saved_models/my_kmrf_model.pkl')
        >>> 
        >>> # Generate predictions immediately (no training needed)
        >>> predictions = kmrf.predict()
        >>> 
        >>> # Or load with verbose output
        >>> kmrf = KMRF.load_model('saved_models/my_kmrf_model.pkl', verbose=True)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"LOADING SAVED KMRF MODEL")
            print(f"{'='*80}")
            print(f"Model path: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved parameters
        # Note: use_data_type is inferred from saved data presence
        kmrf = cls(
            asset_name=model_data['asset_name'],
            asset_class=model_data['asset_class'],
            end_date=model_data['end_date'],
            use_data_type='master',  # Default, will be overridden by restored data
            validation_start=model_data.get('validation_start'),
            validation_end=model_data.get('validation_end'),
            test_start=model_data.get('test_start'),
            random_seed=model_data.get('random_seed', 1010),
            classification_type=model_data.get('classification_type', 'adapted'),
            feature_window_size=model_data.get('feature_window_size', 1),
            feature_asset_classes=model_data.get('feature_asset_classes'),
            cross_asset_specific=model_data.get('cross_asset_specific'),
            xgb_params=model_data.get('xgb_params'),
            use_boruta_selection=model_data.get('use_boruta_selection', False),
            use_consensus_selection=model_data.get('use_consensus_selection', False)
        )
        
        # Restore model components
        kmrf.xgb_model = model_data['xgb_model']
        kmrf.selected_features = model_data.get('selected_features')
        
        # Restore label mappings for adapted classification
        kmrf._label_mapping = model_data.get('label_mapping')
        kmrf._inverse_label_mapping = model_data.get('inverse_label_mapping')
        
        # Restore all data
        kmrf.raw_data = model_data.get('raw_data')
        kmrf.raw_ohlc = model_data.get('ohlc_data')
        kmrf.features = model_data.get('features')
        kmrf.labels = model_data.get('labels')
        kmrf.adapted_labels = model_data.get('adapted_labels')
        kmrf.macro_data = model_data.get('macro_data')
        kmrf.cross_asset_features = model_data.get('cross_asset_features')
        
        # Restore train/val/test splits
        kmrf.X_train = model_data.get('X_train')
        kmrf.y_train = model_data.get('y_train')
        kmrf.X_val = model_data.get('X_val')
        kmrf.X_test = model_data.get('X_test')
        kmrf.y_val_proba = model_data.get('y_val_proba')
        kmrf.y_test_proba = model_data.get('y_test_proba')
        
        if verbose:
            print(f"\n✓ Model loaded successfully")
            print(f"  Asset: {kmrf.asset_name}")
            print(f"  Asset class: {kmrf.asset_class}")
            print(f"  Classification type: {kmrf.classification_type}")
            print(f"  Training end date: {kmrf.end_date}")
            print(f"  Features: {len(kmrf.selected_features) if kmrf.selected_features else 'all'}")
            print(f"  Feature window size: {kmrf.feature_window_size}")
            print(f"  Feature asset classes: {kmrf.feature_asset_classes}")

        if verbose:
            if kmrf.X_train is not None:
                print(f"\n  Restored data splits:")
                print(f"    Training samples: {len(kmrf.X_train)} (with labels)")
                if kmrf.X_val is not None:
                    print(f"    Validation samples: {len(kmrf.X_val)} (features only)")
            if kmrf.X_test is not None:
                print(f"    Test samples: {len(kmrf.X_test)} (features only)")
        
        if verbose:
            print(f"\n  Model is ready for predictions!")
            print(f"  You can now call predict() without retraining.")
            print(f"{'='*80}")
        
        return kmrf
