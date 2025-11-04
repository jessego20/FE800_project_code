"""
KMRF (KAMA+MSR+RF) - Regime Prediction Model

This module implements the KMRF regime prediction model, which combines:
- KAMA (Kaufman's Adaptive Moving Average) for trend detection
- MSR (Markov-Switching Regression) for volatility regime detection
- RF (Random Forest) for ex-ante regime prediction

Based on the papers by Pomorski & Gorse:
- "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes"
- "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework"
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import pickle
from glob import glob

# Import the TimeSeriesDerivedFields class for feature engineering
# import derive_features as dd # feature engineering is completed in feature_engineering.ipynb and saved in 'ready' data 

# Import for feature selection
try:
    from boruta import BorutaPy
except ImportError:
    BorutaPy = None
    warnings.warn("boruta not installed. Feature selection will be skipped. Install with: pip install boruta")

warnings.filterwarnings('ignore')


class KMRF:
    """
    KMRF: KAMA+MSR+RF Regime Prediction Model
    
    This class handles the complete pipeline for regime prediction including:
    - Loading multi-asset data from CSV files
    - Loading macroeconomic data and aligning it
    - Computing/loading technical features
    - Loading KAMA+MSR regime labels from saved models
    - Optionally adapting 4-regime labels to 3-class KMRF labels
    - Feature selection using Boruta algorithm
    - Random Forest training for regime prediction
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
        use_ready_data: bool = True,
        validation_start: str = '2019-04-01',
        validation_end: str = '2019-09-30',
        test_start: str = '2020-01-01',
        random_seed: int = 1010,
        classification_type: str = 'adapted'
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
        use_ready_data : bool, default=True
            Whether to use pre-computed features from 'ready' folder
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
        """
        self.asset_name = asset_name
        self.asset_class = asset_class
        self.classification_type = classification_type
        self.end_date = end_date
        self.random_seed = random_seed
        self.use_ready_data = use_ready_data
        
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
            if use_ready_data:
                data_path_map = {
                    'us_equity': 'data/ready/us_equity.csv',
                    'commodity': 'data/ready/commodity.csv',
                    'int_equity': 'data/ready/int_equity.csv',
                    'us_treasury': 'data/ready/us_treasury.csv'
                }
            else:
                raise NotImplementedError("Only implemented to use pre-computed 'ready' data")
            self.data_path = Path(data_path_map.get(asset_class, ''))
        else:
            self.data_path = Path(data_path)
        
        if kama_msr_model_dir is None:
            self.kama_msr_model_dir = Path(f'saved_models/KAMA_MSR/{asset_class}/{end_date}/')
        else:
            self.kama_msr_model_dir = Path(kama_msr_model_dir)
        
        # Initialize data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None  # Original KAMA+MSR 4-regime labels
        self.adapted_labels: Optional[pd.Series] = None  # Adapted 3-class KMRF labels
        self.macro_data: Optional[pd.DataFrame] = None
        self.kama_msr_model: Optional[object] = None
        self.selected_features: Optional[List[str]] = None
        
        # Data splits
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None  # None for val/test (no labels available)
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None  # None for val/test (no labels available)
        
        # Model components
        self.feature_selector = None
        self.rf_model = None
        self.scaler = None
        self.performance_metrics: Dict = {}
        
        print(f"KMRF model initialized")
        print(f"  Asset: {self.asset_name}")
        print(f"  Asset class: {self.asset_class}")
        print(f"  Classification type: {self.classification_type}")
        print(f"  End date: {self.end_date}")
        print(f"  Using pre-computed features: {self.use_ready_data}")
        print(f"  Data path: {self.data_path}")
        print(f"  KAMA+MSR model directory: {self.kama_msr_model_dir}")
        print(f"  Validation period: {validation_start} to {validation_end}")
        print(f"  Test start: {test_start}")
        print(f"  Random seed: {self.random_seed}")
    
    def load_data(self, rename_map: Optional[Dict] = None) -> pd.DataFrame:
        """Load data for the specific asset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"\nLoading data from: {self.data_path}")
        
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
        
        self.raw_data = full_data.xs(self.asset_name, level=0, axis=1)
        
        print(f"Loaded data for: {self.asset_name}")
        print(f"  Rows: {self.raw_data.shape[0]}")
        print(f"  Columns: {self.raw_data.shape[1]}")
        print(f"  Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        
        return self.raw_data
    
    def get_features(self) -> pd.DataFrame:
        """Get features for the asset."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_ready_data:
            print(f"\nExtracting pre-computed features for {self.asset_name}...")
            self.features = self.raw_data.copy()
            print(f"  Features shape: {self.features.shape}")
        else:
            raise NotImplementedError("Feature computation from raw data not yet implemented.")
        
        return self.features
    
    def load_kama_msr_labels(self) -> pd.Series:
        """Load KAMA+MSR regime labels from saved model for this asset."""
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
            macro_data_path = Path('data/ready/macro_data_all.csv')
        else:
            macro_data_path = Path(macro_data_path)
        
        if not macro_data_path.exists():
            print(f"Warning: Macro data file not found at {macro_data_path}")
            print("Continuing without macroeconomic features...")
            return None
        
        print(f"\nLoading macroeconomic data from: {macro_data_path}")
        
        macro_df = pd.read_csv(macro_data_path, index_col=0, header=[0, 1], parse_dates=True)
        macro_df.index = pd.to_datetime(macro_df.index)
        
        if self.features is not None:
            macro_df = macro_df.reindex(self.features.index).ffill()
            print(f"Aligned macro data to features index")
        
        self.macro_data = macro_df
        
        print(f"Macro data shape: {macro_df.shape}")
        print(f"Macro indicators: {len(macro_df.columns.get_level_values(0).unique())}")
        
        return self.macro_data
    
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
        select_features: bool = False,
        boruta_params: Optional[Dict] = None,
        split_data: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for model training.
        
        This comprehensive method:
        1. Optionally loads and combines macroeconomic data
        2. Uses the classification type specified during initialization
        3. Aligns feature and label indices
        4. Optionally applies Boruta feature selection (on training data only)
        5. Optionally splits data into train/validation/test sets
        
        Parameters
        ----------
        include_macro : bool, default=True
            Load and include macroeconomic features
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
        
        # Step 1: Start with technical features
        X = self.features.copy()
        
        print(f"\nStep 1: Technical Features")
        print(f"  Features Shape: {X.shape}")
        
        # Step 2: Load and combine macro data if requested
        if include_macro:
            print(f"\nStep 2: Loading Macroeconomic Data")
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
            print(f"\nStep 2: Skipping macroeconomic data")
        
        # Step 3: Get labels based on classification type
        if self.classification_type == 'adapted':
            if self.adapted_labels is not None:
                # Use pre-adapted labels
                print(f"\nStep 3: Using Pre-Adapted Labels")
                print(f"  (3-class labels from previous adapt_regime_labels() call)")
                y = self.adapted_labels.copy()
            else:
                # Adapt now
                print(f"\nStep 3: Adapting Labels")
                print("  4-regime → 3-class (Bullish=1, Bearish=-1, Other=0)")
                y = self.adapt_regime_labels(price_data=None, labels=self.labels)
        else:  # original
            print(f"\nStep 3: Using Original 4-Regime Labels")
            print(f"  (LV Bullish=0, LV Bearish=1, HV Bullish=2, HV Bearish=3)")
            y = self.labels.copy()
        
        print(f"  Labels Shape: ({len(y)},)")
        
        # Step 4: Clean features
        print(f"\nStep 4: Cleaning Features - start when labels exist")
        X = X.loc[y.index[0]:]
        print(f"  Features date range: {X.index[0]} to {X.index[-1]}")
        print(f"  Labels date range: {y.index[0]} to {y.index[-1]}")
        
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
        
        print(f"  Features ready across full date range")
        
        # Step 5: Split data if requested
        if split_data:
            print(f"\nStep 5: Splitting Data")
            self.split_train_val_test(X, y, select_features=select_features, boruta_params=boruta_params)
            return self.X_train, self.y_train
        else:
            print(f"\nStep 5: No data splitting")
            
            # Step 6: Feature selection on full dataset
            if select_features:
                print(f"\nStep 6: Feature Selection (Boruta)")
                if boruta_params is None:
                    boruta_params = {'max_iter': 100}
                
                selected_features = self.select_features_boruta(X, y, **boruta_params)
                X = X[selected_features]
            else:
                print(f"\nStep 6: Skipping feature selection")
            
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
    
    def __repr__(self) -> str:
        """String representation of the KMRF model."""
        status = [f"KMRF('{self.asset_name}', {self.asset_class})"]
        
        if self.raw_data is not None:
            status.append(f"Data: {self.raw_data.shape}")
        
        if self.features is not None:
            status.append(f"Features: {self.features.shape}")
        
        if self.labels is not None:
            status.append(f"4-Regime Labels: ({len(self.labels)},)")
        
        if self.adapted_labels is not None:
            status.append(f"3-Class Labels: ({len(self.adapted_labels)},)")
        
        if self.X_train is not None:
            status.append(f"Train: {self.X_train.shape}")
        
        if self.X_val is not None:
            status.append(f"Val: {self.X_val.shape}")
        
        if self.X_test is not None:
            status.append(f"Test: {self.X_test.shape}")
        
        return " | ".join(status)
    
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
        rf_params: Optional[Dict] = None
    ) -> 'KMRF':
        """
        Fit Random Forest classifier for regime prediction.
        
        This method trains the RF component of the KMRF model using prepared features
        and adapted labels. The trained model can then generate ex-ante regime predictions.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix. If None, uses output from prepare_training_data()
        y : pd.DataFrame, optional
            Target labels. If None, uses output from prepare_training_data()
        rf_params : dict, optional
            Random Forest hyperparameters. Defaults based on paper Table 2:
            - n_estimators: 100-300
            - max_depth: 1-20
            - min_samples_split: 1-100
            - min_samples_leaf: 1-100
            
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
        >>> kmrf.fit()  # Uses data from prepare_training_data()
        >>> kmrf.fit(X_train, y_train, rf_params={'n_estimators': 200})
        """
        from sklearn.ensemble import RandomForestClassifier
        
        if X is None or y is None:
            raise ValueError("Features and labels required. Call prepare_training_data() first.")
        
        print(f"\n{'='*80}")
        print(f"TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'='*80}")
        
        # Default RF parameters based on paper
        if self.asset_class == 'commodity':
            default_params = {
                'n_estimators': 280,  # Paper Table 2 values
                'max_depth': 3,
                'min_samples_split': 18,
                'min_samples_leaf': 95,
                'max_samples': 0.125,
                'max_features': 0.4,
                'min_weight_fraction_leaf': 0.02,
                'random_state': self.random_seed,
                'n_jobs': -1
            }
        else:    
            default_params = {
                'n_estimators': 220,  # Paper Table 2 values
                'max_depth': 13,
                'min_samples_split': 76,
                'min_samples_leaf': 95,
                'max_samples': 0.36,
                'max_features': 0.25,
                'min_weight_fraction_leaf': 0.045,
                'random_state': self.random_seed,
                'n_jobs': -1
            }
        
        if rf_params:
            default_params.update(rf_params)
        
        print(f"RF Parameters:")
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
        
        # Train Random Forest
        print(f"\nFitting Random Forest...")
        self.rf_model = RandomForestClassifier(**default_params, class_weight='balanced')
        self.rf_model.fit(X_clean.values, y_clean.values)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Model ready for prediction")
        
        return self
    
    def predict(
        self,
        X: Optional[pd.DataFrame] = None,
        return_proba: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Generate regime predictions for new data.
        
        This method produces ex-ante regime predictions for the next trading day.
        Returns probabilities for each of the regime classes per asset.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for prediction. If None, uses X_test from train/val/test split
        return_proba : bool, default=True
            Return probabilities instead of class predictions
            
        Returns
        -------
        pd.DataFrame or np.ndarray
            If return_proba=True: DataFrame with columns ['P(Bullish)', 'P(Other)', 'P(Bearish)']
            If return_proba=False: Array of predicted classes (1, 0, -1 for adapted; 0,1,2,3 for original)
            
        Notes
        -----
        Predictions should be interpreted with CONTRARIAN strategy (per paper):
        - High P(Bullish) → SHORT signal (market overbought)
        - High P(Bearish) → LONG signal (market oversold)  
        - High P(Other) → CLOSE position
        
        Examples
        --------
        >>> predictions = kmrf.predict()  # Get probabilities for X_test
        >>> predictions = kmrf.predict(X_test, return_proba=True)  # Explicit
        >>> classes = kmrf.predict(X_test, return_proba=False)  # Get class predictions
        """
        if self.rf_model is None:
            raise ValueError("Model not trained. Call fit() first or load a saved model.")
        
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available. Either provide X or run prepare_training_data() with split_data=True.")
            X = self.X_test
        
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
        
        if return_proba:
            # Get probabilities for each class
            proba = self.rf_model.predict_proba(X_clean.values)
            
            # Create DataFrame with meaningful column names
            # Classes are sorted, so we need to map them correctly
            classes = self.rf_model.classes_
            
            if self.classification_type == 'adapted':
                class_names = {-1: 'P(Bearish)', 0: 'P(Other)', 1: 'P(Bullish)'}
            else:  # original
                class_names = {0: 'P(LV_Bull)', 1: 'P(LV_Bear)', 2: 'P(HV_Bull)', 3: 'P(HV_Bear)'}
            
            result = pd.DataFrame(
                proba,
                index=X.index,
                columns=[class_names.get(c, f'P(Class_{c})') for c in classes]
            )
            
            print(f"✓ Generated probability predictions: {result.shape}")
            return result
        else:
            # Get class predictions
            predictions = self.rf_model.predict(X_clean.values)
            print(f"✓ Generated class predictions: {len(predictions)}")
            return predictions
    
    def save_model(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        boruta_used: bool = False
    ) -> Path:
        """
        Save trained KMRF model to disk.
        
        Saves the complete model including:
        - Random Forest classifier
        - Selected features (if feature selection was used)
        - Train/val/test splits (X_train, y_train, X_val, X_test)
        - Model metadata (asset class, end_date, classification_type, etc.)
        
        This allows you to load the model later without retraining or re-running feature selection.
        
        Parameters
        ----------
        model_dir : str or Path, optional
            Directory to save model. If None, uses 'saved_models/KMRF/{classification_type}_labels/{asset_class}/'
        model_name : str, optional
            Model filename. If None, auto-generates name
        boruta_used : bool, default=False
            Whether Boruta feature selection was used (adds to filename)
            
        Returns
        -------
        Path
            Full path to saved model file
            
        Examples
        --------
        >>> kmrf.fit()
        >>> model_path = kmrf.save_model(boruta_used=True)
        >>> # Load later: kmrf = KMRF.load_model(model_path)
        """
        if self.rf_model is None:
            raise ValueError("No model to save. Train model first with fit().")
        
        # Set default directory
        if model_dir is None:
            model_dir = Path(f'saved_models/KMRF/{self.classification_type}_labels/{self.asset_class}/')
        else:
            model_dir = Path(model_dir)
        
        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-generate filename
        if model_name is None:
            model_name = f"KMRF_{('-').join(self.asset_name.split())}_{self.end_date}"
            if boruta_used:
                model_name += f"_boruta.pkl"
            else:
                model_name += f".pkl"
        
        model_path = model_dir / model_name
        
        # Package model data - save everything needed to skip retraining
        model_data = {
            'rf_model': self.rf_model,
            'selected_features': self.selected_features,
            'asset_class': self.asset_class,
            'asset_name': self.asset_name,
            'end_date': self.end_date,
            'classification_type': self.classification_type,
            'random_seed': self.random_seed,
            'model_params': self.rf_model.get_params(),
            'validation_start': self.validation_start,
            'validation_end': self.validation_end,
            'test_start': self.test_start,
            # Save train/val/test splits
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'boruta_used': boruta_used
        }
        
        # Save
        print(f"\nSaving model to: {model_path}")
        print(f"  Asset: {self.asset_name}")
        print(f"  Classification type: {self.classification_type}")
        print(f"  Features: {len(self.selected_features) if self.selected_features else 'all'}")
        print(f"  Training samples: {len(self.X_train) if self.X_train is not None else 0}")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved successfully")
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path: Union[str, Path], use_ready_data: bool = True) -> 'KMRF':
        """
        Load a saved KMRF model.
        
        This loads a complete trained model, including:
        - Trained Random Forest classifier
        - Selected features (if feature selection was used)
        - Train/val/test data splits
        - All model metadata
        
        After loading, you can immediately use predict() without retraining.
        
        Parameters
        ----------
        model_path : str or Path
            Path to saved model file
        use_ready_data : bool, default=True
            Whether to use ready data when initializing
            
        Returns
        -------
        KMRF
            Loaded model instance with all state restored
            
        Examples
        --------
        >>> # Load a previously trained model
        >>> kmrf = KMRF.load_model('saved_models/KMRF/adapted_labels/us_equity/KMRF_SPY_20190101_boruta.pkl')
        >>> 
        >>> # Load data for prediction period
        >>> kmrf.load_data()
        >>> features = kmrf.get_features()
        >>> 
        >>> # Generate predictions immediately (no training needed)
        >>> predictions = kmrf.predict(kmrf.X_test, return_proba=True)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"\n{'='*80}")
        print(f"LOADING SAVED KMRF MODEL")
        print(f"{'='*80}")
        print(f"Model path: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved parameters
        kmrf = cls(
            asset_name=model_data['asset_name'],
            asset_class=model_data['asset_class'],
            end_date=model_data['end_date'],
            use_ready_data=use_ready_data,
            validation_start=model_data.get('validation_start'),
            validation_end=model_data.get('validation_end'),
            test_start=model_data.get('test_start'),
            random_seed=model_data.get('random_seed', 42),
            classification_type=model_data.get('classification_type', 'adapted')
        )
        
        # Restore model components
        kmrf.rf_model = model_data['rf_model']
        kmrf.selected_features = model_data['selected_features']
        
        # Restore train/val/test splits
        kmrf.X_train = model_data.get('X_train')
        kmrf.y_train = model_data.get('y_train')
        kmrf.X_val = model_data.get('X_val')
        kmrf.X_test = model_data.get('X_test')
        kmrf.y_val = model_data.get('y_val')
        kmrf.y_test = model_data.get('y_test')
        
        print(f"\n✓ Model loaded successfully")
        print(f"  Asset: {kmrf.asset_name}")
        print(f"  Asset class: {kmrf.asset_class}")
        print(f"  Classification type: {kmrf.classification_type}")
        print(f"  Training end date: {kmrf.end_date}")
        print(f"  Features: {len(kmrf.selected_features) if kmrf.selected_features else 'all'}")
        print(f"  Boruta used: {model_data.get('boruta_used', 'Unknown')}")
        
        if kmrf.X_train is not None:
            print(f"\n  Restored data splits:")
            print(f"    Training samples: {len(kmrf.X_train)} (with labels)")
            if kmrf.X_val is not None:
                print(f"    Validation samples: {len(kmrf.X_val)} (features only)")
            if kmrf.X_test is not None:
                print(f"    Test samples: {len(kmrf.X_test)} (features only)")
        
        print(f"\n  Model is ready for predictions!")
        print(f"  You can now call predict() without retraining.")
        print(f"{'='*80}")
        
        return kmrf
