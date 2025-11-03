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
import derive_data as dd

# Import for feature selection
try:
    from BorutaShap import BorutaShap
except ImportError:
    BorutaShap = None
    warnings.warn("BorutaShap not installed. Feature selection will be skipped.")

warnings.filterwarnings('ignore')


class KMRF:
    """
    KMRF: KAMA+MSR+RF Regime Prediction Model
    
    This class handles the complete pipeline for regime prediction including:
    - Loading multi-asset data from CSV files
    - Loading macroeconomic data and aligning it
    - Computing/loading technical features
    - Loading KAMA+MSR regime labels from saved models
    - Adapting 4-regime labels to 3-class KMRF labels
    - Feature selection using BorutaShap
    - Random Forest training for regime prediction
    - Performance evaluation
    
    The model predicts three regime classes:
    - Bullish (1): LV bullish + extension to peak of next HV bullish
    - Bearish (-1): HV bearish + extension to trough of next LV bearish
    - Other (0): Remaining periods
    """
    
    def __init__(
        self,
        asset_class: str = 'us_equity',
        data_path: Optional[Union[str, Path]] = None,
        kama_msr_model_dir: Optional[Union[str, Path]] = None,
        end_date: str = '20190101',
        use_ready_data: bool = True,
        random_seed: int = 42
    ):
        """Initialize the KMRF model."""
        self.asset_class = asset_class
        self.end_date = end_date
        self.random_seed = random_seed
        self.use_ready_data = use_ready_data
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Set default paths
        if data_path is None:
            if use_ready_data:
                data_path_map = {
                    'us_equity': 'data/ready/us_equity.csv',
                    'commodity': 'data/ready/commodity.csv',
                    'int_equity': 'data/ready/int_equity.csv'
                }
            else:
                data_path_map = {
                    'us_equity': 'data/processed/us_equity_all_data.csv',
                    'commodity': 'data/processed/commodity_data.csv',
                    'int_equity': 'data/processed/us_equity_all_data.csv'
                }
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
        self.labels: Optional[pd.DataFrame] = None
        self.macro_data: Optional[pd.DataFrame] = None
        self.asset_names: List[str] = []
        self.kama_msr_models: Dict = {}
        self.selected_features: Optional[List[str]] = None
        
        # Model components
        self.feature_selector = None
        self.rf_model = None
        self.scaler = None
        self.performance_metrics: Dict = {}
        
        print(f"KMRF model initialized")
        print(f"  Asset class: {self.asset_class}")
        print(f"  End date: {self.end_date}")
        print(f"  Using pre-computed features: {self.use_ready_data}")
        print(f"  Data path: {self.data_path}")
        print(f"  KAMA+MSR model directory: {self.kama_msr_model_dir}")
        print(f"  Random seed: {self.random_seed}")
    
    def load_data(self, rename_map: Optional[Dict] = None) -> pd.DataFrame:
        """Load multi-asset data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"\nLoading data from: {self.data_path}")
        
        self.raw_data = pd.read_csv(
            self.data_path,
            index_col=0,
            header=[0, 1],
            parse_dates=True
        )
        
        self.raw_data.index = pd.to_datetime(self.raw_data.index)
        
        if rename_map:
            self.raw_data.rename(columns=rename_map, level=0, inplace=True)
        
        self.asset_names = self.raw_data.columns.get_level_values(0).unique().tolist()
        
        print(f"Loaded {self.raw_data.shape[0]} rows, {len(self.asset_names)} assets")
        print(f"Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        print(f"Assets: {', '.join(self.asset_names[:5])}{'...' if len(self.asset_names) > 5 else ''}")
        
        return self.raw_data
    
    def get_features(self, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """Get features for specified assets."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_ready_data:
            print(f"\nExtracting pre-computed features...")
            
            if assets is None:
                self.features = self.raw_data
            else:
                asset_columns = []
                for asset in assets:
                    if asset in self.asset_names:
                        asset_cols = [col for col in self.raw_data.columns if col[0] == asset]
                        asset_columns.extend(asset_cols)
                
                if not asset_columns:
                    raise ValueError(f"No matching assets found: {assets}")
                
                self.features = self.raw_data[asset_columns]
            
            print(f"Features shape: {self.features.shape}")
            print(f"Assets: {len(self.features.columns.get_level_values(0).unique())}")
        else:
            raise NotImplementedError("Feature computation from raw data not yet implemented.")
        
        return self.features
    
    def load_kama_msr_labels(self, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """Load KAMA+MSR regime labels from saved models."""
        if not self.kama_msr_model_dir.exists():
            raise FileNotFoundError(f"KAMA+MSR model directory not found: {self.kama_msr_model_dir}")
        
        if assets is None:
            assets = self.asset_names
        
        print(f"\n{'='*80}")
        print(f"LOADING KAMA+MSR LABELS FOR {len(assets)} ASSETS")
        print(f"{'='*80}")
        print(f"Model directory: {self.kama_msr_model_dir}")
        
        labels_list = []
        loaded_count = 0
        
        for i, asset in enumerate(assets):
            model_pattern = f"{asset}_KAMA-MSR_4-regimes.pkl"
            model_files = list(self.kama_msr_model_dir.glob(model_pattern))
            
            if not model_files:
                asset_safe = asset.replace(' ', '_')
                model_pattern = f"{asset_safe}_KAMA-MSR_4-regimes.pkl"
                model_files = list(self.kama_msr_model_dir.glob(model_pattern))
            
            if not model_files:
                print(f"[{i+1}/{len(assets)}] ✗ Model not found for: {asset}")
                continue
            
            model_file = model_files[0]
            
            try:
                with open(model_file, 'rb') as f:
                    kama_msr_model = pickle.load(f)
                
                self.kama_msr_models[asset] = kama_msr_model
                
                if hasattr(kama_msr_model, 'regime_labels'):
                    regime_labels = kama_msr_model.regime_labels.copy()
                else:
                    print(f"[{i+1}/{len(assets)}] ✗ No regime_labels attribute for: {asset}")
                    continue
                
                asset_labels = pd.DataFrame({'regime_label': regime_labels})
                asset_labels.columns = pd.MultiIndex.from_product([[asset], asset_labels.columns])
                
                labels_list.append(asset_labels)
                loaded_count += 1
                
                print(f"[{i+1}/{len(assets)}] ✓ Loaded labels for: {asset}")
                
            except Exception as e:
                print(f"[{i+1}/{len(assets)}] ✗ Error loading {asset}: {str(e)}")
                continue
        
        if not labels_list:
            raise ValueError("No labels loaded successfully")
        
        self.labels = pd.concat(labels_list, axis=1)
        
        print(f"\n{'='*80}")
        print(f"LABEL LOADING COMPLETE")
        print(f"{'='*80}")
        print(f"Successfully loaded: {loaded_count}/{len(assets)} assets")
        print(f"Label date range: {self.labels.index[0]} to {self.labels.index[-1]}")
        
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
        y: pd.DataFrame,
        n_trials: int = 100,
        sample: bool = False,
        train_or_test: str = 'test',
        importance_measure: str = 'shap',
        classification: bool = True,
        percentile: int = 100,
        pvalue: float = 0.05,
        verbose: int = 0
    ) -> List[str]:
        """
        Perform feature selection using BorutaShap algorithm.
        
        This implements BorutaShap adapted for time-series data to avoid data leakage.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (multi-index format)
        y : pd.DataFrame
            Target labels (multi-index format)
        n_trials : int, default=100
            Number of BorutaShap trials
        
        Returns
        -------
        list
            Selected feature column names
            
        Notes
        -----
        TODO: Implement PGTS (Purged Group Time-Series Split) cross-validation
        """
        if BorutaShap is None:
            raise ImportError("BorutaShap not installed. Install with: pip install BorutaShap")
        
        print(f"\n{'='*80}")
        print(f"BORUTA FEATURE SELECTION")
        print(f"{'='*80}")
        print(f"Initial features: {X.shape[1]}")
        print(f"Running BorutaShap with {n_trials} trials...")
        print("WARNING: Time-series cross-validation (PGTS) not yet implemented")
        print("         Feature selection may have data leakage risk")
        
        # Flatten multi-index for BorutaShap
        X_flat = X.copy()
        X_flat.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) 
                          for col in X.columns]
        
        # For labels, take first asset's labels
        y_flat = y.iloc[:, 0] if len(y.shape) > 1 else y
        
        # Remove NaN values
        valid_idx = ~(X_flat.isna().any(axis=1) | y_flat.isna())
        X_clean = X_flat[valid_idx]
        y_clean = y_flat[valid_idx]
        
        print(f"Clean samples: {len(X_clean)}")
        
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_seed
        )
        
        Feature_Selector = BorutaShap(
            model=model,
            importance_measure=importance_measure,
            classification=classification,
            percentile=percentile,
            pvalue=pvalue
        )
        
        print("Fitting BorutaShap...")
        Feature_Selector.fit(
            X=X_clean.values,
            y=y_clean.values,
            n_trials=n_trials,
            sample=sample,
            train_or_test=train_or_test,
            verbose=verbose
        )
        
        # Get selected features
        selected_features_bool = Feature_Selector.Subset().values
        selected_feature_names = X_clean.columns[selected_features_bool].tolist()
        
        # Map back to original multi-index column names
        selected_original_names = []
        for flat_name in selected_feature_names:
            for orig_col in X.columns:
                col_str = '_'.join(map(str, orig_col)) if isinstance(orig_col, tuple) else str(orig_col)
                if col_str == flat_name:
                    selected_original_names.append(orig_col)
                    break
        
        self.selected_features = selected_original_names
        
        print(f"\n{'='*80}")
        print(f"FEATURE SELECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Reduction: {100 * (1 - len(self.selected_features) / X.shape[1]):.1f}%")
        
        return self.selected_features
    
    def adapt_regime_labels(
        self,
        price_data: Optional[pd.DataFrame] = None,
        labels: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Adapt 4-regime KAMA+MSR labels to 3-class KMRF labels.
        
        Implements the label transformation from the paper:
        1. Bullish (1): LV bullish + extension to peak of next HV bullish
        2. Bearish (-1): HV bearish + extension to trough of next LV bearish
        3. Other (0): Remaining periods (including post-peak HV bullish and post-trough LV bearish)
        """
        if price_data is None:
            if self.raw_data is None:
                raise ValueError("No price data available. Load data first.")
            price_data = self.raw_data
        
        if labels is None:
            if self.labels is None:
                raise ValueError("No labels available. Load KAMA+MSR labels first.")
            labels = self.labels
        
        print(f"\n{'='*80}")
        print(f"ADAPTING REGIME LABELS")
        print(f"{'='*80}")
        
        asset_names = labels.columns.get_level_values(0).unique()
        adapted_labels_list = []
        
        for asset in asset_names:
            print(f"\nProcessing: {asset}")
            
            try:
                asset_labels = labels.xs(asset, level=0, axis=1)['regime_label'].copy()
                
                # Get close price
                try:
                    asset_prices = price_data.xs(asset, level=0, axis=1)
                    if 'close' in asset_prices.columns:
                        asset_close = asset_prices['close']
                    else:
                        numeric_cols = asset_prices.select_dtypes(include=[np.number]).columns
                        asset_close = asset_prices[numeric_cols[0]] if len(numeric_cols) > 0 else None
                except:
                    asset_close = None
                
                # Initialize as Other (0)
                adapted = pd.Series(0, index=asset_labels.index, dtype=int)
                
                # Process regimes
                i = 0
                while i < len(asset_labels):
                    current_regime = asset_labels.iloc[i]
                    
                    if pd.isna(current_regime):
                        i += 1
                        continue
                    
                    current_regime = int(current_regime)
                    
                    # Find regime end
                    j = i + 1
                    while j < len(asset_labels) and asset_labels.iloc[j] == current_regime:
                        j += 1
                    
                    regime_start = i
                    regime_end = j
                    
                    # LV Bullish (0) → Extend to peak of HV Bullish (2)
                    if current_regime == 0:
                        extension_end = regime_end
                        
                        k = regime_end
                        while k < len(asset_labels):
                            next_regime = asset_labels.iloc[k]
                            if pd.isna(next_regime):
                                k += 1
                                continue
                            
                            next_regime = int(next_regime)
                            
                            if next_regime == 2:  # HV Bullish
                                hv_start = k
                                while k < len(asset_labels) and int(asset_labels.iloc[k]) == 2:
                                    k += 1
                                hv_end = k
                                
                                if asset_close is not None:
                                    hv_indices = asset_labels.index[hv_start:hv_end]
                                    hv_prices = asset_close.loc[hv_indices]
                                    if len(hv_prices) > 0:
                                        peak_idx = hv_prices.idxmax()
                                        peak_pos = asset_labels.index.get_loc(peak_idx)
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
                        while k < len(asset_labels):
                            next_regime = asset_labels.iloc[k]
                            if pd.isna(next_regime):
                                k += 1
                                continue
                            
                            next_regime = int(next_regime)
                            
                            if next_regime == 1:  # LV Bearish
                                lv_start = k
                                while k < len(asset_labels) and int(asset_labels.iloc[k]) == 1:
                                    k += 1
                                lv_end = k
                                
                                if asset_close is not None:
                                    lv_indices = asset_labels.index[lv_start:lv_end]
                                    lv_prices = asset_close.loc[lv_indices]
                                    if len(lv_prices) > 0:
                                        trough_idx = lv_prices.idxmin()
                                        trough_pos = asset_labels.index.get_loc(trough_idx)
                                        extension_end = trough_pos + 1
                                        
                                        # Mark remaining LV Bearish after trough as Other
                                        adapted.iloc[extension_end:lv_end] = 0
                                break
                            k += 1
                        
                        # Mark as Bearish up to extension
                        adapted.iloc[regime_start:extension_end] = -1
                    
                    i = regime_end
                
                # Create DataFrame
                asset_adapted = pd.DataFrame({'adapted_regime': adapted})
                asset_adapted.columns = pd.MultiIndex.from_product([[asset], asset_adapted.columns])
                
                adapted_labels_list.append(asset_adapted)
                
                # Print distribution
                dist = adapted.value_counts().sort_index()
                print(f"  Label distribution:")
                label_map = {-1: 'Bearish', 0: 'Other', 1: 'Bullish'}
                for label_val, count in dist.items():
                    pct = (count / len(adapted)) * 100
                    print(f"    {label_map[label_val]:>8} ({label_val:>2}): {count:>5} ({pct:>5.1f}%)")
                
            except Exception as e:
                print(f"  ✗ Error processing {asset}: {str(e)}")
                continue
        
        if not adapted_labels_list:
            raise ValueError("No labels adapted successfully")
        
        adapted_labels = pd.concat(adapted_labels_list, axis=1)
        
        print(f"\n{'='*80}")
        print(f"LABEL ADAPTATION COMPLETE")
        print(f"{'='*80}")
        print(f"Adapted {len(asset_names)} assets")
        print(f"Label shape: {adapted_labels.shape}")
        
        return adapted_labels
    
    def prepare_training_data(
        self,
        transform_labels: bool = True,
        include_macro: bool = True,
        select_features: bool = False,
        boruta_params: Optional[Dict] = None,
        align_indices: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and labels for model training.
        
        This comprehensive method:
        1. Optionally loads and combines macroeconomic data
        2. Transforms 4-regime KAMA+MSR labels to 3-class KMRF labels
        3. Aligns feature and label indices
        4. Optionally applies BorutaShap feature selection
        5. Removes rows with missing values
        
        Parameters
        ----------
        transform_labels : bool, default=True
            Transform KAMA+MSR 4-regime to KMRF 3-class labels
        include_macro : bool, default=True
            Load and include macroeconomic features
        select_features : bool, default=False
            Apply BorutaShap feature selection
        boruta_params : dict, optional
            Parameters for BorutaShap (n_trials, etc.)
        align_indices : bool, default=True
            Align feature and label indices
            
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            (features, labels) ready for training
            
        Notes
        -----
        Feature combination order:
        1. Technical features (from asset data)
        2. Macroeconomic features (if include_macro=True)
        
        Feature selection (if select_features=True):
        - Uses BorutaShap algorithm
        - Should use PGTS cross-validation (TODO)
        - Reduces dimensionality by selecting top features
        """
        if self.features is None or self.labels is None:
            raise ValueError(
                "Features and labels must be available first. "
                "Call get_features() and load_kama_msr_labels()"
            )
        
        print(f"\n{'='*80}")
        print(f"PREPARING TRAINING DATA")
        print(f"{'='*80}")
        
        # Step 1: Start with technical features
        X = self.features.copy()
        y = self.labels.copy()
        
        print(f"\nStep 1: Technical Features")
        print(f"  Shape: {X.shape}")
        
        # Step 2: Load and combine macro data if requested
        if include_macro:
            print(f"\nStep 2: Loading Macroeconomic Data")
            if self.macro_data is None:
                self.load_macro_data()
            
            if self.macro_data is not None:
                # Combine with existing features
                X = pd.concat([X, self.macro_data], axis=1)
                print(f"  Combined shape: {X.shape}")
            else:
                print(f"  Macro data not available, skipping...")
        else:
            print(f"\nStep 2: Skipping macroeconomic data")
        
        # Step 3: Transform labels
        if transform_labels:
            print(f"\nStep 3: Transforming Labels")
            print("  4-regime → 3-class (Bullish=1, Bearish=-1, Other=0)")
            y = self.adapt_regime_labels(price_data=self.raw_data, labels=y)
        else:
            print(f"\nStep 3: Keeping original 4-regime labels")
        
        # Step 4: Align indices
        if align_indices:
            print(f"\nStep 4: Aligning Indices")
            print(f"  Features date range: {X.index[0]} to {X.index[-1]}")
            print(f"  Labels date range: {y.index[0]} to {y.index[-1]}")
            
            # Reindex features to match label dates
            X = X.reindex(y.index)
            
            # Identify columns with too many NaN values (> 50% of label period)
            nan_threshold = 0.5
            nan_counts = X.isna().sum()
            total_rows = len(X)
            bad_cols = nan_counts[nan_counts > total_rows * nan_threshold].index.tolist()
            
            if bad_cols:
                print(f"  Dropping {len(bad_cols)} features with >{nan_threshold*100:.0f}% NaN values")
                X = X.drop(columns=bad_cols)
            
            # Forward-fill and backward-fill remaining NaN values
            # This handles features calculated on later data
            X = X.ffill().bfill()
            
            # Drop any remaining rows with NaN (should be minimal now)
            valid_mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
            rows_before = len(X)
            X = X[valid_mask]
            y = y[valid_mask]
            rows_dropped = rows_before - len(X)
            
            print(f"  Final date range: {X.index[0]} to {X.index[-1]}")
            print(f"  Total training rows: {len(X)}")
            if rows_dropped > 0:
                print(f"  (Dropped {rows_dropped} rows with remaining NaN)")
        
        # Step 5: Handle missing values
        print(f"\nStep 5: Handling Missing Values")
        X_missing = X.isna().sum().sum()
        y_missing = y.isna().sum().sum()
        print(f"  Missing - Features: {X_missing}, Labels: {y_missing}")
        
        if X_missing > 0 or y_missing > 0:
            print("  Dropping rows with NaN...")
            valid_mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            print(f"  Remaining rows: {len(X)}")
        
        # Step 6: Feature selection
        if select_features:
            print(f"\nStep 6: Feature Selection (BorutaShap)")
            if boruta_params is None:
                boruta_params = {'n_trials': 100}
            
            selected_features = self.select_features_boruta(X, y, **boruta_params)
            X = X[selected_features]
        else:
            print(f"\nStep 6: Skipping feature selection")
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"TRAINING DATA READY")
        print(f"{'='*80}")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        if transform_labels:
            print(f"\nOverall label distribution:")
            label_counts = {-1: 0, 0: 0, 1: 0}
            for col in y.columns:
                col_name = col[1] if isinstance(col, tuple) else col
                if 'adapted_regime' in str(col_name) or 'regime' in str(col_name):
                    asset_labels = y[col]
                    for val in [-1, 0, 1]:
                        label_counts[val] += (asset_labels == val).sum()
            
            total = sum(label_counts.values())
            if total > 0:
                print(f"  Bullish  ( 1): {label_counts[1]:>6} ({100*label_counts[1]/total:>5.1f}%)")
                print(f"  Other    ( 0): {label_counts[0]:>6} ({100*label_counts[0]/total:>5.1f}%)")
                print(f"  Bearish (-1): {label_counts[-1]:>6} ({100*label_counts[-1]/total:>5.1f}%)")
        
        return X, y
    
    def __repr__(self) -> str:
        """String representation of the KMRF model."""
        status = [f"KMRF({self.asset_class}, {self.end_date})"]
        
        if self.raw_data is not None:
            status.append(f"Data: {self.raw_data.shape[0]}×{len(self.asset_names)} assets")
        
        if self.features is not None:
            status.append(f"Features: {self.features.shape}")
        
        if self.labels is not None:
            status.append(f"Labels: {self.labels.shape}")
        
        return " | ".join(status)