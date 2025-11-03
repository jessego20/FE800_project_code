import pandas as pd
import numpy as np
import warnings
from typing import Union, Optional, Dict, List
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
warnings.filterwarnings('ignore')

class TimeSeriesDerivedFields:
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize with price data

        Parameters:
        price_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        Index should be datetime
        """
        self.data = price_data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.sort_index()

        # Ensure required columns exist (case insensitive)
        self.data.columns = [col.lower() for col in self.data.columns]
        required_cols = ['close']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Convert all columns to numeric, handling commas and strings
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Try to convert strings with commas to numeric
                self.data[col] = pd.to_numeric(self.data[col].astype(str).str.replace(',', ''), errors='coerce')
                print(f"Converted column '{col}' from object to numeric")
    
    def compute_returns(self) -> pd.DataFrame:
        """
        Compute various return measures using log returns only
        """
        returns_data = pd.DataFrame(index=self.data.index)
        
        # Log returns (single period)
        returns_data['log_ret'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Multi-period log returns
        for period in [5, 10, 20, 60, 120, 252]:
            returns_data[f'log_ret_{period}d'] = np.log(
                self.data['close'] / self.data['close'].shift(period)
            )
        
        return returns_data
    
    def compute_volatility(self, window_sizes: list = [20, 60, 120, 252]) -> pd.DataFrame:
        """
        Compute various volatility measures
        """
        vol_data = pd.DataFrame(index=self.data.index)
        
        # Get log returns only
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Standard volatility (annualized) - using log returns
        for window in window_sizes:
            vol_data[f'vol_{window}d'] = log_returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (uses high/low)
        if all(col in self.data.columns for col in ['high', 'low']):
            for window in window_sizes:
                parkinson_vol = np.sqrt(
                    (1 / (4 * np.log(2))) *
                    (np.log(self.data['high'] / self.data['low']) ** 2).rolling(window).mean() * 252
                )
                vol_data[f'parkinson_vol_{window}d'] = parkinson_vol
        
        # Garman-Klass volatility
        if all(col in self.data.columns for col in ['open', 'high', 'low', 'close']):
            for window in window_sizes:
                gk_vol = np.sqrt(
                    (0.5 * (np.log(self.data['high'] / self.data['low']) ** 2) -
                     (2 * np.log(2) - 1) * (np.log(self.data['close'] / self.data['open']) ** 2)
                    ).rolling(window).mean() * 252
                )
                vol_data[f'gk_vol_{window}d'] = gk_vol
        
        return vol_data
    
    def compute_momentum(self, window_sizes: list = [10, 20, 60, 120, 252]) -> pd.DataFrame:
        """
        Compute momentum indicators using log returns
        """
        momentum_data = pd.DataFrame(index=self.data.index)
        
        # Price momentum (log returns)
        for window in window_sizes:
            momentum_data[f'momentum_{window}d'] = np.log(
                self.data['close'] / self.data['close'].shift(window)
            )
        
        # EMA ratios (price/EMA)
        for window in window_sizes:
            ema = self.data['close'].ewm(span=window).mean()
            momentum_data[f'ema_ratio_{window}d'] = self.data['close'] / ema
        
        # RSI (Relative Strength Index) - using log returns
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        for window in [14, 30]:
            gain = (log_returns.where(log_returns > 0, 0)).rolling(window=window).mean()
            loss = (-log_returns.where(log_returns < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            momentum_data[f'rsi_{window}d'] = 100 - (100 / (1 + rs))
        
        return momentum_data
    
    def compute_technical_indicators(self) -> pd.DataFrame:
        """
        Compute technical indicators commonly used in financial modeling
        """
        tech_data = pd.DataFrame(index=self.data.index)
        
        # EMA moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            tech_data[f'ema_{window}'] = self.data['close'].ewm(span=window).mean()
        
        # Bollinger Bands (20-day, 2 standard deviations) using EMA
        ema_20 = self.data['close'].ewm(span=20).mean()
        std_20 = self.data['close'].rolling(20).std()
        tech_data['bb_upper'] = ema_20 + (2 * std_20)
        tech_data['bb_lower'] = ema_20 - (2 * std_20)
        tech_data['bb_width'] = (tech_data['bb_upper'] - tech_data['bb_lower']) / ema_20
        tech_data['bb_position'] = (self.data['close'] - tech_data['bb_lower']) / (
            tech_data['bb_upper'] - tech_data['bb_lower']
        )
        
        # MACD (12, 26, 9)
        ema_12 = self.data['close'].ewm(span=12).mean()
        ema_26 = self.data['close'].ewm(span=26).mean()
        tech_data['macd'] = ema_12 - ema_26
        tech_data['macd_signal'] = tech_data['macd'].ewm(span=9).mean()
        tech_data['macd_histogram'] = tech_data['macd'] - tech_data['macd_signal']
        
        return tech_data
    
    def compute_market_microstructure(self) -> pd.DataFrame:
        """
        Compute market microstructure variables
        """
        micro_data = pd.DataFrame(index=self.data.index)
        
        if all(col in self.data.columns for col in ['high', 'low', 'close', 'open']):
            # Price ranges
            micro_data['daily_range'] = (self.data['high'] - self.data['low']) / self.data['close']
            micro_data['open_to_close'] = (self.data['close'] - self.data['open']) / self.data['open']
            
            # True Range and Average True Range
            prev_close = self.data['close'].shift(1)
            true_range = np.maximum(
                self.data['high'] - self.data['low'],
                np.maximum(
                    abs(self.data['high'] - prev_close),
                    abs(self.data['low'] - prev_close)
                )
            )
            micro_data['true_range'] = true_range / self.data['close']
            micro_data['atr_14'] = true_range.rolling(14).mean() / self.data['close']
        
        if 'volume' in self.data.columns:
            # Volume indicators
            micro_data['volume_ma_20'] = self.data['volume'].rolling(20).mean()
            micro_data['volume_ratio'] = self.data['volume'] / micro_data['volume_ma_20']
            
            # Price-Volume indicators
            micro_data['volume_weighted_price'] = (
                (self.data['volume'] * self.data['close']).rolling(20).sum() /
                self.data['volume'].rolling(20).sum()
            )
            
            # On-Balance Volume - using log returns
            log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
            obv = (log_returns.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) *
                   self.data['volume']).cumsum()
            micro_data['obv'] = obv
        
        return micro_data
    
    def compute_regime_variables(self, window_sizes: list = [20, 60, 120]) -> pd.DataFrame:
        """
        Compute variables for regime identification using log returns
        """
        regime_data = pd.DataFrame(index=self.data.index)
        
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Rolling statistics
        for window in window_sizes:
            regime_data[f'skew_{window}d'] = log_returns.rolling(window).skew()
            regime_data[f'kurt_{window}d'] = log_returns.rolling(window).kurt()
            regime_data[f'var_95_{window}d'] = log_returns.rolling(window).quantile(0.05)
            regime_data[f'var_99_{window}d'] = log_returns.rolling(window).quantile(0.01)
        
        # Drawdown measures - using log returns
        cumulative_returns = np.exp(log_returns.expanding().sum())
        running_max = cumulative_returns.expanding().max()
        regime_data['drawdown'] = (cumulative_returns - running_max) / running_max
        regime_data['max_drawdown_1y'] = regime_data['drawdown'].rolling(252).min()
        
        return regime_data
    
    def compute_tsfresh_features(self,
                                  columns: Optional[List[str]] = None,
                                  window_size: int = 20,
                                  shift_periods: List[int] = [1]) -> pd.DataFrame:
        """
        Compute tsfresh statistical features using MinimalFCParameters.
        
        This method extracts time-series features from the price data using tsfresh's
        minimal feature set, which includes statistical properties like mean, variance,
        skewness, kurtosis, and other time series characteristics.
        
        Parameters:
        -----------
        columns : List[str], optional
            List of column names to extract features from. If None, uses ['close', 'volume']
            if available, otherwise just ['close']
        window_size : int, default=20
            Size of the rolling window for feature extraction
        shift_periods : List[int], default=[1]
            List of periods to shift/lag the features by (e.g., [1, 5, 10] for 1-day,
            5-day, and 10-day lagged features)
        
        Returns:
        --------
        pd.DataFrame : DataFrame containing tsfresh-extracted features
        
        Notes:
        ------
        - Uses MinimalFCParameters() which includes ~20 basic statistical features
        - Features are computed on rolling windows to maintain time-series structure
        - Automatically handles missing values
        - Column names will be prefixed with 'tsfresh_'
        """
        # Determine which columns to process
        if columns is None:
            columns = ['close']
            # Only add volume if it exists AND has non-NaN values
            if 'volume' in self.data.columns:
                if not self.data['volume'].isna().all():
                    columns.append('volume')
                else:
                    print("Volume column exists but contains only NaN values. Skipping volume features.")
        else:
            # Validate columns exist and have valid data
            valid_columns = []
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
                elif self.data[col].isna().all():
                    print(f"Warning: Column '{col}' contains only NaN values. Skipping.")
                else:
                    valid_columns.append(col)
            columns = valid_columns
            
            if len(columns) == 0:
                raise ValueError("No valid columns with data available for feature extraction")
        
        print(f"Computing tsfresh features for columns: {columns}")
        print(f"Window size: {window_size}, Shift periods: {shift_periods}")
        
        # Prepare data for tsfresh (requires specific format)
        # tsfresh doesn't accept NaN values, so we need to handle them
        tsfresh_data = []
        valid_indices = []
        
        for idx in range(window_size, len(self.data)):
            window_data = self.data.iloc[idx-window_size:idx]
            
            # Check if window has any NaN values for the columns we're processing
            has_nan = False
            for col in columns:
                if window_data[col].isna().any():
                    has_nan = True
                    break
            
            # Only include windows without NaN values
            if not has_nan:
                valid_indices.append(idx)
                for col in columns:
                    for time_idx, value in enumerate(window_data[col].values):
                        tsfresh_data.append({
                            'id': idx,
                            'time': time_idx,
                            'value': value,
                            'kind': col
                        })
        
        if len(tsfresh_data) == 0:
            raise ValueError(
                f"No valid windows found without NaN values. "
                f"Data may have too many NaN values for window_size={window_size}. "
                f"Try using a smaller window_size or ensuring data has fewer NaN values."
            )
        
        tsfresh_df = pd.DataFrame(tsfresh_data)
        
        print(f"Valid windows for feature extraction: {len(valid_indices)} out of {len(self.data) - window_size}")
        
        # Extract features using MinimalFCParameters
        print("Extracting tsfresh features...")
        extracted_features = extract_features(
            tsfresh_df,
            column_id='id',
            column_sort='time',
            column_kind='kind',
            column_value='value',
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=False,
            n_jobs=1  # Set to 1 to avoid multiprocessing issues; can be increased
        )
        
        # Align with original index using valid_indices
        # Map the extracted features to their corresponding dates
        feature_index = self.data.index[valid_indices]
        extracted_features.index = feature_index
        
        # Handle duplicate indices if they exist
        if extracted_features.index.has_duplicates:
            print(f"Warning: Extracted features have {extracted_features.index.duplicated().sum()} duplicate indices. Aggregating...")
            # Group by index and take the mean for duplicate indices
            extracted_features = extracted_features.groupby(level=0).mean()
        
        # Reindex to match original data length (fill with NaN for periods without valid windows)
        # Also handle duplicates in the target index
        if self.data.index.has_duplicates:
            print(f"Warning: Original data has {self.data.index.duplicated().sum()} duplicate indices. Using unique indices...")
            # Get unique indices from original data
            unique_index = self.data.index.unique()
            aligned_features = extracted_features.reindex(unique_index)
            # Then reindex again to match the full original index (with duplicates)
            aligned_features = aligned_features.reindex(self.data.index, method='ffill')
        else:
            aligned_features = extracted_features.reindex(self.data.index)
        
        # Rename columns to add 'tsfresh_' prefix
        aligned_features.columns = [f'tsfresh_{col}' for col in aligned_features.columns]
        
        # Create shifted versions if requested
        all_features = [aligned_features]
        
        for shift in shift_periods:
            if shift > 0:
                shifted = aligned_features.shift(shift)
                shifted.columns = [f'{col}_lag{shift}' for col in aligned_features.columns]
                all_features.append(shifted)
        
        # Combine all features
        final_features = pd.concat(all_features, axis=1)
        
        print(f"Generated {final_features.shape[1]} tsfresh features")
        
        return final_features
    
    def compute_all_derived_fields(self, include_tsfresh: bool = False,
                                     tsfresh_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute all derived fields and return combined DataFrame
        
        Parameters:
        -----------
        include_tsfresh : bool, default=False
            Whether to include tsfresh statistical features
        tsfresh_params : Dict, optional
            Parameters to pass to compute_tsfresh_features().
            Defaults to {'window_size': 20, 'shift_periods': [1]}
        
        Returns:
        --------
        pd.DataFrame : Combined dataframe with all features
        """
        all_fields = [self.data]
        
        # print("Computing returns...")
        all_fields.append(self.compute_returns())
        
        # print("Computing volatility measures...")
        all_fields.append(self.compute_volatility())
        
        # print("Computing momentum indicators...")
        all_fields.append(self.compute_momentum())
        
        # print("Computing technical indicators...")
        all_fields.append(self.compute_technical_indicators())
        
        # print("Computing market microstructure variables...")
        all_fields.append(self.compute_market_microstructure())
        
        # print("Computing regime variables...")
        all_fields.append(self.compute_regime_variables())
        
        # Optionally compute tsfresh features
        if include_tsfresh:
            # print("Computing tsfresh features...")
            if tsfresh_params is None:
                tsfresh_params = {'window_size': 20, 'shift_periods': [1]}
            all_fields.append(self.compute_tsfresh_features(**tsfresh_params))
        
        # Combine all DataFrames
        combined_data = pd.concat(all_fields, axis=1)
        
        # Remove duplicate columns
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
        
        # print(f"Total derived fields computed: {combined_data.shape[1] - len(self.data.columns)}")
        return combined_data