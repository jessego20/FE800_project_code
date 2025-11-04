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
        price_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'vwap']
        Index should be datetime
        
        Notes:
        ------
        - All raw price columns except 'open' are lagged by 1 day with '_lag1d' suffix
        - Original 'open' column is kept unlagged (available at day t)
        - Lagged columns: high_lag1d, low_lag1d, close_lag1d, volume_lag1d, vwap_lag1d
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
        
        # Create lagged versions of raw price columns
        # Keep original 'open' unlagged, lag all others
        lagged_data = pd.DataFrame(index=self.data.index)
        
        # Keep original open (not lagged - available at day t)
        if 'open' in self.data.columns:
            lagged_data['open'] = self.data['open']
        
        # Lag all other columns by 1 day
        columns_to_lag = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for col in columns_to_lag:
            if col in self.data.columns:
                lagged_data[f'{col}_lag1d'] = self.data[col].shift(1)
        
        # Replace self.data with the new structure
        self.data = lagged_data
    
    def compute_returns(self) -> pd.DataFrame:
        """
        Compute various return measures using log returns only.
        Only overnight returns are NOT lagged (use only open price of day t).
        All other returns are LAGGED by 1 day and have _lag1d suffix.
        """
        returns_data = pd.DataFrame(index=self.data.index)
        
        # Overnight returns (open t / close t-1) - NOT LAGGED (uses only day t open price)
        # Note: close_lag1d already represents close at t-1
        if 'open' in self.data.columns and 'close_lag1d' in self.data.columns:
            returns_data['log_ret_overnight'] = np.log(self.data['open'] / self.data['close_lag1d'])
        
        # Intraday returns (close t / open t) - LAGGED (uses close price of day t)
        # Use close_lag1d (close at t-1) and open_lag1d (open at t-1)
        if 'close_lag1d' in self.data.columns and 'open_lag1d' in self.data.columns:
            log_ret_intraday = np.log(self.data['close_lag1d'] / self.data['open_lag1d'])
            returns_data['log_ret_intraday_lag1d'] = log_ret_intraday
        
        # Standard log returns (close t / close t-1) - LAGGED
        # At time t, we only know returns up to t-1
        if 'close_lag1d' in self.data.columns:
            log_ret = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(1))
            returns_data['log_ret_lag1d'] = log_ret
        
            # Multi-period log returns - LAGGED
            for period in [5, 10, 20, 60, 120, 252]:
                multi_period_ret = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(period))
                returns_data[f'log_ret_{period}d_lag1d'] = multi_period_ret
        
        return returns_data
    
    def compute_volatility(self, window_sizes: list = [20, 60, 120, 252]) -> pd.DataFrame:
        """
        Compute various volatility measures.
        All volatility measures are LAGGED by 1 day and have _lag1d suffix.
        """
        vol_data = pd.DataFrame(index=self.data.index)
        
        if 'close_lag1d' not in self.data.columns:
            return vol_data
        
        # Get log returns from lagged close
        log_returns = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(1))
        
        # Standard volatility (annualized) - LAGGED
        for window in window_sizes:
            vol = log_returns.rolling(window).std() * np.sqrt(252)
            vol_data[f'vol_{window}d_lag1d'] = vol
        
        # Parkinson volatility (uses high/low) - LAGGED
        if all(col in self.data.columns for col in ['high_lag1d', 'low_lag1d']):
            for window in window_sizes:
                parkinson_vol = np.sqrt(
                    (1 / (4 * np.log(2))) *
                    (np.log(self.data['high_lag1d'] / self.data['low_lag1d']) ** 2).rolling(window).mean() * 252
                )
                vol_data[f'parkinson_vol_{window}d_lag1d'] = parkinson_vol
        
        # Garman-Klass volatility - LAGGED
        if all(col in self.data.columns for col in ['open_lag1d', 'high_lag1d', 'low_lag1d', 'close_lag1d']):
            for window in window_sizes:
                gk_vol = np.sqrt(
                    (0.5 * (np.log(self.data['high_lag1d'] / self.data['low_lag1d']) ** 2) -
                     (2 * np.log(2) - 1) * (np.log(self.data['close_lag1d'] / self.data['open_lag1d']) ** 2)
                    ).rolling(window).mean() * 252
                )
                vol_data[f'gk_vol_{window}d_lag1d'] = gk_vol
        
        return vol_data
    
    def compute_momentum(self, window_sizes: list = [10, 20, 60, 120, 252]) -> pd.DataFrame:
        """
        Compute momentum indicators using log returns.
        All momentum indicators are LAGGED by 1 day and have _lag1d suffix.
        """
        momentum_data = pd.DataFrame(index=self.data.index)
        
        if 'close_lag1d' not in self.data.columns:
            return momentum_data
        
        # Price momentum (log returns) - LAGGED
        for window in window_sizes:
            momentum = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(window))
            momentum_data[f'momentum_{window}d_lag1d'] = momentum
        
        # EMA ratios (price/EMA) - LAGGED
        for window in window_sizes:
            ema = self.data['close_lag1d'].ewm(span=window).mean()
            ema_ratio = self.data['close_lag1d'] / ema
            momentum_data[f'ema_ratio_{window}d_lag1d'] = ema_ratio
        
        # RSI (Relative Strength Index) - LAGGED
        log_returns = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(1))
        for window in [14, 30]:
            gain = (log_returns.where(log_returns > 0, 0)).rolling(window=window).mean()
            loss = (-log_returns.where(log_returns < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            momentum_data[f'rsi_{window}d_lag1d'] = rsi
        
        return momentum_data
    
    def compute_technical_indicators(self) -> pd.DataFrame:
        """
        Compute technical indicators commonly used in financial modeling.
        All technical indicators are LAGGED by 1 day and have _lag1d suffix.
        """
        tech_data = pd.DataFrame(index=self.data.index)
        
        if 'close_lag1d' not in self.data.columns:
            return tech_data
        
        # EMA moving averages - LAGGED
        for window in [5, 10, 20, 50, 100, 200]:
            ema = self.data['close_lag1d'].ewm(span=window).mean()
            tech_data[f'ema_{window}_lag1d'] = ema
        
        # Bollinger Bands (20-day, 2 standard deviations) using EMA - LAGGED
        ema_20 = self.data['close_lag1d'].ewm(span=20).mean()
        std_20 = self.data['close_lag1d'].rolling(20).std()
        bb_upper = ema_20 + (2 * std_20)
        bb_lower = ema_20 - (2 * std_20)
        bb_width = (bb_upper - bb_lower) / ema_20
        bb_position = (self.data['close_lag1d'] - bb_lower) / (bb_upper - bb_lower)
        
        tech_data['bb_upper_lag1d'] = bb_upper
        tech_data['bb_lower_lag1d'] = bb_lower
        tech_data['bb_width_lag1d'] = bb_width
        tech_data['bb_position_lag1d'] = bb_position
        
        # MACD (12, 26, 9) - LAGGED
        ema_12 = self.data['close_lag1d'].ewm(span=12).mean()
        ema_26 = self.data['close_lag1d'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        tech_data['macd_lag1d'] = macd
        tech_data['macd_signal_lag1d'] = macd_signal
        tech_data['macd_histogram_lag1d'] = macd_histogram
        
        return tech_data
    
    def compute_market_microstructure(self) -> pd.DataFrame:
        """
        Compute market microstructure variables.
        All features are LAGGED by 1 day and have _lag1d suffix.
        """
        micro_data = pd.DataFrame(index=self.data.index)
        
        if all(col in self.data.columns for col in ['high_lag1d', 'low_lag1d', 'close_lag1d', 'open_lag1d']):
            # Intraday price ranges - LAGGED (use close price from day t)
            daily_range = (self.data['high_lag1d'] - self.data['low_lag1d']) / self.data['close_lag1d']
            open_to_close = (self.data['close_lag1d'] - self.data['open_lag1d']) / self.data['open_lag1d']
            
            micro_data['daily_range_lag1d'] = daily_range
            micro_data['open_to_close_lag1d'] = open_to_close
            
            # True Range and Average True Range - LAGGED
            prev_close = self.data['close_lag1d'].shift(1)
            true_range = np.maximum(
                self.data['high_lag1d'] - self.data['low_lag1d'],
                np.maximum(
                    abs(self.data['high_lag1d'] - prev_close),
                    abs(self.data['low_lag1d'] - prev_close)
                )
            )
            tr_normalized = true_range / self.data['close_lag1d']
            atr_14 = true_range.rolling(14).mean() / self.data['close_lag1d']
            
            micro_data['true_range_lag1d'] = tr_normalized
            micro_data['atr_14_lag1d'] = atr_14
        
        if 'volume_lag1d' in self.data.columns:
            # Volume indicators - LAGGED
            volume_ma_20 = self.data['volume_lag1d'].rolling(20).mean()
            volume_ratio = self.data['volume_lag1d'] / volume_ma_20
            
            micro_data['volume_ma_20_lag1d'] = volume_ma_20
            micro_data['volume_ratio_lag1d'] = volume_ratio
            
            # Price-Volume indicators - LAGGED
            if 'close_lag1d' in self.data.columns:
                vwp = (
                    (self.data['volume_lag1d'] * self.data['close_lag1d']).rolling(20).sum() /
                    self.data['volume_lag1d'].rolling(20).sum()
                )
                micro_data['volume_weighted_price_lag1d'] = vwp
                
                # On-Balance Volume - LAGGED
                log_returns = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(1))
                obv = (log_returns.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) *
                       self.data['volume_lag1d']).cumsum()
                micro_data['obv_lag1d'] = obv
        
        return micro_data
    
    def compute_regime_variables(self, window_sizes: list = [20, 60, 120]) -> pd.DataFrame:
        """
        Compute variables for regime identification using log returns.
        All regime variables are LAGGED by 1 day and have _lag1d suffix.
        """
        regime_data = pd.DataFrame(index=self.data.index)
        
        if 'close_lag1d' not in self.data.columns:
            return regime_data
        
        log_returns = np.log(self.data['close_lag1d'] / self.data['close_lag1d'].shift(1))
        
        # Rolling statistics - LAGGED
        for window in window_sizes:
            skew = log_returns.rolling(window).skew()
            kurt = log_returns.rolling(window).kurt()
            var_95 = log_returns.rolling(window).quantile(0.05)
            var_99 = log_returns.rolling(window).quantile(0.01)
            
            regime_data[f'skew_{window}d_lag1d'] = skew
            regime_data[f'kurt_{window}d_lag1d'] = kurt
            regime_data[f'var_95_{window}d_lag1d'] = var_95
            regime_data[f'var_99_{window}d_lag1d'] = var_99
        
        # Drawdown measures - LAGGED
        cumulative_returns = np.exp(log_returns.expanding().sum())
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown_1y = drawdown.rolling(252).min()
        
        regime_data['drawdown_lag1d'] = drawdown
        regime_data['max_drawdown_1y_lag1d'] = max_drawdown_1y
        
        return regime_data
    
    def compute_tsfresh_features(self,
                                  columns: Optional[List[str]] = None,
                                  window_size: int = 20,
                                  shift_periods: List[int] = [0]) -> pd.DataFrame:
        """
        Compute tsfresh statistical features using MinimalFCParameters.
        
        This method extracts time-series features from the price data using tsfresh's
        minimal feature set, which includes statistical properties like mean, variance,
        skewness, kurtosis, and other time series characteristics.
        
        Parameters:
        -----------
        columns : List[str], optional
            List of column names to extract features from. If None, uses ['close_lag1d', 'volume_lag1d']
            if available, otherwise just ['close_lag1d']
        window_size : int, default=20
            Size of the rolling window for feature extraction
        shift_periods : List[int], default=[0]
            List of ADDITIONAL periods to shift/lag the features beyond the base lag.
            Default [0] means no additional shifting (features already use _lag1d data).
            Use [1, 5] for additional lags creating _lag2d, _lag6d versions.
        
        Returns:
        --------
        pd.DataFrame : DataFrame containing tsfresh-extracted features
        
        Notes:
        ------
        - Uses MinimalFCParameters() which includes ~20 basic statistical features
        - Features are computed on rolling windows to maintain time-series structure
        - Automatically handles missing values
        - Column names will be prefixed with 'tsfresh_' and suffixed with '_lag1d'
        - Input columns should already be lagged (e.g., 'close_lag1d')
        """
        # Determine which columns to process
        if columns is None:
            columns = ['close_lag1d']
            # Only add volume if it exists AND has non-NaN values
            if 'volume_lag1d' in self.data.columns:
                if not self.data['volume_lag1d'].isna().all():
                    columns.append('volume_lag1d')
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
                # Calculate total lag (base lag from input data + additional shift)
                # Since input is already _lag1d, shift of 1 means total lag of 2 days
                total_lag = shift + 1
                shifted.columns = [col.replace('_lag1d', f'_lag{total_lag}d') for col in aligned_features.columns]
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
            Defaults to {'window_size': 20, 'shift_periods': [0]}
        
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
                tsfresh_params = {'window_size': 20, 'shift_periods': [0]}
            all_fields.append(self.compute_tsfresh_features(**tsfresh_params))
        
        # Combine all DataFrames
        combined_data = pd.concat(all_fields, axis=1)
        
        # Remove duplicate columns
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
        
        # print(f"Total derived fields computed: {combined_data.shape[1] - len(self.data.columns)}")
        return combined_data