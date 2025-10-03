import pandas as pd
import numpy as np
import warnings
from typing import Union, Optional
warnings.filterwarnings('ignore')

class TimeSeriesDerivedFields:
    """
    Class to compute comprehensive derived fields from financial time series data
    commonly used in academic financial modeling and research
    """
    
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
        Compute various return measures
        """
        returns_data = pd.DataFrame(index=self.data.index)
        
        # Simple returns
        returns_data['ret'] = self.data['close'].pct_change()
        
        # Log returns
        returns_data['log_ret'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Overnight returns (if open price available)
        if 'open' in self.data.columns:
            returns_data['overnight_ret'] = (self.data['open'] / self.data['close'].shift(1)) - 1
            returns_data['intraday_ret'] = (self.data['close'] / self.data['open']) - 1
        
        # Multi-period returns
        for period in [5, 10, 20, 60, 120, 252]:
            returns_data[f'ret_{period}d'] = self.data['close'].pct_change(period)
            returns_data[f'log_ret_{period}d'] = np.log(
                self.data['close'] / self.data['close'].shift(period)
            )
        
        return returns_data
    
    def compute_volatility(self, window_sizes: list = [20, 60, 120, 252]) -> pd.DataFrame:
        """
        Compute various volatility measures
        """
        vol_data = pd.DataFrame(index=self.data.index)
        
        # Get returns first
        returns = self.data['close'].pct_change()
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Standard volatility (annualized)
        for window in window_sizes:
            vol_data[f'vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
            vol_data[f'log_vol_{window}d'] = log_returns.rolling(window).std() * np.sqrt(252)
        
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
        Compute momentum indicators
        """
        momentum_data = pd.DataFrame(index=self.data.index)
        
        # Price momentum (past returns)
        for window in window_sizes:
            momentum_data[f'momentum_{window}d'] = (
                self.data['close'] / self.data['close'].shift(window) - 1
            )
        
        # Moving average ratios
        for window in window_sizes:
            ma = self.data['close'].rolling(window).mean()
            momentum_data[f'ma_ratio_{window}d'] = self.data['close'] / ma
        
        # RSI (Relative Strength Index)
        for window in [14, 30]:
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            momentum_data[f'rsi_{window}d'] = 100 - (100 / (1 + rs))
        
        return momentum_data
    
    def compute_technical_indicators(self) -> pd.DataFrame:
        """
        Compute technical indicators commonly used in financial modeling
        """
        tech_data = pd.DataFrame(index=self.data.index)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            tech_data[f'sma_{window}'] = self.data['close'].rolling(window).mean()
            tech_data[f'ema_{window}'] = self.data['close'].ewm(span=window).mean()
        
        # Bollinger Bands (20-day, 2 standard deviations)
        sma_20 = self.data['close'].rolling(20).mean()
        std_20 = self.data['close'].rolling(20).std()
        tech_data['bb_upper'] = sma_20 + (2 * std_20)
        tech_data['bb_lower'] = sma_20 - (2 * std_20)
        tech_data['bb_width'] = (tech_data['bb_upper'] - tech_data['bb_lower']) / sma_20
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
            returns = self.data['close'].pct_change()
            micro_data['volume_weighted_price'] = (
                (self.data['volume'] * self.data['close']).rolling(20).sum() / 
                self.data['volume'].rolling(20).sum()
            )
            
            # On-Balance Volume
            obv = (returns.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) * 
                   self.data['volume']).cumsum()
            micro_data['obv'] = obv
        
        return micro_data
    
    def compute_regime_variables(self, window_sizes: list = [20, 60, 120]) -> pd.DataFrame:
        """
        Compute variables for regime identification
        """
        regime_data = pd.DataFrame(index=self.data.index)
        
        returns = self.data['close'].pct_change()
        
        # Rolling statistics
        for window in window_sizes:
            regime_data[f'skew_{window}d'] = returns.rolling(window).skew()
            regime_data[f'kurt_{window}d'] = returns.rolling(window).kurt()
            regime_data[f'var_95_{window}d'] = returns.rolling(window).quantile(0.05)
            regime_data[f'var_99_{window}d'] = returns.rolling(window).quantile(0.01)
        
        # Drawdown measures
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        regime_data['drawdown'] = (cumulative_returns - running_max) / running_max
        regime_data['max_drawdown_1y'] = regime_data['drawdown'].rolling(252).min()
        
        return regime_data
    
    def compute_all_derived_fields(self) -> pd.DataFrame:
        """
        Compute all derived fields and return combined DataFrame
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
        
        # Combine all DataFrames
        combined_data = pd.concat(all_fields, axis=1)
        
        # Remove duplicate columns
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
        
        # print(f"Total derived fields computed: {combined_data.shape[1] - len(self.data.columns)}")
        return combined_data