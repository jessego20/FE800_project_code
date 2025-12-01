"""
BACKTEST Module for Portfolio Strategy Backtesting

This module provides the BACKTEST class which runs historical backtests of 
portfolio strategies using the OPTIMIZER_INPUTS pipeline.

Key Features:
- Flexible rebalancing schedules (daily, weekly, monthly, quarterly, custom)
- Regime-based rebalancing triggers (optional)
- Portfolio performance tracking
- Transaction cost modeling
- Comprehensive performance metrics

Author: Jesse Goodman
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path
import pickle
from datetime import datetime
import warnings
import os
from joblib import Parallel, delayed

from OPTIMIZER_INPUTS import OPTIMIZER_INPUTS, get_KM_model_dates
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

TRADING_DAYS = CustomBusinessDay(calendar=USFederalHolidayCalendar())

warnings.filterwarnings('ignore')


class BACKTEST:
    """
    Backtesting framework for regime-based portfolio optimization strategies.
    
    This class runs historical backtests by:
    1. Generating rebalance dates based on specified frequency
    2. At each rebalance date: running OPTIMIZER_INPUTS to get mu/Sigma, then optimizing
    3. Tracking portfolio value using actual (realized) returns between rebalances
    4. Optionally triggering rebalances based on regime changes
    
    Parameters
    ----------
    backtest_start : str or pd.Timestamp
        Start date of the backtest (YYYY-MM-DD or YYYYMMDD format)
    backtest_end : str or pd.Timestamp
        End date of the backtest (YYYY-MM-DD or YYYYMMDD format)
    asset_list : List[str]
        List of asset names to include in the portfolio
    rebalance_frequency : str, default='monthly'
        How often to rebalance. Options:
        - 'daily': Every trading day
        - 'weekly': Every 5 trading days
        - 'biweekly': Every 10 trading days  
        - 'monthly': Every 21 trading days
        - 'quarterly': Every 63 trading days
        - 'custom': Use custom_interval parameter
    custom_interval : int, optional
        Number of trading days between rebalances (used with frequency='custom')
    objective : str, default='max_sharpe'
        Optimization objective: 'max_sharpe', 'max_sortino', 'risk_aversion', 'min_variance'
    allow_short : bool, default=False
        Whether to allow short selling
    gross_exposure : float, default=1.0
        Maximum gross exposure. Set > 1 for leverage (e.g., 1.5 for 150% gross)
    risk_aversion : float, default=0.5
        Risk aversion parameter (only used if objective='risk_aversion')
    n_days : int, default=21
        Forecast horizon in trading days for simulation
    n_simulations : int, default=1000
        Number of Monte Carlo simulations per optimization
    market_asset : str, default='SPDR S&P 500 ETF'
        Asset to use as market regime indicator
    initial_capital : float, default=100000.0
        Starting portfolio value
    transaction_cost_bps : float, default=5.0
        Transaction costs in basis points (e.g., 10 = 0.10%)
    use_regime_trigger : bool, default=False
        If True, also rebalance when regime predictions change significantly
    regime_change_threshold : float, default=0.3
        Threshold for regime change (max probability shift to trigger rebalance)
    random_seed : int, default=123
        Random seed for reproducibility
        
    Attributes
    ----------
    rebalance_dates : List[str]
        Generated rebalance dates in YYYYMMDD format
    optimization_results : Dict[str, OPTIMIZER_INPUTS]
        OPTIMIZER_INPUTS objects for each rebalance date
    weights_history : pd.DataFrame
        Portfolio weights over time
    portfolio_value : pd.Series
        Daily portfolio value
    returns : pd.Series
        Daily portfolio returns
    """
    
    def __init__(
        self,
        backtest_start: str | pd.Timestamp,
        backtest_end: str | pd.Timestamp,
        asset_list: List[str],
        rebalance_frequency: str = 'monthly',
        custom_interval: Optional[int] = None,
        objective: str = 'max_sharpe',
        allow_short: bool = False,
        gross_exposure: float = 1.0,
        risk_aversion: float = 0.5,
        n_days: int = 21,
        n_simulations: int = 1000,
        market_asset: str = 'SPDR S&P 500 ETF',
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,
        use_regime_trigger: bool = False,
        regime_change_threshold: float = 0.3,
        max_intermediate_rebalances: int = 3,
        random_seed: int = 123
    ):
        # Validate inputs
        if not asset_list:
            raise ValueError("asset_list must be provided and cannot be empty")
        
        valid_objectives = ['max_sharpe', 'max_sortino', 'risk_aversion', 'min_variance']
        if objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
        
        valid_frequencies = ['daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'custom']
        if rebalance_frequency not in valid_frequencies:
            raise ValueError(f"rebalance_frequency must be one of {valid_frequencies}")
        
        if rebalance_frequency == 'custom' and custom_interval is None:
            raise ValueError("custom_interval required when rebalance_frequency='custom'")
        
        # Store parameters
        self.backtest_start = pd.Timestamp(backtest_start)
        self.backtest_end = pd.Timestamp(backtest_end)
        self.asset_list = asset_list
        self.rebalance_frequency = rebalance_frequency
        self.custom_interval = custom_interval
        self.objective = objective
        self.allow_short = allow_short
        self.gross_exposure = gross_exposure if allow_short else None
        self.risk_aversion = risk_aversion
        self.n_days = n_days
        self.n_simulations = n_simulations
        self.market_asset = market_asset
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.use_regime_trigger = use_regime_trigger
        self.regime_change_threshold = regime_change_threshold
        self.max_intermediate_rebalances = max_intermediate_rebalances
        self.random_seed = random_seed
        
        # Load price data for returns calculation
        self._load_price_data()
        
        # Load risk-free rates
        self._load_risk_free_rates()
        
        # Generate rebalance dates
        self.rebalance_dates = self._generate_rebalance_dates()
        
        # Precompute regime check dates if using regime trigger
        self._regime_check_schedule: Optional[Dict[str, Dict]] = None
        if self.use_regime_trigger:
            self._precompute_regime_check_schedule()
        
        # Initialize result containers
        self.optimization_results: Dict[str, OPTIMIZER_INPUTS] = {}
        self.weights_history: Optional[pd.DataFrame] = None
        self.portfolio_value: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.daily_weights: Optional[pd.DataFrame] = None
        self.transaction_costs: Optional[pd.Series] = None
        self.turnover: Optional[pd.Series] = None
        
        # Regime tracking (for regime-triggered rebalancing)
        # Stores market asset's forward predictions (all horizons) from the most recent rebalance
        self._last_rebal_market_fwd_probs: Optional[pd.DataFrame] = None
        self._last_rebal_date: Optional[pd.Timestamp] = None
        self._intermediate_rebal_count: int = 0  # Count for current standard period
        
        # Store all trading days for quick lookups
        self._all_trading_days: Optional[List[pd.Timestamp]] = None
        
    def _precompute_regime_check_schedule(self):
        """
        Precompute which days to check for regime changes between standard rebalances.
        
        For each standard rebalance period, creates a schedule of:
        - check_date: The date we're checking from
        - next_standard_rebal: The next standard rebalance date
        - days_to_next_rebal: Trading days until next standard rebalance
        - standard_rebal_idx: Index of the preceding standard rebalance
        
        Note: The horizon for triggered rebalances will be calculated dynamically
        based on days remaining to next standard rebalance.
        """
        # Get all trading days in backtest period
        trading_days = self._close_prices.index
        mask = (trading_days >= self.backtest_start) & (trading_days <= self.backtest_end)
        self._all_trading_days = trading_days[mask].tolist()
        
        # Convert rebalance dates to timestamps
        rebal_ts_list = [pd.Timestamp(d) for d in self.rebalance_dates]
        
        self._regime_check_schedule = {}
        
        for i, rebal_date in enumerate(rebal_ts_list[:-1]):  # Don't process last rebalance
            next_rebal_date = rebal_ts_list[i + 1]
            
            # Get trading days between this rebalance and next (exclusive of rebal dates)
            check_dates = [d for d in self._all_trading_days 
                          if d > rebal_date and d < next_rebal_date]
            
            for check_date in check_dates:
                # Calculate days to next standard rebalance
                days_to_next = len([d for d in self._all_trading_days 
                                   if d > check_date and d <= next_rebal_date])
                
                self._regime_check_schedule[check_date.strftime('%Y%m%d')] = {
                    'check_date': check_date,
                    'next_standard_rebal': next_rebal_date,
                    'days_to_next_rebal': days_to_next,
                    'standard_rebal_idx': i  # Index of the preceding standard rebalance
                }
        
    def _load_price_data(self):
        """Load price data for returns calculation."""
        # Symbol to name mapping (same as in KMRF)
        etf_symbol_name_dict = {
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
            
            # Commodity ETFs
            'GLD': 'SPDR Gold Shares',
            'SLV': 'iShares Silver Trust',
            'USO': 'United States Oil Fund',
            'UNG': 'United States Natural Gas Fund',
            'DBA': 'Invesco DB Agriculture',
            
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
            
            'IYR': 'iShares U.S. Real Estate ETF',
            
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
            
            # INTERNATIONAL
            'VXUS': 'Vanguard Total International Stock ETF',
            'VEA': 'Vanguard FTSE Developed Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF',
            'VGK': 'Vanguard FTSE Europe ETF',
            'VPL': 'Vanguard FTSE Pacific ETF',
            'FXI': 'iShares China Large-Cap ETF',
            'EWJ': 'iShares MSCI Japan ETF',
            'INDA': 'iShares MSCI India ETF',
            'EFA': 'iShares MSCI EAFE ETF',
            'EEM': 'iShares MSCI Emerging Markets ETF',
            
            # BONDS (additional)
            'AGG': 'iShares Core U.S. Aggregate Bond ETF',
            'BND': 'Vanguard Total Bond Market ETF',
            'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
            'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'DBC': 'Invesco DB Commodity Index Tracking Fund',
        }
        
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
        
        data_path = Path('data/processed/all_etf_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Price data not found at {data_path}")
        
        etf_data = pd.read_csv(data_path, index_col=0, header=[0, 1], parse_dates=True)
        etf_data.index = pd.to_datetime(etf_data.index)
        
        # Rename columns using symbol to name mapping
        etf_data.rename(columns=etf_symbol_name_dict, level=0, inplace=True)
        
        # Extract close prices
        close_cols = etf_data.columns[etf_data.columns.get_level_values(1) == 'close']
        close_prices = etf_data[close_cols].droplevel(1, axis=1)
        
        self._close_prices = close_prices
        
        # Also load universe ETF data if available
        universe_path = Path('data/processed/universe_etfs.csv')
        if universe_path.exists():
            universe_data = pd.read_csv(universe_path, index_col=0, header=[0, 1], parse_dates=True)
            universe_data.index = pd.to_datetime(universe_data.index)
            universe_data.rename(columns=universe_symbol_name_dict, level=0, inplace=True)
            universe_close_cols = universe_data.columns[universe_data.columns.get_level_values(1) == 'close']
            universe_close = universe_data[universe_close_cols].droplevel(1, axis=1)
            # Combine with main data (avoid duplicating columns)
            for col in universe_close.columns:
                if col not in self._close_prices.columns:
                    self._close_prices[col] = universe_close[col]
        
        # Calculate simple returns from close prices
        self._asset_returns = self._close_prices.pct_change()
        
    def _load_risk_free_rates(self):
        """Load risk-free rate data."""
        rf_path = Path('data/risk_free_rates.csv')
        if rf_path.exists():
            rf_data = pd.read_csv(rf_path, index_col=0, parse_dates=True)
            # Use 3-month rate by default, annualized
            self._rf_rates = rf_data['RF_3M']
        else:
            # Default to 0 if no risk-free data
            self._rf_rates = None
    
    def _get_rf_rate(self, date: pd.Timestamp) -> float:
        """Get risk-free rate for a given date (annualized decimal)."""
        if self._rf_rates is None:
            return 0.0
        
        # Find nearest available rate
        valid_rates = self._rf_rates[self._rf_rates.index <= date]
        if len(valid_rates) == 0:
            return 0.0
        return float(valid_rates.iloc[-1])
        
    def _generate_rebalance_dates(self) -> List[str]:
        """
        Generate rebalance dates based on frequency.
        
        Returns dates in YYYYMMDD format.
        
        IMPORTANT: Rebalance dates represent the date on which:
        1. We observe market close and regime predictions
        2. We run optimization overnight
        3. We execute trades at next day's open
        
        So returns are calculated from day after rebalance to next rebalance.
        """
        # Get trading days from price data
        trading_days = self._close_prices.index
        
        # Filter to backtest period
        mask = (trading_days >= self.backtest_start) & (trading_days <= self.backtest_end)
        period_days = trading_days[mask]
        
        if len(period_days) == 0:
            raise ValueError(f"No trading days found between {self.backtest_start} and {self.backtest_end}")
        
        # Map frequency to interval
        frequency_map = {
            'daily': 1,
            'weekly': 5,
            'biweekly': 10,
            'monthly': 21,
            'quarterly': 63,
            'custom': self.custom_interval
        }
        interval = frequency_map[self.rebalance_frequency]
        
        # Generate rebalance dates at specified interval
        # Start from first date, then every interval days
        rebal_indices = list(range(0, len(period_days), interval))
        rebal_dates = period_days[rebal_indices]
        
        # Convert to YYYYMMDD format
        return [d.strftime('%Y%m%d') for d in rebal_dates]
    
    def _get_asset_returns(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Get daily returns for portfolio assets between two dates.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Start of return period (exclusive - returns start from day after)
        end_date : pd.Timestamp
            End of return period (inclusive)
            
        Returns
        -------
        pd.DataFrame
            Daily log returns indexed by date, columns are asset names
        """
        # Get returns from OPTIMIZER_INPUTS simulator objects (they have the actual prices)
        # For now, compute from close prices
        
        # The challenge is mapping asset names to price columns
        # We'll do this based on the first optimization result
        if not self.optimization_results:
            raise ValueError("Must run backtest first to get asset mappings")
        
        # Get first optimizer to access simulator objects
        first_opt = list(self.optimization_results.values())[0]
        
        returns_dict = {}
        for asset_name in self.asset_list:
            if asset_name in first_opt.simulator_objects:
                sim = first_opt.simulator_objects[asset_name]
                asset_returns = sim.km_model.returns
                
                # Filter to date range
                mask = (asset_returns.index > start_date) & (asset_returns.index <= end_date)
                returns_dict[asset_name] = asset_returns[mask]
        
        return pd.DataFrame(returns_dict)
    
    def run(self, verbose: bool = True) -> 'BACKTEST':
        """
        Run the backtest.
        
        For each standard rebalance period:
        1. Run optimization at start of period
        2. Check each trading day for regime triggers (if enabled)
        3. Calculate period returns using actual realized returns
        4. Move to next period
        
        Parameters
        ----------
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        if verbose:
            print("=" * 80)
            print("RUNNING BACKTEST")
            print("=" * 80)
            print(f"Period: {self.backtest_start.strftime('%Y-%m-%d')} to {self.backtest_end.strftime('%Y-%m-%d')}")
            print(f"Assets: {len(self.asset_list)}")
            print(f"Standard Rebalances: {len(self.rebalance_dates)}")
            print(f"Objective: {self.objective}")
            print(f"Allow Short: {self.allow_short}, Gross Exposure: {self.gross_exposure}")
            if self.use_regime_trigger:
                print(f"Regime Trigger: Enabled (threshold={self.regime_change_threshold})")
                print(f"Max Intermediate Rebalances per Period: {self.max_intermediate_rebalances}")
            print("=" * 80)
        
        # Get all trading days in backtest period
        all_days = self._close_prices.loc[self.backtest_start:self.backtest_end].index.tolist()
        
        # Storage for results
        weights_list = []  # List of {date, rebal_type, asset weights}
        all_transaction_costs = []
        daily_returns_list = []  # List of {date, return}
        daily_values_list = []  # List of {date, value}
        
        current_weights = None  # pd.Series
        current_portfolio_value = self.initial_capital
        
        # Convert rebalance dates to timestamps
        rebal_ts_list = [pd.Timestamp(d) for d in self.rebalance_dates]
        
        # Process each standard rebalance period
        for period_idx in range(len(rebal_ts_list)):
            period_start = rebal_ts_list[period_idx]
            period_end = rebal_ts_list[period_idx + 1] if period_idx < len(rebal_ts_list) - 1 else pd.Timestamp(self.backtest_end)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Period {period_idx+1}/{len(rebal_ts_list)}] {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
                print(f"{'='*60}")
            
            # Get trading days in this period (inclusive of start, exclusive of end for returns)
            period_days = [d for d in all_days if d >= period_start and d < period_end]
            if period_idx == len(rebal_ts_list) - 1:
                # Last period: include end date
                period_days = [d for d in all_days if d >= period_start and d <= period_end]
            
            if len(period_days) == 0:
                continue
            
            # === STANDARD REBALANCE at period start ===
            rebal_date_str = period_start.strftime('%Y%m%d')
            
            if verbose:
                print(f"\n  [STANDARD] Rebalancing on {rebal_date_str}...")
            
            try:
                new_weights, opt_inputs = self._run_optimization(
                    rebal_date_str, 
                    n_days=self.n_days,
                    seed_offset=period_idx,
                    verbose=verbose
                )
                
                # Store optimization result
                self.optimization_results[rebal_date_str] = opt_inputs
                
                # Calculate and apply transaction costs
                tc = self._calculate_transaction_cost(current_weights, new_weights, current_portfolio_value)
                all_transaction_costs.append({'date': period_start, 'tc': tc, 'type': 'standard'})
                current_portfolio_value -= tc
                
                # Store weights
                weights_list.append({
                    'date': period_start,
                    'rebal_type': 'standard',
                    **{asset: new_weights.get(asset, 0.0) for asset in self.asset_list}
                })
                current_weights = new_weights
                
                # Store baseline for regime checking
                if self.use_regime_trigger:
                    self._store_market_baseline_prediction(period_start, opt_inputs)
                
                if verbose:
                    print(f"    ✓ Optimized. TC: ${tc:.2f}")
                    
            except Exception as e:
                if verbose:
                    print(f"    ✗ Optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                if current_weights is None:
                    # First period failed - use equal weights
                    current_weights = pd.Series({asset: 1.0/len(self.asset_list) for asset in self.asset_list})
                weights_list.append({
                    'date': period_start,
                    'rebal_type': 'standard',
                    **{asset: current_weights.get(asset, 0.0) for asset in self.asset_list}
                })
                continue
            
            # === PROCESS EACH DAY IN PERIOD ===
            intermediate_rebal_count = 0
            last_rebal_date = period_start
            
            for day_idx, day in enumerate(period_days):
                # Calculate return for this day
                # Note: Even the period_start day has returns (portfolio set at prior day's close)
                day_return = self._calculate_daily_return(day, current_weights, opt_inputs)
                current_portfolio_value = current_portfolio_value * (1 + day_return)
                
                daily_returns_list.append({'date': day, 'return': day_return})
                daily_values_list.append({'date': day, 'value': current_portfolio_value})
                
                # === CHECK FOR REGIME TRIGGER (if enabled and not last day of period) ===
                if (self.use_regime_trigger and 
                    intermediate_rebal_count < self.max_intermediate_rebalances and
                    day < period_end):
                    
                    should_trigger, prob_change = self._check_regime_trigger(day, opt_inputs)
                    
                    if should_trigger:
                        # Calculate days remaining to period end
                        days_remaining = len([d for d in period_days if d > day])
                        
                        if days_remaining < 2:
                            # Not enough days remaining for rebalance to matter
                            continue
                        
                        if verbose:
                            print(f"\n  [TRIGGERED] Regime change on {day.strftime('%Y-%m-%d')}")
                            print(f"    Prob change: {prob_change:.3f} >= threshold {self.regime_change_threshold}")
                            print(f"    Days remaining in period: {days_remaining}")
                        
                        try:
                            # Run optimization with shortened horizon
                            triggered_weights, triggered_opt = self._run_optimization(
                                day.strftime('%Y%m%d'),
                                n_days=min(days_remaining, self.n_days),
                                seed_offset=period_idx * 100 + intermediate_rebal_count,
                                verbose=False
                            )
                            
                            # Store result
                            self.optimization_results[day.strftime('%Y%m%d')] = triggered_opt
                            
                            # Calculate and apply transaction costs
                            tc = self._calculate_transaction_cost(current_weights, triggered_weights, current_portfolio_value)
                            all_transaction_costs.append({'date': day, 'tc': tc, 'type': 'triggered'})
                            current_portfolio_value -= tc
                            
                            # Store weights
                            weights_list.append({
                                'date': day,
                                'rebal_type': 'triggered',
                                **{asset: triggered_weights.get(asset, 0.0) for asset in self.asset_list}
                            })
                            current_weights = triggered_weights
                            
                            # Update baseline
                            self._store_market_baseline_prediction(day, triggered_opt)
                            
                            intermediate_rebal_count += 1
                            last_rebal_date = day
                            
                            if verbose:
                                print(f"    ✓ Triggered rebalance complete. TC: ${tc:.2f}")
                                print(f"    Intermediate rebalances this period: {intermediate_rebal_count}/{self.max_intermediate_rebalances}")
                            
                        except Exception as e:
                            if verbose:
                                print(f"    ✗ Triggered rebalance failed: {e}")
        
        # === COMPILE RESULTS ===
        if verbose:
            print(f"\n{'='*80}")
            print("COMPILING RESULTS")
            print(f"{'='*80}")
        
        # Convert to DataFrames
        if weights_list:
            weights_df = pd.DataFrame(weights_list)
            self.weights_history = weights_df.drop(columns=['rebal_type']).set_index('date')
            self._rebalance_types = weights_df.set_index('date')['rebal_type']
        
        if daily_values_list:
            values_df = pd.DataFrame(daily_values_list).set_index('date')
            self.portfolio_value = values_df['value']
        
        if daily_returns_list:
            returns_df = pd.DataFrame(daily_returns_list).set_index('date')
            self.returns = returns_df['return']
        
        if all_transaction_costs:
            tc_df = pd.DataFrame(all_transaction_costs)
            self.transaction_costs = tc_df.set_index('date')['tc']
        
        # Count rebalances
        n_standard = len(self.rebalance_dates)
        n_triggered = len(weights_list) - n_standard if weights_list else 0
        
        if verbose:
            print(f"\n  Standard rebalances: {n_standard}")
            print(f"  Triggered rebalances: {n_triggered}")
            print(f"  Total transaction costs: ${self.transaction_costs.sum() if self.transaction_costs is not None else 0:.2f}")
            print("\n✓ Backtest complete!")
            self.print_summary()
        
        return self
    
    def run_parallel(self, n_cpus: int = None, verbose: bool = True) -> 'BACKTEST':
        """
        Run the backtest using parallel processing for entire rebalance periods.
        
        Each period (including standard rebalance + any triggered rebalances) is
        processed in parallel. This is possible because:
        1. Regime trigger checks only depend on predictions (deterministic)
        2. Daily returns are calculated from preloaded price data
        3. Only portfolio value accumulation needs to be sequential (done at the end)
        
        Parameters
        ----------
        n_cpus : int, optional
            Number of CPU cores to use. Defaults to (cpu_count - 1) or 1.
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        if n_cpus is None:
            n_cpus = max(1, os.cpu_count() - 1)
        
        if verbose:
            print("=" * 80)
            print("RUNNING BACKTEST (PARALLEL MODE)")
            print("=" * 80)
            print(f"Period: {self.backtest_start.strftime('%Y-%m-%d')} to {self.backtest_end.strftime('%Y-%m-%d')}")
            print(f"Assets: {len(self.asset_list)}")
            print(f"Standard Rebalances: {len(self.rebalance_dates)}")
            print(f"Objective: {self.objective}")
            print(f"Allow Short: {self.allow_short}, Gross Exposure: {self.gross_exposure}")
            if self.use_regime_trigger:
                print(f"Regime Trigger: Enabled (threshold={self.regime_change_threshold})")
                print(f"Max Intermediate Rebalances per Period: {self.max_intermediate_rebalances}")
            print(f"Parallel Workers: {n_cpus}")
            print("=" * 80)
        
        # Get all trading days in backtest period
        all_days = self._close_prices.loc[self.backtest_start:self.backtest_end].index.tolist()
        
        # Convert rebalance dates to timestamps
        rebal_ts_list = [pd.Timestamp(d) for d in self.rebalance_dates]
        
        # Prepare period info for parallel processing
        period_args = []
        for period_idx in range(len(rebal_ts_list)):
            period_start = rebal_ts_list[period_idx]
            period_end = rebal_ts_list[period_idx + 1] if period_idx < len(rebal_ts_list) - 1 else pd.Timestamp(self.backtest_end)
            
            # Get trading days in this period
            if period_idx == len(rebal_ts_list) - 1:
                period_days = [d for d in all_days if d >= period_start and d <= period_end]
            else:
                period_days = [d for d in all_days if d >= period_start and d < period_end]
            
            rf_rate = self._get_rf_rate(period_start)
            period_args.append((period_idx, period_start, period_end, period_days, rf_rate))
        
        # === Run all periods in parallel ===
        if verbose:
            print(f"\n[Parallel] Processing {len(period_args)} periods with optimizations and regime triggers...")
        
        period_results = Parallel(n_jobs=n_cpus, verbose=10 if verbose else 0)(
            delayed(self._process_period_parallel)(
                period_idx, period_start, period_end, period_days, rf_rate
            )
            for period_idx, period_start, period_end, period_days, rf_rate in period_args
        )
        
        # Sort results by period_idx
        period_results = sorted(period_results, key=lambda x: x['period_idx'])
        
        if verbose:
            n_success = sum(1 for r in period_results if r['success'])
            total_triggered = sum(r['n_triggered'] for r in period_results)
            print(f"\n  Periods complete: {n_success}/{len(period_args)} successful")
            print(f"  Total triggered rebalances: {total_triggered}")
        
        # === Combine results and compute portfolio values sequentially ===
        if verbose:
            print(f"\n[Sequential] Computing portfolio values...")
        
        weights_list = []
        all_transaction_costs = []
        daily_returns_list = []
        daily_values_list = []
        
        current_weights = None
        current_portfolio_value = self.initial_capital
        
        for result in period_results:
            if not result['success']:
                # Use equal weights for failed periods
                if current_weights is None:
                    current_weights = pd.Series({asset: 1.0/len(self.asset_list) for asset in self.asset_list})
                # Still need to process days with current weights
                for day, day_return in result.get('daily_returns', []):
                    current_portfolio_value *= (1 + day_return)
                    daily_returns_list.append({'date': day, 'return': day_return})
                    daily_values_list.append({'date': day, 'value': current_portfolio_value})
                continue
            
            # Store optimization results
            for date_str, opt_inputs in result['optimization_results'].items():
                self.optimization_results[date_str] = opt_inputs
            
            # Process each rebalance in this period
            for rebal_info in result['rebalances']:
                new_weights = rebal_info['weights']
                
                # Calculate transaction cost based on current portfolio value
                tc = self._calculate_transaction_cost(current_weights, new_weights, current_portfolio_value)
                all_transaction_costs.append({
                    'date': rebal_info['date'], 
                    'tc': tc, 
                    'type': rebal_info['rebal_type']
                })
                current_portfolio_value -= tc
                
                # Store weights
                weights_list.append({
                    'date': rebal_info['date'],
                    'rebal_type': rebal_info['rebal_type'],
                    **{asset: new_weights.get(asset, 0.0) for asset in self.asset_list}
                })
                current_weights = new_weights
            
            # Process daily returns for this period
            for day, day_return in result['daily_returns']:
                current_portfolio_value *= (1 + day_return)
                daily_returns_list.append({'date': day, 'return': day_return})
                daily_values_list.append({'date': day, 'value': current_portfolio_value})
        
        # === Compile Results ===
        if verbose:
            print(f"\n{'='*80}")
            print("COMPILING RESULTS")
            print(f"{'='*80}")
        
        if weights_list:
            weights_df = pd.DataFrame(weights_list)
            self.weights_history = weights_df.drop(columns=['rebal_type']).set_index('date')
            self._rebalance_types = weights_df.set_index('date')['rebal_type']
        
        if daily_values_list:
            values_df = pd.DataFrame(daily_values_list).set_index('date')
            self.portfolio_value = values_df['value']
        
        if daily_returns_list:
            returns_df = pd.DataFrame(daily_returns_list).set_index('date')
            self.returns = returns_df['return']
        
        if all_transaction_costs:
            tc_df = pd.DataFrame(all_transaction_costs)
            self.transaction_costs = tc_df.set_index('date')['tc']
        
        n_standard = len(self.rebalance_dates)
        n_triggered = sum(r['n_triggered'] for r in period_results)
        
        if verbose:
            print(f"\n  Standard rebalances: {n_standard}")
            print(f"  Triggered rebalances: {n_triggered}")
            print(f"  Total transaction costs: ${self.transaction_costs.sum() if self.transaction_costs is not None else 0:.2f}")
            print("\n✓ Backtest complete!")
            self.print_summary()
        
        return self
    
    def _process_period_parallel(
        self,
        period_idx: int,
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
        period_days: List[pd.Timestamp],
        rf_rate: float
    ) -> Dict:
        """
        Process an entire rebalance period including triggered rebalances.
        
        This method is designed to be called in parallel for each period.
        It runs the standard rebalance optimization, checks for regime triggers,
        and runs triggered rebalance optimizations as needed.
        
        Parameters
        ----------
        period_idx : int
            Index of this period
        period_start : pd.Timestamp
            Start date of the period (standard rebalance date)
        period_end : pd.Timestamp
            End date of the period
        period_days : List[pd.Timestamp]
            Trading days in this period
        rf_rate : float
            Risk-free rate for this period
            
        Returns
        -------
        Dict
            Dictionary containing:
            - period_idx: int
            - success: bool
            - rebalances: List of dicts with date, weights, rebal_type
            - daily_returns: List of (date, return) tuples
            - optimization_results: Dict of date_str -> OPTIMIZER_INPUTS
            - n_triggered: int count of triggered rebalances
            - error: Optional error message
        """
        try:
            if len(period_days) == 0:
                return {
                    'period_idx': period_idx,
                    'success': True,
                    'rebalances': [],
                    'daily_returns': [],
                    'optimization_results': {},
                    'n_triggered': 0,
                    'error': None
                }
            
            rebalances = []
            daily_returns = []
            optimization_results = {}
            n_triggered = 0
            
            # === Standard Rebalance ===
            rebal_date_str = period_start.strftime('%Y%m%d')
            
            opt_inputs = OPTIMIZER_INPUTS(
                opt_date=rebal_date_str,
                asset_list=self.asset_list,
                n_days=self.n_days,
                n_simulations=self.n_simulations,
                market_asset=self.market_asset,
                random_seed=self.random_seed + period_idx,
                risk_free_rate=rf_rate
            )
            
            # Run simulation pipeline
            opt_inputs.load_simulator_objects(verbose=False)
            opt_inputs.estimate_regime_correlations(verbose=False)
            opt_inputs.estimate_regime_concordance(verbose=False)
            opt_inputs.simulate_market_regime_paths(verbose=False)
            opt_inputs.simulate_asset_regime_paths(verbose=False)
            opt_inputs.simulate_returns_copula(verbose=False)
            opt_inputs.compute_portfolio_inputs(verbose=False)
            
            opt_inputs.optimize_portfolio(
                objective=self.objective,
                allow_short=self.allow_short,
                gross_exposure=self.gross_exposure,
                risk_aversion=self.risk_aversion
            )
            
            current_weights = opt_inputs.optimal_weights.copy()
            optimization_results[rebal_date_str] = opt_inputs
            
            rebalances.append({
                'date': period_start,
                'weights': current_weights,
                'rebal_type': 'standard'
            })
            
            # Store baseline for regime checking
            baseline_fwd_probs = None
            if self.use_regime_trigger:
                try:
                    market_sim = opt_inputs.market_simulator if hasattr(opt_inputs, 'market_simulator') else opt_inputs.simulator_objects.get(self.market_asset)
                    if market_sim is not None:
                        baseline_fwd_probs = market_sim.get_forward_regime_probs(period_start, self.n_days)
                except Exception:
                    pass
            
            # === Process each day ===
            intermediate_rebal_count = 0
            current_opt_inputs = opt_inputs
            
            for day in period_days:
                # Calculate daily return
                day_return = self._calculate_daily_return(day, current_weights, current_opt_inputs)
                daily_returns.append((day, day_return))
                
                # Check for regime trigger
                if (self.use_regime_trigger and 
                    baseline_fwd_probs is not None and
                    intermediate_rebal_count < self.max_intermediate_rebalances and
                    day < period_end and
                    day != period_start):  # Don't check on rebalance day itself
                    
                    should_trigger, _ = self._check_regime_trigger_static(
                        day, current_opt_inputs, baseline_fwd_probs
                    )
                    
                    if should_trigger:
                        days_remaining = len([d for d in period_days if d > day])
                        
                        if days_remaining >= 2:
                            try:
                                # Run triggered optimization
                                triggered_rf = self._get_rf_rate(day)
                                triggered_opt = OPTIMIZER_INPUTS(
                                    opt_date=day.strftime('%Y%m%d'),
                                    asset_list=self.asset_list,
                                    n_days=min(days_remaining, self.n_days),
                                    n_simulations=self.n_simulations,
                                    market_asset=self.market_asset,
                                    random_seed=self.random_seed + period_idx * 100 + intermediate_rebal_count,
                                    risk_free_rate=triggered_rf
                                )
                                
                                triggered_opt.load_simulator_objects(verbose=False)
                                triggered_opt.estimate_regime_correlations(verbose=False)
                                triggered_opt.estimate_regime_concordance(verbose=False)
                                triggered_opt.simulate_market_regime_paths(verbose=False)
                                triggered_opt.simulate_asset_regime_paths(verbose=False)
                                triggered_opt.simulate_returns_copula(verbose=False)
                                triggered_opt.compute_portfolio_inputs(verbose=False)
                                
                                triggered_opt.optimize_portfolio(
                                    objective=self.objective,
                                    allow_short=self.allow_short,
                                    gross_exposure=self.gross_exposure,
                                    risk_aversion=self.risk_aversion
                                )
                                
                                current_weights = triggered_opt.optimal_weights.copy()
                                current_opt_inputs = triggered_opt
                                optimization_results[day.strftime('%Y%m%d')] = triggered_opt
                                
                                rebalances.append({
                                    'date': day,
                                    'weights': current_weights,
                                    'rebal_type': 'triggered'
                                })
                                
                                # Update baseline
                                try:
                                    market_sim = triggered_opt.market_simulator if hasattr(triggered_opt, 'market_simulator') else triggered_opt.simulator_objects.get(self.market_asset)
                                    if market_sim is not None:
                                        baseline_fwd_probs = market_sim.get_forward_regime_probs(day, self.n_days)
                                except Exception:
                                    pass
                                
                                intermediate_rebal_count += 1
                                n_triggered += 1
                                
                            except Exception:
                                pass  # Skip failed triggered rebalances
            
            return {
                'period_idx': period_idx,
                'success': True,
                'rebalances': rebalances,
                'daily_returns': daily_returns,
                'optimization_results': optimization_results,
                'n_triggered': n_triggered,
                'error': None
            }
            
        except Exception as e:
            import traceback
            return {
                'period_idx': period_idx,
                'success': False,
                'rebalances': [],
                'daily_returns': [],
                'optimization_results': {},
                'n_triggered': 0,
                'error': str(e) + '\n' + traceback.format_exc()
            }
    
    def _check_regime_trigger_static(
        self,
        check_date: pd.Timestamp,
        opt_inputs: OPTIMIZER_INPUTS,
        baseline_fwd_probs: pd.DataFrame
    ) -> Tuple[bool, float]:
        """
        Check if regime has changed enough to trigger rebalance (static version for parallel).
        
        This is a static version that takes the baseline predictions as a parameter
        instead of using instance state, making it safe for parallel execution.
        
        Parameters
        ----------
        check_date : pd.Timestamp
            Date to check
        opt_inputs : OPTIMIZER_INPUTS
            Optimizer with market simulator
        baseline_fwd_probs : pd.DataFrame
            Forward predictions from the last rebalance
            
        Returns
        -------
        Tuple[bool, float]
            (should_trigger, max_probability_change)
        """
        if baseline_fwd_probs is None or len(baseline_fwd_probs) == 0:
            return False, 0.0
        
        try:
            # Get current horizon-1 prediction
            market_sim = opt_inputs.market_simulator if hasattr(opt_inputs, 'market_simulator') else opt_inputs.simulator_objects.get(self.market_asset)
            if market_sim is None:
                return False, 0.0
            
            current_probs = market_sim.get_forward_regime_probs(check_date, 1)
            
            if len(current_probs) == 0:
                return False, 0.0
            
            target_date = current_probs.index[0]
            current_h1_probs = current_probs.iloc[0].values
            
            if target_date not in baseline_fwd_probs.index:
                return False, 0.0
            
            baseline_probs = baseline_fwd_probs.loc[target_date].values
            
            prob_change = np.abs(current_h1_probs - baseline_probs).max()
            should_trigger = prob_change >= self.regime_change_threshold
            
            return should_trigger, prob_change
            
        except Exception:
            return False, 0.0
    
    def _run_optimization(
        self, 
        rebal_date: str, 
        n_days: int,
        seed_offset: int = 0,
        verbose: bool = False
    ) -> Tuple[pd.Series, OPTIMIZER_INPUTS]:
        """
        Run portfolio optimization for a given date.
        
        Parameters
        ----------
        rebal_date : str
            Date in YYYYMMDD format
        n_days : int
            Forecast horizon
        seed_offset : int
            Offset to add to random seed for reproducibility
        verbose : bool
            Print progress
            
        Returns
        -------
        Tuple[pd.Series, OPTIMIZER_INPUTS]
            (optimal_weights, optimizer_instance)
        """
        rf_rate = self._get_rf_rate(pd.Timestamp(rebal_date))
        
        opt_inputs = OPTIMIZER_INPUTS(
            opt_date=rebal_date,
            asset_list=self.asset_list,
            n_days=n_days,
            n_simulations=self.n_simulations,
            market_asset=self.market_asset,
            random_seed=self.random_seed + seed_offset,
            risk_free_rate=rf_rate
        )
        
        # Run simulation pipeline
        if verbose:
            print("    Loading simulators...")
        opt_inputs.load_simulator_objects(verbose=False)
        
        if verbose:
            print("    Running simulation...")
        opt_inputs.estimate_regime_correlations(verbose=False)
        opt_inputs.estimate_regime_concordance(verbose=False)
        opt_inputs.simulate_market_regime_paths(verbose=False)
        opt_inputs.simulate_asset_regime_paths(verbose=False)
        opt_inputs.simulate_returns_copula(verbose=False)
        opt_inputs.compute_portfolio_inputs(verbose=False)
        
        if verbose:
            print("    Optimizing portfolio...")
        opt_inputs.optimize_portfolio(
            objective=self.objective,
            allow_short=self.allow_short,
            gross_exposure=self.gross_exposure,
            risk_aversion=self.risk_aversion
        )
        
        return opt_inputs.optimal_weights.copy(), opt_inputs
    
    def _run_single_optimization(
        self,
        period_idx: int,
        rebal_date_str: str,
        rf_rate: float
    ) -> Dict:
        """
        Run optimization for a single rebalance period (used by joblib for parallel execution).
        
        Parameters
        ----------
        period_idx : int
            Index of the rebalance period
        rebal_date_str : str
            Date in YYYYMMDD format
        rf_rate : float
            Risk-free rate for the period
            
        Returns
        -------
        Dict
            Dictionary with period_idx, rebal_date_str, weights, opt_inputs, success, error
        """
        try:
            opt_inputs = OPTIMIZER_INPUTS(
                opt_date=rebal_date_str,
                asset_list=self.asset_list,
                n_days=self.n_days,
                n_simulations=self.n_simulations,
                market_asset=self.market_asset,
                random_seed=self.random_seed + period_idx,
                risk_free_rate=rf_rate
            )
            
            # Run simulation pipeline
            opt_inputs.load_simulator_objects(verbose=False)
            opt_inputs.estimate_regime_correlations(verbose=False)
            opt_inputs.estimate_regime_concordance(verbose=False)
            opt_inputs.simulate_market_regime_paths(verbose=False)
            opt_inputs.simulate_asset_regime_paths(verbose=False)
            opt_inputs.simulate_returns_copula(verbose=False)
            opt_inputs.compute_portfolio_inputs(verbose=False)
            
            opt_inputs.optimize_portfolio(
                objective=self.objective,
                allow_short=self.allow_short,
                gross_exposure=self.gross_exposure,
                risk_aversion=self.risk_aversion
            )
            
            return {
                'period_idx': period_idx,
                'rebal_date_str': rebal_date_str,
                'weights': opt_inputs.optimal_weights.copy(),
                'opt_inputs': opt_inputs,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            import traceback
            return {
                'period_idx': period_idx,
                'rebal_date_str': rebal_date_str,
                'weights': None,
                'opt_inputs': None,
                'success': False,
                'error': str(e) + '\n' + traceback.format_exc()
            }
    
    def _calculate_transaction_cost(
        self,
        old_weights: Optional[pd.Series],
        new_weights: pd.Series,
        portfolio_value: float
    ) -> float:
        """
        Calculate transaction cost for a rebalance.
        
        Transaction cost = portfolio_value * turnover * cost_bps / 10000
        where turnover = sum(|new_weight - old_weight|) / 2
        
        Parameters
        ----------
        old_weights : pd.Series or None
            Previous weights (None for initial allocation)
        new_weights : pd.Series
            New weights
        portfolio_value : float
            Current portfolio value
            
        Returns
        -------
        float
            Transaction cost in dollars
        """
        if old_weights is None:
            # Initial allocation: cost on full position
            turnover = new_weights.abs().sum() / 2  # Divide by 2 since only buying
        else:
            # Rebalance: cost on weight changes
            # Align indices
            all_assets = set(old_weights.index) | set(new_weights.index)
            old_aligned = pd.Series({a: old_weights.get(a, 0.0) for a in all_assets})
            new_aligned = pd.Series({a: new_weights.get(a, 0.0) for a in all_assets})
            turnover = (new_aligned - old_aligned).abs().sum() / 2
        
        return portfolio_value * turnover * (self.transaction_cost_bps / 10000)
    
    def _calculate_daily_return(
        self,
        day: pd.Timestamp,
        weights: pd.Series,
        opt_inputs: OPTIMIZER_INPUTS
    ) -> float:
        """
        Calculate portfolio return for a single day using preloaded price data.
        
        Parameters
        ----------
        day : pd.Timestamp
            The day to calculate return for
        weights : pd.Series
            Current portfolio weights
        opt_inputs : OPTIMIZER_INPUTS
            Optimizer with loaded simulators (not used, kept for interface compatibility)
            
        Returns
        -------
        float
            Portfolio return for the day (simple return)
        """
        day_return = 0.0
        
        # Normalize the day to match index (remove time component if present)
        day_normalized = pd.Timestamp(day.date())
        
        for asset_name in self.asset_list:
            if asset_name in self._asset_returns.columns:
                if day_normalized in self._asset_returns.index:
                    asset_return = self._asset_returns.loc[day_normalized, asset_name]
                    if pd.notna(asset_return):
                        weight = weights.get(asset_name, 0.0)
                        day_return += weight * asset_return
        
        return day_return
    
    def _check_regime_trigger(
        self,
        check_date: pd.Timestamp,
        opt_inputs: OPTIMIZER_INPUTS
    ) -> Tuple[bool, float]:
        """
        Check if regime has changed enough to trigger rebalance.
        
        Compares current horizon-1 prediction (for tomorrow) against the prediction
        that was made at the last rebalance for THIS SAME target date.
        
        The logic: At the last rebalance, we predicted regime probs for the next N days.
        Now we're at a future date. If the new prediction for tomorrow differs significantly
        from what we predicted for tomorrow back at the last rebalance, trigger rebalance.
        
        Parameters
        ----------
        check_date : pd.Timestamp
            Date to check (we compare predictions for check_date + 1 day)
        opt_inputs : OPTIMIZER_INPUTS
            Optimizer with market simulator
            
        Returns
        -------
        Tuple[bool, float]
            (should_trigger, max_probability_change)
        """
        if not hasattr(self, '_last_rebal_market_fwd_probs') or self._last_rebal_market_fwd_probs is None:
            return False, 0.0
        
        try:
            # Get current horizon-1 prediction (for tomorrow)
            market_sim = opt_inputs.market_simulator
            current_probs = market_sim.get_forward_regime_probs(check_date, 1)
            
            if len(current_probs) == 0:
                return False, 0.0
            
            # The target date is the index of current_probs (prediction_date)
            target_date = current_probs.index[0]
            current_h1_probs = current_probs.iloc[0].values
            
            # Find the prediction for this same target date from the last rebalance
            # The stored forward probs are indexed by prediction_date (target date)
            if target_date not in self._last_rebal_market_fwd_probs.index:
                # Target date not in our stored predictions (beyond horizon)
                return False, 0.0
            
            baseline_probs = self._last_rebal_market_fwd_probs.loc[target_date].values
            
            # Compare predictions for the same target date
            prob_change = np.abs(current_h1_probs - baseline_probs).max()
            
            should_trigger = prob_change >= self.regime_change_threshold
            return should_trigger, prob_change
            
        except Exception as e:
            return False, 0.0
    
    def print_summary(self):
        """Print backtest performance summary."""
        if self.returns is None or len(self.returns) == 0:
            print("No returns calculated yet. Run backtest first.")
            return
        
        # Calculate metrics
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        n_days = len(self.returns)
        n_years = n_days / 252
        
        # Annualized return
        ann_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
        
        # Annualized volatility
        ann_vol = self.returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (using average risk-free rate)
        avg_rf = self._rf_rates.loc[self.returns.index].mean() if self._rf_rates is not None else 0
        sharpe = (ann_return/100 - avg_rf) / (ann_vol/100) if ann_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        # Win rate
        win_rate = (self.returns > 0).mean() * 100
        
        # Sortino ratio
        neg_returns = self.returns[self.returns < 0]
        downside_std = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0
        sortino = (ann_return/100 - avg_rf) / downside_std if downside_std > 0 else 0
        
        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Period:             {self.backtest_start.strftime('%Y-%m-%d')} to {self.backtest_end.strftime('%Y-%m-%d')}")
        print(f"Trading Days:       {n_days}")
        
        # Count standard vs triggered rebalances
        n_standard = len(self.rebalance_dates)
        n_total = len(self.weights_history) if self.weights_history is not None else n_standard
        n_triggered = n_total - n_standard
        
        print(f"Standard Rebalances: {n_standard}")
        if n_triggered > 0:
            print(f"Triggered Rebalances: {n_triggered}")
            print(f"Total Rebalances:   {n_total}")
        print("-" * 60)
        print(f"Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"Final Value:        ${self.portfolio_value.iloc[-1]:,.2f}")
        print(f"Total Return:       {total_return:.2f}%")
        print(f"Annualized Return:  {ann_return:.2f}%")
        print(f"Annualized Vol:     {ann_vol:.2f}%")
        print("-" * 60)
        print(f"Sharpe Ratio:       {sharpe:.3f}")
        print(f"Sortino Ratio:      {sortino:.3f}")
        print(f"Max Drawdown:       {max_dd:.2f}%")
        print(f"Win Rate:           {win_rate:.1f}%")
        print("=" * 60)
    
    def get_metrics(self, include_benchmarks: bool = True) -> pd.DataFrame:
        """
        Get comprehensive performance metrics comparing strategy to benchmarks.
        
        Parameters
        ----------
        include_benchmarks : bool, default=True
            Whether to include benchmark comparisons
        
        Returns
        -------
        pd.DataFrame
            DataFrame with metrics for strategy and benchmarks
        """
        if self.returns is None or len(self.returns) == 0:
            return pd.DataFrame()
        
        def calc_metrics(returns: pd.Series, values: pd.Series, name: str, n_rebalances: int = None) -> Dict:
            """Calculate metrics for a single return series."""
            n_days = len(returns)
            n_years = n_days / 252
            
            total_return = (values.iloc[-1] / values.iloc[0] - 1) if len(values) > 0 else 0
            ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
            ann_vol = returns.std() * np.sqrt(252)
            
            avg_rf = 0
            if self._rf_rates is not None:
                rf_aligned = self._rf_rates.reindex(returns.index)
                avg_rf = rf_aligned.mean() if len(rf_aligned.dropna()) > 0 else 0
            
            sharpe = (ann_return - avg_rf) / ann_vol if ann_vol > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            neg_returns = returns[returns < 0]
            downside_std = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0
            sortino = (ann_return - avg_rf) / downside_std if downside_std > 0 else 0
            
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
            
            metrics = {
                'Total Return': total_return,
                'Annualized Return': ann_return,
                'Annualized Volatility': ann_vol,
                'Sharpe Ratio': sharpe,
                'Sortino Ratio': sortino,
                'Max Drawdown': max_dd,
                'Calmar Ratio': calmar,
                'Win Rate': (returns > 0).mean(),
                'Avg Daily Return': returns.mean(),
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                'N Rebalances': n_rebalances if n_rebalances is not None else np.nan
            }
            
            return metrics
        
        # Calculate strategy metrics - count total rebalances including triggered
        n_total_rebalances = len(self.weights_history) if self.weights_history is not None else len(self.rebalance_dates)
        results = {}
        results['Regime Strategy'] = calc_metrics(self.returns, self.portfolio_value, 'Regime Strategy', n_total_rebalances)
        
        if include_benchmarks:
            # Ensure benchmarks are computed
            if not hasattr(self, 'benchmark_returns') or self.benchmark_returns is None:
                self.compute_benchmarks()
            
            # Benchmarks only rebalance on standard dates
            n_standard_rebalances = len(self.rebalance_dates)
            
            for bench_name, bench_returns in self.benchmark_returns.items():
                bench_values = self.benchmark_values[bench_name]
                # S&P 500 has no rebalances (buy and hold), MV portfolios rebalance on standard dates
                if bench_name == 'S&P 500':
                    n_rebal = 0  # Buy and hold
                else:
                    n_rebal = n_standard_rebalances
                results[bench_name] = calc_metrics(bench_returns, bench_values, bench_name, n_rebal)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(results)
        
        return metrics_df
    
    def plot_performance(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot cumulative returns comparing strategy to benchmarks.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(12, 6)
            Figure size
        """
        if self.portfolio_value is None:
            raise ValueError("Run backtest first")
        
        # Ensure benchmarks are computed
        if not hasattr(self, 'benchmark_returns') or self.benchmark_returns is None:
            self.compute_benchmarks()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for consistency
        colors = {
            'Regime Strategy': 'blue',
            'S&P 500': 'gray',
            'MV Long-Only': 'green',
            'MV Long-Short': 'orange'
        }
        
        # Plot strategy cumulative returns
        cum_returns = self.portfolio_value / self.initial_capital
        ax.plot(cum_returns.index, cum_returns.values, label='Regime Strategy', 
                linewidth=2.5, color=colors['Regime Strategy'])
        
        # Plot benchmark cumulative returns
        for bench_name, bench_values in self.benchmark_values.items():
            bench_cum = bench_values / self.initial_capital
            ax.plot(bench_cum.index, bench_cum.values, label=bench_name, 
                    linewidth=1.5, alpha=0.8, color=colors.get(bench_name, 'purple'))
        
        ax.set_title('Cumulative Returns: Regime Strategy vs Benchmarks', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Growth of $1')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_analysis(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Plot detailed analysis charts including drawdowns, rolling Sharpe, 
        return distribution, weight allocation, and metrics summary.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(16, 12)
            Figure size
        """
        if self.portfolio_value is None:
            raise ValueError("Run backtest first")
        
        # Ensure benchmarks are computed
        if not hasattr(self, 'benchmark_returns') or self.benchmark_returns is None:
            self.compute_benchmarks()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Define colors for consistency
        colors = {
            'Regime Strategy': 'blue',
            'S&P 500': 'gray',
            'MV Long-Only': 'green',
            'MV Long-Short': 'orange'
        }
        
        # 1. Drawdown Comparison
        ax = axes[0, 0]
        
        # Strategy drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, 
                        color=colors['Regime Strategy'], label='Regime Strategy')
        
        # Benchmark drawdowns (just lines)
        for bench_name, bench_returns in self.benchmark_returns.items():
            bench_cum = (1 + bench_returns).cumprod()
            bench_max = bench_cum.expanding().max()
            bench_dd = (bench_cum - bench_max) / bench_max * 100
            ax.plot(bench_dd.index, bench_dd.values, linewidth=1, alpha=0.7,
                    color=colors.get(bench_name, 'purple'), label=bench_name)
        
        ax.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Comparison
        ax = axes[0, 1]
        window = min(126, len(self.returns) // 2)  # Use 6-month or half the data
        
        if window >= 20:
            rolling_mean = self.returns.rolling(window).mean() * 252
            rolling_std = self.returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_mean / rolling_std
            ax.plot(rolling_sharpe.dropna().index, rolling_sharpe.dropna().values, 
                    linewidth=1.5, color=colors['Regime Strategy'], label='Regime Strategy')
            
            for bench_name, bench_returns in self.benchmark_returns.items():
                bench_roll_mean = bench_returns.rolling(window).mean() * 252
                bench_roll_std = bench_returns.rolling(window).std() * np.sqrt(252)
                bench_roll_sharpe = bench_roll_mean / bench_roll_std
                ax.plot(bench_roll_sharpe.dropna().index, bench_roll_sharpe.dropna().values, 
                        linewidth=1, alpha=0.7, color=colors.get(bench_name, 'purple'), label=bench_name)
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for rolling Sharpe', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        # 3. Return Distribution
        ax = axes[1, 0]
        ax.hist(self.returns * 100, bins=50, edgecolor='black', alpha=0.7, 
                color=colors['Regime Strategy'], label='Regime Strategy')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=self.returns.mean() * 100, color='green', linestyle='--', 
                   label=f'Mean: {self.returns.mean()*100:.3f}%')
        ax.set_title('Daily Return Distribution (Strategy)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Weight Allocation Over Time
        ax = axes[1, 1]
        if self.weights_history is not None and len(self.weights_history) > 0:
            has_shorts = (self.weights_history < 0).any().any()
            if has_shorts:
                self.weights_history.plot(ax=ax, alpha=0.8, linewidth=1.5)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.set_title('Portfolio Weight Allocation (Line - has shorts)', fontsize=12, fontweight='bold')
            else:
                self.weights_history.plot.area(ax=ax, alpha=0.7, stacked=True)
                ax.set_title('Portfolio Weight Allocation', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No weight history available', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def _compute_mv_weights(
        self,
        returns_lookback: pd.DataFrame,
        allow_short: bool = False,
        gross_exposure: float = 1.0
    ) -> pd.Series:
        """
        Compute mean-variance optimal weights using historical returns.
        
        Parameters
        ----------
        returns_lookback : pd.DataFrame
            Historical returns for the lookback period (assets as columns)
        allow_short : bool
            Whether to allow short positions
        gross_exposure : float
            Maximum gross exposure (sum of absolute weights)
            
        Returns
        -------
        pd.Series
            Optimal weights
        """
        from scipy.optimize import minimize
        
        # Calculate expected returns and covariance
        mu = returns_lookback.mean() * 252  # Annualized
        Sigma = returns_lookback.cov() * 252  # Annualized
        
        n_assets = len(mu)
        
        # Objective: maximize Sharpe (minimize negative Sharpe)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
            if port_vol < 1e-10:
                return 1e10
            return -port_return / port_vol
        
        # Constraints
        constraints = []
        
        if allow_short:
            # Net exposure = 1 (fully invested)
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            # Gross exposure constraint
            constraints.append({'type': 'ineq', 'fun': lambda w: gross_exposure - np.sum(np.abs(w))})
            bounds = [(-gross_exposure, gross_exposure) for _ in range(n_assets)]
        else:
            # Long only: weights sum to 1, all >= 0
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weight
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = pd.Series(result.x, index=mu.index)
        else:
            # Fallback to equal weight
            weights = pd.Series(np.ones(n_assets) / n_assets, index=mu.index)
        
        return weights
    
    def compute_benchmarks(self, lookback_days: int = 252) -> None:
        """
        Compute benchmark portfolio returns.
        
        Creates:
        - SPY: S&P 500 ETF returns
        - MV Long-Only: Mean-variance optimized long-only portfolio
        - MV Long-Short: Mean-variance optimized with shorts (if strategy allows shorts)
        
        All MV portfolios are rebalanced only on standard rebalance dates.
        
        Parameters
        ----------
        lookback_days : int, default=252
            Number of trading days for historical estimation
        """
        if self.returns is None:
            raise ValueError("Run backtest first")
        
        # Get all trading days in backtest period
        backtest_days = self.returns.index
        
        # === 1. S&P 500 Benchmark ===
        spy_returns = self._asset_returns['SPDR S&P 500 ETF'].reindex(backtest_days).fillna(0)
        
        # === 2. Mean-Variance Long-Only Benchmark ===
        # Get returns for portfolio assets
        asset_returns = self._asset_returns[self.asset_list].copy()
        
        mv_long_values = [self.initial_capital]
        mv_long_weights = None
        current_mv_long_value = self.initial_capital
        
        # Also track long-short if strategy allows shorts
        if self.allow_short:
            mv_ls_values = [self.initial_capital]
            mv_ls_weights = None
            current_mv_ls_value = self.initial_capital
        
        for i, day in enumerate(backtest_days):
            # Check if this is a standard rebalance date
            day_str = day.strftime('%Y%m%d')
            is_rebal_day = day_str in self.rebalance_dates
            
            if is_rebal_day or mv_long_weights is None:
                # Get lookback returns
                lookback_end_idx = asset_returns.index.get_loc(day)
                lookback_start_idx = max(0, lookback_end_idx - lookback_days)
                returns_lookback = asset_returns.iloc[lookback_start_idx:lookback_end_idx].dropna(how='all')
                
                if len(returns_lookback) >= 60:  # Minimum 60 days of data
                    # Compute MV long-only weights
                    mv_long_weights = self._compute_mv_weights(
                        returns_lookback,
                        allow_short=False,
                        gross_exposure=1.0
                    )
                    
                    # Compute MV long-short weights if applicable
                    if self.allow_short:
                        mv_ls_weights = self._compute_mv_weights(
                            returns_lookback,
                            allow_short=True,
                            gross_exposure=self.gross_exposure
                        )
                else:
                    # Not enough data, use equal weight
                    mv_long_weights = pd.Series(1/len(self.asset_list), index=self.asset_list)
                    if self.allow_short:
                        mv_ls_weights = pd.Series(1/len(self.asset_list), index=self.asset_list)
            
            # Calculate daily returns
            day_returns = asset_returns.loc[day].fillna(0)
            
            # MV Long-only return
            mv_long_ret = (mv_long_weights * day_returns).sum()
            current_mv_long_value *= (1 + mv_long_ret)
            mv_long_values.append(current_mv_long_value)
            
            # MV Long-short return
            if self.allow_short:
                mv_ls_ret = (mv_ls_weights * day_returns).sum()
                current_mv_ls_value *= (1 + mv_ls_ret)
                mv_ls_values.append(current_mv_ls_value)
        
        # Store benchmark data
        self.benchmark_returns = {
            'S&P 500': spy_returns,
            'MV Long-Only': pd.Series(mv_long_values[1:], index=backtest_days).pct_change().fillna(0)
        }
        
        self.benchmark_values = {
            'S&P 500': self.initial_capital * (1 + spy_returns).cumprod(),
            'MV Long-Only': pd.Series(mv_long_values[1:], index=backtest_days)
        }
        
        if self.allow_short:
            self.benchmark_returns['MV Long-Short'] = pd.Series(mv_ls_values[1:], index=backtest_days).pct_change().fillna(0)
            self.benchmark_values['MV Long-Short'] = pd.Series(mv_ls_values[1:], index=backtest_days)
        
        # Fix first day returns (should be calculated from values)
        for name in self.benchmark_returns:
            values = self.benchmark_values[name]
            returns = values.pct_change()
            returns.iloc[0] = (values.iloc[0] / self.initial_capital) - 1
            self.benchmark_returns[name] = returns
    
    def get_benchmark_returns(self, benchmark_name: str = None) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Get benchmark returns for comparison.
        
        Parameters
        ----------
        benchmark_name : str, optional
            Specific benchmark name. If None, returns all benchmarks.
            Options: 'S&P 500', 'MV Long-Only', 'MV Long-Short'
            
        Returns
        -------
        pd.Series or Dict[str, pd.Series]
            Daily returns of benchmark(s)
        """
        if not hasattr(self, 'benchmark_returns') or self.benchmark_returns is None:
            self.compute_benchmarks()
        
        if benchmark_name is not None:
            if benchmark_name not in self.benchmark_returns:
                raise ValueError(f"Benchmark '{benchmark_name}' not found. Available: {list(self.benchmark_returns.keys())}")
            return self.benchmark_returns[benchmark_name]
        
        return self.benchmark_returns
    
    def compare_to_benchmark(self, benchmark_asset: str = 'SPDR S&P 500 ETF') -> pd.DataFrame:
        """
        Compare strategy performance to benchmark.
        
        Parameters
        ----------
        benchmark_asset : str, default='SPDR S&P 500 ETF'
            Asset name to use as benchmark
            
        Returns
        -------
        pd.DataFrame
            Comparison metrics for strategy vs benchmark
        """
        if self.returns is None:
            raise ValueError("Run backtest first")
        
        bench_returns = self.get_benchmark_returns(benchmark_asset)
        
        # Align returns
        common_idx = self.returns.index.intersection(bench_returns.index)
        strat_returns = self.returns.loc[common_idx]
        bench_returns = bench_returns.loc[common_idx]
        
        n_years = len(strat_returns) / 252
        
        def calc_metrics(rets, name):
            total_ret = (1 + rets).prod() - 1
            ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            cum = (1 + rets).cumprod()
            max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
            
            neg_rets = rets[rets < 0]
            downside = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 0
            sortino = ann_ret / downside if downside > 0 else 0
            
            return {
                'name': name,
                'total_return': total_ret,
                'ann_return': ann_ret,
                'ann_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'win_rate': (rets > 0).mean()
            }
        
        strat_metrics = calc_metrics(strat_returns, 'Strategy')
        bench_metrics = calc_metrics(bench_returns, 'Benchmark')
        
        comparison = pd.DataFrame([strat_metrics, bench_metrics]).set_index('name').T
        
        # Add relative metrics
        comparison['Difference'] = comparison['Strategy'] - comparison['Benchmark']
        
        # Alpha (simple approximation)
        beta = strat_returns.cov(bench_returns) / bench_returns.var()
        alpha = comparison.loc['ann_return', 'Strategy'] - beta * comparison.loc['ann_return', 'Benchmark']
        
        comparison.loc['beta', :] = [beta, 1.0, beta - 1.0]
        comparison.loc['alpha', :] = [alpha, 0.0, alpha]
        
        # Information ratio
        tracking_error = (strat_returns - bench_returns).std() * np.sqrt(252)
        excess_return = comparison.loc['ann_return', 'Strategy'] - comparison.loc['ann_return', 'Benchmark']
        info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        comparison.loc['tracking_error', :] = [tracking_error, 0.0, tracking_error]
        comparison.loc['information_ratio', :] = [info_ratio, 0.0, info_ratio]
        
        return comparison
    
    def _store_market_baseline_prediction(
        self,
        rebal_date: pd.Timestamp,
        opt_inputs: OPTIMIZER_INPUTS
    ):
        """
        Store the market asset's forward regime predictions from the most recent rebalance.
        
        Stores the full forward prediction DataFrame (all horizons) so we can compare
        predictions for the SAME target date when checking for regime triggers.
        
        Parameters
        ----------
        rebal_date : pd.Timestamp
            The rebalance date
        opt_inputs : OPTIMIZER_INPUTS
            OPTIMIZER_INPUTS instance with loaded simulators
        """
        if not self.use_regime_trigger:
            return
        
        self._last_rebal_date = rebal_date
        
        # Get market asset's forward regime probs for all horizons
        # This gives us predictions indexed by target date
        try:
            market_sim = opt_inputs.market_simulator
            fwd_probs = market_sim.get_forward_regime_probs(rebal_date, self.n_days)
            if len(fwd_probs) > 0:
                # Store the full DataFrame indexed by prediction_date (target date)
                self._last_rebal_market_fwd_probs = fwd_probs.copy()
        except Exception:
            # If market_simulator not available, try from simulator_objects
            if self.market_asset in opt_inputs.simulator_objects:
                sim = opt_inputs.simulator_objects[self.market_asset]
                fwd_probs = sim.get_forward_regime_probs(rebal_date, self.n_days)
                if len(fwd_probs) > 0:
                    self._last_rebal_market_fwd_probs = fwd_probs.copy()
    
    def get_turnover(self) -> pd.Series:
        """
        Calculate portfolio turnover at each rebalance.
        
        Turnover = sum of absolute weight changes / 2
        (divided by 2 because sells = buys)
        
        Returns
        -------
        pd.Series
            Turnover at each rebalance date
        """
        if self.weights_history is None or len(self.weights_history) < 2:
            return pd.Series(dtype=float)
        
        weight_changes = self.weights_history.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2
        
        return turnover.iloc[1:]  # Skip first (no prior weights)
    
    def save(self, filepath: str):
        """
        Save backtest results to pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to save the backtest results
        """
        save_data = {
            'backtest_start': self.backtest_start,
            'backtest_end': self.backtest_end,
            'asset_list': self.asset_list,
            'rebalance_frequency': self.rebalance_frequency,
            'objective': self.objective,
            'allow_short': self.allow_short,
            'gross_exposure': self.gross_exposure,
            'n_days': self.n_days,
            'n_simulations': self.n_simulations,
            'initial_capital': self.initial_capital,
            'transaction_cost_bps': self.transaction_cost_bps,
            'rebalance_dates': self.rebalance_dates,
            'weights_history': self.weights_history,
            'portfolio_value': self.portfolio_value,
            'returns': self.returns,
            'daily_weights': self.daily_weights,
            'metrics': self.get_metrics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Backtest saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BACKTEST':
        """
        Load backtest results from pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved backtest file
            
        Returns
        -------
        BACKTEST
            Loaded backtest instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance with saved parameters
        bt = cls(
            backtest_start=data['backtest_start'],
            backtest_end=data['backtest_end'],
            asset_list=data['asset_list'],
            rebalance_frequency=data['rebalance_frequency'],
            objective=data['objective'],
            allow_short=data['allow_short'],
            gross_exposure=data['gross_exposure'],
            n_days=data['n_days'],
            n_simulations=data['n_simulations'],
            initial_capital=data['initial_capital'],
            transaction_cost_bps=data['transaction_cost_bps']
        )
        
        # Restore results
        bt.rebalance_dates = data['rebalance_dates']
        bt.weights_history = data['weights_history']
        bt.portfolio_value = data['portfolio_value']
        bt.returns = data['returns']
        bt.daily_weights = data['daily_weights']
        
        return bt


# Utility function for generating rebalance dates
def generate_rebalance_dates(
    start_date: str,
    end_date: str,
    frequency: str = 'monthly',
    custom_interval: Optional[int] = None,
    price_data: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Generate rebalance dates based on start/end dates and frequency.
    
    This is a standalone utility function that can be used outside the BACKTEST class.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' or 'YYYYMMDD' format
    end_date : str
        End date in 'YYYY-MM-DD' or 'YYYYMMDD' format
    frequency : str, default='monthly'
        Rebalancing frequency. Options:
        - 'daily': Every trading day
        - 'weekly': Every 5 trading days
        - 'biweekly': Every 10 trading days
        - 'monthly': Every 21 trading days
        - 'quarterly': Every 63 trading days
        - 'custom': Use custom_interval parameter
    custom_interval : int, optional
        Number of trading days between rebalances (used with frequency='custom')
    price_data : pd.DataFrame, optional
        DataFrame with DatetimeIndex of trading days
        
    Returns
    -------
    List[str]
        List of rebalance dates in 'YYYYMMDD' format
    """
    if price_data is None:
        # Load default price data
        data_path = Path('data/processed/all_etf_data.csv')
        if data_path.exists():
            etf_data = pd.read_csv(data_path, index_col=0, header=[0, 1], parse_dates=True)
            close_cols = etf_data.columns[etf_data.columns.get_level_values(1) == 'close']
            price_data = etf_data[close_cols].droplevel(1, axis=1)
        else:
            raise FileNotFoundError("No price data found and none provided")
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    trading_days = price_data.loc[start_ts:end_ts].index
    
    if len(trading_days) == 0:
        raise ValueError(f"No trading days found between {start_date} and {end_date}")
    
    frequency_map = {
        'daily': 1,
        'weekly': 5,
        'biweekly': 10,
        'monthly': 21,
        'quarterly': 63,
        'custom': custom_interval
    }
    
    if frequency not in frequency_map:
        raise ValueError(f"Unknown frequency: {frequency}. Choose from: {list(frequency_map.keys())}")
    
    interval = frequency_map[frequency]
    
    if interval is None:
        raise ValueError("Must provide custom_interval when using frequency='custom'")
    
    rebal_dates = trading_days[::interval]
    
    return [d.strftime('%Y%m%d') for d in rebal_dates]
