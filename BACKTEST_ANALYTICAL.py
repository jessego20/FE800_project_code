"""
BACKTEST.py

Backtesting framework for portfolio optimization strategies using the 
ANALYTICAL_INPUTS → PORTFOLIO_OPTIMIZER pipeline.

Key Features:
- Flexible rebalancing schedules (1 to 21 trading days)
- Portfolio performance tracking with daily returns
- Transaction cost modeling
- Comprehensive performance metrics (Sharpe, Sortino, Drawdown, etc.)
- Benchmark comparisons (S&P 500, Equal Weight, Mean-Variance)
- Visualization tools

Author: Jesse Goodman
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path
from datetime import datetime
import warnings
from joblib import Parallel, delayed

from ANALYTICAL_INPUTS import ANALYTICAL_INPUTS
from PORTFOLIO_OPTIMIZER import PORTFOLIO_OPTIMIZER
from MODEL_INFO import get_KM_model_dates

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

TRADING_DAYS = CustomBusinessDay(calendar=USFederalHolidayCalendar())

warnings.filterwarnings('ignore')


class BACKTEST:
    """
    Backtesting framework for regime-based portfolio optimization strategies.
    
    This class runs historical backtests by:
    1. Generating rebalance dates based on specified frequency
    2. At each rebalance date: running ANALYTICAL_INPUTS to get μ/Σ, then optimizing
    3. Tracking portfolio value using actual (realized) returns between rebalances
    4. Computing comprehensive performance metrics
    
    Parameters
    ----------
    asset_list : List[str]
        List of asset names to include in the portfolio
    start_date : str or pd.Timestamp
        First rebalance date (inclusive), YYYY-MM-DD or YYYYMMDD format
    end_date : str or pd.Timestamp
        End date of backtest (exclusive), YYYY-MM-DD or YYYYMMDD format
    rebalance_frequency : int, default=21
        Number of trading days between rebalances (1 to 21)
    objective : str, default='max_sharpe'
        Optimization objective: 'max_sharpe', 'min_variance', 'risk_parity', 'mean_variance'
    allow_short_selling : bool, default=False
        Whether to allow short positions
    gross_exposure_limit : float, default=1.0
        Maximum gross exposure (sum of |weights|). Only used if allow_short_selling=True.
        E.g., 1.5 allows 150% long + 50% short = 100% net, 200% gross
    min_weight : float, default=0.0
        Minimum weight per asset. Only used if allow_short_selling=True.
    max_weight : float, default=1.0
        Maximum weight per asset
    max_turnover : float, optional
        Maximum allowed turnover per rebalance. None = no constraint.
    risk_aversion : float, default=1.0
        Risk aversion parameter for 'mean_variance' objective
    n_days : int, default=1
        Forecast horizon in trading days for ANALYTICAL_INPUTS (for KMRF predictions)
    initial_capital : float, default=100000.0
        Starting portfolio value
    transaction_cost_bps : float, default=5.0
        Transaction costs in basis points (e.g., 5 = 0.05%)
        
    Attributes
    ----------
    rebalance_dates : List[pd.Timestamp]
        Generated rebalance dates
    weights_history : pd.DataFrame
        Portfolio weights at each rebalance date
    daily_weights : pd.DataFrame
        Portfolio weights for each trading day
    portfolio_value : pd.Series
        Daily portfolio value
    returns : pd.Series
        Daily portfolio returns
    """
    
    # Valid optimization objectives
    VALID_OBJECTIVES = ['max_sharpe', 'min_variance', 'risk_parity', 'mean_variance']
    
    def __init__(
        self,
        asset_list: List[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        rebalance_frequency: int = 21,
        objective: str = 'max_sharpe',
        allow_short_selling: bool = False,
        gross_exposure_limit: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        max_turnover: Optional[float] = None,
        risk_aversion: float = 1.0,
        n_days: int = 1,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0
    ):
        # Validate inputs
        if not asset_list:
            raise ValueError("asset_list must be provided and cannot be empty")
        
        if objective not in self.VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {self.VALID_OBJECTIVES}")
        
        if not 1 <= rebalance_frequency <= 21:
            raise ValueError("rebalance_frequency must be between 1 and 21 trading days")
        
        if allow_short_selling and gross_exposure_limit < 1.0:
            raise ValueError("gross_exposure_limit must be >= 1.0 when short selling is allowed")
        
        # Store parameters
        self.asset_list = asset_list
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.rebalance_frequency = rebalance_frequency
        self.objective = objective
        self.allow_short_selling = allow_short_selling
        self.gross_exposure_limit = gross_exposure_limit
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_turnover = max_turnover
        self.risk_aversion = risk_aversion
        self.n_days = n_days
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        
        # Load price data
        self._load_price_data()
        
        # Load risk-free rates
        self._load_risk_free_rates()
        
        # Get available model dates
        self._available_model_dates = get_KM_model_dates()
        
        # Generate rebalance dates
        self.rebalance_dates = self._generate_rebalance_dates()
        
        # Initialize result containers
        self.weights_history: Optional[pd.DataFrame] = None
        self.daily_weights: Optional[pd.DataFrame] = None
        self.portfolio_value: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.transaction_costs: Optional[pd.Series] = None
        self.turnover_history: Optional[pd.Series] = None
        
        # Benchmark results
        self.benchmark_values: Optional[Dict[str, pd.Series]] = None
        self.benchmark_returns: Optional[Dict[str, pd.Series]] = None
        
        # Optimization details (for analysis)
        self._optimization_inputs: Dict[pd.Timestamp, ANALYTICAL_INPUTS] = {}
        self._optimizers: Dict[pd.Timestamp, PORTFOLIO_OPTIMIZER] = {}
    
    def _load_price_data(self):
        """Load price data for returns calculation."""
        # Symbol to name mapping
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
            'IWM': 'iShares Russell 2000 ETF',
            'IWB': 'iShares Russell 1000 ETF',
            'DIA': 'SPDR Dow Jones Industrial Average ETF',
            'VTI': 'Vanguard Total Stock Market ETF',
            
            # S&P 500 SECTOR ETFS
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
            'IWF': 'iShares Russell 1000 Growth ETF',
            'IWO': 'iShares Russell 2000 Growth ETF',
            'VUG': 'Vanguard Growth ETF',
            
            # VALUE ETFs
            'IVE': 'iShares S&P 500 Value ETF',
            'IWD': 'iShares Russell 1000 Value ETF',
            'IWN': 'iShares Russell 2000 Value ETF',
            'VTV': 'Vanguard Value ETF',
            
            # SIZE ETFs
            'IWR': 'iShares Russell Mid-Cap ETF',
            'IWC': 'iShares Micro-Cap ETF',
            'IJH': 'iShares Core S&P Mid-Cap ETF',
            'IJR': 'iShares Core S&P Small-Cap ETF',
            'MDY': 'SPDR S&P MidCap 400 ETF',
            
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
            
            # BONDS
            'AGG': 'iShares Core U.S. Aggregate Bond ETF',
            'BND': 'Vanguard Total Bond Market ETF',
            'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
            'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'DBC': 'Invesco DB Commodity Index Tracking Fund',
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
        self._close_prices = etf_data[close_cols].droplevel(1, axis=1)
        
        # Calculate simple returns
        self._asset_returns = self._close_prices.pct_change()
    
    def _load_risk_free_rates(self):
        """Load risk-free rate data."""
        rf_path = Path('data/risk_free_rates.csv')
        if rf_path.exists():
            rf_data = pd.read_csv(rf_path, index_col=0, parse_dates=True)
            self._rf_rates = rf_data['RF_3M']
        else:
            self._rf_rates = None
    
    def _get_rf_rate(self, date: pd.Timestamp) -> float:
        """Get risk-free rate for a given date (annualized decimal)."""
        if self._rf_rates is None:
            return 0.0
        
        valid_rates = self._rf_rates[self._rf_rates.index <= date]
        if len(valid_rates) == 0:
            return 0.0
        return float(valid_rates.iloc[-1])
    
    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate rebalance dates based on frequency."""
        # Get trading days from price data
        trading_days = self._close_prices.index
        
        # Filter to backtest period
        mask = (trading_days >= self.start_date) & (trading_days < self.end_date)
        period_days = trading_days[mask]
        
        if len(period_days) == 0:
            raise ValueError(f"No trading days found between {self.start_date} and {self.end_date}")
        
        # Generate rebalance dates at specified interval
        rebal_indices = list(range(0, len(period_days), self.rebalance_frequency))
        rebal_dates = period_days[rebal_indices]
        
        return list(rebal_dates)
    
    def run(self, verbose: bool = True) -> 'BACKTEST':
        """
        Execute the backtest.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print progress updates
            
        Returns
        -------
        BACKTEST
            Self, for method chaining
        """
        if verbose:
            print(f"\n{'='*80}")
            print("RUNNING BACKTEST")
            print(f"{'='*80}")
            print(f"  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            print(f"  Assets: {len(self.asset_list)}")
            print(f"  Rebalance frequency: Every {self.rebalance_frequency} trading days")
            print(f"  Rebalance dates: {len(self.rebalance_dates)}")
            print(f"  Objective: {self.objective}")
            print(f"  Short selling: {'Allowed' if self.allow_short_selling else 'Not allowed'}")
            if self.allow_short_selling:
                print(f"  Gross exposure limit: {self.gross_exposure_limit:.1%}")
            print(f"{'='*80}\n")
        
        # Get all trading days for the backtest period
        trading_days = self._close_prices.index
        mask = (trading_days >= self.start_date) & (trading_days < self.end_date)
        backtest_days = trading_days[mask]
        
        if len(backtest_days) == 0:
            raise ValueError("No trading days in backtest period")
        
        # Initialize containers
        weights_dict = {}
        daily_weights_list = []
        portfolio_values = [self.initial_capital]
        returns_list = []
        transaction_costs_list = []
        turnover_list = []
        
        previous_weights = None
        current_portfolio_value = self.initial_capital
        
        # Iterate through rebalance dates
        for i, rebal_date in enumerate(self.rebalance_dates):
            if verbose:
                print(f"  [{i+1}/{len(self.rebalance_dates)}] Rebalancing on {rebal_date.strftime('%Y-%m-%d')}...", end=' ')
            
            try:
                # Run ANALYTICAL_INPUTS pipeline
                inputs = ANALYTICAL_INPUTS(
                    opt_date=rebal_date.strftime('%Y%m%d'),
                    asset_list=self.asset_list,
                    n_days=self.n_days,
                    annualize=True
                )
                inputs.run_full_pipeline(verbose=False)
                
                # Get risk-free rate
                rf_rate = self._get_rf_rate(rebal_date)
                
                # Create optimizer
                optimizer = PORTFOLIO_OPTIMIZER.from_analytical_inputs(
                    analytical_inputs=inputs,
                    risk_free_rate=rf_rate
                )
                
                # Run optimization
                weights = optimizer.optimize(
                    objective=self.objective,
                    min_weight=self.min_weight,
                    max_weight=self.max_weight,
                    allow_short_selling=self.allow_short_selling,
                    gross_exposure_limit=self.gross_exposure_limit,
                    max_turnover=self.max_turnover,
                    previous_weights=previous_weights,
                    risk_aversion=self.risk_aversion,
                    verbose=False
                )
                
                # Store optimization objects for analysis
                self._optimization_inputs[rebal_date] = inputs
                self._optimizers[rebal_date] = optimizer
                
                if verbose:
                    print(f"✓ (μ={optimizer.portfolio_return:.2%}, σ={optimizer.portfolio_volatility:.2%})")
                
            except Exception as e:
                if verbose:
                    print(f"✗ Error: {str(e)[:50]}...")
                # Use equal weights as fallback
                weights = pd.Series(1.0 / len(self.asset_list), index=self.asset_list)
            
            # Store weights
            weights_dict[rebal_date] = weights
            
            # Calculate turnover and transaction costs
            if previous_weights is not None:
                # Align weights
                aligned_prev = previous_weights.reindex(weights.index).fillna(0)
                turnover = np.sum(np.abs(weights - aligned_prev)) / 2
                tc = turnover * current_portfolio_value * (self.transaction_cost_bps / 10000)
                current_portfolio_value -= tc
            else:
                turnover = 1.0  # Initial allocation
                tc = current_portfolio_value * (self.transaction_cost_bps / 10000)
                current_portfolio_value -= tc
            
            turnover_list.append(turnover)
            transaction_costs_list.append(tc)
            
            # Determine next rebalance date
            if i + 1 < len(self.rebalance_dates):
                next_rebal = self.rebalance_dates[i + 1]
            else:
                next_rebal = self.end_date
            
            # Get trading days between rebalances
            days_mask = (backtest_days > rebal_date) & (backtest_days <= next_rebal)
            holding_days = backtest_days[days_mask]
            
            # Calculate returns for each day in holding period
            for day in holding_days:
                # Get asset returns for this day
                day_returns = self._asset_returns.loc[day, self.asset_list].fillna(0)
                
                # Portfolio return
                port_return = (weights * day_returns).sum()
                returns_list.append(port_return)
                
                # Update portfolio value
                current_portfolio_value *= (1 + port_return)
                portfolio_values.append(current_portfolio_value)
                
                # Store daily weights (weights don't change within holding period)
                daily_weights_list.append(weights)
            
            previous_weights = weights.copy()
        
        # Build result DataFrames
        self.weights_history = pd.DataFrame(weights_dict).T
        self.weights_history.index.name = 'rebalance_date'
        
        # Daily weights (excluding first day which is rebalance day)
        all_days = backtest_days[1:]  # Skip first day
        self.daily_weights = pd.DataFrame(daily_weights_list, index=all_days[:len(daily_weights_list)])
        
        # Portfolio value series
        self.portfolio_value = pd.Series(
            portfolio_values[:len(all_days)+1],
            index=[backtest_days[0]] + list(all_days[:len(portfolio_values)-1])
        )
        
        # Returns series
        self.returns = pd.Series(returns_list, index=all_days[:len(returns_list)])
        
        # Transaction costs
        self.transaction_costs = pd.Series(transaction_costs_list, index=self.rebalance_dates)
        self.turnover_history = pd.Series(turnover_list, index=self.rebalance_dates)
        
        if verbose:
            print(f"\n{'='*80}")
            print("BACKTEST COMPLETE")
            print(f"{'='*80}")
            print(f"  Total transaction costs: ${self.transaction_costs.sum():,.2f}")
            print(f"  Average turnover: {self.turnover_history.mean():.2%}")
        
        return self
    
    def run_parallel(self, n_jobs: int = -1, verbose: bool = True) -> 'BACKTEST':
        """
        Execute the backtest with parallelized optimization computations.
        
        This method parallelizes the expensive ANALYTICAL_INPUTS and optimization
        computations across rebalance dates using joblib. The return calculations
        are still done sequentially since they depend on the order of weights.
        
        Parameters
        ----------
        n_jobs : int, default=-1
            Number of parallel jobs. -1 uses all available cores.
            1 = sequential (useful for debugging)
        verbose : bool, default=True
            Print progress updates
            
        Returns
        -------
        BACKTEST
            Self, for method chaining
        """
        if verbose:
            print(f"\n{'='*80}")
            print("RUNNING BACKTEST (PARALLEL)")
            print(f"{'='*80}")
            print(f"  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            print(f"  Assets: {len(self.asset_list)}")
            print(f"  Rebalance frequency: Every {self.rebalance_frequency} trading days")
            print(f"  Rebalance dates: {len(self.rebalance_dates)}")
            print(f"  Objective: {self.objective}")
            print(f"  Short selling: {'Allowed' if self.allow_short_selling else 'Not allowed'}")
            if self.allow_short_selling:
                print(f"  Gross exposure limit: {self.gross_exposure_limit:.1%}")
            print(f"  Parallel jobs: {n_jobs if n_jobs > 0 else 'all cores'}")
            print(f"{'='*80}\n")
        
        # Get all trading days for the backtest period
        trading_days = self._close_prices.index
        mask = (trading_days >= self.start_date) & (trading_days < self.end_date)
        backtest_days = trading_days[mask]
        
        if len(backtest_days) == 0:
            raise ValueError("No trading days in backtest period")
        
        # Pre-compute risk-free rates for all dates
        rf_rates = {date: self._get_rf_rate(date) for date in self.rebalance_dates}
        
        # Define the optimization function for a single rebalance date
        def optimize_for_date(rebal_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Series, Optional[float], Optional[float], Optional[Exception]]:
            """
            Run optimization for a single rebalance date.
            
            Returns tuple of (date, weights, expected_return, expected_vol, error)
            """
            try:
                # Run ANALYTICAL_INPUTS pipeline
                inputs = ANALYTICAL_INPUTS(
                    opt_date=rebal_date.strftime('%Y%m%d'),
                    asset_list=self.asset_list,
                    n_days=self.n_days,
                    annualize=True
                )
                inputs.run_full_pipeline(verbose=False)
                
                # Get risk-free rate
                rf_rate = rf_rates[rebal_date]
                
                # Create optimizer
                optimizer = PORTFOLIO_OPTIMIZER.from_analytical_inputs(
                    analytical_inputs=inputs,
                    risk_free_rate=rf_rate
                )
                
                # Run optimization (without turnover constraint - will be applied in sequential pass if needed)
                weights = optimizer.optimize(
                    objective=self.objective,
                    min_weight=self.min_weight,
                    max_weight=self.max_weight,
                    allow_short_selling=self.allow_short_selling,
                    gross_exposure_limit=self.gross_exposure_limit,
                    max_turnover=None,  # Can't apply turnover in parallel
                    previous_weights=None,
                    risk_aversion=self.risk_aversion,
                    verbose=False
                )
                
                return (rebal_date, weights, optimizer.portfolio_return, optimizer.portfolio_volatility, None)
                
            except Exception as e:
                # Return equal weights as fallback
                weights = pd.Series(1.0 / len(self.asset_list), index=self.asset_list)
                return (rebal_date, weights, None, None, e)
        
        # Run optimizations in parallel
        if verbose:
            print("  Computing optimizations in parallel...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(optimize_for_date)(date) for date in self.rebalance_dates
        )
        
        # Sort results by date and extract weights
        results_dict = {r[0]: r for r in results}
        
        if verbose:
            print("\n  Parallel optimization complete. Assembling results...\n")
            # Print summary of each optimization
            for i, rebal_date in enumerate(self.rebalance_dates):
                result = results_dict[rebal_date]
                _, weights, exp_ret, exp_vol, error = result
                if error is None:
                    print(f"  [{i+1}/{len(self.rebalance_dates)}] {rebal_date.strftime('%Y-%m-%d')}: "
                          f"✓ (μ={exp_ret:.2%}, σ={exp_vol:.2%})")
                else:
                    print(f"  [{i+1}/{len(self.rebalance_dates)}] {rebal_date.strftime('%Y-%m-%d')}: "
                          f"✗ Error: {str(error)[:50]}...")
        
        # Initialize containers
        weights_dict = {}
        daily_weights_list = []
        portfolio_values = [self.initial_capital]
        returns_list = []
        transaction_costs_list = []
        turnover_list = []
        
        previous_weights = None
        current_portfolio_value = self.initial_capital
        
        # Sequential pass: calculate returns and track portfolio value
        for i, rebal_date in enumerate(self.rebalance_dates):
            # Get pre-computed weights
            result = results_dict[rebal_date]
            _, weights, exp_ret, exp_vol, error = result
            
            # Apply turnover constraint if specified (re-optimize if needed)
            if self.max_turnover is not None and previous_weights is not None:
                # Check if turnover exceeds limit
                aligned_prev = previous_weights.reindex(weights.index).fillna(0)
                current_turnover = np.sum(np.abs(weights - aligned_prev)) / 2
                
                if current_turnover > self.max_turnover:
                    # Re-run optimization with turnover constraint
                    try:
                        inputs = ANALYTICAL_INPUTS(
                            opt_date=rebal_date.strftime('%Y%m%d'),
                            asset_list=self.asset_list,
                            n_days=self.n_days,
                            annualize=True
                        )
                        inputs.run_full_pipeline(verbose=False)
                        
                        rf_rate = rf_rates[rebal_date]
                        optimizer = PORTFOLIO_OPTIMIZER.from_analytical_inputs(
                            analytical_inputs=inputs,
                            risk_free_rate=rf_rate
                        )
                        
                        weights = optimizer.optimize(
                            objective=self.objective,
                            min_weight=self.min_weight,
                            max_weight=self.max_weight,
                            allow_short_selling=self.allow_short_selling,
                            gross_exposure_limit=self.gross_exposure_limit,
                            max_turnover=self.max_turnover,
                            previous_weights=previous_weights,
                            risk_aversion=self.risk_aversion,
                            verbose=False
                        )
                    except Exception:
                        pass  # Keep original weights
            
            # Store weights
            weights_dict[rebal_date] = weights
            
            # Calculate turnover and transaction costs
            if previous_weights is not None:
                aligned_prev = previous_weights.reindex(weights.index).fillna(0)
                turnover = np.sum(np.abs(weights - aligned_prev)) / 2
                tc = turnover * current_portfolio_value * (self.transaction_cost_bps / 10000)
                current_portfolio_value -= tc
            else:
                turnover = 1.0
                tc = current_portfolio_value * (self.transaction_cost_bps / 10000)
                current_portfolio_value -= tc
            
            turnover_list.append(turnover)
            transaction_costs_list.append(tc)
            
            # Determine next rebalance date
            if i + 1 < len(self.rebalance_dates):
                next_rebal = self.rebalance_dates[i + 1]
            else:
                next_rebal = self.end_date
            
            # Get trading days between rebalances
            days_mask = (backtest_days > rebal_date) & (backtest_days <= next_rebal)
            holding_days = backtest_days[days_mask]
            
            # Calculate returns for each day in holding period
            for day in holding_days:
                day_returns = self._asset_returns.loc[day, self.asset_list].fillna(0)
                port_return = (weights * day_returns).sum()
                returns_list.append(port_return)
                current_portfolio_value *= (1 + port_return)
                portfolio_values.append(current_portfolio_value)
                daily_weights_list.append(weights)
            
            previous_weights = weights.copy()
        
        # Build result DataFrames
        self.weights_history = pd.DataFrame(weights_dict).T
        self.weights_history.index.name = 'rebalance_date'
        
        all_days = backtest_days[1:]
        self.daily_weights = pd.DataFrame(daily_weights_list, index=all_days[:len(daily_weights_list)])
        
        self.portfolio_value = pd.Series(
            portfolio_values[:len(all_days)+1],
            index=[backtest_days[0]] + list(all_days[:len(portfolio_values)-1])
        )
        
        self.returns = pd.Series(returns_list, index=all_days[:len(returns_list)])
        self.transaction_costs = pd.Series(transaction_costs_list, index=self.rebalance_dates)
        self.turnover_history = pd.Series(turnover_list, index=self.rebalance_dates)
        
        if verbose:
            print(f"\n{'='*80}")
            print("BACKTEST COMPLETE (PARALLEL)")
            print(f"{'='*80}")
            print(f"  Total transaction costs: ${self.transaction_costs.sum():,.2f}")
            print(f"  Average turnover: {self.turnover_history.mean():.2%}")
        
        return self
    
    # ====================================================================================
    # PERFORMANCE METRICS
    # ====================================================================================
    
    def get_metrics(self, include_benchmarks: bool = True) -> pd.DataFrame:
        """
        Get comprehensive performance metrics.
        
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
            raise ValueError("Must run backtest first")
        
        def calc_metrics(returns: pd.Series, values: pd.Series, name: str) -> Dict:
            """Calculate metrics for a single return series."""
            n_days = len(returns)
            n_years = n_days / 252
            
            total_return = (values.iloc[-1] / values.iloc[0] - 1)
            ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
            ann_vol = returns.std() * np.sqrt(252)
            
            # Get average risk-free rate
            avg_rf = 0
            if self._rf_rates is not None:
                rf_aligned = self._rf_rates.reindex(returns.index)
                avg_rf = rf_aligned.mean() if len(rf_aligned.dropna()) > 0 else 0
            
            sharpe = (ann_return - avg_rf) / ann_vol if ann_vol > 0 else 0
            
            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            # Sortino ratio
            neg_returns = returns[returns < 0]
            downside_std = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0
            sortino = (ann_return - avg_rf) / downside_std if downside_std > 0 else 0
            
            # Calmar ratio
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
            
            return {
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
            }
        
        # Calculate strategy metrics
        results = {}
        results['Strategy'] = calc_metrics(self.returns, self.portfolio_value, 'Strategy')
        
        if include_benchmarks:
            self._compute_benchmarks()
            for bench_name, bench_returns in self.benchmark_returns.items():
                bench_values = self.benchmark_values[bench_name]
                results[bench_name] = calc_metrics(bench_returns, bench_values, bench_name)
        
        return pd.DataFrame(results)
    
    def _compute_benchmarks(self, lookback_days: int = 252):
        """
        Compute benchmark portfolios.
        
        Creates:
        - S&P 500: Buy and hold S&P 500 ETF
        - Equal Weight: Equal weight all assets, rebalanced on same schedule
        - MV Long-Only: Mean-variance long-only using historical returns
        """
        if self.benchmark_values is not None:
            return
        
        backtest_days = self.returns.index
        
        self.benchmark_values = {}
        self.benchmark_returns = {}
        
        # === 1. S&P 500 Benchmark ===
        spy_returns = self._asset_returns['SPDR S&P 500 ETF'].reindex(backtest_days).fillna(0)
        spy_values = (1 + spy_returns).cumprod() * self.initial_capital
        self.benchmark_returns['S&P 500'] = spy_returns
        self.benchmark_values['S&P 500'] = spy_values
        
        # === 2. Equal Weight Benchmark ===
        n_assets = len(self.asset_list)
        equal_weights = pd.Series(1.0 / n_assets, index=self.asset_list)
        
        eq_returns_list = []
        for day in backtest_days:
            day_returns = self._asset_returns.loc[day, self.asset_list].fillna(0)
            eq_returns_list.append((equal_weights * day_returns).sum())
        
        eq_returns = pd.Series(eq_returns_list, index=backtest_days)
        eq_values = (1 + eq_returns).cumprod() * self.initial_capital
        self.benchmark_returns['Equal Weight'] = eq_returns
        self.benchmark_values['Equal Weight'] = eq_values
        
        # === 3. MV Long-Only Benchmark ===
        self._compute_mv_benchmark(lookback_days, allow_short=False)
    
    def _compute_mv_benchmark(self, lookback_days: int, allow_short: bool):
        """Compute mean-variance benchmark."""
        from scipy.optimize import minimize
        
        backtest_days = self.returns.index
        asset_returns = self._asset_returns[self.asset_list].copy()
        
        mv_returns_list = []
        mv_weights = None
        rebal_idx = 0
        
        for i, day in enumerate(backtest_days):
            # Check if we need to rebalance
            if rebal_idx < len(self.rebalance_dates) and day > self.rebalance_dates[rebal_idx]:
                rebal_date = self.rebalance_dates[rebal_idx]
                rebal_idx += 1
                
                # Get lookback data
                lookback_end = rebal_date
                lookback_start = lookback_end - pd.Timedelta(days=int(lookback_days * 1.5))
                lookback_mask = (asset_returns.index >= lookback_start) & (asset_returns.index < lookback_end)
                returns_lookback = asset_returns.loc[lookback_mask].dropna()
                
                if len(returns_lookback) >= 20:
                    mu = returns_lookback.mean() * 252
                    Sigma = returns_lookback.cov() * 252
                    
                    n = len(mu)
                    
                    def neg_sharpe(w):
                        ret = np.dot(w, mu)
                        vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
                        return -ret / vol if vol > 1e-10 else 1e10
                    
                    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
                    bounds = [(0, 1) for _ in range(n)]
                    w0 = np.ones(n) / n
                    
                    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
                    
                    if result.success:
                        mv_weights = pd.Series(result.x, index=mu.index)
                    else:
                        mv_weights = pd.Series(1.0 / n, index=mu.index)
                else:
                    mv_weights = pd.Series(1.0 / len(self.asset_list), index=self.asset_list)
            
            if mv_weights is None:
                mv_weights = pd.Series(1.0 / len(self.asset_list), index=self.asset_list)
            
            day_returns = asset_returns.loc[day].fillna(0)
            mv_returns_list.append((mv_weights * day_returns).sum())
        
        mv_returns = pd.Series(mv_returns_list, index=backtest_days)
        mv_values = (1 + mv_returns).cumprod() * self.initial_capital
        
        name = 'MV Long-Only'
        self.benchmark_returns[name] = mv_returns
        self.benchmark_values[name] = mv_values
    
    def print_summary(self):
        """Print backtest performance summary."""
        if self.returns is None or len(self.returns) == 0:
            print("No returns calculated yet. Run backtest first.")
            return
        
        metrics = self.get_metrics(include_benchmarks=True)
        
        print(f"\n{'='*80}")
        print("BACKTEST PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Trading Days: {len(self.returns)}")
        print(f"Rebalances: {len(self.rebalance_dates)}")
        print(f"{'='*80}")
        
        # Format and print metrics
        format_pct = lambda x: f"{x*100:.2f}%"
        format_ratio = lambda x: f"{x:.3f}"
        
        print(f"\n{'Metric':<25} {'Strategy':>15} {'S&P 500':>15} {'Equal Weight':>15} {'MV Long-Only':>15}")
        print("-" * 85)
        
        for metric in ['Total Return', 'Annualized Return', 'Annualized Volatility', 
                       'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Win Rate']:
            row = f"{metric:<25}"
            for col in metrics.columns:
                val = metrics.loc[metric, col]
                if metric in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
                    row += f" {format_ratio(val):>15}"
                else:
                    row += f" {format_pct(val):>15}"
            print(row)
        
        print(f"\n{'='*80}")
        print(f"Total Transaction Costs: ${self.transaction_costs.sum():,.2f}")
        print(f"Average Turnover per Rebalance: {self.turnover_history.mean():.2%}")
        print(f"{'='*80}")
    
    # ====================================================================================
    # VISUALIZATION
    # ====================================================================================
    
    def plot_performance(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Plot cumulative returns comparing strategy to benchmarks.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(14, 6)
            Figure size
        """
        if self.portfolio_value is None:
            raise ValueError("Run backtest first")
        
        self._compute_benchmarks()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {
            'Strategy': '#2E86AB',
            'S&P 500': '#A23B72',
            'Equal Weight': '#F18F01',
            'MV Long-Only': '#C73E1D'
        }
        
        # Plot strategy
        cum_returns = self.portfolio_value / self.initial_capital
        ax.plot(cum_returns.index, cum_returns.values, label='Strategy', 
                linewidth=2.5, color=colors['Strategy'])
        
        # Plot benchmarks
        for bench_name, bench_values in self.benchmark_values.items():
            bench_cum = bench_values / self.initial_capital
            ax.plot(bench_cum.index, bench_cum.values, label=bench_name,
                    linewidth=1.5, alpha=0.8, color=colors.get(bench_name, 'gray'))
        
        ax.set_title('Cumulative Returns: Strategy vs Benchmarks', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Growth of $1')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_analysis(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Plot detailed analysis charts including drawdowns, rolling metrics,
        return distribution, and weight allocation.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(16, 12)
            Figure size
        """
        if self.portfolio_value is None:
            raise ValueError("Run backtest first")
        
        self._compute_benchmarks()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        colors = {
            'Strategy': '#2E86AB',
            'S&P 500': '#A23B72',
            'Equal Weight': '#F18F01',
            'MV Long-Only': '#C73E1D'
        }
        
        # 1. Drawdown Comparison
        ax = axes[0, 0]
        
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3,
                        color=colors['Strategy'], label='Strategy')
        
        for bench_name, bench_returns in self.benchmark_returns.items():
            bench_cum = (1 + bench_returns).cumprod()
            bench_max = bench_cum.expanding().max()
            bench_dd = (bench_cum - bench_max) / bench_max * 100
            ax.plot(bench_dd.index, bench_dd.values, linewidth=1, alpha=0.7,
                    color=colors.get(bench_name, 'gray'), label=bench_name)
        
        ax.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Ratio
        ax = axes[0, 1]
        window = min(126, len(self.returns) // 2)
        
        if window >= 20:
            rolling_mean = self.returns.rolling(window).mean() * 252
            rolling_std = self.returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_mean / rolling_std
            ax.plot(rolling_sharpe.dropna().index, rolling_sharpe.dropna().values,
                    linewidth=1.5, color=colors['Strategy'], label='Strategy')
            
            for bench_name, bench_returns in self.benchmark_returns.items():
                bench_roll_mean = bench_returns.rolling(window).mean() * 252
                bench_roll_std = bench_returns.rolling(window).std() * np.sqrt(252)
                bench_roll_sharpe = bench_roll_mean / bench_roll_std
                ax.plot(bench_roll_sharpe.dropna().index, bench_roll_sharpe.dropna().values,
                        linewidth=1, alpha=0.7, color=colors.get(bench_name, 'gray'), label=bench_name)
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        # 3. Return Distribution
        ax = axes[1, 0]
        ax.hist(self.returns * 100, bins=50, edgecolor='black', alpha=0.7,
                color=colors['Strategy'], label='Strategy')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=self.returns.mean() * 100, color='green', linestyle='--',
                   label=f'Mean: {self.returns.mean()*100:.3f}%')
        ax.set_title('Daily Return Distribution', fontsize=12, fontweight='bold')
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
                ax.set_title('Portfolio Weight Allocation (has shorts)', fontsize=12, fontweight='bold')
            else:
                self.weights_history.plot.area(ax=ax, alpha=0.7, stacked=True)
                ax.set_title('Portfolio Weight Allocation', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weights_heatmap(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot weights as a heatmap over time.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(14, 8)
            Figure size
        """
        if self.weights_history is None:
            raise ValueError("Run backtest first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Transpose for better visualization
        weights_plot = self.weights_history.T
        weights_plot.columns = [d.strftime('%Y-%m-%d') for d in weights_plot.columns]
        
        # Create heatmap
        cmap = 'RdYlGn' if self.allow_short_selling else 'YlGn'
        vmin = -0.5 if self.allow_short_selling else 0
        
        sns.heatmap(weights_plot, annot=True, fmt='.1%', cmap=cmap,
                    vmin=vmin, vmax=0.5, center=0, ax=ax,
                    cbar_kws={'label': 'Weight'})
        
        ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rebalance Date')
        ax.set_ylabel('Asset')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # ====================================================================================
    # UTILITY METHODS
    # ====================================================================================
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get trade log showing weight changes at each rebalance.
        
        Returns
        -------
        pd.DataFrame
            Trade log with columns: date, asset, old_weight, new_weight, trade_size
        """
        if self.weights_history is None:
            raise ValueError("Run backtest first")
        
        trades = []
        prev_weights = None
        
        for date in self.weights_history.index:
            curr_weights = self.weights_history.loc[date]
            
            for asset in self.asset_list:
                old_w = prev_weights[asset] if prev_weights is not None else 0
                new_w = curr_weights[asset]
                trade_size = new_w - old_w
                
                if abs(trade_size) > 1e-6:
                    trades.append({
                        'date': date,
                        'asset': asset,
                        'old_weight': old_w,
                        'new_weight': new_w,
                        'trade_size': trade_size
                    })
            
            prev_weights = curr_weights
        
        return pd.DataFrame(trades)
    
    def get_exposure_summary(self) -> pd.DataFrame:
        """
        Get long/short/net/gross exposure summary over time.
        
        Returns
        -------
        pd.DataFrame
            Exposure summary with columns: long, short, net, gross
        """
        if self.weights_history is None:
            raise ValueError("Run backtest first")
        
        exposures = []
        for date in self.weights_history.index:
            w = self.weights_history.loc[date]
            long_exp = w[w > 0].sum()
            short_exp = -w[w < 0].sum()
            net_exp = long_exp - short_exp
            gross_exp = long_exp + short_exp
            
            exposures.append({
                'date': date,
                'long': long_exp,
                'short': short_exp,
                'net': net_exp,
                'gross': gross_exp
            })
        
        return pd.DataFrame(exposures).set_index('date')
