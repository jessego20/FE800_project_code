"""
Rules-Based Regime Trading Strategy
====================================

State-Machine Strategy Using Original 4 Regimes

This strategy uses explicit trading rules based on regime transitions:
- LV_Bull (0): Long position
- LV_Bear (1): Context-dependent (short or long)
- HV_Bull (2): Uninvested (neutral)
- HV_Bear (3): Short position

Base States:
-----------
- UNINVESTED: No position (cash)
- LONG: 100% long position
- SHORT: 100% short position

Trading Rules:
-------------
1. LV Bull → LONG
   - If in HV Bull → close to uninvested
   
2. HV Bull → UNINVESTED (close any position)

3. Bear Regimes → SHORT
   
   A. LV Bear Short:
      - Cover at HV Bear or HV Bull → uninvested
      - Cover at LV Bull → go LONG
      - If back to LV Bull → repeat from rule 1
      - If cover at HV Bear → go long at next LV regime
      - If long from LV Bear and stay in LV Bear > X periods → cover to uninvested
   
   B. HV Bear Short:
      - Hold until regime exit
      - Exit to HV Bull → uninvested
      - Exit to LV Bull → repeat from rule 1
      - Exit to LV Bear → go LONG (recovery trade)
      - If long from HV Bear→LV Bear and don't reach LV Bull in X periods → uninvested

Key Parameters:
--------------
- lv_bear_patience: Max periods to hold long after entering from LV Bear
- hv_bear_patience: Max periods to hold long after entering from HV Bear
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum


class Position(Enum):
    """Trading position states."""
    UNINVESTED = 0
    LONG = 1
    SHORT = -1


class RegimeState(Enum):
    """Regime states (original 4-regime model)."""
    LV_BULL = 0
    LV_BEAR = 1
    HV_BULL = 2
    HV_BEAR = 3


@dataclass
class StrategyParameters:
    """Parameters for rules-based regime trading strategy."""
    
    # Patience parameters (how long to hold recovery longs)
    lv_bear_patience: int = 5  # Days to hold long after LV Bear short cover
    hv_bear_patience: int = 5  # Days to hold long after HV Bear exit to LV Bear
    
    # Position sizing
    position_size: float = 1.0  # Full investment (100%)
    
    # Transaction costs
    transaction_cost: float = 0.0004  # 4 basis points per trade
    
    def __repr__(self):
        return (f"StrategyParameters:\n" +
                f"  lv_bear_patience: {self.lv_bear_patience} periods\n" +
                f"  hv_bear_patience: {self.hv_bear_patience} periods\n" +
                f"  position_size: {self.position_size}\n" +
                f"  transaction_cost: {self.transaction_cost}")


class RulesBasedRegimeStrategy:
    """
    Rules-based trading strategy using state machine logic.
    Follows explicit rules for each regime transition.
    """
    
    def __init__(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        params: Optional[StrategyParameters] = None
    ):
        """
        Initialize the rules-based strategy.
        
        Parameters
        ----------
        predictions : pd.DataFrame
            Regime probability predictions with columns:
            ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']
        prices : pd.DataFrame
            Price DataFrame with columns 'Open' and 'Close'
        params : StrategyParameters, optional
            Strategy parameters. If None, uses defaults.
        """
        self.predictions = predictions
        self.prices = prices
        self.params = params or StrategyParameters()
        
        # Validate data alignment
        if not self.predictions.index.equals(self.prices.index):
            raise ValueError("Predictions and prices must have the same index")
        
        # Validate prediction columns
        required_cols = ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"Predictions must have columns: {required_cols}")
        
        # Storage for computed features
        self.signals = None
        self.returns = None
        self.positions = None
        self.performance_metrics = None
    
    def _get_regime(self, probs: pd.Series) -> RegimeState:
        """Get the regime with highest probability."""
        regime_cols = ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']
        regime_idx = probs[regime_cols].values.argmax()
        return RegimeState(regime_idx)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals using state machine rules.
        
        Returns
        -------
        pd.DataFrame
            Signals with columns:
            - regime: Current regime prediction
            - position: Current position (LONG=1, SHORT=-1, UNINVESTED=0)
            - signal_type: Description of the rule applied
            - days_in_position: Days since entering current position
            - recovery_counter: Counter for recovery long patience
        """
        signals = pd.DataFrame(index=self.predictions.index)
        
        # Get predicted regime for each day
        signals['regime'] = self.predictions.apply(
            lambda row: self._get_regime(row).value, axis=1
        )
        
        # Initialize tracking variables
        signals['position'] = 0.0
        signals['signal_type'] = ''
        signals['days_in_position'] = 0
        signals['recovery_counter'] = 0
        signals['prev_regime'] = signals['regime'].shift(1).fillna(signals['regime'].iloc[0])
        
        # State machine variables
        current_position = Position.UNINVESTED
        days_in_position = 0
        recovery_counter = 0
        recovery_mode = None  # 'LV_BEAR' or 'HV_BEAR' or None
        prev_regime = RegimeState(int(signals['regime'].iloc[0]))
        
        # Process each day using state machine logic
        for i in range(len(signals)):
            regime = RegimeState(int(signals.iloc[i]['regime']))
            
            # Default: maintain current position
            new_position = current_position
            signal_type = f"HOLD_{current_position.name}"
            
            # ================================================================
            # RULE 1: LV BULL → LONG
            # ================================================================
            if regime == RegimeState.LV_BULL:
                if current_position != Position.LONG:
                    new_position = Position.LONG
                    signal_type = "LV_BULL_LONG"
                    recovery_mode = None
                    recovery_counter = 0
                else:
                    signal_type = "LV_BULL_HOLD_LONG"
            
            # ================================================================
            # RULE 2: HV BULL → UNINVESTED
            # ================================================================
            elif regime == RegimeState.HV_BULL:
                if current_position != Position.UNINVESTED:
                    new_position = Position.UNINVESTED
                    signal_type = "HV_BULL_CLOSE_TO_UNINVESTED"
                    recovery_mode = None
                    recovery_counter = 0
                else:
                    signal_type = "HV_BULL_STAY_UNINVESTED"
            
            # ================================================================
            # RULE 3A: LV BEAR → SHORT (with recovery logic)
            # ================================================================
            elif regime == RegimeState.LV_BEAR:
                
                # If we're in recovery mode from previous LV Bear
                if recovery_mode == 'LV_BEAR' and current_position == Position.LONG:
                    recovery_counter += 1
                    
                    # Check patience
                    if recovery_counter > self.params.lv_bear_patience:
                        # Patience exhausted → uninvested
                        new_position = Position.UNINVESTED
                        signal_type = "LV_BEAR_RECOVERY_TIMEOUT_UNINVESTED"
                        recovery_mode = None
                        recovery_counter = 0
                    else:
                        signal_type = f"LV_BEAR_RECOVERY_LONG_WAIT_{recovery_counter}"
                
                # Coming from HV Bear short → go long (recovery)
                elif prev_regime == RegimeState.HV_BEAR and current_position == Position.SHORT:
                    new_position = Position.LONG
                    signal_type = "LV_BEAR_FROM_HV_BEAR_RECOVERY_LONG"
                    recovery_mode = 'LV_BEAR'
                    recovery_counter = 0
                
                # Coming from LV Bear short cover at HV Bear → go long (recovery)
                elif recovery_mode == 'FROM_HV_BEAR_COVER' and current_position == Position.UNINVESTED:
                    new_position = Position.LONG
                    signal_type = "LV_BEAR_RECOVERY_LONG_AFTER_HV_BEAR"
                    recovery_mode = 'LV_BEAR'
                    recovery_counter = 0
                
                # Normal LV Bear → short
                elif current_position != Position.SHORT and recovery_mode != 'LV_BEAR':
                    new_position = Position.SHORT
                    signal_type = "LV_BEAR_SHORT"
                    recovery_mode = None
                    recovery_counter = 0
            
            # ================================================================
            # RULE 3B: HV BEAR → SHORT (hold until exit)
            # ================================================================
            elif regime == RegimeState.HV_BEAR:
                
                # If we're in recovery mode from HV Bear exit
                if recovery_mode == 'HV_BEAR' and current_position == Position.LONG:
                    recovery_counter += 1
                    
                    # Check patience
                    if recovery_counter > self.params.hv_bear_patience:
                        # Patience exhausted → uninvested
                        new_position = Position.UNINVESTED
                        signal_type = "HV_BEAR_RECOVERY_TIMEOUT_UNINVESTED"
                        recovery_mode = None
                        recovery_counter = 0
                    else:
                        signal_type = f"HV_BEAR_RECOVERY_LONG_WAIT_{recovery_counter}"
                
                # Coming from LV Bear short → go long (recovery)
                elif prev_regime == RegimeState.LV_BEAR and current_position == Position.SHORT:
                    new_position = Position.LONG
                    signal_type = "HV_BEAR_FROM_LV_BEAR_RECOVERY_LONG"
                    recovery_mode = 'HV_BEAR'
                    recovery_counter = 0
                
                # Normal HV Bear → short
                elif current_position != Position.SHORT and recovery_mode != 'HV_BEAR':
                    new_position = Position.SHORT
                    signal_type = "HV_BEAR_SHORT"
                    recovery_mode = None
                    recovery_counter = 0
            
            # ================================================================
            # Special case: Exiting LV Bear short
            # ================================================================
            if prev_regime == RegimeState.LV_BEAR and current_position == Position.SHORT:
                if regime == RegimeState.HV_BEAR:
                    # Cover short, prepare for recovery long at next LV regime
                    new_position = Position.UNINVESTED
                    signal_type = "LV_BEAR_SHORT_COVER_AT_HV_BEAR"
                    recovery_mode = 'FROM_HV_BEAR_COVER'
                    recovery_counter = 0
                elif regime == RegimeState.HV_BULL:
                    # Cover short to uninvested
                    new_position = Position.UNINVESTED
                    signal_type = "LV_BEAR_SHORT_COVER_AT_HV_BULL"
                    recovery_mode = None
                    recovery_counter = 0
            
            # Update position tracking
            if new_position != current_position:
                days_in_position = 0
            else:
                days_in_position += 1
            
            # Store signals
            signals.iloc[i, signals.columns.get_loc('position')] = new_position.value
            signals.iloc[i, signals.columns.get_loc('signal_type')] = signal_type
            signals.iloc[i, signals.columns.get_loc('days_in_position')] = days_in_position
            signals.iloc[i, signals.columns.get_loc('recovery_counter')] = recovery_counter
            
            # Update state for next iteration
            current_position = new_position
            prev_regime = regime
        
        # Convert position to position size
        signals['position_size'] = signals['position'] * self.params.position_size
        
        self.signals = signals
        return signals
    
    def backtest(self) -> Dict:
        """
        Backtest the strategy and compute performance metrics.
        
        Returns
        -------
        Dict
            Performance metrics
        """
        if self.signals is None:
            self.generate_signals()
        
        # Compute returns
        returns = pd.DataFrame(index=self.prices.index)
        
        # Market returns (buy and hold)
        returns['market_return'] = self.prices['Close'].pct_change()
        
        # Position from signals
        position = self.signals['position_size']
        
        # Strategy returns
        returns['strategy_return'] = position * returns['market_return']
        
        # Transaction costs
        position_change = position.diff().abs()
        transaction_costs = position_change * self.params.transaction_cost
        returns['strategy_return'] -= transaction_costs
        
        # Cumulative returns
        returns['market_cumulative'] = (1 + returns['market_return']).cumprod() - 1
        returns['strategy_cumulative'] = (1 + returns['strategy_return']).cumprod() - 1
        
        self.returns = returns
        self.positions = position
        
        # Compute metrics
        metrics = self._compute_metrics(returns)
        self.performance_metrics = metrics
        
        return metrics
    
    def _compute_metrics(self, returns: pd.DataFrame) -> Dict:
        """Compute performance metrics."""
        metrics = {}
        
        # Total returns
        metrics['buy_hold_total_return'] = returns['market_cumulative'].iloc[-1]
        metrics['total_return'] = returns['strategy_cumulative'].iloc[-1]
        metrics['excess_return'] = metrics['total_return'] - metrics['buy_hold_total_return']
        
        # Annualized returns
        n_years = len(returns) / 252
        if n_years > 0:
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (1/n_years) - 1
            metrics['buy_hold_annualized_return'] = (1 + metrics['buy_hold_total_return']) ** (1/n_years) - 1
        else:
            metrics['annualized_return'] = 0
            metrics['buy_hold_annualized_return'] = 0
        
        # Volatility
        metrics['volatility'] = returns['strategy_return'].std() * np.sqrt(252)
        metrics['buy_hold_volatility'] = returns['market_return'].std() * np.sqrt(252)
        
        # Sharpe ratio
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        metrics['buy_hold_sharpe'] = metrics['buy_hold_annualized_return'] / metrics['buy_hold_volatility'] if metrics['buy_hold_volatility'] > 0 else 0
        
        # Maximum drawdown
        cum_strategy = (1 + returns['strategy_return']).cumprod()
        cum_market = (1 + returns['market_return']).cumprod()
        
        metrics['max_drawdown'] = (cum_strategy / cum_strategy.cummax() - 1).min()
        metrics['buy_hold_max_dd'] = (cum_market / cum_market.cummax() - 1).min()
        
        # Win rate
        winning_days = (returns['strategy_return'] > 0).sum()
        total_trading_days = (returns['strategy_return'] != 0).sum()
        metrics['win_rate'] = winning_days / total_trading_days if total_trading_days > 0 else 0
        
        # Trade statistics
        position_changes = self.positions.diff().abs()
        metrics['num_trades'] = (position_changes > 0.01).sum()
        
        # Position distribution
        long_days = (self.signals['position'] == 1).sum()
        short_days = (self.signals['position'] == -1).sum()
        uninvested_days = (self.signals['position'] == 0).sum()
        total_days = len(self.signals)
        
        metrics['long_pct'] = long_days / total_days if total_days > 0 else 0
        metrics['short_pct'] = short_days / total_days if total_days > 0 else 0
        metrics['uninvested_pct'] = uninvested_days / total_days if total_days > 0 else 0
        
        return metrics
    
    def plot_performance(self, title: str = 'Rules-Based Regime Strategy'):
        """Plot strategy performance."""
        if self.returns is None:
            self.backtest()
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # 1. Cumulative Returns
        ax = axes[0]
        ax.plot(self.returns.index, (1 + self.returns['strategy_cumulative']) * 100, 
                label='Strategy', linewidth=2)
        ax.plot(self.returns.index, (1 + self.returns['market_cumulative']) * 100,
                label='Buy & Hold', linewidth=2, alpha=0.7)
        ax.set_ylabel('Portfolio Value ($)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. Price with Positions
        ax = axes[1]
        ax.plot(self.prices.index, self.prices['Close'], 
                color='black', linewidth=1, alpha=0.7, label='Price')
        
        # Color-code by position
        long_mask = self.signals['position'] == 1
        short_mask = self.signals['position'] == -1
        uninvested_mask = self.signals['position'] == 0
        
        ax.fill_between(self.prices.index, 
                        self.prices['Close'].min(), self.prices['Close'].max(),
                        where=long_mask, alpha=0.2, color='green', label='Long')
        ax.fill_between(self.prices.index,
                        self.prices['Close'].min(), self.prices['Close'].max(),
                        where=short_mask, alpha=0.2, color='red', label='Short')
        ax.fill_between(self.prices.index,
                        self.prices['Close'].min(), self.prices['Close'].max(),
                        where=uninvested_mask, alpha=0.2, color='gray', label='Uninvested')
        
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.set_title('Positions Over Time', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 3. Regime Predictions
        ax = axes[2]
        regime_names = ['LV_Bull', 'LV_Bear', 'HV_Bull', 'HV_Bear']
        colors = ['green', 'orange', 'blue', 'red']
        
        for i, (name, color) in enumerate(zip(regime_names, colors)):
            mask = self.signals['regime'] == i
            ax.fill_between(self.signals.index, i, i+1,
                           where=mask, alpha=0.7, color=color, label=name)
        
        ax.set_ylabel('Regime', fontsize=10)
        ax.set_title('Regime Predictions', fontsize=11)
        ax.set_ylim(0, 4)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(regime_names)
        ax.legend(loc='best', ncol=4)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Position Allocation
        ax = axes[3]
        ax.plot(self.signals.index, self.signals['position'],
                linewidth=1.5, color='black')
        ax.fill_between(self.signals.index, 0, self.signals['position'],
                        where=self.signals['position']>0, alpha=0.3, color='green', label='Long')
        ax.fill_between(self.signals.index, 0, self.signals['position'],
                        where=self.signals['position']<0, alpha=0.3, color='red', label='Short')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel('Position', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_title('Position Allocation', fontsize=11)
        ax.set_ylim(-1.2, 1.2)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_summary(self) -> str:
        """Get a text summary of strategy performance."""
        if self.performance_metrics is None:
            self.backtest()
        
        m = self.performance_metrics
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║         RULES-BASED REGIME STRATEGY SUMMARY                  ║
╚══════════════════════════════════════════════════════════════╝

STRATEGY PERFORMANCE:
  Total Return:       {m['total_return']:>8.2%}
  Annualized Return:  {m['annualized_return']:>8.2%}
  Volatility:         {m['volatility']:>8.2%}
  Sharpe Ratio:       {m['sharpe_ratio']:>8.3f}
  Max Drawdown:       {m['max_drawdown']:>8.2%}
  
BUY & HOLD PERFORMANCE:
  Total Return:       {m['buy_hold_total_return']:>8.2%}
  Annualized Return:  {m['buy_hold_annualized_return']:>8.2%}
  Volatility:         {m['buy_hold_volatility']:>8.2%}
  Sharpe Ratio:       {m['buy_hold_sharpe']:>8.3f}
  Max Drawdown:       {m['buy_hold_max_dd']:>8.2%}

EXCESS PERFORMANCE:
  Excess Return:      {m['excess_return']:>8.2%}
  
TRADING STATISTICS:
  Number of Trades:   {m['num_trades']:>8.0f}
  Win Rate:           {m['win_rate']:>8.2%}

POSITION ALLOCATION:
  Long Days:          {m['long_pct']:>8.2%}
  Short Days:         {m['short_pct']:>8.2%}
  Uninvested Days:    {m['uninvested_pct']:>8.2%}
  
PARAMETERS:
  LV Bear Patience:   {self.params.lv_bear_patience:>8d} periods
  HV Bear Patience:   {self.params.hv_bear_patience:>8d} periods
  Position Size:      {self.params.position_size:>8.1f}

════════════════════════════════════════════════════════════════
"""
        return summary
