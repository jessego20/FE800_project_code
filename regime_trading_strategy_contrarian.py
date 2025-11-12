"""
Regime-Based Trading Strategy
==============================

Adaptive Strategy: Regime Stability-Based Contrarian/Momentum Switching

This module implements an adaptive trading strategy that:
1. Uses adapted regime labels (Bull, Other, Bear) from KMRF model
2. Computes multi-dimensional Regime Stability Metric
3. Automatically switches between Contrarian and Momentum modes
4. Generates signals based on highest probability regime

Strategy Philosophy:
-------------------
REGIME STABILITY → TRADING MODE:
  - LOW STABILITY (unstable regimes): CONTRARIAN signals
    → Rapid regime changes, balanced probabilities, high "Other"
    → Market is confused, mean-reverting
    → Bull prediction → SHORT (overbought), Bear → LONG (oversold)
  
  - HIGH STABILITY (stable regimes): MOMENTUM signals  
    → Consistent predictions, clear winner, low "Other"
    → Market has conviction, trending
    → Bull prediction → LONG, Bear → SHORT

Regime Stability Metric Components:
-----------------------------------
1. PREDICTION CONSISTENCY: How often predictions flip (Bull→Bear→Bull)
2. PROBABILITY CONCENTRATION: How concentrated vs. balanced are the probabilities
3. "OTHER" REGIME LEVEL: High "Other" = uncertainty/instability
4. "OTHER" REGIME VOLATILITY: Standard deviation of "Other" probability (choppy = unstable)

Key Insight for Choppy Markets:
- Longer lookback (21 days) captures regime consistency better
- "Other" volatility (std dev) detects choppy sideways markets
- Shorter window (5 days) for "Other" volatility is more responsive

Adaptive Signal Logic:
---------------------
1. Determine highest probability regime: Bull, Other, or Bear
2. Compute regime stability metric (0 = unstable, 1 = stable)
3. If stability < threshold: Use CONTRARIAN mapping
4. If stability ≥ threshold: Use MOMENTUM mapping
5. Other regime always means HOLD current position
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import entropy


@dataclass
class StrategyParameters:
    """Parameters for adaptive regime-based trading strategy."""
    
    # Regime stability parameters
    stability_lookback: int = 42  # Days to look back for stability calculation
    max_changes_threshold: int = 10
    stability_threshold: float = 0.7  # Below this = contrarian, above = momentum
    
    # Stability metric weights (should sum to 1.0)
    weight_consistency: float = 0.30  # Weight for prediction consistency
    weight_concentration: float = 0.30  # Weight for probability concentration
    weight_other_level: float = 0.10  # Weight for "Other" probability level
    weight_other_volatility: float = 0.30  # Weight for "Other" probability volatility
    
    # Other volatility calculation window (shorter than lookback)
    other_vol_window: int = 10  # Window for "Other" std dev (< stability_lookback)
    
    # Signal generation parameters
    min_prob_threshold: float = 0.40  # Minimum probability to generate signal
    hold_on_other: bool = True  # If "Other" is highest, hold position
    
    # Position sizing
    base_position: float = 1.0  # Full investment (100%)
    
    # Transaction costs
    transaction_cost: float = 0.0004  # 4 basis points per trade
    
    def __post_init__(self):
        """Validate parameters."""
        # Ensure weights sum to 1.0
        total_weight = (self.weight_consistency + self.weight_concentration + 
                       self.weight_other_level + self.weight_other_volatility)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Stability metric weights must sum to 1.0, got {total_weight}")
        
        # Ensure other_vol_window < stability_lookback
        if self.other_vol_window >= self.stability_lookback:
            raise ValueError(
                f"other_vol_window ({self.other_vol_window}) must be < "
                f"stability_lookback ({self.stability_lookback})"
            )
    
    def __repr__(self):
        return (f"StrategyParameters:\n" +
                f"  stability_lookback: {self.stability_lookback}\n" +
                f"  consistency_chnages_threshold: {self.max_changes_threshold}\n" +
                f"  stability_threshold: {self.stability_threshold}\n" +
                f"  min_prob_threshold: {self.min_prob_threshold}\n" +
                f"  other_vol_window: {self.other_vol_window}\n" +
                f"  Stability weights: consistency={self.weight_consistency}, " +
                f"concentration={self.weight_concentration}, " +
                f"other_level={self.weight_other_level}, " +
                f"other_volatility={self.weight_other_volatility}")


class RegimeTradingStrategy:
    """
    Adaptive trading strategy that switches between contrarian and momentum modes
    based on regime stability.
    """
    
    def __init__(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        params: Optional[StrategyParameters] = None
    ):
        """
        Initialize the trading strategy.
        
        Parameters
        ----------
        predictions : pd.DataFrame
            Regime probability predictions with columns:
            ['P(Bull)', 'P(Other)', 'P(Bear)'] OR
            ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']
            
            If using 4-regime format, they will be mapped to adapted labels:
            - Bull = LV_Bull
            - Other = LV_Bear + HV_Bull  
            - Bear = HV_Bear
            
        prices : pd.DataFrame
            Price DataFrame with columns 'Open' and 'Close'
        params : StrategyParameters, optional
            Strategy parameters. If None, uses defaults.
        """
        self.predictions = self._prepare_predictions(predictions)
        self.prices = prices
        self.params = params or StrategyParameters()
        
        # Validate data alignment
        if not self.predictions.index.equals(self.prices.index):
            raise ValueError("Predictions and prices must have the same index")
        
        # Storage for computed features
        self.features = None
        self.stability_metrics = None
        self.signals = None
        self.returns = None
        self.positions = None
        self.performance_metrics = None
    
    def _prepare_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare predictions to Bull/Other/Bear format.
        
        If predictions have 4 regimes, map to adapted labels:
        - Bull = LV_Bull (regime 0)
        - Other = LV_Bear + HV_Bull (regimes 1, 2)
        - Bear = HV_Bear (regime 3)
        """
        # Check if already in Bull/Other/Bear format
        if all(col in predictions.columns for col in ['P(Bull)', 'P(Other)', 'P(Bear)']):
            return predictions[['P(Bull)', 'P(Other)', 'P(Bear)']].copy()
        
        # Check if in 4-regime format
        regime_cols = ['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)']
        if all(col in predictions.columns for col in regime_cols):
            adapted = pd.DataFrame(index=predictions.index)
            adapted['P(Bull)'] = predictions['P(LV_Bull)']
            adapted['P(Other)'] = predictions['P(LV_Bear)'] + predictions['P(HV_Bull)']
            adapted['P(Bear)'] = predictions['P(HV_Bear)']
            return adapted
        
        raise ValueError(
            "Predictions must have either ['P(Bull)', 'P(Other)', 'P(Bear)'] or "
            "['P(LV_Bull)', 'P(LV_Bear)', 'P(HV_Bull)', 'P(HV_Bear)'] columns"
        )
    
    def compute_stability_metrics(self) -> pd.DataFrame:
        """
        Compute multi-dimensional regime stability metric.
        
        Returns a stability score from 0 (unstable) to 1 (stable) based on:
        1. Prediction consistency (how often predictions flip)
        2. Probability concentration (entropy-based measure)
        3. "Other" probability level (high = uncertain)
        4. "Other" probability volatility (high std dev = unstable)
        
        Returns
        -------
        pd.DataFrame
            Stability metrics with columns:
            - predicted_regime: 0=Bull, 1=Other, 2=Bear
            - consistency_score: [0, 1] - high = stable predictions
            - concentration_score: [0, 1] - high = clear winner
            - other_level_score: [0, 1] - high = low "Other" prob
            - other_volatility_score: [0, 1] - high = low "Other" volatility
            - stability: [0, 1] - overall stability metric
            - is_contrarian: True if stability < threshold
        """
        metrics = pd.DataFrame(index=self.predictions.index)
        
        # Get predicted regime (0=Bull, 1=Other, 2=Bear)
        regime_cols = ['P(Bull)', 'P(Other)', 'P(Bear)']
        metrics['predicted_regime'] = self.predictions[regime_cols].values.argmax(axis=1)
        
        # Store individual probabilities
        metrics['bull_prob'] = self.predictions['P(Bull)']
        metrics['other_prob'] = self.predictions['P(Other)']
        metrics['bear_prob'] = self.predictions['P(Bear)']
        
        lookback = self.params.stability_lookback
        
        # ================================================================
        # 1. PREDICTION CONSISTENCY
        # ================================================================
        # How often predictions change in the lookback window
        # High score = stable predictions (few changes)
        
        regime_changes = metrics['predicted_regime'].diff().abs().fillna(0)
        regime_changes_binary = (regime_changes > 0).astype(int)
        
        # Count changes over lookback window
        changes_count = regime_changes_binary.rolling(window=lookback, min_periods=1).sum()
        
        # Normalize: 0 changes = 1.0 (stable), many changes = 0.0 (unstable)
        # max_possible_changes = lookback - 1
        metrics['consistency_score'] = 1.0 - (changes_count / self.params.max_changes_threshold)
        
        # ================================================================
        # 2. PROBABILITY CONCENTRATION  
        # ================================================================
        # How concentrated the probabilities are (using entropy)
        # High score = one clear winner (low entropy)
        # Low score = balanced/uncertain (high entropy)
        
        def calc_concentration(row):
            probs = [row['P(Bull)'], row['P(Other)'], row['P(Bear)']]
            # Entropy of uniform distribution (max uncertainty)
            max_entropy = -np.log(1/3)
            # Actual entropy
            actual_entropy = entropy(probs, base=np.e)
            # Normalize: 0 entropy = 1.0 (concentrated), max entropy = 0.0 (balanced)
            return 1.0 - (actual_entropy / max_entropy)
        
        metrics['concentration_score'] = self.predictions.apply(calc_concentration, axis=1)
        
        # Smooth over lookback window
        metrics['concentration_score'] = metrics['concentration_score'].rolling(
            window=lookback, min_periods=1
        ).mean()
        
        # ================================================================
        # 3. "OTHER" PROBABILITY LEVEL
        # ================================================================
        # High "Other" probability indicates uncertainty/instability
        # High score = low "Other" probability (stable)
        
        metrics['other_level_score'] = 1.0 - metrics['other_prob']
        
        # ================================================================
        # 4. "OTHER" PROBABILITY VOLATILITY
        # ================================================================
        # High volatility in "Other" indicates choppy/uncertain regime
        # High score = low volatility in "Other" (stable)
        
        other_vol_window = self.params.other_vol_window
        
        # Calculate rolling standard deviation of "Other" probability
        other_std = metrics['other_prob'].rolling(window=other_vol_window, min_periods=1).std()
        
        # Normalize: std can range from 0 to ~0.5 (max possible for probabilities)
        # We'll scale by max observed std or use a reasonable max
        max_std = 0.3  # Reasonable max for probability std dev
        normalized_std = np.clip(other_std / max_std, 0, 1)
        
        # Invert: low std = high score (stable)
        metrics['other_volatility_score'] = 1.0 - normalized_std
        
        # ================================================================
        # AGGREGATE STABILITY METRIC
        # ================================================================
        # Weighted combination of all components
        
        metrics['stability'] = (
            self.params.weight_consistency * metrics['consistency_score'] +
            self.params.weight_concentration * metrics['concentration_score'] +
            self.params.weight_other_level * metrics['other_level_score'] +
            self.params.weight_other_volatility * metrics['other_volatility_score']
        )
        
        # Determine trading mode
        metrics['is_contrarian'] = metrics['stability'] < self.params.stability_threshold
        
        self.stability_metrics = metrics
        return metrics
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on regime predictions and stability.
        
        Signal Logic:
        ------------
        1. Identify highest probability regime (Bull, Other, Bear)
        2. If "Other" is highest → HOLD current position
        3. If stability < threshold (CONTRARIAN mode):
           - Bull prediction → SHORT (market overbought)
           - Bear prediction → LONG (market oversold)
        4. If stability ≥ threshold (MOMENTUM mode):
           - Bull prediction → LONG (trend following)
           - Bear prediction → SHORT (trend following)
        
        Returns
        -------
        pd.DataFrame
            Signals with columns:
            - raw_regime: 0=Bull, 1=Other, 2=Bear (before mapping)
            - trading_mode: 'CONTRARIAN' or 'MOMENTUM'
            - signal: -1 (short), 0 (hold), 1 (long)
            - signal_type: Description of signal
            - position_size: Position size [-1, 1]
        """
        if self.stability_metrics is None:
            self.compute_stability_metrics()
        
        signals = pd.DataFrame(index=self.predictions.index)
        metrics = self.stability_metrics
        
        # Get raw regime prediction
        signals['raw_regime'] = metrics['predicted_regime']
        signals['trading_mode'] = metrics['is_contrarian'].map({
            True: 'CONTRARIAN',
            False: 'MOMENTUM'
        })
        
        # Initialize signals
        signals['signal'] = 0
        signals['signal_type'] = 'HOLD'
        signals['position_size'] = 0.0
        
        # Store probabilities for reference
        signals['bull_prob'] = metrics['bull_prob']
        signals['other_prob'] = metrics['other_prob']
        signals['bear_prob'] = metrics['bear_prob']
        signals['stability'] = metrics['stability']
        
        # ================================================================
        # SIGNAL GENERATION
        # ================================================================
        
        for idx in signals.index:
            regime = signals.loc[idx, 'raw_regime']
            is_contrarian = metrics.loc[idx, 'is_contrarian']
            
            # Get probabilities
            bull_p = signals.loc[idx, 'bull_prob']
            other_p = signals.loc[idx, 'other_prob']
            bear_p = signals.loc[idx, 'bear_prob']
            
            # Check if probability meets threshold
            max_prob = max(bull_p, other_p, bear_p)
            if max_prob < self.params.min_prob_threshold:
                # Low confidence - hold
                signals.loc[idx, 'signal'] = 0
                signals.loc[idx, 'signal_type'] = 'LOW_CONFIDENCE_HOLD'
                continue
            
            # OTHER regime always means HOLD
            if regime == 1:  # Other
                signals.loc[idx, 'signal'] = 0
                signals.loc[idx, 'signal_type'] = 'OTHER_HOLD'
                continue
            
            # BULL regime
            if regime == 0:
                if is_contrarian:
                    # Contrarian: Bull = overbought → SHORT
                    signals.loc[idx, 'signal'] = -1
                    signals.loc[idx, 'signal_type'] = 'CONTRARIAN_BULL_SHORT'
                else:
                    # Momentum: Bull = uptrend → LONG
                    signals.loc[idx, 'signal'] = 1
                    signals.loc[idx, 'signal_type'] = 'MOMENTUM_BULL_LONG'
            
            # BEAR regime
            elif regime == 2:
                if is_contrarian:
                    # Contrarian: Bear = oversold → LONG
                    signals.loc[idx, 'signal'] = 1
                    signals.loc[idx, 'signal_type'] = 'CONTRARIAN_BEAR_LONG'
                else:
                    # Momentum: Bear = downtrend → SHORT
                    signals.loc[idx, 'signal'] = -1
                    signals.loc[idx, 'signal_type'] = 'MOMENTUM_BEAR_SHORT'
        
        # ================================================================
        # POSITION SIZING
        # ================================================================
        
        # For now, use full position when signal is active
        signals['position_size'] = signals['signal'] * self.params.base_position
        
        # Handle HOLD signals - maintain previous position
        prev_position = 0.0
        for i in range(len(signals)):
            if signals.iloc[i]['signal'] == 0:  # HOLD
                signals.iloc[i, signals.columns.get_loc('position_size')] = prev_position
            else:
                prev_position = signals.iloc[i]['position_size']
        
        self.signals = signals
        return signals
    
    def backtest(self) -> Dict:
        """
        Backtest the strategy and compute performance metrics.
        
        Returns
        -------
        Dict
            Performance metrics including:
            - total_return, annualized_return, sharpe_ratio
            - max_drawdown, win_rate, num_trades
            - buy_hold metrics for comparison
        """
        if self.signals is None:
            self.generate_signals()
        
        # Compute returns
        returns = pd.DataFrame(index=self.prices.index)
        
        # Market returns (buy and hold) - close to close
        returns['market_return'] = self.prices['Close'].pct_change()
        
        # Position from signals
        position = self.signals['position_size']
        
        # Strategy returns: position * market return
        returns['strategy_return'] = position * returns['market_return']
        
        # Transaction costs when position changes
        position_change = position.diff().abs()
        transaction_costs = position_change * self.params.transaction_cost
        returns['strategy_return'] -= transaction_costs
        
        # Cumulative returns
        returns['market_cumulative'] = (1 + returns['market_return']).cumprod() - 1
        returns['strategy_cumulative'] = (1 + returns['strategy_return']).cumprod() - 1
        
        self.returns = returns
        self.positions = position
        
        # Compute performance metrics
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
        
        # Annualized returns (assuming 252 trading days)
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
        
        # Sharpe ratio (assuming 0% risk-free rate)
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
        
        # Average trade return
        trade_returns = returns.loc[self.positions.shift(1).abs() > 0, 'strategy_return']
        metrics['avg_trade'] = trade_returns.mean() if len(trade_returns) > 0 else 0
        
        # Mode statistics
        if self.signals is not None:
            contrarian_days = (self.signals['trading_mode'] == 'CONTRARIAN').sum()
            momentum_days = (self.signals['trading_mode'] == 'MOMENTUM').sum()
            total_days = len(self.signals)
            metrics['contrarian_pct'] = contrarian_days / total_days if total_days > 0 else 0
            metrics['momentum_pct'] = momentum_days / total_days if total_days > 0 else 0
        
        return metrics
    
    def plot_performance(self, title: str = 'Adaptive Regime Strategy Performance'):
        """
        Plot strategy performance with regime stability and trading modes.
        
        Parameters
        ----------
        title : str
            Plot title
        """
        if self.returns is None:
            self.backtest()
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # ================================================================
        # 1. Cumulative Returns
        # ================================================================
        ax = axes[0]
        ax.plot(self.returns.index, (1 + self.returns['strategy_cumulative']) * 100, 
                label='Strategy', linewidth=2)
        ax.plot(self.returns.index, (1 + self.returns['market_cumulative']) * 100,
                label='Buy & Hold', linewidth=2, alpha=0.7)
        ax.set_ylabel('Portfolio Value ($)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 2. Price with Trading Modes
        # ================================================================
        ax = axes[1]
        ax.plot(self.prices.index, self.prices['Close'], 
                label='Price', color='black', linewidth=1, alpha=0.7)
        
        # Highlight contrarian vs momentum periods
        contrarian_mask = self.signals['trading_mode'] == 'CONTRARIAN'
        momentum_mask = self.signals['trading_mode'] == 'MOMENTUM'
        
        ax.fill_between(self.prices.index, 
                        self.prices['Close'].min(), self.prices['Close'].max(),
                        where=contrarian_mask, alpha=0.2, color='orange', 
                        label='Contrarian Mode')
        ax.fill_between(self.prices.index,
                        self.prices['Close'].min(), self.prices['Close'].max(),
                        where=momentum_mask, alpha=0.2, color='blue',
                        label='Momentum Mode')
        
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.set_title('Trading Modes', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 3. Regime Stability Metric
        # ================================================================
        ax = axes[2]
        ax.plot(self.stability_metrics.index, self.stability_metrics['stability'],
                label='Stability Score', color='green', linewidth=2)
        ax.axhline(y=self.params.stability_threshold, color='red', 
                  linestyle='--', label=f'Threshold ({self.params.stability_threshold})')
        ax.fill_between(self.stability_metrics.index, 0, self.params.stability_threshold,
                       alpha=0.1, color='orange', label='Contrarian Zone')
        ax.fill_between(self.stability_metrics.index, self.params.stability_threshold, 1,
                       alpha=0.1, color='blue', label='Momentum Zone')
        ax.set_ylabel('Stability Score', fontsize=10)
        ax.set_title('Regime Stability Metric', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 4. Regime Probabilities
        # ================================================================
        ax = axes[3]
        ax.plot(self.predictions.index, self.predictions['P(Bull)'], 
               label='P(Bull)', linewidth=1.5, alpha=0.8)
        ax.plot(self.predictions.index, self.predictions['P(Other)'],
               label='P(Other)', linewidth=1.5, alpha=0.8)
        ax.plot(self.predictions.index, self.predictions['P(Bear)'],
               label='P(Bear)', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_title('Regime Probabilities', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_stability_components(self, title: str = 'Regime Stability Components'):
        """
        Plot each component of the regime stability metric over time.
        
        This visualization shows how each of the 4 components contributes
        to the overall stability score, helping to diagnose which factors
        are driving the contrarian vs momentum mode switching.
        
        Parameters
        ----------
        title : str
            Plot title
        """
        if self.stability_metrics is None:
            self.compute_stability_metrics()
        
        fig, axes = plt.subplots(6, 1, figsize=(16, 14))
        
        metrics = self.stability_metrics
        
        # ================================================================
        # 1. Overall Stability Score
        # ================================================================
        ax = axes[0]
        ax.plot(metrics.index, metrics['stability'],
                label='Overall Stability', color='green', linewidth=2)
        ax.axhline(y=self.params.stability_threshold, color='red', 
                  linestyle='--', linewidth=1.5, label=f'Threshold ({self.params.stability_threshold})')
        ax.fill_between(metrics.index, 0, self.params.stability_threshold,
                       alpha=0.1, color='orange', label='Contrarian Zone')
        ax.fill_between(metrics.index, self.params.stability_threshold, 1,
                       alpha=0.1, color='blue', label='Momentum Zone')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 2. Consistency Score (Regime Flips)
        # ================================================================
        ax = axes[1]
        ax.plot(metrics.index, metrics['consistency_score'],
                label=f'Consistency Score (weight={self.params.weight_consistency:.0%})', 
                color='purple', linewidth=1.5)
        
        # Show regime changes as vertical lines
        regime_changes = metrics['predicted_regime'].diff().abs()
        change_dates = metrics.index[regime_changes > 0]
        for date in change_dates:
            ax.axvline(x=date, color='red', alpha=0.2, linewidth=0.5)
        
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Component 1: Prediction Consistency (Few Regime Flips = High)', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 3. Concentration Score (Entropy)
        # ================================================================
        ax = axes[2]
        ax.plot(metrics.index, metrics['concentration_score'],
                label=f'Concentration Score (weight={self.params.weight_concentration:.0%})', 
                color='navy', linewidth=1.5)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Component 2: Probability Concentration (Clear Winner = High)', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 4. Other Level Score
        # ================================================================
        ax = axes[3]
        ax.plot(metrics.index, metrics['other_level_score'],
                label=f'Other Level Score (weight={self.params.weight_other_level:.0%})', 
                color='brown', linewidth=1.5)
        # Show actual Other probability on secondary axis
        ax2 = ax.twinx()
        ax2.plot(metrics.index, metrics['other_prob'],
                label='P(Other)', color='gray', linewidth=1, alpha=0.5, linestyle='--')
        ax2.set_ylabel('P(Other)', fontsize=9, color='gray')
        ax2.set_ylim(0, 1)
        
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Component 3: Other Probability Level (Low Other = High)', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 5. Other Volatility Score
        # ================================================================
        ax = axes[4]
        ax.plot(metrics.index, metrics['other_volatility_score'],
                label=f'Other Volatility Score (weight={self.params.weight_other_volatility:.0%})', 
                color='darkorange', linewidth=1.5)
        
        # Show actual Other std dev on secondary axis
        ax2 = ax.twinx()
        other_std = metrics['other_prob'].rolling(window=self.params.other_vol_window, min_periods=1).std()
        ax2.plot(metrics.index, other_std,
                label=f'Other StdDev ({self.params.other_vol_window}d)', 
                color='red', linewidth=1, alpha=0.5, linestyle='--')
        ax2.set_ylabel('Std Dev', fontsize=9, color='red')
        
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Component 4: Other Probability Volatility (Low StdDev = High)', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # ================================================================
        # 6. Weighted Contributions
        # ================================================================
        ax = axes[5]
        
        # Calculate weighted contributions
        weighted_consistency = metrics['consistency_score'] * self.params.weight_consistency
        weighted_concentration = metrics['concentration_score'] * self.params.weight_concentration
        weighted_other_level = metrics['other_level_score'] * self.params.weight_other_level
        weighted_other_vol = metrics['other_volatility_score'] * self.params.weight_other_volatility
        
        # Stacked area plot
        ax.fill_between(metrics.index, 0, weighted_consistency,
                       label=f'Consistency ({self.params.weight_consistency:.0%})', 
                       alpha=0.7, color='purple')
        ax.fill_between(metrics.index, weighted_consistency, 
                       weighted_consistency + weighted_concentration,
                       label=f'Concentration ({self.params.weight_concentration:.0%})', 
                       alpha=0.7, color='navy')
        ax.fill_between(metrics.index, 
                       weighted_consistency + weighted_concentration,
                       weighted_consistency + weighted_concentration + weighted_other_level,
                       label=f'Other Level ({self.params.weight_other_level:.0%})', 
                       alpha=0.7, color='brown')
        ax.fill_between(metrics.index,
                       weighted_consistency + weighted_concentration + weighted_other_level,
                       metrics['stability'],
                       label=f'Other Volatility ({self.params.weight_other_volatility:.0%})', 
                       alpha=0.7, color='darkorange')
        
        # Add overall stability line
        ax.plot(metrics.index, metrics['stability'],
                color='green', linewidth=2, label='Total Stability', alpha=0.8)
        
        ax.axhline(y=self.params.stability_threshold, color='red', 
                  linestyle='--', linewidth=1.5, label=f'Threshold ({self.params.stability_threshold})')
        
        ax.set_ylabel('Weighted Score', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_title('Weighted Component Contributions (Stacked)', fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=8, ncol=2)
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
║         ADAPTIVE REGIME TRADING STRATEGY SUMMARY             ║
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
  Avg Trade Return:   {m['avg_trade']:>8.4%}

REGIME STABILITY:
  Contrarian Days:    {m.get('contrarian_pct', 0):>8.2%}
  Momentum Days:      {m.get('momentum_pct', 0):>8.2%}
  
PARAMETERS:
  Stability Lookback: {self.params.stability_lookback:>8d} days
  Stability Threshold:{self.params.stability_threshold:>8.2f}
  Min Prob Threshold: {self.params.min_prob_threshold:>8.2f}

════════════════════════════════════════════════════════════════
"""
        return summary
