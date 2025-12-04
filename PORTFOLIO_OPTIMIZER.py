"""
PORTFOLIO_OPTIMIZER.py

Portfolio optimization class that takes μ (expected returns) and Σ (covariance matrix)
and produces optimal weights with various constraints.

Supported Objectives:
- Maximum Sharpe Ratio
- Maximum Sortino Ratio  
- Minimum Variance
- Risk Parity
- Mean-Variance with risk aversion parameter

Constraints:
- Short selling (boolean toggle with gross exposure limit)
- Minimum/maximum weight per asset
- Maximum turnover from previous weights
- Gross exposure limit (sum of absolute weights)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from typing import Dict, List, Optional, Tuple, Union
import warnings
from ANALYTICAL_INPUTS import ANALYTICAL_INPUTS

warnings.filterwarnings('ignore')


class PORTFOLIO_OPTIMIZER:
    """
    Portfolio optimizer with multiple objectives and constraints.
    
    Takes pre-computed expected returns (μ) and covariance matrix (Σ) and
    produces optimal portfolio weights.
    
    Parameters
    ----------
    mu : pd.Series
        Expected returns for each asset, indexed by asset names
    Sigma : pd.DataFrame
        Covariance matrix of returns, index and columns are asset names
    risk_free_rate : float, default=0.0
        Annual risk-free rate for Sharpe ratio calculation
        
    Attributes
    ----------
    optimal_weights : pd.Series
        Computed optimal weights after calling optimize()
    portfolio_return : float
        Expected portfolio return
    portfolio_volatility : float
        Expected portfolio volatility (standard deviation)
    sharpe_ratio : float
        Portfolio Sharpe ratio
    """
    
    # Valid optimization objectives
    VALID_OBJECTIVES = ['max_sharpe', 'min_variance', 'risk_parity', 'mean_variance']
    
    def __init__(
        self,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        risk_free_rate: float = 0.0
    ):
        # Validate inputs
        if not isinstance(mu, pd.Series):
            raise TypeError("mu must be a pandas Series")
        if not isinstance(Sigma, pd.DataFrame):
            raise TypeError("Sigma must be a pandas DataFrame")
        
        # Ensure consistent ordering
        common_assets = mu.index.intersection(Sigma.index)
        if len(common_assets) != len(mu) or len(common_assets) != len(Sigma):
            raise ValueError("mu and Sigma must have the same assets")
        
        self.mu = mu.loc[common_assets]
        self.Sigma = Sigma.loc[common_assets, common_assets]
        self.assets = list(common_assets)
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        
        # Results (set after optimization)
        self.optimal_weights: Optional[pd.Series] = None
        self.portfolio_return: Optional[float] = None
        self.portfolio_volatility: Optional[float] = None
        self.sharpe_ratio: Optional[float] = None
        
        # Optimization metadata
        self._optimization_status: Optional[str] = None
        self._objective_used: Optional[str] = None
        self._constraints_used: Optional[Dict] = None
    
    def optimize(
        self,
        objective: str = 'max_sharpe',
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        allow_short_selling: bool = False,
        gross_exposure_limit: float = 1.0,
        max_turnover: Optional[float] = None,
        previous_weights: Optional[pd.Series] = None,
        risk_aversion: float = 1.0,
        verbose: bool = False
    ) -> pd.Series:
        """
        Compute optimal portfolio weights.
        
        Parameters
        ----------
        objective : str, default='max_sharpe'
            Optimization objective:
            - 'max_sharpe': Maximize Sharpe ratio
            - 'min_variance': Minimize portfolio variance
            - 'risk_parity': Equal risk contribution
            - 'mean_variance': Mean-variance with risk aversion
        min_weight : float, default=0.0
            Minimum weight per asset. Only used if allow_short_selling=True.
            E.g., -0.3 allows up to 30% short position per asset.
            If allow_short_selling=False, this is ignored and 0.0 is used.
        max_weight : float, default=1.0
            Maximum weight per asset. E.g., 0.4 limits any asset to 40%.
        allow_short_selling : bool, default=False
            If True, allows negative weights (short positions) bounded by min_weight.
            If False, all weights are constrained to be >= 0 (long-only).
        gross_exposure_limit : float, default=1.0
            Maximum gross exposure: sum(|weights|) <= gross_exposure_limit.
            Only enforced when allow_short_selling=True.
            E.g., 1.5 allows 150% long + 50% short = 100% net, 200% gross.
            Must be >= 1.0 when short selling is allowed.
        max_turnover : float, optional
            Maximum allowed turnover from previous_weights.
            Turnover = sum(|w_new - w_old|) / 2
            E.g., 0.3 means at most 30% of portfolio can change.
            Ignored if previous_weights is None.
        previous_weights : pd.Series, optional
            Previous portfolio weights for turnover constraint.
            Must be indexed by same asset names.
        risk_aversion : float, default=1.0
            Risk aversion parameter for 'mean_variance' objective.
            Higher values → more risk-averse portfolios.
        verbose : bool, default=False
            Print optimization details.
            
        Returns
        -------
        pd.Series
            Optimal weights indexed by asset names
            
        Examples
        --------
        >>> optimizer = PORTFOLIO_OPTIMIZER(mu, Sigma)
        >>> 
        >>> # Long-only max Sharpe
        >>> weights = optimizer.optimize(objective='max_sharpe')
        >>> 
        >>> # Allow short selling with 150% gross exposure limit
        >>> weights = optimizer.optimize(
        ...     allow_short_selling=True,
        ...     min_weight=-0.3,
        ...     max_weight=0.4,
        ...     gross_exposure_limit=1.5
        ... )
        >>> 
        >>> # With turnover constraint
        >>> new_weights = optimizer.optimize(
        ...     max_turnover=0.25,
        ...     previous_weights=old_weights
        ... )
        """
        # Validate objective
        if objective not in self.VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {self.VALID_OBJECTIVES}")
        
        # Handle short selling logic
        if not allow_short_selling:
            # Long-only: override min_weight to 0
            min_weight = 0.0
            gross_exposure_limit = 1.0  # Not applicable for long-only
        else:
            # Short selling enabled: validate gross exposure
            if gross_exposure_limit < 1.0:
                raise ValueError("gross_exposure_limit must be >= 1.0 when short selling is allowed")
        
        # Validate constraints
        if min_weight > max_weight:
            raise ValueError("min_weight must be <= max_weight")
        if max_weight > 1.0:
            raise ValueError("max_weight must be <= 1.0")
        if min_weight < -1.0:
            raise ValueError("min_weight must be >= -1.0")
        
        # Validate turnover constraint
        if max_turnover is not None:
            if max_turnover <= 0 or max_turnover > 1.0:
                raise ValueError("max_turnover must be in (0, 1]")
            if previous_weights is None:
                warnings.warn("max_turnover specified but previous_weights is None. Ignoring turnover constraint.")
                max_turnover = None
        
        # Align previous weights if provided
        if previous_weights is not None:
            # Ensure same assets, fill missing with 0
            prev_w = pd.Series(0.0, index=self.assets)
            common = previous_weights.index.intersection(self.assets)
            prev_w.loc[common] = previous_weights.loc[common]
            previous_weights = prev_w
        
        # Store constraints used
        self._constraints_used = {
            'min_weight': min_weight,
            'max_weight': max_weight,
            'allow_short_selling': allow_short_selling,
            'gross_exposure_limit': gross_exposure_limit,
            'max_turnover': max_turnover,
            'has_previous_weights': previous_weights is not None
        }
        self._objective_used = objective
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PORTFOLIO OPTIMIZATION")
            print(f"{'='*70}")
            print(f"  Objective: {objective}")
            print(f"  Assets: {self.n_assets}")
            print(f"  Short selling: {'Allowed' if allow_short_selling else 'Not allowed (long-only)'}")
            print(f"  Weight bounds: [{min_weight:.1%}, {max_weight:.1%}]")
            if allow_short_selling:
                print(f"  Gross exposure limit: {gross_exposure_limit:.1%}")
            if max_turnover is not None:
                print(f"  Max turnover: {max_turnover:.1%}")
            print(f"{'='*70}")
        
        # Dispatch to appropriate optimizer
        if objective == 'max_sharpe':
            weights = self._optimize_max_sharpe(
                min_weight, max_weight, allow_short_selling, gross_exposure_limit,
                max_turnover, previous_weights, verbose
            )
        elif objective == 'min_variance':
            weights = self._optimize_min_variance(
                min_weight, max_weight, allow_short_selling, gross_exposure_limit,
                max_turnover, previous_weights, verbose
            )
        elif objective == 'risk_parity':
            weights = self._optimize_risk_parity(
                min_weight, max_weight, allow_short_selling, gross_exposure_limit,
                max_turnover, previous_weights, verbose
            )
        else:  # mean_variance
            weights = self._optimize_mean_variance(
                min_weight, max_weight, allow_short_selling, gross_exposure_limit,
                max_turnover, previous_weights, risk_aversion, verbose
            )
        
        # Compute portfolio statistics
        self._compute_portfolio_stats()
        
        if verbose:
            self._print_results()
        
        return weights
    
    def _optimize_max_sharpe(
        self,
        min_weight: float,
        max_weight: float,
        allow_short_selling: bool,
        gross_exposure_limit: float,
        max_turnover: Optional[float],
        previous_weights: Optional[pd.Series],
        verbose: bool
    ) -> pd.Series:
        """
        Maximize Sharpe ratio using SLSQP with multiple starting points.
        Falls back to mean-variance if optimization fails.
        """
        mu_arr = self.mu.values
        Sigma_arr = self.Sigma.values
        
        def neg_sharpe(w):
            ret = mu_arr @ w - self.risk_free_rate
            vol = np.sqrt(w @ Sigma_arr @ w)
            return -ret / vol if vol > 1e-10 else 1e10
        
        # Build constraints
        constraints = []
        
        # Weights sum to 1 (net exposure = 100%)
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Gross exposure constraint (only for short selling)
        if allow_short_selling:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: gross_exposure_limit - np.sum(np.abs(w))
            })
        
        # Turnover constraint
        if max_turnover is not None and previous_weights is not None:
            prev_w = previous_weights.values
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_turnover - np.sum(np.abs(w - prev_w)) / 2
            })
        
        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        
        # Multiple starting points
        best_result = None
        best_sharpe = -np.inf
        
        np.random.seed(42)
        n_starts = 20
        
        for i in range(n_starts):
            # Generate starting point
            if i == 0:
                w0 = np.ones(self.n_assets) / self.n_assets  # Equal weight
            elif i == 1 and previous_weights is not None:
                w0 = previous_weights.values.copy()  # Previous weights
            else:
                # Random weights satisfying constraints
                w0 = np.random.dirichlet(np.ones(self.n_assets))
                w0 = np.clip(w0, max(min_weight, 0.01), max_weight)
                w0 = w0 / w0.sum()
            
            try:
                result = minimize(
                    neg_sharpe,
                    w0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-10, 'maxiter': 1000}
                )
                
                if result.success and -result.fun > best_sharpe:
                    # Verify constraints
                    w = result.x
                    if np.abs(w.sum() - 1.0) < 1e-6:
                        if max_turnover is None or previous_weights is None or \
                           np.sum(np.abs(w - previous_weights.values)) / 2 <= max_turnover + 1e-6:
                            best_result = result
                            best_sharpe = -result.fun
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            self.optimal_weights = pd.Series(best_result.x, index=self.assets)
            self._optimization_status = 'optimal'
        else:
            # Fallback to mean-variance with high risk aversion
            if verbose:
                print("  ⚠ Max Sharpe failed, falling back to mean-variance")
            return self._optimize_mean_variance(
                min_weight, max_weight, allow_short_selling, gross_exposure_limit,
                max_turnover, previous_weights, risk_aversion=1.0, verbose=False
            )
        
        return self.optimal_weights
    
    def _optimize_min_variance(
        self,
        min_weight: float,
        max_weight: float,
        allow_short_selling: bool,
        gross_exposure_limit: float,
        max_turnover: Optional[float],
        previous_weights: Optional[pd.Series],
        verbose: bool
    ) -> pd.Series:
        """Minimize portfolio variance using convex optimization."""
        w = cp.Variable(self.n_assets)
        Sigma_arr = self.Sigma.values
        
        # Regularize covariance if needed
        min_eig = np.linalg.eigvalsh(Sigma_arr).min()
        if min_eig < 1e-8:
            Sigma_arr = Sigma_arr + np.eye(self.n_assets) * (1e-8 - min_eig)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, Sigma_arr))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight
        ]
        
        # Gross exposure constraint (only for short selling)
        if allow_short_selling:
            constraints.append(cp.norm(w, 1) <= gross_exposure_limit)
        
        # Turnover constraint
        if max_turnover is not None and previous_weights is not None:
            prev_w = previous_weights.values
            constraints.append(cp.norm(w - prev_w, 1) <= 2 * max_turnover)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except:
            problem.solve(solver=cp.SCS)
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            self.optimal_weights = pd.Series(w.value, index=self.assets)
            self._optimization_status = problem.status
        else:
            raise RuntimeError(f"Min variance optimization failed: {problem.status}")
        
        return self.optimal_weights
    
    def _optimize_mean_variance(
        self,
        min_weight: float,
        max_weight: float,
        allow_short_selling: bool,
        gross_exposure_limit: float,
        max_turnover: Optional[float],
        previous_weights: Optional[pd.Series],
        risk_aversion: float,
        verbose: bool
    ) -> pd.Series:
        """Mean-variance optimization with risk aversion parameter."""
        w = cp.Variable(self.n_assets)
        mu_arr = self.mu.values
        Sigma_arr = self.Sigma.values
        
        # Regularize covariance if needed
        min_eig = np.linalg.eigvalsh(Sigma_arr).min()
        if min_eig < 1e-8:
            Sigma_arr = Sigma_arr + np.eye(self.n_assets) * (1e-8 - min_eig)
        
        # Objective: maximize return - gamma * variance
        portfolio_return = mu_arr @ w
        portfolio_variance = cp.quad_form(w, Sigma_arr)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight
        ]
        
        # Gross exposure constraint (only for short selling)
        if allow_short_selling:
            constraints.append(cp.norm(w, 1) <= gross_exposure_limit)
        
        # Turnover constraint
        if max_turnover is not None and previous_weights is not None:
            prev_w = previous_weights.values
            constraints.append(cp.norm(w - prev_w, 1) <= 2 * max_turnover)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except:
            problem.solve(solver=cp.SCS)
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            self.optimal_weights = pd.Series(w.value, index=self.assets)
            self._optimization_status = problem.status
        else:
            raise RuntimeError(f"Mean-variance optimization failed: {problem.status}")
        
        return self.optimal_weights
    
    def _optimize_risk_parity(
        self,
        min_weight: float,
        max_weight: float,
        allow_short_selling: bool,
        gross_exposure_limit: float,
        max_turnover: Optional[float],
        previous_weights: Optional[pd.Series],
        verbose: bool
    ) -> pd.Series:
        """
        Risk parity: equal risk contribution from each asset.
        Uses SLSQP to minimize sum of squared differences in risk contributions.
        Note: Risk parity typically requires positive weights for meaningful interpretation.
        """
        Sigma_arr = self.Sigma.values
        
        def risk_contribution(w):
            """Compute risk contribution of each asset."""
            port_var = w @ Sigma_arr @ w
            if port_var < 1e-10:
                return np.zeros(self.n_assets)
            marginal_risk = Sigma_arr @ w
            risk_contrib = w * marginal_risk / np.sqrt(port_var)
            return risk_contrib
        
        def objective(w):
            """Minimize squared differences from equal risk contribution."""
            rc = risk_contribution(w)
            target_rc = np.sqrt(w @ Sigma_arr @ w) / self.n_assets
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Gross exposure constraint (only for short selling)
        if allow_short_selling:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: gross_exposure_limit - np.sum(np.abs(w))
            })
        
        if max_turnover is not None and previous_weights is not None:
            prev_w = previous_weights.values
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_turnover - np.sum(np.abs(w - prev_w)) / 2
            })
        
        # Bounds (risk parity typically requires positive weights)
        lb = max(min_weight, 1e-4)  # Small positive lower bound for stability
        bounds = [(lb, max_weight) for _ in range(self.n_assets)]
        
        # Starting point: inverse volatility
        vols = np.sqrt(np.diag(Sigma_arr))
        w0 = (1 / vols) / (1 / vols).sum()
        w0 = np.clip(w0, lb, max_weight)
        w0 = w0 / w0.sum()
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )
        
        if result.success:
            self.optimal_weights = pd.Series(result.x, index=self.assets)
            self._optimization_status = 'optimal'
        else:
            # Fallback to equal weight
            if verbose:
                print("  ⚠ Risk parity failed, falling back to equal weight")
            self.optimal_weights = pd.Series(1 / self.n_assets, index=self.assets)
            self._optimization_status = 'fallback_equal_weight'
        
        return self.optimal_weights
    
    def _compute_portfolio_stats(self):
        """Compute portfolio return, volatility, and Sharpe ratio."""
        if self.optimal_weights is None:
            return
        
        w = self.optimal_weights.values
        self.portfolio_return = self.mu.values @ w
        self.portfolio_volatility = np.sqrt(w @ self.Sigma.values @ w)
        
        if self.portfolio_volatility > 1e-10:
            self.sharpe_ratio = (self.portfolio_return - self.risk_free_rate) / self.portfolio_volatility
        else:
            self.sharpe_ratio = 0.0
    
    def _print_results(self):
        """Print optimization results summary."""
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"  Status: {self._optimization_status}")
        print(f"  Expected Return: {self.portfolio_return:.4f} ({self.portfolio_return*100:.2f}%)")
        print(f"  Volatility: {self.portfolio_volatility:.4f} ({self.portfolio_volatility*100:.2f}%)")
        print(f"  Sharpe Ratio: {self.sharpe_ratio:.4f}")
        
        print(f"\n  Weights:")
        for asset in self.assets:
            w = self.optimal_weights[asset]
            bar = '█' * int(abs(w) * 50)
            sign = '+' if w >= 0 else '-'
            print(f"    {asset:40s}: {sign}{abs(w):6.2%} {bar}")
        
        # Summary stats
        long_exposure = self.optimal_weights[self.optimal_weights > 0].sum()
        short_exposure = -self.optimal_weights[self.optimal_weights < 0].sum()
        print(f"\n  Long exposure: {long_exposure:.2%}")
        print(f"  Short exposure: {short_exposure:.2%}")
        print(f"  Net exposure: {long_exposure - short_exposure:.2%}")
        print(f"{'='*70}")
    
    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of optimization results.
        
        Returns
        -------
        pd.DataFrame
            Summary with weights, expected returns, volatilities, and contributions
        """
        if self.optimal_weights is None:
            raise ValueError("Must call optimize() first")
        
        # Individual asset stats
        vols = np.sqrt(np.diag(self.Sigma.values))
        
        # Risk contribution
        w = self.optimal_weights.values
        port_vol = self.portfolio_volatility
        marginal_risk = self.Sigma.values @ w
        risk_contrib = w * marginal_risk / port_vol if port_vol > 1e-10 else np.zeros(self.n_assets)
        
        summary = pd.DataFrame({
            'Weight': self.optimal_weights,
            'Expected Return': self.mu,
            'Volatility': pd.Series(vols, index=self.assets),
            'Return Contribution': self.optimal_weights * self.mu,
            'Risk Contribution': pd.Series(risk_contrib, index=self.assets)
        })
        
        # Add totals row
        totals = pd.DataFrame({
            'Weight': [summary['Weight'].sum()],
            'Expected Return': [self.portfolio_return],
            'Volatility': [self.portfolio_volatility],
            'Return Contribution': [summary['Return Contribution'].sum()],
            'Risk Contribution': [summary['Risk Contribution'].sum()]
        }, index=['PORTFOLIO'])
        
        return pd.concat([summary, totals])
    
    def get_turnover(self, previous_weights: pd.Series) -> float:
        """
        Compute turnover from previous weights.
        
        Turnover = sum(|w_new - w_old|) / 2
        
        Parameters
        ----------
        previous_weights : pd.Series
            Previous portfolio weights
            
        Returns
        -------
        float
            Turnover (0 to 1)
        """
        if self.optimal_weights is None:
            raise ValueError("Must call optimize() first")
        
        # Align weights
        prev_w = pd.Series(0.0, index=self.assets)
        common = previous_weights.index.intersection(self.assets)
        prev_w.loc[common] = previous_weights.loc[common]
        
        return np.sum(np.abs(self.optimal_weights - prev_w)) / 2
    
    @classmethod
    def from_analytical_inputs(
        cls,
        analytical_inputs: ANALYTICAL_INPUTS,
        risk_free_rate: Optional[float] = None
    ) -> 'PORTFOLIO_OPTIMIZER':
        """
        Create optimizer from ANALYTICAL_INPUTS instance.
        
        Parameters
        ----------
        analytical_inputs : ANALYTICAL_INPUTS
            Instance with computed mu and Sigma
        risk_free_rate : float, optional
            Override risk-free rate (uses analytical_inputs.risk_free_rate if None)
            
        Returns
        -------
        PORTFOLIO_OPTIMIZER
            Optimizer instance ready for optimization
        """
        if analytical_inputs.mu is None or analytical_inputs.Sigma is None:
            raise ValueError("ANALYTICAL_INPUTS must have computed mu and Sigma")
        
        rf = risk_free_rate if risk_free_rate is not None else analytical_inputs.risk_free_rate
        
        return cls(
            mu=analytical_inputs.mu,
            Sigma=analytical_inputs.Sigma,
            risk_free_rate=rf
        )

    def plot_efficient_frontier(
        self,
        n_points: int = 50,
        show_assets: bool = True,
        show_optimal: bool = True,
        show_capital_market_line: bool = True,
        allow_short_selling: bool = False,
        gross_exposure_limit: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None
    ):
        """
        Plot the efficient frontier with optional capital market line and individual assets.
        
        Parameters
        ----------
        n_points : int, default=50
            Number of points to compute on the efficient frontier
        show_assets : bool, default=True
            Whether to show individual assets on the plot
        show_optimal : bool, default=True
            Whether to highlight the current optimal portfolio (if computed)
        show_capital_market_line : bool, default=True
            Whether to show the capital market line (tangent from risk-free rate)
        allow_short_selling : bool, default=False
            Whether to allow short selling when computing frontier
        gross_exposure_limit : float, default=1.0
            Gross exposure limit for short selling
        min_weight : float, default=0.0
            Minimum weight per asset
        max_weight : float, default=1.0
            Maximum weight per asset
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        title : str, optional
            Custom title for the plot
        """
        import matplotlib.pyplot as plt
        
        # Compute efficient frontier points
        frontier_vols, frontier_rets, frontier_weights = self._compute_efficient_frontier(
            n_points=n_points,
            allow_short_selling=allow_short_selling,
            gross_exposure_limit=gross_exposure_limit,
            min_weight=min_weight,
            max_weight=max_weight
        )
        
        # Find the max Sharpe portfolio on the frontier
        sharpe_ratios = (frontier_rets - self.risk_free_rate) / frontier_vols
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_ret = frontier_rets[max_sharpe_idx]
        max_sharpe_vol = frontier_vols[max_sharpe_idx]
        max_sharpe = sharpe_ratios[max_sharpe_idx]
        
        # Find minimum variance portfolio
        min_var_idx = np.argmin(frontier_vols)
        min_var_ret = frontier_rets[min_var_idx]
        min_var_vol = frontier_vols[min_var_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        ax.plot(frontier_vols, frontier_rets, 'b-', linewidth=2.5, label='Efficient Frontier')
        
        # Plot individual assets
        if show_assets:
            asset_vols = np.sqrt(np.diag(self.Sigma.values))
            asset_rets = self.mu.values
            ax.scatter(asset_vols, asset_rets, c='gray', s=80, alpha=0.7, 
                      edgecolors='black', linewidths=1, zorder=5)
            
            # Label assets
            for i, asset in enumerate(self.assets):
                # Truncate long names
                label = asset[:20] + '...' if len(asset) > 20 else asset
                ax.annotate(label, (asset_vols[i], asset_rets[i]), 
                           fontsize=8, alpha=0.7,
                           xytext=(5, 5), textcoords='offset points')
        
        # Plot minimum variance portfolio
        ax.scatter([min_var_vol], [min_var_ret], c='green', s=150, marker='s',
                  edgecolors='black', linewidths=2, zorder=10, label='Min Variance')
        
        # Plot max Sharpe portfolio
        ax.scatter([max_sharpe_vol], [max_sharpe_ret], c='red', s=150, marker='*',
                  edgecolors='black', linewidths=1, zorder=10, 
                  label=f'Max Sharpe (SR={max_sharpe:.2f})')
        
        # Plot capital market line
        if show_capital_market_line:
            # CML from risk-free rate tangent to efficient frontier
            cml_x = np.linspace(0, max(frontier_vols) * 1.2, 100)
            cml_y = self.risk_free_rate + max_sharpe * cml_x
            ax.plot(cml_x, cml_y, 'r--', linewidth=1.5, alpha=0.7, 
                   label='Capital Market Line')
            
            # Plot risk-free rate point
            ax.scatter([0], [self.risk_free_rate], c='gold', s=100, marker='D',
                      edgecolors='black', linewidths=1, zorder=10,
                      label=f'Risk-Free Rate ({self.risk_free_rate:.1%})')
        
        # Plot current optimal portfolio if computed
        if show_optimal and self.optimal_weights is not None:
            ax.scatter([self.portfolio_volatility], [self.portfolio_return], 
                      c='blue', s=200, marker='o', edgecolors='white', linewidths=2,
                      zorder=15, label=f'Current Optimal ({self._objective_used})')
        
        # Formatting
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Expected Return', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        # Set axis limits with some padding
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def _compute_efficient_frontier(
        self,
        n_points: int = 50,
        allow_short_selling: bool = False,
        gross_exposure_limit: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Compute points on the efficient frontier.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]
            (volatilities, returns, list of weight vectors)
        """
        from scipy.optimize import minimize
        
        n = self.n_assets
        mu = self.mu.values
        Sigma = self.Sigma.values
        
        # Handle constraints
        if not allow_short_selling:
            min_weight = 0.0
            bounds = [(0.0, max_weight) for _ in range(n)]
        else:
            bounds = [(min_weight, max_weight) for _ in range(n)]
        
        # First, find the range of achievable returns
        # Min return portfolio
        def neg_return(w):
            return -np.dot(w, mu)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        if allow_short_selling and gross_exposure_limit < 2.0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: gross_exposure_limit - np.sum(np.abs(w))
            })
        
        w0 = np.ones(n) / n
        
        # Find min return
        result_min = minimize(neg_return, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result_min.success:
            min_ret = np.dot(result_min.x, mu)
        else:
            min_ret = np.min(mu)
        
        # Find max return
        def pos_return(w):
            return np.dot(w, mu)
        
        result_max = minimize(lambda w: -pos_return(w), w0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result_max.success:
            max_ret = np.dot(result_max.x, mu)
        else:
            max_ret = np.max(mu)
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_vols = []
        frontier_rets = []
        frontier_weights = []
        
        for target in target_returns:
            # Minimize variance subject to target return
            def portfolio_variance(w):
                return np.dot(w, np.dot(Sigma, w))
            
            constraints_with_target = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}
            ]
            
            if allow_short_selling and gross_exposure_limit < 2.0:
                constraints_with_target.append({
                    'type': 'ineq',
                    'fun': lambda w: gross_exposure_limit - np.sum(np.abs(w))
                })
            
            result = minimize(portfolio_variance, w0, method='SLSQP', 
                            bounds=bounds, constraints=constraints_with_target,
                            options={'maxiter': 1000})
            
            if result.success:
                w_opt = result.x
                vol = np.sqrt(np.dot(w_opt, np.dot(Sigma, w_opt)))
                ret = np.dot(w_opt, mu)
                
                frontier_vols.append(vol)
                frontier_rets.append(ret)
                frontier_weights.append(w_opt)
        
        return np.array(frontier_vols), np.array(frontier_rets), frontier_weights
