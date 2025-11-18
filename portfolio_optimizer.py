"""
Portfolio Optimization using CVXPortfolio

This module provides a class for computing optimal portfolio weights using 
mean-variance optimization with various constraints via the CVXPortfolio package.
"""

import pandas as pd
import numpy as np
import cvxportfolio as cvx
from typing import Optional, Dict, List, Union
import matplotlib.pyplot as plt
import seaborn as sns


class PortfolioOptimizer:
    """
    Portfolio optimizer with multiple objective functions.
    
    Parameters
    ----------
    mu : pd.Series
        Expected returns for each asset (indexed by asset names)
    Sigma : pd.DataFrame
        Covariance matrix of returns (index and columns are asset names)
    allow_short : bool, default=False
        Whether to allow short selling
    gross_exposure : float, optional
        Maximum gross exposure (sum of absolute weights) if shorting is allowed
        If None and allow_short=True, no gross exposure limit is applied
    objective : str, default='max_sharpe'
        Optimization objective:
        - 'max_sharpe': Maximize Sharpe ratio (uses non-convex solver)
        - 'max_sortino': Maximize Sortino ratio (uses non-convex solver)
        - 'risk_aversion': Mean-variance with risk aversion parameter
    risk_aversion : float, optional
        Risk aversion parameter (gamma) for 'risk_aversion' objective
        Required if objective='risk_aversion'
        Higher values → more risk-averse → lower risk, lower return
    simulated_returns : pd.DataFrame, optional
        Simulated return paths for Sortino ratio calculation
        Required if objective='max_sortino'
        Shape: (n_days, n_simulations) for each asset
    """
    
    def __init__(
        self,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        assets: Optional[List[str]] = None,
        raw_ohlc: Optional[Dict[str, pd.DataFrame]] = None,
        opt_date: Optional[pd.Timestamp] = None,
        allow_short: bool = False,
        gross_exposure: Optional[float] = None,
        objective: str = 'max_sharpe',
        risk_aversion: Optional[float] = 0.5,
        simulated_returns: Optional[Dict[str, pd.DataFrame]] = None
    ):
        # Validate inputs
        if not isinstance(mu, pd.Series):
            raise TypeError("mu must be a pandas Series")
        if not isinstance(Sigma, pd.DataFrame):
            raise TypeError("Sigma must be a pandas DataFrame")
        if not all(mu.index == Sigma.index):
            raise ValueError("mu and Sigma must have the same asset names")
        if not all(Sigma.index == Sigma.columns):
            raise ValueError("Sigma must be symmetric with matching index/columns")
        
        # Validate objective
        valid_objectives = ['max_sharpe', 'max_sortino', 'risk_aversion']
        if objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
        
        if objective == 'risk_aversion' and risk_aversion is None:
            raise ValueError("risk_aversion parameter required when objective='risk_aversion'")
        
        if objective == 'max_sortino' and simulated_returns is None:
            raise ValueError("simulated_returns required when objective='max_sortino'")
        
        self.mu = mu
        self.Sigma = Sigma
        self.assets = assets if assets is not None else mu.index.tolist()
        self.n_assets = len(self.assets)
        self.raw_ohlc = raw_ohlc
        self.opt_date = opt_date
        
        # Optimization parameters
        self.objective = objective
        self.risk_aversion = risk_aversion
        self.simulated_returns = simulated_returns
        
        # Constraints
        self.allow_short = allow_short
        self.gross_exposure = gross_exposure
        
        # Validate constraint combinations
        if gross_exposure is not None and not allow_short:
            raise ValueError("gross_exposure only applies when allow_short=True")
        if gross_exposure is not None and gross_exposure <= 1.0:
            raise ValueError("gross_exposure must be > 1.0 (e.g., 1.3 for 130/30)")
        
        # Results storage (regime-based)
        self.optimal_weights = None
        self.portfolio_return = None
        self.portfolio_risk = None
        self.sharpe_ratio = None
        
    def optimize(self) -> pd.Series:
        """
        Compute optimal portfolio weights based on specified objective.
        
        Returns
        -------
        pd.Series
            Optimal weights indexed by asset names
        """
        if self.objective == 'max_sharpe':
            return self._optimize_max_sharpe()
        elif self.objective == 'max_sortino':
            return self._optimize_max_sortino()
        else:  # risk_aversion
            return self._optimize_risk_aversion()
    
    def _optimize_risk_aversion(self) -> pd.Series:
        """
        Mean-variance optimization with risk aversion parameter.
        Uses convex optimization (fast and reliable).
        """
        import cvxpy as cp
        
        w = cp.Variable(self.n_assets)
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values
        
        # Objective: maximize return - gamma * variance
        portfolio_return = mu_array @ w
        portfolio_variance = cp.quad_form(w, Sigma_array)
        objective = cp.Minimize(self.risk_aversion * portfolio_variance - portfolio_return)
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        if not self.allow_short:
            constraints.append(w >= 0)
        if self.allow_short and self.gross_exposure is not None:
            constraints.append(cp.norm(w, 1) <= self.gross_exposure)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        # Store results
        self.optimal_weights = pd.Series(w.value, index=self.assets)
        self._compute_portfolio_stats()
        
        return self.optimal_weights
    
    def _optimize_max_sharpe(self) -> pd.Series:
        """
        Maximize Sharpe ratio using non-convex optimization.
        Uses scipy's SLSQP solver with multiple random starting points.
        """
        from scipy.optimize import minimize, Bounds, LinearConstraint
        
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values
        
        def negative_sharpe(w):
            """Negative Sharpe ratio (to minimize)."""
            ret = mu_array @ w
            risk = np.sqrt(w @ Sigma_array @ w)
            return -ret / risk if risk > 1e-8 else 1e10
        
        def sharpe_gradient(w):
            """Gradient of negative Sharpe ratio."""
            ret = mu_array @ w
            var = w @ Sigma_array @ w
            risk = np.sqrt(var)
            
            if risk < 1e-8:
                return np.zeros(self.n_assets)
            
            # d(-ret/risk)/dw = -(mu/risk - ret * Sigma*w / risk^3)
            return -(mu_array / risk - ret * (Sigma_array @ w) / (risk ** 3))
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append(LinearConstraint(
            np.ones(self.n_assets), 
            lb=1.0, 
            ub=1.0
        ))
        
        # Bounds on weights
        if not self.allow_short:
            bounds = Bounds(lb=0, ub=1)
        else:
            if self.gross_exposure is not None:
                # For gross exposure, we need a nonlinear constraint
                # We'll handle this separately
                bounds = Bounds(lb=-2, ub=2)  # Reasonable bounds
            else:
                bounds = Bounds(lb=-1, ub=1)  # No shorting limit
        
        # Add gross exposure as nonlinear constraint if needed
        if self.allow_short and self.gross_exposure is not None:
            from scipy.optimize import NonlinearConstraint
            constraints.append(NonlinearConstraint(
                lambda w: np.sum(np.abs(w)),
                lb=0,
                ub=self.gross_exposure
            ))
        
        # Try multiple random starting points
        best_result = None
        best_sharpe = -np.inf
        
        np.random.seed(42)
        n_tries = 10
        
        for i in range(n_tries):
            # Generate random starting point
            if i == 0:
                # First try: equal weight
                w0 = np.ones(self.n_assets) / self.n_assets
            else:
                # Random weights
                w0 = np.random.randn(self.n_assets)
                w0 = w0 / np.sum(w0)  # Normalize to sum to 1
                
                if not self.allow_short:
                    w0 = np.abs(w0) / np.sum(np.abs(w0))
            
            # Optimize
            result = minimize(
                negative_sharpe,
                w0,
                method='SLSQP',
                jac=sharpe_gradient,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                sharpe = -result.fun
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
        
        if best_result is None or not best_result.success:
            raise RuntimeError("Sharpe ratio optimization failed to converge")
        
        # Store results
        self.optimal_weights = pd.Series(best_result.x, index=self.assets)
        self._compute_portfolio_stats()
        
        return self.optimal_weights
    
    def _optimize_max_sortino(self) -> pd.Series:
        """
        Maximize Sortino ratio using non-convex optimization.
        Sortino ratio uses downside deviation instead of total volatility.
        """
        from scipy.optimize import minimize, Bounds, LinearConstraint
        
        mu_array = self.mu.values
        
        # Compute downside covariance from simulated returns
        def compute_downside_deviation(w):
            """Compute downside deviation for portfolio with weights w."""
            # Get portfolio returns for each simulation
            portfolio_returns = np.zeros(list(self.simulated_returns.values())[0].shape)
            
            for i, asset in enumerate(self.assets):
                portfolio_returns += w[i] * self.simulated_returns[asset].values
            
            # Compute downside deviation (only negative returns)
            downside_returns = np.minimum(portfolio_returns, 0)
            downside_dev = np.sqrt(np.mean(downside_returns ** 2))
            
            return downside_dev
        
        def negative_sortino(w):
            """Negative Sortino ratio (to minimize)."""
            ret = mu_array @ w
            downside_dev = compute_downside_deviation(w)
            return -ret / downside_dev if downside_dev > 1e-8 else 1e10
        
        # Constraints
        constraints = []
        constraints.append(LinearConstraint(
            np.ones(self.n_assets), 
            lb=1.0, 
            ub=1.0
        ))
        
        # Bounds
        if not self.allow_short:
            bounds = Bounds(lb=0, ub=1)
        else:
            bounds = Bounds(lb=-2, ub=2)
        
        if self.allow_short and self.gross_exposure is not None:
            from scipy.optimize import NonlinearConstraint
            constraints.append(NonlinearConstraint(
                lambda w: np.sum(np.abs(w)),
                lb=0,
                ub=self.gross_exposure
            ))
        
        # Try multiple starting points
        best_result = None
        best_sortino = -np.inf
        
        np.random.seed(42)
        n_tries = 10
        
        for i in range(n_tries):
            if i == 0:
                w0 = np.ones(self.n_assets) / self.n_assets
            else:
                w0 = np.random.randn(self.n_assets)
                w0 = w0 / np.sum(w0)
                if not self.allow_short:
                    w0 = np.abs(w0) / np.sum(np.abs(w0))
            
            result = minimize(
                negative_sortino,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                sortino = -result.fun
                if sortino > best_sortino:
                    best_sortino = sortino
                    best_result = result
        
        if best_result is None or not best_result.success:
            raise RuntimeError("Sortino ratio optimization failed to converge")
        
        # Store results
        self.optimal_weights = pd.Series(best_result.x, index=self.assets)
        self._compute_portfolio_stats()
        
        return self.optimal_weights
    
    def _compute_portfolio_stats(self):
        """Compute portfolio statistics after optimization."""
        self.portfolio_return = float(self.mu @ self.optimal_weights)
        self.portfolio_risk = float(np.sqrt(self.optimal_weights @ self.Sigma @ self.optimal_weights))
        self.sharpe_ratio = self.portfolio_return / self.portfolio_risk if self.portfolio_risk > 0 else 0
        
        # Compute Sortino ratio if simulated returns available
        if self.simulated_returns is not None:
            portfolio_returns = np.zeros(list(self.simulated_returns.values())[0].shape)
            for i, asset in enumerate(self.assets):
                portfolio_returns += self.optimal_weights.iloc[i] * self.simulated_returns[asset].values
            
            downside_returns = np.minimum(portfolio_returns, 0)
            downside_dev = np.sqrt(np.mean(downside_returns ** 2))
            self.sortino_ratio = self.portfolio_return / downside_dev if downside_dev > 1e-8 else 0
        else:
            self.sortino_ratio = None
    
    def summary(self) -> pd.DataFrame:
        """
        Get summary statistics of the optimal portfolio.
        
        Returns
        -------
        pd.DataFrame
            Summary with weights, returns contribution, and risk contribution
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimize() first")
        
        # Compute marginal risk contribution
        portfolio_variance = self.optimal_weights @ self.Sigma @ self.optimal_weights
        marginal_risk = (self.Sigma @ self.optimal_weights) / np.sqrt(portfolio_variance)
        risk_contribution = self.optimal_weights * marginal_risk
        
        summary = pd.DataFrame({
            'Weight': self.optimal_weights,
            'Expected Return': self.mu,
            'Return Contribution': self.optimal_weights * self.mu,
            'Marginal Risk': marginal_risk,
            'Risk Contribution': risk_contribution,
            'Risk Contribution %': risk_contribution / risk_contribution.sum() * 100
        })
        
        # Sort by absolute weight
        summary = summary.reindex(summary['Weight'].abs().sort_values(ascending=False).index)
        
        return summary
    
    def portfolio_statistics(self) -> Dict[str, float]:
        """
        Get portfolio-level statistics.
        
        Returns
        -------
        dict
            Portfolio return, risk, Sharpe ratio, and other metrics
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimize() first")
        
        stats = {
            'Portfolio Return': self.portfolio_return,
            'Portfolio Risk': self.portfolio_risk,
            'Sharpe Ratio': self.sharpe_ratio,
            'Number of Assets': self.n_assets,
            'Number of Positions': (self.optimal_weights.abs() > 1e-4).sum(),
            'Long Positions': (self.optimal_weights > 1e-4).sum(),
            'Short Positions': (self.optimal_weights < -1e-4).sum(),
            'Gross Exposure': self.optimal_weights.abs().sum(),
            'Net Exposure': self.optimal_weights.sum(),
            'Max Long Position': self.optimal_weights.max(),
            'Max Short Position': self.optimal_weights.min(),
            'Effective N': 1 / (self.optimal_weights ** 2).sum()  # Diversification ratio
        }
        
        # Add Sortino ratio if available
        if self.sortino_ratio is not None:
            stats['Sortino Ratio'] = self.sortino_ratio
        
        return stats
    
    def plot_weights(self, figsize: tuple = (12, 6), top_n: Optional[int] = None):
        """
        Plot optimal portfolio weights.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 6)
            Figure size
        top_n : int, optional
            Show only top N positions by absolute weight
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimize() first")
        
        weights = self.optimal_weights.copy()
        
        # Filter to top N if requested
        if top_n is not None:
            top_assets = weights.abs().nlargest(top_n).index
            weights = weights[top_assets]
        
        # Sort by weight value
        weights = weights.sort_values()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red' if w < 0 else 'green' for w in weights]
        weights.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        
        ax.set_xlabel('Weight', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        ax.set_title('Optimal Portfolio Weights', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(axis='x', alpha=0.3)
        
        # Add weight labels
        for i, (asset, weight) in enumerate(weights.items()):
            ax.text(weight, i, f' {weight:.3f}', 
                   va='center', ha='left' if weight > 0 else 'right',
                   fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_risk_contribution(self, figsize: tuple = (12, 6), top_n: Optional[int] = None):
        """
        Plot risk contribution by asset.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 6)
            Figure size
        top_n : int, optional
            Show only top N contributors
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimize() first")
        
        summary = self.summary()
        risk_contrib = summary['Risk Contribution %']
        
        # Filter to top N if requested
        if top_n is not None:
            risk_contrib = risk_contrib.nlargest(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        risk_contrib.sort_values().plot(kind='barh', ax=ax, color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Risk Contribution (%)', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        ax.set_title('Portfolio Risk Contribution by Asset', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (asset, contrib) in enumerate(risk_contrib.sort_values().items()):
            ax.text(contrib, i, f' {contrib:.1f}%', 
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def efficient_frontier(
        self, 
        n_points: int = 50,
        return_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.
        
        Parameters
        ----------
        n_points : int, default=50
            Number of points on the frontier
        return_range : tuple, optional
            (min_return, max_return) to explore
            If None, uses range from min-variance to max-return portfolios
            
        Returns
        -------
        pd.DataFrame
            Frontier points with returns, risks, and weights
        """
        import cvxpy as cp
        
        # Convert to numpy
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values
        
        # Find return range if not specified
        if return_range is None:
            # Minimum variance portfolio return
            w_minvar = cp.Variable(self.n_assets)
            obj_minvar = cp.Minimize(cp.quad_form(w_minvar, Sigma_array))
            constraints_minvar = [cp.sum(w_minvar) == 1]
            if not self.allow_short:
                constraints_minvar.append(w_minvar >= 0)
            if self.allow_short and self.gross_exposure is not None:
                constraints_minvar.append(cp.norm(w_minvar, 1) <= self.gross_exposure)
            cp.Problem(obj_minvar, constraints_minvar).solve()
            min_return = float(mu_array @ w_minvar.value)
            
            # Maximum return portfolio
            max_return = float(self.mu.max() if not self.allow_short else 
                             self.mu.max() * self.gross_exposure if self.gross_exposure else 
                             self.mu.max())
            
            return_range = (min_return, max_return)
        
        # Generate target returns
        target_returns = np.linspace(return_range[0], return_range[1], n_points)
        
        # Compute frontier
        frontier_risks = []
        frontier_weights = []
        
        for target_ret in target_returns:
            w = cp.Variable(self.n_assets)
            
            # Minimize variance subject to target return
            obj = cp.Minimize(cp.quad_form(w, Sigma_array))
            constraints = [
                cp.sum(w) == 1,
                mu_array @ w >= target_ret
            ]
            
            if not self.allow_short:
                constraints.append(w >= 0)
            if self.allow_short and self.gross_exposure is not None:
                constraints.append(cp.norm(w, 1) <= self.gross_exposure)
            
            prob = cp.Problem(obj, constraints)
            prob.solve()
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                risk = np.sqrt(prob.value)
                frontier_risks.append(risk)
                frontier_weights.append(w.value)
            else:
                # If infeasible, stop
                break
        
        # Create DataFrame
        frontier_df = pd.DataFrame({
            'Return': target_returns[:len(frontier_risks)],
            'Risk': frontier_risks,
            'Sharpe': np.array(target_returns[:len(frontier_risks)]) / np.array(frontier_risks)
        })
        
        # Add weights
        for i, asset in enumerate(self.assets):
            frontier_df[f'Weight_{asset}'] = [w[i] for w in frontier_weights]
        
        return frontier_df
    
    def plot_efficient_frontier(
        self, 
        n_points: int = 50,
        figsize: tuple = (10, 7),
        show_assets: bool = False,
        show_optimal: bool = True
    ):
        """
        Plot the efficient frontier.
        
        Parameters
        ----------
        n_points : int, default=50
            Number of points on the frontier
        figsize : tuple, default=(10, 7)
            Figure size
        show_assets : bool, default=True
            Whether to show individual assets
        show_optimal : bool, default=True
            Whether to highlight the optimal portfolio
        """
        frontier = self.efficient_frontier(n_points=n_points)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot frontier
        ax.plot(frontier['Risk'], frontier['Return'], 
               'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot individual assets
        if show_assets:
            asset_risks = np.sqrt(np.diag(self.Sigma))
            ax.scatter(asset_risks, self.mu, 
                      c='gray', marker='o', s=100, alpha=0.6, label='Individual Assets')
            
            # Label assets
            for asset, ret, risk in zip(self.assets, self.mu, asset_risks):
                ax.annotate(asset, (risk, ret), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Plot optimal portfolio
        if show_optimal and self.optimal_weights is not None:
            ax.scatter(self.portfolio_risk, self.portfolio_return,
                      c='red', marker='*', s=500, 
                      label=f'Optimal Portfolio (SR={self.sharpe_ratio:.2f})',
                      edgecolors='black', linewidths=1.5, zorder=5)
        
        ax.set_xlabel('Risk (Volatility)', fontsize=12)
        ax.set_ylabel('Expected Return', fontsize=12)
        ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @classmethod
    def from_optimizer_inputs(
        cls,
        results: Dict,
        allow_short: bool = False,
        gross_exposure: Optional[float] = None,
        objective: str = 'max_sharpe',
        risk_aversion: Optional[float] = 0.5
    ):
        """
        Create optimizer from PortfolioOptimizerInputs results.
        
        Parameters
        ----------
        results : dict
            Results dictionary from PortfolioOptimizerInputs.quick_run()
        allow_short : bool, default=False
            Whether to allow short selling
        gross_exposure : float, optional
            Maximum gross exposure if shorting allowed
        objective : str, default='max_sharpe'
            Optimization objective: 'max_sharpe', 'max_sortino', or 'risk_aversion'
        risk_aversion : float, optional
            Risk aversion parameter (required if objective='risk_aversion')
            
        Returns
        -------
        PortfolioOptimizer
            Initialized optimizer instance
        """
        mu = results['inputs']['mu']
        Sigma = results['inputs']['Sigma']
        assets = results['instance'].asset_names
        raw_ohlc = {}
        for asset in assets:
            raw_ohlc[asset] = results['instance'].load_models(asset)[1].raw_ohlc
        opt_date = results['instance'].load_models(results['instance'].asset_names[0])[0].returns.index[-1]
        
        # Get simulated returns if available (for Sortino ratio)
        simulated_returns = results.get('asset_simulations', None)
        
        return cls(
            mu=mu,
            Sigma=Sigma,
            assets=assets,
            raw_ohlc=raw_ohlc,
            opt_date=opt_date,
            allow_short=allow_short,
            gross_exposure=gross_exposure,
            objective=objective,
            risk_aversion=risk_aversion,
            simulated_returns=simulated_returns
        )
