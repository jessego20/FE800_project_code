"""
Minimal Example: 5-Line Portfolio Optimization
Phase 4+ Workflow
"""

from portfolio_optimizer_inputs import PortfolioOptimizerInputs
from portfolio_optimizer import PortfolioOptimizer

# 1. Initialize portfolio
portfolio = PortfolioOptimizerInputs(
    asset_names=['SPDR S&P 500 ETF', 'iShares Russell 2000 ETF'],
    asset_class='us_equity',
    end_date='20241101'
)

# 2. Simulate (uses copula by default - preserves correlations)
portfolio.simulate_all_assets()

# 3. Optimize (one-liner)
optimizer = PortfolioOptimizer.quick_optimize(portfolio)

# 4. View results
print("\nOptimal Weights:")
print(optimizer.optimal_weights)

print("\nPortfolio Statistics:")
print(f"Return: {optimizer.portfolio_return:.2%}")
print(f"Risk: {optimizer.portfolio_risk:.2%}")
print(f"Sharpe: {optimizer.sharpe_ratio:.2f}")

# 5. Visualize
optimizer.plot_weights()
optimizer.plot_efficient_frontier()
