"""
Portfolio Optimizer Inputs Generator

This module generates mean-variance optimization inputs (μ, Σ) for a portfolio of assets
using Bayesian forward simulations from regime-switching models.

For each asset:
1. Loads pre-fitted KAMA+MSR and KMRF models
2. Runs Bayesian forward simulation to generate return paths
3. Computes expected returns and covariance matrix from simulated paths

Author: Jesse Goodman
Date: November 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
import contextlib
import io

from bayesian_forward_simulator import BayesianForwardSimulator
from kama_msr import KAMA_MSR
from kmrf import KMRF

warnings.filterwarnings('ignore')


class PortfolioOptimizerInputs:
    """
    Generate mean-variance optimization inputs from Bayesian forward simulations.
    
    This class:
    - Loads pre-fitted models for multiple assets
    - Runs Bayesian forward simulations for each asset
    - Computes expected returns (μ) and covariance matrix (Σ)
    - Provides ready-to-use inputs for portfolio optimization
    
    Parameters
    ----------
    asset_names : List[str]
        List of asset names (must match saved model filenames)
    asset_class : str
        Asset class folder name ('us_equity', 'commodity', 'int_equity', 'universe', etc.)
    end_date : str
        End date in YYYYMMDD format (e.g., '20181231')
    models_base_path : str, default='saved_models'
        Base directory containing saved models
    n_days : int, default=21
        Forecast horizon in days
    n_simulations : int, default=10000
        Number of Monte Carlo simulations per asset
    alpha_confidence : float, default=0.75
        Bayesian confidence weight (0=HMM only, 1=KMRF only)
    significance_level : float, default=0.05
        Significance level for distribution selection
    random_seed : int, default=1010
        Random seed for reproducibility
    retrain_kmrf : bool, default=False
        If True, retrain KMRF models using all data available in KAMA_MSR
        If False, load pre-trained KMRF models from disk
    use_boruta_selection : bool, default=False
        If True and retrain_kmrf=True, use Boruta feature selection when training KMRF
    use_consensus_selection : bool, default=False
        If True and retrain_kmrf=True, use consensus feature selection when training KMRF
        Overrides use_boruta_selection if both are True
    
    Attributes
    ----------
    asset_simulations : Dict[str, pd.DataFrame]
        {asset_name: simulated_daily_returns} for each asset
    mu : pd.Series
        Expected returns for each asset (horizon return)
    Sigma : pd.DataFrame
        Covariance matrix of returns
    correlation_matrix : pd.DataFrame
        Correlation matrix of returns
    simulators : Dict[str, BayesianForwardSimulator]
        {asset_name: simulator_instance} for each asset
    """
    
    def __init__(
        self,
        asset_names: List[str],
        asset_class: str,
        end_date: str,
        models_base_path: str = 'saved_models',
        n_days: int = 21,
        n_simulations: int = 10000,
        alpha_confidence: float = 1.0,
        significance_level: float = 0.05,
        random_seed: int = 1010,
        retrain_kmrf: bool = False,
        use_boruta_selection: bool = False,
        use_consensus_selection: bool = False
    ):
        self.asset_names = asset_names
        self.asset_class = asset_class
        self.end_date = end_date
        self.models_base_path = Path(models_base_path)
        self.n_days = n_days
        self.n_simulations = n_simulations
        self.alpha = alpha_confidence
        self.sig_level = significance_level
        self.random_seed = random_seed
        self.retrain_kmrf = retrain_kmrf
        self.use_boruta_selection = use_boruta_selection
        self.use_consensus_selection = use_consensus_selection
        
        # Storage for results
        self.asset_simulations: Dict[str, pd.DataFrame] = {}
        self.simulators: Dict[str, BayesianForwardSimulator] = {}
        self.kmrf_models: Dict[str, KMRF] = {}
        self.mu: Optional[pd.Series] = None
        self.Sigma: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Validate paths
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate that base paths exist."""
        if not self.models_base_path.exists():
            raise FileNotFoundError(
                f"Models base path does not exist: {self.models_base_path}"
            )
        
        kama_msr_path = self.models_base_path / 'KAMA_MSR' / self.asset_class / self.end_date
        if not kama_msr_path.exists():
            raise FileNotFoundError(
                f"KAMA_MSR path does not exist: {kama_msr_path}"
            )
        
        # Only check KMRF path if not retraining
        if not self.retrain_kmrf:
            kmrf_path = self.models_base_path / 'KMRF_new' / 'original' / self.asset_class
            if not kmrf_path.exists():
                warnings.warn(
                    f"KMRF path does not exist: {kmrf_path}. "
                    "Set retrain_kmrf=True to train models dynamically."
                )
    
    def _get_kama_msr_path(self, asset_name: str) -> Path:
        """Get path to KAMA+MSR model file."""
        filename = f"{asset_name}_KAMA-MSR_4-regimes.pkl"
        return self.models_base_path / 'KAMA_MSR' / self.asset_class / self.end_date / filename
    
    def _get_kmrf_path(self, asset_name: str) -> Path:
        """Get path to KMRF model file."""
        # KMRF models are typically stored by asset class, not date
        # Keep asset name as-is, no space replacement
        filename = f"{asset_name}_KMRF_model.pkl"
        return self.models_base_path / 'KMRF_new' / 'original' / self.asset_class / filename

    def load_models(self, asset_name: str, verbose: bool = False, kmrf: bool = False) -> Tuple[KAMA_MSR, KMRF]:
        """
        Load pre-fitted KAMA+MSR and KMRF models for an asset.
        If retrain_kmrf=True, trains a new KMRF model using data from KAMA_MSR.
        
        Parameters
        ----------
        asset_name : str
            Asset name
        verbose : bool, default=False
            Print training progress if retraining KMRF
            
        Returns
        -------
        tuple
            (kama_msr, kmrf) model instances
        """
        kama_msr_path = self._get_kama_msr_path(asset_name)
        
        # Check if KAMA_MSR file exists
        if not kama_msr_path.exists():
            raise FileNotFoundError(
                f"KAMA+MSR model not found for {asset_name}: {kama_msr_path}"
            )
        
        # Load KAMA+MSR
        with open(kama_msr_path, 'rb') as f:
            kama_msr = pickle.load(f)
        
        if not kmrf:
            return kama_msr, None
        
        # Load or retrain KMRF
        if self.retrain_kmrf:
            # Train new KMRF using all data available in KAMA_MSR
            if verbose:
                print(f"  Training new KMRF model using KAMA_MSR data...")
            
            kmrf = self._train_kmrf_from_kama_msr(kama_msr, asset_name, verbose=verbose)
        else:
            # Load pre-trained KMRF
            kmrf_path = self._get_kmrf_path(asset_name)
            
            if not kmrf_path.exists():
                raise FileNotFoundError(
                    f"KMRF model not found for {asset_name}: {kmrf_path}"
                )
            
            # Load KMRF (suppress output)
            with contextlib.redirect_stdout(io.StringIO()):
                kmrf = KMRF.load_model(str(kmrf_path))
        
        return kama_msr, kmrf
    
    def _train_kmrf_from_kama_msr(
        self, 
        kama_msr: KAMA_MSR, 
        asset_name: str,
        verbose: bool = False
    ) -> KMRF:
        """
        Train a new KMRF model using data and regime labels from KAMA_MSR.
        Uses the KMRF pipeline method for complete training workflow.
        
        Parameters
        ----------
        kama_msr : KAMA_MSR
            Fitted KAMA+MSR model containing price data and regime labels
        asset_name : str
            Asset name for KMRF initialization
        verbose : bool, default=False
            Print training progress
            
        Returns
        -------
        KMRF
            Trained KMRF model
        """
        # Initialize KMRF with feature selection options
        kmrf = KMRF(
            asset_name=asset_name,
            asset_class=self.asset_class,
            end_date=self.end_date,
            random_seed=self.random_seed,
            classification_type='original',  # Use 4-regime labels from KAMA_MSR
            use_data_type='master',
            use_boruta_selection=self.use_boruta_selection,
            use_consensus_selection=self.use_consensus_selection
        )
        
        # Run the complete pipeline
        if verbose:
            # Show training output
            kmrf.pipeline()
        else:
            # Suppress training output
            with contextlib.redirect_stdout(io.StringIO()):
                kmrf.pipeline()
        
        return kmrf
    
    def simulate_asset(
        self, 
        asset_name: str, 
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run Bayesian forward simulation for a single asset.
        
        Parameters
        ----------
        asset_name : str
            Asset name
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        pd.DataFrame
            Simulated daily returns (n_days × n_simulations)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"SIMULATING: {asset_name}")
            print(f"{'='*80}")
        
        # Load models
        if verbose:
            print(f"Loading models...")
        kama_msr, kmrf = self.load_models(asset_name, kmrf=True)
        
        # Create simulator
        simulator = BayesianForwardSimulator(
            kama_msr=kama_msr,
            kmrf=kmrf,
            n_days=self.n_days,
            alpha_confidence=self.alpha,
            significance_level=self.sig_level
        )
        
        # Run simulation
        if verbose:
            print(f"Computing forward probabilities...")
        simulator.compute_forward_regime_probs()
        
        if verbose:
            print(f"Fitting regime distributions...")
        simulator.fit_regime_distributions(verbose=verbose)
        
        if verbose:
            print(f"Validating parameters...")
        simulator.validate_distributions()
        
        if verbose:
            print(f"Running {self.n_simulations:,} simulations...")
        simulated_returns = simulator.simulate(
            n_simulations=self.n_simulations,
            random_seed=self.random_seed
        )
        
        # Store results
        self.asset_simulations[asset_name] = simulated_returns
        self.simulators[asset_name] = simulator
        self.kmrf_models[asset_name] = kmrf
        
        if verbose:
            # Compute some summary stats
            terminal_returns = simulated_returns.add(1).cumprod(axis=0).sub(1).iloc[-1]
            print(f"\nTerminal {self.n_days}-day return statistics:")
            print(f"  Mean: {terminal_returns.mean():.4f}")
            print(f"  Std:  {terminal_returns.std():.4f}")
            print(f"  Min:  {terminal_returns.min():.4f}")
            print(f"  Max:  {terminal_returns.max():.4f}")
        
        return simulated_returns
    
    def simulate_all_assets(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run Bayesian forward simulations for all assets.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        dict
            {asset_name: simulated_returns} for all assets
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"PORTFOLIO SIMULATION: {len(self.asset_names)} ASSETS")
            print(f"{'='*80}")
            print(f"Asset Class: {self.asset_class}")
            print(f"End Date: {self.end_date}")
            print(f"Horizon: {self.n_days} days")
            print(f"Simulations per asset: {self.n_simulations:,}")
        
        for i, asset_name in enumerate(self.asset_names, 1):
            if verbose:
                print(f"\n[{i}/{len(self.asset_names)}] Processing {asset_name}...")
            
            try:
                self.simulate_asset(asset_name, verbose=verbose)
            except Exception as e:
                print(f"\n⚠️  ERROR simulating {asset_name}: {e}")
                print(f"Skipping {asset_name}...")
                continue
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✓ Successfully simulated {len(self.asset_simulations)}/{len(self.asset_names)} assets")
            print(f"{'='*80}")
        
        return self.asset_simulations
    
    def compute_portfolio_inputs(
        self, 
        method: str = 'path_covariance',
        annualization_factor: Optional[float] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute expected returns (μ) and covariance matrix (Σ) for portfolio optimization.
        
        Parameters
        ----------
        method : str, default='terminal'
            Method for computing returns:
            - 'terminal': Use terminal (day n_days) cumulative return
            - 'daily_avg': Use average daily return × n_days
            - 'path_covariance': Compute covariance for each path, then average
        annualization_factor : float, optional
            Factor to annualize returns (e.g., 252/n_days for daily data)
            If None, returns are for the forecast horizon
            
        Returns
        -------
        tuple
            (mu, Sigma) where:
            - mu: pd.Series of expected returns
            - Sigma: pd.DataFrame of return covariance matrix
        """
        if not self.asset_simulations:
            raise ValueError("No simulations available. Run simulate_all_assets() first.")
        
        assets = list(self.asset_simulations.keys())
        n_assets = len(assets)
        
        if method == 'terminal':
            # Use terminal cumulative returns across all paths
            terminal_returns = {}
            for asset in assets:
                sim = self.asset_simulations[asset]
                # Cumulative return at end of horizon
                terminal_returns[asset] = sim.add(1).cumprod(axis=0).iloc[-1] - 1
            
            # Create DataFrame: rows=simulations, columns=assets
            returns_df = pd.DataFrame(terminal_returns)
            
            # Expected returns and covariance
            mu = returns_df.mean()
            Sigma = returns_df.cov()
            
        elif method == 'daily_avg':
            # Use average daily return scaled by horizon
            daily_returns = {}
            for asset in assets:
                sim = self.asset_simulations[asset]
                # Average daily return across time and simulations
                daily_returns[asset] = sim.mean(axis=0)  # Average across days for each sim
            
            returns_df = pd.DataFrame(daily_returns)
            
            # Scale to horizon
            mu = returns_df.mean() * self.n_days
            Sigma = returns_df.cov() * self.n_days
        
        elif method == 'path_covariance':
            # For each simulation path, compute the time-series covariance
            # Then average covariance matrices across all paths
            
            # Stack all asset returns: dict of DataFrames -> 3D structure
            # Create list of DataFrames, one per simulation
            path_covariances = []
            terminal_returns = {}
            
            for asset in assets:
                terminal_returns[asset] = self.asset_simulations[asset].add(1).cumprod(axis=0).iloc[-1] - 1
            
            # For each simulation
            for sim_idx in range(self.n_simulations):
                # Extract the path for this simulation across all assets
                path_data = {}
                for asset in assets:
                    # Daily returns for this simulation
                    path_data[asset] = self.asset_simulations[asset].iloc[:, sim_idx]
                
                # Create DataFrame: rows=days, columns=assets
                path_df = pd.DataFrame(path_data)
                
                # Compute covariance matrix for this path (time-series covariance)
                path_cov = path_df.cov()
                path_covariances.append(path_cov.values)
            
            # Average covariance matrices across all simulations
            # Note: This is DAILY covariance, needs to be scaled by n_days for horizon covariance
            Sigma = np.mean(path_covariances, axis=0) * self.n_days
            Sigma = pd.DataFrame(Sigma, index=assets, columns=assets)
            
            # Expected returns: use terminal returns (same as terminal method)
            returns_df = pd.DataFrame(terminal_returns)
            mu = returns_df.mean()
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'terminal', 'daily_avg', or 'path_covariance'")
        
        # Annualize if requested
        if annualization_factor is not None:
            # Use linear scaling for all methods
            mu = mu * annualization_factor
            Sigma = Sigma * annualization_factor
        
        # Store results
        self.mu = mu
        self.Sigma = Sigma
        
        # Also compute correlation matrix
        std = np.sqrt(np.diag(Sigma))
        self.correlation_matrix = Sigma / np.outer(std, std)
        self.correlation_matrix = pd.DataFrame(
            self.correlation_matrix,
            index=assets,
            columns=assets
        )
        
        return mu, Sigma
    
    def get_optimization_inputs(
        self,
        method: str = 'path_covariance',
        annualize: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all inputs needed for portfolio optimization in a convenient format.
        
        Parameters
        ----------
        method : str, default='terminal'
            Method for computing returns:
            - 'terminal': Covariance of terminal returns across simulations
            - 'daily_avg': Covariance of averaged daily returns (scaled)
            - 'path_covariance': Average of within-path time-series covariances
        annualize : bool, default=False
            Whether to annualize returns (assumes 252 trading days)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'mu': Expected returns (pd.Series)
            - 'Sigma': Covariance matrix (pd.DataFrame)
            - 'correlation': Correlation matrix (pd.DataFrame)
            - 'volatility': Standard deviations (pd.Series)
        
        Notes
        -----
        Method comparison:
        - 'terminal': Treats each simulation's terminal return as independent sample.
          Best for: Buy-and-hold optimization, capturing tail risk.
          
        - 'daily_avg': Averages daily returns within each path, then computes covariance.
          Best for: Quick approximation, comparison with i.i.d. models.
          
        - 'path_covariance': Computes time-series covariance within each path, then averages.
          Best for: Capturing intra-path co-movement, regime-dependent correlations.
          This preserves the correlation structure across time while averaging out
          simulation uncertainty.
        """
        if self.mu is None or self.Sigma is None:
            annualization_factor = 252 / self.n_days if annualize else None
            self.compute_portfolio_inputs(
                method=method,
                annualization_factor=annualization_factor
            )
        
        volatility = pd.Series(
            np.sqrt(np.diag(self.Sigma)),
            index=self.Sigma.index,
            name='volatility'
        )
        
        return {
            'mu': self.mu,
            'Sigma': self.Sigma,
            'correlation': self.correlation_matrix,
            'volatility': volatility
        }
    
    def summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for all assets.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for each asset
        """
        if not self.asset_simulations:
            raise ValueError("No simulations available. Run simulate_all_assets() first.")
        
        summary_data = []
        
        for asset_name, sim in self.asset_simulations.items():
            # Compute terminal returns
            terminal_returns = sim.add(1).cumprod(axis=0).sub(1).iloc[-1]
            
            summary_data.append({
                'asset': asset_name,
                'mean_return': terminal_returns.mean(),
                'std_return': terminal_returns.std(),
                'sharpe_ratio': terminal_returns.mean() / terminal_returns.std() if terminal_returns.std() > 0 else 0,
                'min_return': terminal_returns.min(),
                'max_return': terminal_returns.max(),
                'median_return': terminal_returns.median(),
                'skewness': terminal_returns.skew(),
                'kurtosis': terminal_returns.kurtosis(),
                'var_95': terminal_returns.quantile(0.05),
                'cvar_95': terminal_returns[terminal_returns <= terminal_returns.quantile(0.05)].mean()
            })
        
        df = pd.DataFrame(summary_data).set_index('asset')
        
        print("\n" + "="*80)
        print(f"PORTFOLIO SUMMARY STATISTICS ({self.n_days}-DAY HORIZON)")
        print("="*80)
        print(df.to_string())
        print("="*80)
        
        return df
    
    def plot_correlation_heatmap(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrix heatmap.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.correlation_matrix is None:
            raise ValueError("Must call compute_portfolio_inputs() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        ax.set_title(
            f'Asset Return Correlations ({self.n_days}-Day Horizon)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_efficient_frontier_inputs(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot risk-return scatter of individual assets.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt
        
        if self.mu is None or self.Sigma is None:
            raise ValueError("Must call compute_portfolio_inputs() first")
        
        volatility = np.sqrt(np.diag(self.Sigma))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(
            volatility,
            self.mu,
            s=100,
            alpha=0.7,
            c=range(len(self.mu)),
            cmap='viridis',
            edgecolors='black',
            linewidth=1.5
        )
        
        # Annotate points
        for i, asset in enumerate(self.mu.index):
            ax.annotate(
                asset,
                (volatility[i], self.mu.iloc[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        ax.set_xlabel('Volatility (Std Dev)', fontsize=12)
        ax.set_ylabel('Expected Return', fontsize=12)
        ax.set_title(
            f'Risk-Return Profile of Assets ({self.n_days}-Day Horizon)',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    '''
    def save_results(self, output_path: str, filename: str = 'portfolio_simulation_results.pkl'):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / filename, 'wb') as f:
            pickle.dump(self.results, f)
        
        # # Save simulations
        # sims_path = output_path / 'simulations'
        # sims_path.mkdir(exist_ok=True)
        
        # for asset_name, sim in self.asset_simulations.items():
        #     filename = f"{asset_name.replace(' ', '_')}_simulations.csv"
        #     sim.to_csv(sims_path / filename)
        
        # # Save optimization inputs
        # if self.mu is not None:
        #     self.mu.to_csv(output_path / 'expected_returns.csv', header=['mu'])
        
        # if self.Sigma is not None:
        #     self.Sigma.to_csv(output_path / 'covariance_matrix.csv')
        
        # if self.correlation_matrix is not None:
        #     self.correlation_matrix.to_csv(output_path / 'correlation_matrix.csv')
        
        # # Save summary
        # summary = self.summary()
        # summary.to_csv(output_path / 'summary_statistics.csv')
        
        # print(f"\n✓ Results saved to {output_path}")
        # print(f"  - {len(self.asset_simulations)} simulation files")
        # print(f"  - Expected returns: expected_returns.csv")
        # print(f"  - Covariance matrix: covariance_matrix.csv")
        # print(f"  - Correlation matrix: correlation_matrix.csv")
        # print(f"  - Summary statistics: summary_statistics.csv")
    '''
    @classmethod
    def quick_run(
        cls,
        asset_names: List[str],
        asset_class: str,
        end_date: str,
        n_days: int = 21,
        n_simulations: int = 10000,
        method: str = 'terminal',
        annualize: bool = True,
        verbose: bool = True,
        random_seed: int = 1010,
        retrain_kmrf: bool = False,
        use_boruta_selection: bool = False,
        use_consensus_selection: bool = False
    ) -> Dict:
        """
        Quick run: simulate all assets and return optimization inputs.
        
        Parameters
        ----------
        asset_names : List[str]
            Asset names
        asset_class : str
            Asset class
        end_date : str
            End date (YYYYMMDD)
        n_days : int, default=21
            Forecast horizon
        n_simulations : int, default=10000
            Number of simulations
        method : str, default='terminal'
            Return computation method ('terminal', 'daily_avg', 'path_covariance')
        annualize : bool, default=False
            Annualize returns
        verbose : bool, default=True
            Print progress
        retrain_kmrf : bool, default=False
            If True, retrain KMRF models using KAMA_MSR data instead of loading pre-trained
        use_boruta_selection : bool, default=False
            If True and retrain_kmrf=True, use Boruta feature selection
        use_consensus_selection : bool, default=False
            If True and retrain_kmrf=True, use consensus feature selection (overrides Boruta)
            
        Returns
        -------
        dict
            Optimization inputs and instance
        """
        # Create instance
        portfolio_gen = cls(
            asset_names=asset_names,
            asset_class=asset_class,
            end_date=end_date,
            n_days=n_days,
            n_simulations=n_simulations,
            random_seed=random_seed,
            retrain_kmrf=retrain_kmrf,
            use_boruta_selection=use_boruta_selection,
            use_consensus_selection=use_consensus_selection
        )
        
        # Run simulations
        portfolio_gen.simulate_all_assets(verbose=verbose)
        
        # Compute inputs
        inputs = portfolio_gen.get_optimization_inputs(
            method=method,
            annualize=annualize
        )
        
        # Return everything
        return {
            'inputs': inputs,
            'instance': portfolio_gen,
            'summary': portfolio_gen.summary(),
            'asset_simulations': portfolio_gen.asset_simulations  # Add for Sortino ratio
        }
