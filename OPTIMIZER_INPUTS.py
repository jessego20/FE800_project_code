import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Union, List
from scipy import stats as scipy_stats
from scipy.stats import skewnorm, t, norminvgauss, norm
from scipy.optimize import minimize, curve_fit
import warnings
import contextlib
import io
from pathlib import Path

from kama_msr import KAMA_MSR
from kmrf import KMRF
from SIMULATOR import SIMULATOR

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

TRADING_DAYS = CustomBusinessDay(calendar=USFederalHolidayCalendar())

warnings.filterwarnings('ignore')

KM_MODEL_BASE_PATH = Path('saved_models/KAMA_MSR/us_equity')
KMRF_PREDICTIONS_BASE_PATH = Path('data/multi_horizon_predictions')

# Using saved KAMA+MSR models
def get_KM_model_dates(KM_MODEL_BASE_PATH = Path('saved_models/KAMA_MSR/us_equity')) -> pd.Series:
    return pd.Series([f.stem for f in list(KM_MODEL_BASE_PATH.glob('*'))]).sort_values().iloc[1:].reset_index(drop=True)

def get_KM_model_paths(MODEL_DATE: str, KM_MODEL_BASE_PATH = Path('saved_models/KAMA_MSR/us_equity')) ->  pd.Series:
    return pd.Series(list((KM_MODEL_BASE_PATH / MODEL_DATE).glob('*'))).sort_values().reset_index(drop=True)

# Using saved KMRF predictions
def get_asset_names(KMRF_PREDICTIONS_BASE_PATH = Path('data/multi_horizon_predictions')) -> pd.Series:
    kmrf_preds_paths = list(KMRF_PREDICTIONS_BASE_PATH.glob('*'))
    return pd.Series([f.stem.split('multi')[0][:-1].replace('_', ' ') for f in kmrf_preds_paths]).sort_values().reset_index(drop=True)

def get_KMRF_prediction_paths(KMRF_PREDICTIONS_BASE_PATH = Path('data/multi_horizon_predictions')) -> pd.Series:
    return pd.Series(list(KMRF_PREDICTIONS_BASE_PATH.glob('*'))).sort_values().reset_index(drop=True)


class OPTIMIZER_INPUTS:
    """
    Generate mean-variance optimization inputs (mu, Sigma) using Gaussian copula simulations.
    
    This class:
    - Loads pre-fitted KAMA_MSR models and KMRF predictions for multiple assets
    - Estimates regime-dependent correlation matrices
    - Estimates regime concordance (P(asset_regime | market_regime))
    - Runs Gaussian copula simulations with KMRF-based forward regime probabilities
    - Computes expected returns (μ) and covariance matrix (Σ)
    
    Parameters
    ----------
    opt_date : str or pd.Timestamp
        Optimization/rebalancing date (YYYYMMDD format)
    asset_list : List[str]
        List of asset names to include in portfolio
    n_days : int, default=21
        Forecast horizon in trading days
    n_simulations : int, default=10000
        Number of Monte Carlo simulation paths
    market_asset : str, default='SPDR S&P 500 ETF'
        Asset to use as market regime indicator
    random_seed : int, default=1010
        Random seed for reproducibility
        
    Attributes
    ----------
    simulator_objects : Dict[str, SIMULATOR]
        SIMULATOR instances for each asset
    regime_correlations : Dict[int, pd.DataFrame]
        Correlation matrices conditional on market regime (4 matrices, one per regime)
    regime_concordance : Dict[str, np.ndarray]
        P(asset_regime | market_regime) for each asset
    simulated_returns : Dict[str, np.ndarray]
        Simulated daily returns for each asset, shape (n_simulations, n_days)
    mu : pd.Series
        Expected returns for each asset
    Sigma : pd.DataFrame
        Covariance matrix of returns
    """

    def __init__(
        self, 
        opt_date: str | pd.Timestamp, 
        asset_list: List[str],
        n_days: int = 21,
        n_simulations: int = 1000,
        market_asset: str = 'SPDR S&P 500 ETF',
        random_seed: int = 123,
        risk_free_rate: float = 0.0
    ):
        if opt_date is None:
            raise ValueError("opt_date must be provided")
        if not asset_list:
            raise ValueError("asset_list must be provided and cannot be empty")

        self.risk_free_rate = risk_free_rate

        self.ALL_KM_MODEL_DATES = get_KM_model_dates()
        self.ALL_ASSET_NAMES = get_asset_names()

        self.opt_date = pd.Timestamp(opt_date)
        self.n_days = n_days
        self.n_simulations = n_simulations
        self.market_asset = market_asset
        self.random_seed = random_seed

        # Find nearest available model date
        self._model_date = self._find_nearest_model_date(self.opt_date)
        
        self.ALL_KM_MODEL_PATHS = get_KM_model_paths(MODEL_DATE=self._model_date)
        self.ALL_KMRF_PREDICTION_PATHS = get_KMRF_prediction_paths()

        self.asset_names = asset_list
        self.simulator_objects: Dict[str, SIMULATOR] = {}
        
        # Regime structure attributes
        self.market_regime_labels: Optional[pd.Series] = None
        self.regime_correlations: Optional[Dict[int, pd.DataFrame]] = None
        self.regime_concordance: Optional[Dict[str, np.ndarray]] = None
        
        # Simulation outputs
        self.simulated_returns: Optional[Dict[str, np.ndarray]] = None
        self.simulated_market_regimes: Optional[np.ndarray] = None
        self.simulated_asset_regimes: Optional[Dict[str, np.ndarray]] = None
        
        # Optimization inputs
        self.mu: Optional[pd.Series] = None
        self.Sigma: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def _find_nearest_model_date(self, target_date: pd.Timestamp) -> str:
        """Find the nearest available model date <= target_date."""
        available_dates = self.ALL_KM_MODEL_DATES.apply(pd.Timestamp)
        valid_dates = available_dates[available_dates <= target_date]
        
        if len(valid_dates) == 0:
            raise ValueError(f"No models available on or before {target_date}")
        
        nearest_date = valid_dates.max()
        return nearest_date.strftime('%Y%m%d')

    def load_simulator_objects(self, verbose: bool = True):
        """
        Load SIMULATOR objects for all assets and prepare them for simulation.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print progress information
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"LOADING SIMULATORS FOR {len(self.asset_names)} ASSETS")
            print(f"{'='*80}")
            print(f"Optimization Date: {self.opt_date.strftime('%Y-%m-%d')}")
            print(f"Model Date: {self._model_date}")
        
        for i, asset in enumerate(self.asset_names, 1):
            if asset not in self.ALL_ASSET_NAMES.values:
                print(f"⚠️  Asset '{asset}' not found in KMRF predictions. Skipping.")
                continue
            
            if verbose:
                print(f"  [{i}/{len(self.asset_names)}] Loading {asset}...")
            
            asset_idx = self.ALL_ASSET_NAMES[self.ALL_ASSET_NAMES == asset].index[0]
            km_model_path = self.ALL_KM_MODEL_PATHS[asset_idx]
            kmrf_preds_path = self.ALL_KMRF_PREDICTION_PATHS[asset_idx]
            
            sim = SIMULATOR(km_model_path=km_model_path, kmrf_preds_path=kmrf_preds_path)
            sim.prepare_for_simulation()  # Run significance tests, fit distributions, compute transition matrix
            
            self.simulator_objects[asset] = sim
        
        if verbose:
            print(f"\n✓ Successfully loaded {len(self.simulator_objects)}/{len(self.asset_names)} simulators")
        
        # Load market regime simulator
        self.load_market_regime_simulator(verbose=verbose)

    def load_market_regime_simulator(self, verbose: bool = True):
        """Load regime labels for the market asset (S&P 500)."""
        if self.market_asset not in self.simulator_objects:
            if verbose:
                print(f"\n⚠️  Market asset '{self.market_asset}' not in portfolio. Loading separately...")
            
            if self.market_asset not in self.ALL_ASSET_NAMES.values:
                raise ValueError(f"Market asset '{self.market_asset}' not found in available assets")
            
            asset_idx = self.ALL_ASSET_NAMES[self.ALL_ASSET_NAMES == self.market_asset].index[0]
            km_model_path = self.ALL_KM_MODEL_PATHS[asset_idx]
            kmrf_preds_path = self.ALL_KMRF_PREDICTION_PATHS[asset_idx]
            
            market_sim = SIMULATOR(km_model_path=km_model_path, kmrf_preds_path=kmrf_preds_path)
            market_sim.prepare_for_simulation()
            
            # self.market_regime_labels = market_sim.km_model.regime_labels
            self.market_simulator = market_sim
        else:
            # self.market_regime_labels = self.simulator_objects[self.market_asset].km_model.regime_labels
            self.market_simulator = self.simulator_objects[self.market_asset]

    def get_fwd_regime_probs(self, n_days: int = None) -> pd.DataFrame:
        if not hasattr(self, 'simulator_objects') or not hasattr(self, 'market_simulator'):
            raise ValueError("Must call load_simulator_objects() first")

        if n_days is None:
            n_days = self.n_days

        fwd_regime_probs = {}

        if self.market_asset not in self.simulator_objects:

            fwd_regime_probs[self.market_asset] = self.market_simulator.get_forward_regime_probs(self.opt_date, n_days)
        
        for asset_name, sim in self.simulator_objects.items():
            fwd_regime_probs[asset_name] = sim.get_forward_regime_probs(self.opt_date, n_days)

        self.fwd_regime_probs = fwd_regime_probs

####################################################################################################################################################################################
# PHASE 1: Regime Correlation and Concordance Estimation
####################################################################################################################################################################################

    def estimate_regime_correlations(
        self, 
        verbose: bool = True,
        min_observations: int = 30
    ) -> None:
        """
        Estimate correlation matrices for each market regime.
        
        Uses S&P 500 regime labels to define market state, then computes
        correlation matrices for portfolio assets during each regime.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print diagnostic information
        min_observations : int, default=30
            Minimum observations required per regime
            
        Saves
        -----
        self.regime_correlations : Dict[int, pd.DataFrame]
            {regime_id: correlation_matrix} for regimes 0, 1, 2, 3
        """
        if verbose:
            print(f"\n{'='*80}")
            print("ESTIMATING REGIME-DEPENDENT CORRELATIONS")
            print(f"{'='*80}")
        
        if not hasattr(self, 'market_simulator'):
            raise ValueError("Must call load_simulator_objects() first")
        
        market_regime_labels = self.market_simulator.km_model.regime_labels
        
        # Collect returns for all portfolio assets
        asset_returns = {}
        for asset_name, sim in self.simulator_objects.items():
            asset_returns[asset_name] = sim.km_model.returns
        
        returns_df = pd.DataFrame(asset_returns)
        
        # Align with market regime labels
        common_dates = returns_df.index.intersection(market_regime_labels.index)
        aligned_returns = returns_df.loc[common_dates]
        aligned_regimes = market_regime_labels.loc[common_dates]
        
        # Remove NaN regimes
        valid_mask = aligned_regimes.notna()
        aligned_returns = aligned_returns[valid_mask]
        aligned_regimes = aligned_regimes[valid_mask]
        
        if verbose:
            print(f"\nAligned data: {len(aligned_returns)} observations")
        
        # Compute overall correlation as fallback
        overall_corr = aligned_returns.corr()
        
        regime_correlations = {}
        regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        for regime in range(4):
            regime_mask = (aligned_regimes == regime)
            regime_data = aligned_returns[regime_mask]
            n_obs = len(regime_data)
            
            if verbose:
                print(f"\nRegime {regime} ({regime_names[regime]}): {n_obs} observations")
            
            if n_obs < min_observations:
                if verbose:
                    print(f"  ⚠️  Using overall correlation (insufficient data)")
                regime_correlations[regime] = overall_corr.copy()
            else:
                regime_corr = regime_data.corr()
                
                # Ensure positive definiteness
                regime_corr = self._make_positive_definite(regime_corr)
                regime_correlations[regime] = regime_corr
                
                if verbose:
                    mask = ~np.eye(regime_corr.shape[0], dtype=bool)
                    avg_corr = regime_corr.values[mask].mean()
                    print(f"  Avg off-diagonal correlation: {avg_corr:.3f}")
        
        self.regime_correlations = regime_correlations
        
        if verbose:
            print(f"\n{'='*80}")
            print("✓ Regime correlations estimated")
            print(f"{'='*80}")

    def _make_positive_definite(self, corr: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        """Ensure correlation matrix is positive definite."""
        corr_array = corr.values
        
        # Check eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(corr_array)
        
        if eigenvalues.min() >= epsilon:
            return corr
        
        # Fix negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, epsilon)
        corr_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize to ensure diagonal is 1
        d = np.sqrt(np.diag(corr_fixed))
        corr_fixed = corr_fixed / np.outer(d, d)
        
        return pd.DataFrame(corr_fixed, index=corr.index, columns=corr.columns)

    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        """
        Normalize probability array to ensure it sums to 1 and handle edge cases.
        
        Parameters
        ----------
        probs : np.ndarray
            Array of probabilities (may not sum to 1, may contain NaN)
            
        Returns
        -------
        np.ndarray
            Normalized probability array that sums to 1
        """
        probs = np.array(probs, dtype=float)
        
        # Replace NaN with small epsilon
        probs = np.nan_to_num(probs, nan=1e-10)
        
        # Replace negative values with small epsilon
        probs = np.maximum(probs, 1e-10)
        
        # Normalize to sum to 1
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            # Fallback to uniform distribution
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        return probs / total

    def estimate_regime_concordance(
        self,
        verbose: bool = True,
        min_observations: int = 30
    ) -> None:
        """
        Estimate conditional regime probabilities: P(asset_regime | market_regime).
        
        This captures how each asset's regime depends on the S&P 500 market regime,
        enabling realistic co-movement in simulations.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print diagnostic information
        min_observations : int, default=30
            Minimum observations per market regime
            
        Saves
        -----
        self.regime_concordance : Dict[str, np.ndarray]
            For each asset, a 4x4 matrix where [i,j] = P(asset_regime=j | market_regime=i)
        """
        if verbose:
            print(f"\n{'='*80}")
            print("ESTIMATING REGIME CONCORDANCE")
            print(f"{'='*80}")
        
        if not hasattr(self, 'market_simulator'):
            raise ValueError("Must call load_simulator_objects() first")
        
        market_regime_labels = self.market_simulator.km_model.regime_labels
        
        regime_concordance = {}
        regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        for asset_name, sim in self.simulator_objects.items():
            if verbose:
                print(f"\n{asset_name}:")
            
            asset_regimes = sim.km_model.regime_labels
            
            # Align dates
            aligned = pd.DataFrame({
                'market_regime': market_regime_labels,
                'asset_regime': asset_regimes
            }).dropna()
            
            if verbose:
                print(f"  Aligned observations: {len(aligned)}")
            
            # Compute P(asset_regime | market_regime)
            concordance_matrix = np.zeros((4, 4))
            
            for market_regime in range(4):
                mask = aligned['market_regime'] == market_regime
                n_obs = mask.sum()
                
                if n_obs < min_observations:
                    # Uniform fallback
                    concordance_matrix[market_regime, :] = 0.25
                else:
                    asset_regime_counts = aligned.loc[mask, 'asset_regime'].value_counts()
                    for asset_regime in range(4):
                        count = asset_regime_counts.get(asset_regime, 0)
                        concordance_matrix[market_regime, asset_regime] = count / n_obs
                    
                    if verbose:
                        print(f"  Market {regime_names[market_regime]}: {n_obs} obs → " +
                              ", ".join([f"{regime_names[j]}: {concordance_matrix[market_regime, j]:.2f}" 
                                        for j in range(4) if concordance_matrix[market_regime, j] > 0.01]))
            
            regime_concordance[asset_name] = concordance_matrix
        
        self.regime_concordance = regime_concordance
        
        if verbose:
            print(f"\n{'='*80}")
            print("✓ Regime concordance estimated")
            print(f"{'='*80}")

####################################################################################################################################################################################
# PHASE 2: Regime Path Simulation
####################################################################################################################################################################################

    def simulate_market_regime_paths(self, verbose: bool = True) -> None:
        """
        Simulate market asset regime paths using Bayesian updates.
        
        For each simulation path and each day:
        1. Start with KMRF prior for day t
        2. Sample regime from current posterior
        3. Update posterior for day t+1 using: P(regime_t+1) ∝ P(regime_t+1 | regime_t) × P_KMRF(regime_t+1)
        
        Saves
        -----
        self.simulated_market_regimes : np.ndarray
            Market regime paths, shape (n_simulations, n_days)
        """
        np.random.seed(self.random_seed)
        
        if verbose:
            print(f"\n{'='*80}")
            print("SIMULATING MARKET REGIME PATHS")
            print(f"{'='*80}")
        
        # Get KMRF forward probabilities for market asset
        market_kmrf_probs = self.market_simulator.get_forward_regime_probs(self.opt_date, self.n_days).values
        
        # Get transition matrix
        transition_matrix = self.market_simulator.transition_matrix.values
        
        # Initialize output
        market_regime_paths = np.zeros((self.n_simulations, self.n_days), dtype=int)
        
        for sim_idx in range(self.n_simulations):
            # Initialize with KMRF prior for day 0
            current_probs = self._normalize_probs(market_kmrf_probs[0].copy())
            
            for day in range(self.n_days):
                # Sample regime from current posterior
                regime = np.random.choice(4, p=current_probs)
                market_regime_paths[sim_idx, day] = regime
                
                # Bayesian update for next day (if not last day)
                if day < self.n_days - 1:
                    # Transition probabilities from current regime
                    transition_probs = transition_matrix[regime, :]
                    
                    # KMRF prior for next day
                    kmrf_prior = market_kmrf_probs[day + 1]
                    
                    # Bayesian combination: P(regime_t+1) ∝ P(regime_t+1 | regime_t) × P_KMRF(regime_t+1)
                    posterior = transition_probs * kmrf_prior
                    current_probs = self._normalize_probs(posterior)
        
        self.simulated_market_regimes = market_regime_paths
        
        if verbose:
            print(f"  Simulations: {self.n_simulations:,}")
            print(f"  Days: {self.n_days}")
            # Show regime distribution across all simulations
            flat_regimes = market_regime_paths.flatten()
            regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
            print(f"\n  Overall regime distribution:")
            for regime in range(4):
                pct = (flat_regimes == regime).mean() * 100
                print(f"    {regime_names[regime]}: {pct:.1f}%")
            print(f"\n✓ Market regime paths simulated")
            print(f"{'='*80}")

    def simulate_asset_regime_paths(self, verbose: bool = True) -> None:
        """
        Simulate portfolio asset regime paths conditioned on market regime paths.
        
        For each asset, simulation, and day:
        1. Get KMRF prior for asset at day t
        2. Get transition probability from asset's previous regime
        3. Get concordance probability from market regime at day t
        4. Combine: P(regime_t) ∝ P_KMRF(regime_t) × P(regime_t | regime_{t-1}) × P(regime_t | market_regime_t)
        5. Sample regime from combined posterior
        
        Saves
        -----
        self.simulated_asset_regimes : Dict[str, np.ndarray]
            Asset regime paths, {asset_name: array of shape (n_simulations, n_days)}
        """
        if self.simulated_market_regimes is None:
            raise ValueError("Must call simulate_market_regime_paths() first")
        
        if self.regime_concordance is None:
            raise ValueError("Must call estimate_regime_concordance() first")
        
        if verbose:
            print(f"\n{'='*80}")
            print("SIMULATING ASSET REGIME PATHS")
            print(f"{'='*80}")
        
        asset_regime_paths = {}
        
        # Check if market asset is in portfolio
        market_in_portfolio = self.market_asset in self.simulator_objects
        
        for asset_name, sim in self.simulator_objects.items():
            if verbose:
                print(f"\n  {asset_name}...")
            
            # If this is the market asset, use the already-simulated market regimes
            if asset_name == self.market_asset:
                if verbose:
                    print(f"    (Using market regime paths directly)")
                asset_regime_paths[asset_name] = self.simulated_market_regimes.copy()
                continue
            
            # Get KMRF forward probs for this asset
            asset_kmrf_probs = sim.get_forward_regime_probs(self.opt_date, self.n_days).values
            
            # Get asset transition matrix
            asset_transition_matrix = sim.transition_matrix.values
            
            # Get concordance matrix for this asset
            concordance_matrix = self.regime_concordance[asset_name]
            
            # Initialize output
            paths = np.zeros((self.n_simulations, self.n_days), dtype=int)
            
            for sim_idx in range(self.n_simulations):
                # Get this simulation's market regime path
                market_path = self.simulated_market_regimes[sim_idx, :]
                
                # Initialize with KMRF prior for day 0, conditioned on market regime
                kmrf_prior = asset_kmrf_probs[0]
                concordance_prob = concordance_matrix[market_path[0], :]
                
                # Initial posterior: P(regime) ∝ P_KMRF(regime) × P(regime | market_regime)
                current_probs = kmrf_prior * concordance_prob
                current_probs = self._normalize_probs(current_probs)
                
                for day in range(self.n_days):
                    # Sample regime from current posterior
                    regime = np.random.choice(4, p=current_probs)
                    paths[sim_idx, day] = regime
                    
                    # Bayesian update for next day (if not last day)
                    if day < self.n_days - 1:
                        # Component 1: Transition probabilities from current asset regime
                        transition_probs = asset_transition_matrix[regime, :]
                        
                        # Component 2: KMRF prior for next day
                        kmrf_prior = asset_kmrf_probs[day + 1]
                        
                        # Component 3: Concordance with next day's market regime
                        next_market_regime = market_path[day + 1]
                        concordance_prob = concordance_matrix[next_market_regime, :]
                        
                        # Bayesian combination: 
                        # P(regime_t+1) ∝ P(regime_t+1 | regime_t) × P_KMRF(regime_t+1) × P(regime_t+1 | market_regime_t+1)
                        posterior = transition_probs * kmrf_prior * concordance_prob
                        current_probs = self._normalize_probs(posterior)
            
            asset_regime_paths[asset_name] = paths
            
            if verbose:
                flat_regimes = paths.flatten()
                regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
                print(f"    Regime distribution: " + 
                      ", ".join([f"{regime_names[r]}: {(flat_regimes == r).mean()*100:.1f}%" for r in range(4)]))
        
        self.simulated_asset_regimes = asset_regime_paths
        
        if verbose:
            print(f"\n✓ Asset regime paths simulated")
            print(f"{'='*80}")

####################################################################################################################################################################################
# PHASE 3: Return Simulation with Gaussian Copula
####################################################################################################################################################################################

    def simulate_returns_copula(self, verbose: bool = True) -> None:
        """
        Simulate correlated returns using Gaussian copula with market-regime dependent correlations.
        
        For each simulation and day:
        1. Get the market regime for this day (from simulated market regime path)
        2. Use the correlation matrix for that market regime
        3. Generate correlated uniform samples via Gaussian copula
        4. Transform to returns using each asset's regime-specific distribution (inverse CDF)
        
        Saves
        -----
        self.simulated_returns : Dict[str, np.ndarray]
            Simulated returns, {asset_name: array of shape (n_simulations, n_days)}
        """
        if self.simulated_market_regimes is None:
            raise ValueError("Must call simulate_market_regime_paths() first")
        if self.simulated_asset_regimes is None:
            raise ValueError("Must call simulate_asset_regime_paths() first")
        if self.regime_correlations is None:
            raise ValueError("Must call estimate_regime_correlations() first")
        
        np.random.seed(self.random_seed + 1)  # Different seed from regime simulation
        
        if verbose:
            print(f"\n{'='*80}")
            print("SIMULATING RETURNS WITH GAUSSIAN COPULA (VECTORIZED)")
            print(f"{'='*80}")
        
        asset_names = list(self.simulator_objects.keys())
        n_assets = len(asset_names)
        
        # Pre-compute Cholesky decompositions for each regime's correlation matrix
        cholesky_matrices = {}
        for regime_id, corr_matrix in self.regime_correlations.items():
            corr_array = corr_matrix.values
            try:
                cholesky_matrices[regime_id] = np.linalg.cholesky(corr_array)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(corr_array)
                eigenvals = np.maximum(eigenvals, 1e-10)
                cholesky_matrices[regime_id] = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        if verbose:
            print(f"  Assets: {n_assets}")
            print(f"  Simulations: {self.n_simulations:,}")
            print(f"  Days: {self.n_days}")
            print(f"\n  Running vectorized copula simulation...")
        
        # Stack Cholesky matrices for vectorized lookup: shape (4, n_assets, n_assets)
        cholesky_stack = np.stack([cholesky_matrices[r] for r in range(4)])
        
        # Generate ALL independent standard normals at once: shape (n_simulations, n_days, n_assets)
        z_independent = np.random.standard_normal((self.n_simulations, self.n_days, n_assets))
        
        # Get the Cholesky matrix for each (sim, day) based on market regime
        # simulated_market_regimes shape: (n_simulations, n_days)
        # We need to apply the correct Cholesky to each (sim, day)
        
        # Vectorized correlation induction per regime
        # Process by regime to leverage vectorization
        z_correlated = np.zeros((self.n_simulations, self.n_days, n_assets))
        
        for regime in range(4):
            # Find all (sim, day) pairs where market regime == regime
            mask = (self.simulated_market_regimes == regime)
            
            if not mask.any():
                continue
            
            # Get the Cholesky matrix for this regime
            L = cholesky_matrices[regime]
            
            # Extract z values where this regime applies and apply Cholesky
            # z_independent[mask] has shape (n_matches, n_assets)
            z_regime = z_independent[mask]  # shape: (n_matches, n_assets)
            
            # Apply Cholesky: each row gets multiplied by L
            # (n_matches, n_assets) @ (n_assets, n_assets).T = (n_matches, n_assets)
            z_correlated[mask] = z_regime @ L.T
        
        # Transform to uniform [0,1] via standard normal CDF
        u = norm.cdf(z_correlated)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        
        # Initialize output
        simulated_returns = {asset: np.zeros((self.n_simulations, self.n_days)) for asset in asset_names}
        
        # Transform uniforms to returns using asset-specific, regime-specific distributions
        # Vectorize by (asset, regime) combinations
        for asset_idx, asset in enumerate(asset_names):
            sim_obj = self.simulator_objects[asset]
            asset_regimes = self.simulated_asset_regimes[asset]  # shape: (n_simulations, n_days)
            u_asset = u[:, :, asset_idx]  # shape: (n_simulations, n_days)
            
            # Process each regime separately for vectorization
            for regime in range(4):
                mask = (asset_regimes == regime)
                
                if not mask.any():
                    continue
                
                # Get distribution object for this regime
                dist_obj = sim_obj.fitted_distributions[regime]['dist_obj']
                
                # Vectorized inverse CDF (ppf)
                u_regime = u_asset[mask]
                simulated_returns[asset][mask] = dist_obj.ppf(u_regime)
        
        self.simulated_returns = simulated_returns
        
        if verbose:
            print(f"\n  Return statistics:")
            for asset in asset_names:
                returns = simulated_returns[asset]
                mean_ret = returns.mean() * 252  # Annualized
                vol = returns.std() * np.sqrt(252)  # Annualized
                print(f"    {asset}: Ann. Return = {mean_ret:.2%}, Ann. Vol = {vol:.2%}")
            
            print(f"\n✓ Returns simulated via Gaussian copula")
            print(f"{'='*80}")
        
        if verbose:
            print(f"\n  Return statistics:")
            for asset in asset_names:
                returns = simulated_returns[asset]
                mean_ret = returns.mean() * 252  # Annualized
                vol = returns.std() * np.sqrt(252)  # Annualized
                print(f"    {asset}: Ann. Return = {mean_ret:.2%}, Ann. Vol = {vol:.2%}")
            
            print(f"\n✓ Returns simulated via Gaussian copula")
            print(f"{'='*80}")

####################################################################################################################################################################################
# PHASE 4: Compute Portfolio Optimization Inputs (mu, Sigma)
####################################################################################################################################################################################

    def compute_portfolio_inputs(
        self,
        method: str = 'path_covariance',
        verbose: bool = True
    ) -> None:
        """
        Compute expected returns (μ) and covariance matrix (Σ) from simulated returns.
        
        All outputs are ANNUALIZED:
        - mu: annualized expected returns
        - Sigma: annualized covariance matrix
        
        Parameters
        ----------
        method : str, default='path_covariance'
            Method for computing covariance:
            - 'terminal': Covariance of terminal cumulative returns across simulations
            - 'daily_avg': Covariance of average daily returns, scaled by horizon
            - 'path_covariance': Average of within-path time-series covariances (recommended)
        verbose : bool, default=True
            Print summary information
            
        Saves
        -----
        self.mu : pd.Series
            Annualized expected returns
        self.Sigma : pd.DataFrame
            Annualized covariance matrix
        self.correlation_matrix : pd.DataFrame
            Correlation matrix (same whether annualized or not)
        """
        if self.simulated_returns is None:
            raise ValueError("Must call simulate_returns_copula() first")
        
        assets = list(self.simulated_returns.keys())
        
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING PORTFOLIO OPTIMIZATION INPUTS")
            print(f"{'='*80}")
            print(f"\n  Method: {method}")
        
        if method == 'terminal':
            # Compute terminal cumulative returns for each simulation
            terminal_returns = {}
            for asset in assets:
                # Shape: (n_simulations, n_days) -> compound to get terminal return
                cum_returns = np.prod(1 + self.simulated_returns[asset], axis=1) - 1
                terminal_returns[asset] = cum_returns
            
            returns_df = pd.DataFrame(terminal_returns)
            # Annualize: scale by 252/n_days for returns, (252/n_days) for variance
            annualization_factor = 252 / self.n_days
            mu = returns_df.mean() * annualization_factor
            Sigma = returns_df.cov() * annualization_factor
            
        elif method == 'daily_avg':
            # Average daily return across days for each simulation
            daily_returns = {}
            for asset in assets:
                # Mean daily return for each simulation
                daily_returns[asset] = self.simulated_returns[asset].mean(axis=1)
            
            returns_df = pd.DataFrame(daily_returns)
            # Annualize: daily mean * 252, daily cov * 252
            mu = returns_df.mean() * 252
            Sigma = returns_df.cov() * 252
            
        elif method == 'path_covariance':
            # Compute covariance for each path, then average across paths
            # This preserves the time-series covariance structure
            
            # First compute terminal returns for mu
            terminal_returns = {}
            for asset in assets:
                cum_returns = np.prod(1 + self.simulated_returns[asset], axis=1) - 1
                terminal_returns[asset] = cum_returns
            
            returns_df = pd.DataFrame(terminal_returns)
            # Annualize mu: scale terminal returns by 252/n_days
            annualization_factor = 252 / self.n_days
            mu = returns_df.mean() * annualization_factor
            
            # Now compute path covariances
            path_covariances = []
            for sim_idx in range(self.n_simulations):
                # Extract daily returns for this simulation path
                path_data = {
                    asset: self.simulated_returns[asset][sim_idx, :]
                    for asset in assets
                }
                path_df = pd.DataFrame(path_data)
                
                # Compute time-series covariance for this path
                path_cov = path_df.cov()
                path_covariances.append(path_cov.values)
            
            # Average covariances across all paths, annualized (daily cov * 252)
            Sigma = np.mean(path_covariances, axis=0) * 252
            Sigma = pd.DataFrame(Sigma, index=assets, columns=assets)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'terminal', 'daily_avg', or 'path_covariance'")
        
        self.mu = mu
        self.Sigma = Sigma
        
        # Compute correlation matrix
        std = np.sqrt(np.diag(Sigma.values))
        self.correlation_matrix = pd.DataFrame(
            Sigma.values / np.outer(std, std),
            index=assets,
            columns=assets
        )
        
        if verbose:
            print(f"\n  Annualized Expected Returns (μ):")
            for asset in assets:
                print(f"    {asset}: {mu[asset]:.4f} ({mu[asset]*100:.2f}%)")
            
            print(f"\n  Annualized Volatility (σ):")
            for asset in assets:
                vol = np.sqrt(Sigma.loc[asset, asset])
                print(f"    {asset}: {vol:.4f} ({vol*100:.2f}%)")
            
            print(f"\n  Correlation Matrix:")
            print(self.correlation_matrix.round(3).to_string())
            print(f"\n✓ Portfolio inputs computed (annualized)")
            print(f"{'='*80}")

####################################################################################################################################################################################
# CONVENIENCE: Full Pipeline
####################################################################################################################################################################################

    def run_full_pipeline(self, verbose: bool = True) -> None:
        """
        Run the complete pipeline from loading to optimization inputs.
        
        Pipeline:
        1. load_simulator_objects() - Load KAMA_MSR models and KMRF predictions
        2. estimate_regime_correlations() - Compute regime-dependent correlations
        3. estimate_regime_concordance() - Compute P(asset_regime | market_regime)
        4. simulate_market_regime_paths() - Simulate market regime evolution
        5. simulate_asset_regime_paths() - Simulate asset regimes conditioned on market
        6. simulate_returns_copula() - Simulate correlated returns via Gaussian copula
        7. compute_portfolio_inputs() - Compute mu and Sigma
        
        Saves
        -----
        self.mu : pd.Series
            Annualized expected returns
        self.Sigma : pd.DataFrame
            Annualized covariance matrix
        """
        if verbose:
            print(f"\n{'#'*80}")
            print(f"# OPTIMIZER INPUTS FULL PIPELINE")
            print(f"# Optimization Date: {self.opt_date.strftime('%Y-%m-%d')}")
            print(f"# Assets: {len(self.asset_names)}")
            print(f"# Simulations: {self.n_simulations:,}")
            print(f"# Horizon: {self.n_days} days")
            print(f"{'#'*80}")
        
        # Phase 1: Load and prepare
        self.load_simulator_objects(verbose=verbose)
        self.get_fwd_regime_probs()
        self.estimate_regime_correlations(verbose=verbose)
        self.estimate_regime_concordance(verbose=verbose)
        
        # Phase 2: Simulate regimes
        self.simulate_market_regime_paths(verbose=verbose)
        self.simulate_asset_regime_paths(verbose=verbose)
        
        # Phase 3: Simulate returns
        self.simulate_returns_copula(verbose=verbose)
        
        # Phase 4: Compute optimization inputs
        self.compute_portfolio_inputs(verbose=verbose)
        
        if verbose:
            print(f"\n{'#'*80}")
            print(f"# PIPELINE COMPLETE")
            print(f"{'#'*80}")

####################################################################################################################################################################################
# PHASE 5: Portfolio Optimization
####################################################################################################################################################################################

    def optimize_portfolio(
        self,
        objective: str = 'max_sharpe',
        allow_short: bool = False,
        gross_exposure: Optional[float] = None,
        risk_aversion: float = 0.5,
        risk_free_rate: float = None,
        verbose: bool = True
    ) -> None:
        """
        Optimize portfolio based on computed mu and Sigma.
        
        Parameters
        ----------
        objective : str, default='max_sharpe'
            Optimization objective:
            - 'max_sharpe': Maximize Sharpe ratio
            - 'max_sortino': Maximize Sortino ratio (requires simulated_returns)
            - 'risk_aversion': Mean-variance with risk aversion parameter
            - 'min_variance': Minimum variance portfolio
        allow_short : bool, default=False
            Whether to allow short selling
        gross_exposure : float, optional
            Maximum gross exposure if shorting allowed (e.g., 1.3 for 130/30)
        risk_aversion : float, default=0.5
            Risk aversion parameter for 'risk_aversion' objective
            Higher values → more risk-averse → lower risk, lower return
        risk_free_rate : float, default=0.0
            Risk-free rate for Sharpe/Sortino ratio calculations (annualized)
        verbose : bool, default=True
            Print optimization results
            
        Saves
        -----
        self.optimal_weights : pd.Series
            Optimal portfolio weights
        self.portfolio_return : float
            Expected portfolio return (annualized)
        self.portfolio_risk : float
            Portfolio volatility (annualized)
        self.sharpe_ratio : float
            Portfolio Sharpe ratio
        self.sortino_ratio : float (if objective='max_sortino')
            Portfolio Sortino ratio
        """
        if self.mu is None or self.Sigma is None:
            raise ValueError("Must run compute_portfolio_inputs() or run_full_pipeline() first")
        
        # Validate inputs
        valid_objectives = ['max_sharpe', 'max_sortino', 'risk_aversion', 'min_variance']
        if objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
        
        if objective == 'max_sortino' and self.simulated_returns is None:
            raise ValueError("simulated_returns required for max_sortino objective")
        
        if gross_exposure is not None and not allow_short:
            raise ValueError("gross_exposure only applies when allow_short=True")
        
        if gross_exposure is not None and gross_exposure <= 1.0:
            raise ValueError("gross_exposure must be > 1.0 (e.g., 1.3 for 130/30)")
        
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        
        # Store optimization parameters for efficient frontier plotting
        self._allow_short = allow_short
        self._gross_exposure = gross_exposure
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PORTFOLIO OPTIMIZATION")
            print(f"{'='*80}")
            print(f"  Objective: {objective}")
            print(f"  Assets: {len(self.asset_names)}")
            print(f"  Short selling: {'Allowed' if allow_short else 'Not allowed'}")
            if allow_short and gross_exposure:
                print(f"  Gross exposure limit: {gross_exposure:.1%}")
            if objective == 'risk_aversion':
                print(f"  Risk aversion: {risk_aversion}")
        
        # Run optimization based on objective
        if objective == 'max_sharpe':
            self._optimize_max_sharpe(allow_short, gross_exposure)
        elif objective == 'max_sortino':
            self._optimize_max_sortino(allow_short, gross_exposure)
        elif objective == 'risk_aversion':
            self._optimize_risk_aversion(risk_aversion, allow_short, gross_exposure)
        elif objective == 'min_variance':
            self._optimize_min_variance(allow_short, gross_exposure)
        
        # Compute portfolio statistics
        self._compute_portfolio_stats()
        
        if verbose:
            print(f"\n  Optimization Results:")
            print(f"    Portfolio Return: {self.portfolio_return:.2%}")
            print(f"    Portfolio Risk: {self.portfolio_risk:.2%}")
            print(f"    Sharpe Ratio: {self.sharpe_ratio:.3f}")
            if hasattr(self, 'sortino_ratio') and self.sortino_ratio is not None:
                print(f"    Sortino Ratio: {self.sortino_ratio:.3f}")
            
            print(f"\n  Portfolio Weights:")
            sorted_weights = self.optimal_weights.sort_values(ascending=False)
            for asset, weight in sorted_weights.items():
                if abs(weight) > 0.001:
                    print(f"    {asset}: {weight:>8.2%}")
            
            print(f"\n✓ Portfolio optimized")
            print(f"{'='*80}")

    def _optimize_max_sharpe(self, allow_short: bool, gross_exposure: Optional[float]) -> None:
        """Maximize Sharpe ratio using SLSQP with multiple starting points."""
        from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
        
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values.copy()
        n_assets = len(self.asset_names)
        
        # Regularize covariance matrix to improve conditioning
        min_eigenvalue = np.min(np.linalg.eigvalsh(Sigma_array))
        if min_eigenvalue < 1e-8:
            reg_factor = 1e-6 - min_eigenvalue + 1e-8
            Sigma_array = Sigma_array + reg_factor * np.eye(n_assets)
        
        def negative_sharpe(w):
            """Negative Sharpe ratio (to minimize)."""
            ret = mu_array @ w - self.risk_free_rate
            var = w @ Sigma_array @ w
            risk = np.sqrt(max(var, 1e-10))
            return -ret / risk if risk > 1e-8 else 1e10
        
        def sharpe_gradient(w):
            """Gradient of negative Sharpe ratio."""
            ret = mu_array @ w - self.risk_free_rate
            var = w @ Sigma_array @ w
            risk = np.sqrt(max(var, 1e-10))
            
            if risk < 1e-8:
                return np.zeros(n_assets)
            
            return -(mu_array / risk - ret * (Sigma_array @ w) / (risk ** 3))
        
        # Constraints
        constraints = [LinearConstraint(np.ones(n_assets), lb=1.0, ub=1.0)]
        
        # Bounds
        if not allow_short:
            bounds = Bounds(lb=0, ub=1)
        else:
            max_weight = gross_exposure if gross_exposure else 2.0
            bounds = Bounds(lb=-max_weight, ub=max_weight)
            if gross_exposure is not None:
                constraints.append(NonlinearConstraint(
                    lambda w: np.sum(np.abs(w)), lb=0, ub=gross_exposure + 1e-6
                ))
        
        # Try multiple random starting points
        best_result = None
        best_sharpe = -np.inf
        
        np.random.seed(self.random_seed + 100)
        n_tries = 20  # Increased from 10
        
        for i in range(n_tries):
            if i == 0:
                # Equal weight start
                w0 = np.ones(n_assets) / n_assets
            elif i == 1:
                # Max return asset (concentrated start)
                w0 = np.zeros(n_assets)
                w0[np.argmax(mu_array)] = 1.0
            elif i == 2:
                # Min variance approximation
                diag_inv = 1.0 / np.diag(Sigma_array)
                w0 = diag_inv / np.sum(diag_inv)
            else:
                # Random starts
                w0 = np.random.randn(n_assets)
                w0 = w0 / np.sum(w0)
                if not allow_short:
                    w0 = np.abs(w0) / np.sum(np.abs(w0))
            
            try:
                result = minimize(
                    negative_sharpe, w0, method='SLSQP', jac=sharpe_gradient,
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8}
                )
                
                if result.success:
                    sharpe = -result.fun
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = result
            except Exception:
                continue  # Skip failed attempts
        
        if best_result is None:
            # Fallback to convex risk-aversion optimization that approximates max Sharpe
            # Use risk_aversion ≈ (market Sharpe)^2 which is typically around 0.5-2.0
            # A value of 1.0 works well as a general approximation
            import warnings
            warnings.warn("Sharpe optimization failed to converge, falling back to convex risk-aversion optimization")
            try:
                self._optimize_risk_aversion(
                    risk_aversion=1.0,
                    allow_short=allow_short,
                    gross_exposure=gross_exposure
                )
            except Exception as e:
                # Last resort: equal weights
                warnings.warn(f"Risk-aversion fallback also failed ({e}), using equal weights")
                self.optimal_weights = pd.Series(np.ones(n_assets) / n_assets, index=self.asset_names)
        else:
            self.optimal_weights = pd.Series(best_result.x, index=self.asset_names)

    def _optimize_max_sortino(self, allow_short: bool, gross_exposure: Optional[float]) -> None:
        """Maximize Sortino ratio using SLSQP with multiple starting points."""
        from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
        
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values
        n_assets = len(self.asset_names)
        assets = list(self.simulated_returns.keys())
        
        def compute_downside_deviation(w):
            """Compute downside deviation for portfolio with weights w."""
            portfolio_returns = np.zeros_like(self.simulated_returns[assets[0]])
            for i, asset in enumerate(assets):
                portfolio_returns += w[i] * self.simulated_returns[asset]
            
            downside_returns = np.minimum(portfolio_returns, 0)
            return np.sqrt(np.mean(downside_returns ** 2) + 1e-10) * np.sqrt(252)  # Annualized
        
        def negative_sortino(w):
            """Negative Sortino ratio (to minimize)."""
            ret = mu_array @ w - self.risk_free_rate
            downside_dev = compute_downside_deviation(w)
            return -ret / downside_dev if downside_dev > 1e-8 else 1e10
        
        # Constraints
        constraints = [LinearConstraint(np.ones(n_assets), lb=1.0, ub=1.0)]
        
        if not allow_short:
            bounds = Bounds(lb=0, ub=1)
        else:
            max_weight = gross_exposure if gross_exposure else 2.0
            bounds = Bounds(lb=-max_weight, ub=max_weight)
            if gross_exposure is not None:
                constraints.append(NonlinearConstraint(
                    lambda w: np.sum(np.abs(w)), lb=0, ub=gross_exposure + 1e-6
                ))
        
        # Try multiple starting points
        best_result = None
        best_sortino = -np.inf
        
        np.random.seed(self.random_seed + 100)
        n_tries = 20  # Increased from 10
        
        for i in range(n_tries):
            if i == 0:
                # Equal weight start
                w0 = np.ones(n_assets) / n_assets
            elif i == 1:
                # Max return asset (concentrated start)
                w0 = np.zeros(n_assets)
                w0[np.argmax(mu_array)] = 1.0
            elif i == 2:
                # Min variance approximation
                diag_inv = 1.0 / np.diag(Sigma_array)
                w0 = diag_inv / np.sum(diag_inv)
            else:
                # Random starts
                w0 = np.random.randn(n_assets)
                w0 = w0 / np.sum(w0)
                if not allow_short:
                    w0 = np.abs(w0) / np.sum(np.abs(w0))
            
            try:
                result = minimize(
                    negative_sortino, w0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8}
                )
                
                if result.success:
                    sortino = -result.fun
                    if sortino > best_sortino:
                        best_sortino = sortino
                        best_result = result
            except Exception:
                continue  # Skip failed attempts
        
        if best_result is None:
            # Fallback to convex risk-aversion optimization that approximates max Sortino
            # Sortino and Sharpe optimal portfolios are similar, so same risk_aversion works
            import warnings
            warnings.warn("Sortino optimization failed to converge, falling back to convex risk-aversion optimization")
            try:
                self._optimize_risk_aversion(
                    risk_aversion=1.0,
                    allow_short=allow_short,
                    gross_exposure=gross_exposure
                )
            except Exception as e:
                # Last resort: equal weights
                warnings.warn(f"Risk-aversion fallback also failed ({e}), using equal weights")
                n_assets = len(self.asset_names)
                self.optimal_weights = pd.Series(np.ones(n_assets) / n_assets, index=self.asset_names)
        else:
            self.optimal_weights = pd.Series(best_result.x, index=self.asset_names)

    def _optimize_risk_aversion(
        self, risk_aversion: float, allow_short: bool, gross_exposure: Optional[float]
    ) -> None:
        """Mean-variance optimization with risk aversion parameter (convex)."""
        import cvxpy as cp
        
        n_assets = len(self.asset_names)
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values
        
        w = cp.Variable(n_assets)
        
        # Objective: maximize return - gamma * variance
        portfolio_return = mu_array @ w
        portfolio_variance = cp.quad_form(w, Sigma_array)
        objective = cp.Minimize(risk_aversion * portfolio_variance - portfolio_return)
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        if not allow_short:
            constraints.append(w >= 0)
        if allow_short and gross_exposure is not None:
            constraints.append(cp.norm(w, 1) <= gross_exposure)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        self.optimal_weights = pd.Series(w.value, index=self.asset_names)

    def _optimize_min_variance(self, allow_short: bool, gross_exposure: Optional[float]) -> None:
        """Minimum variance portfolio (convex)."""
        import cvxpy as cp
        
        n_assets = len(self.asset_names)
        Sigma_array = self.Sigma.values
        
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, Sigma_array)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        if not allow_short:
            constraints.append(w >= 0)
        if allow_short and gross_exposure is not None:
            constraints.append(cp.norm(w, 1) <= gross_exposure)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        self.optimal_weights = pd.Series(w.value, index=self.asset_names)

    def _compute_portfolio_stats(self) -> None:
        """Compute portfolio statistics after optimization."""
        self.portfolio_return = float(self.mu @ self.optimal_weights)
        self.portfolio_risk = float(np.sqrt(self.optimal_weights @ self.Sigma @ self.optimal_weights))
        
        excess_return = self.portfolio_return - self.risk_free_rate
        self.sharpe_ratio = excess_return / self.portfolio_risk if self.portfolio_risk > 0 else 0
        
        # Compute Sortino ratio if simulated returns available
        if self.simulated_returns is not None:
            assets = list(self.simulated_returns.keys())
            portfolio_returns = np.zeros_like(self.simulated_returns[assets[0]])
            for i, asset in enumerate(assets):
                portfolio_returns += self.optimal_weights.iloc[i] * self.simulated_returns[asset]
            
            downside_returns = np.minimum(portfolio_returns, 0)
            downside_dev = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)  # Annualized
            self.sortino_ratio = excess_return / downside_dev if downside_dev > 1e-8 else 0
        else:
            self.sortino_ratio = None

    def portfolio_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of the optimal portfolio.
        
        Returns
        -------
        pd.DataFrame
            Summary with weights, returns contribution, and risk contribution
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimize_portfolio() first")
        
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
            'Risk Contribution %': risk_contribution / risk_contribution.sum()
        })
        
        # Sort by absolute weight
        summary = summary.reindex(summary['Weight'].abs().sort_values(ascending=False).index)
        
        return summary.round(4).applymap(lambda x: '' if abs(x) < 1e-4 else x)

    def portfolio_statistics(self) -> Dict[str, float]:
        """
        Get portfolio-level statistics.
        
        Returns
        -------
        dict
            Portfolio return, risk, Sharpe ratio, and other metrics
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimize_portfolio() first")
        
        stats = {
            'Portfolio Return': self.portfolio_return,
            'Portfolio Risk': self.portfolio_risk,
            'Sharpe Ratio': self.sharpe_ratio,
            'Risk-Free Rate': self.risk_free_rate,
            'Number of Assets': len(self.asset_names),
            'Number of Positions': (self.optimal_weights.abs() > 1e-4).sum(),
            'Long Positions': (self.optimal_weights > 1e-4).sum(),
            'Short Positions': (self.optimal_weights < -1e-4).sum(),
            'Gross Exposure': self.optimal_weights.abs().sum(),
            'Net Exposure': self.optimal_weights.sum(),
            'Max Long Position': self.optimal_weights.max(),
            'Max Short Position': self.optimal_weights.min(),
            'Effective N': 1 / (self.optimal_weights ** 2).sum()  # Diversification ratio
        }
        
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
            raise ValueError("Must run optimize_portfolio() first")
        
        weights = self.optimal_weights.copy()
        
        if top_n is not None:
            top_assets = weights.abs().nlargest(top_n).index
            weights = weights[top_assets]
        
        weights = weights.sort_values()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red' if w < 0 else 'green' for w in weights]
        weights.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        
        ax.set_xlabel('Weight', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        ax.set_title('Optimal Portfolio Weights', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (asset, weight) in enumerate(weights.items()):
            ax.text(weight, i, f' {weight:.3f}', 
                   va='center', ha='left' if weight > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        return fig, ax

    def plot_efficient_frontier(
        self, 
        n_points: int = 50,
        figsize: tuple = (10, 7),
        show_assets: bool = True,
        show_optimal: bool = True,
        allow_short: Optional[bool] = None,
        gross_exposure: Optional[float] = None
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
        allow_short : bool, optional
            Whether to allow short selling. If None, uses the setting from optimize_portfolio()
        gross_exposure : float, optional
            Maximum gross exposure. If None, uses the setting from optimize_portfolio()
        """
        # Use stored optimization parameters if not specified
        if allow_short is None:
            allow_short = getattr(self, '_allow_short', False)
        if gross_exposure is None:
            gross_exposure = getattr(self, '_gross_exposure', None)
        
        frontier = self.compute_efficient_frontier(
            n_points=n_points, allow_short=allow_short, gross_exposure=gross_exposure
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(frontier['Risk'], frontier['Return'], 
               'b-', linewidth=2, label='Efficient Frontier')
        
        if show_assets:
            asset_risks = np.sqrt(np.diag(self.Sigma))
            ax.scatter(asset_risks, self.mu, 
                      c='gray', marker='o', s=100, alpha=0.6, label='Individual Assets')
            
            for asset, ret, risk in zip(self.asset_names, self.mu, asset_risks):
                ax.annotate(asset, (risk, ret), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        if show_optimal and self.optimal_weights is not None:
            ax.scatter(self.portfolio_risk, self.portfolio_return,
                      c='red', marker='*', s=500, 
                      label=f'Optimal Portfolio (SR={self.sharpe_ratio:.2f})',
                      edgecolors='black', linewidths=1.5, zorder=5)
        
        ax.set_xlabel('Risk (Annualized Volatility)', fontsize=12)
        ax.set_ylabel('Expected Return (Annualized)', fontsize=12)
        ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def compute_efficient_frontier(
        self, 
        n_points: int = 50,
        allow_short: bool = False,
        gross_exposure: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.
        
        Parameters
        ----------
        n_points : int, default=50
            Number of points on the frontier
        allow_short : bool, default=False
            Whether to allow short selling
        gross_exposure : float, optional
            Maximum gross exposure if shorting allowed
            
        Returns
        -------
        pd.DataFrame
            Frontier points with returns, risks, and Sharpe ratios
        """
        import cvxpy as cp
        
        mu_array = self.mu.values
        Sigma_array = self.Sigma.values
        n_assets = len(self.asset_names)
        
        # Find return range
        # Minimum variance portfolio return
        w_minvar = cp.Variable(n_assets)
        obj_minvar = cp.Minimize(cp.quad_form(w_minvar, Sigma_array))
        constraints_minvar = [cp.sum(w_minvar) == 1]
        if not allow_short:
            constraints_minvar.append(w_minvar >= 0)
        if allow_short and gross_exposure is not None:
            constraints_minvar.append(cp.norm(w_minvar, 1) <= gross_exposure)
        cp.Problem(obj_minvar, constraints_minvar).solve()
        min_return = float(mu_array @ w_minvar.value)
        
        # Maximum return portfolio
        w_maxret = cp.Variable(n_assets)
        obj_maxret = cp.Maximize(mu_array @ w_maxret)
        constraints_maxret = [cp.sum(w_maxret) == 1]
        if not allow_short:
            constraints_maxret.append(w_maxret >= 0)
        if allow_short and gross_exposure is not None:
            constraints_maxret.append(cp.norm(w_maxret, 1) <= gross_exposure)
        cp.Problem(obj_maxret, constraints_maxret).solve()
        max_return = float(mu_array @ w_maxret.value)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_risks = []
        frontier_returns = []
        
        for target_ret in target_returns:
            w = cp.Variable(n_assets)
            
            obj = cp.Minimize(cp.quad_form(w, Sigma_array))
            constraints = [
                cp.sum(w) == 1,
                mu_array @ w >= target_ret
            ]
            
            if not allow_short:
                constraints.append(w >= 0)
            if allow_short and gross_exposure is not None:
                constraints.append(cp.norm(w, 1) <= gross_exposure)
            
            prob = cp.Problem(obj, constraints)
            prob.solve()
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                risk = np.sqrt(prob.value)
                frontier_risks.append(risk)
                frontier_returns.append(target_ret)
        
        frontier_df = pd.DataFrame({
            'Return': frontier_returns,
            'Risk': frontier_risks,
            'Sharpe': (np.array(frontier_returns) - self.risk_free_rate) / np.array(frontier_risks)
        })
        
        return frontier_df