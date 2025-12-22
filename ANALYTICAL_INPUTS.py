"""
ANALYTICAL_INPUTS.py

Analytical approach for computing portfolio optimization inputs (μ, Σ) using the 
"Universe Regime" methodology. This replaces Monte Carlo simulation with closed-form
computations based on regime-conditional statistics.

Key Concepts:
- "Universe Regime": A single market state (M=4 regimes) derived by averaging 
  individual asset regime probabilities, reducing dimensionality from N×N to N×M
- Uses Law of Total Variance for covariance computation
- Blends regime-specific correlation matrices weighted by Universe probabilities

Mathematical Framework:
----------------------
1. Universe Probabilities: π_m = (1/N) Σᵢ πᵢ,ₘ  for each regime m
   where πᵢ,ₘ is asset i's probability of being in regime m

2. Expected Returns: μᵢ = Σₘ πᵢ,ₘ × E[rᵢ | regime=m]
   Using individual asset probabilities

3. Expected Volatility: σᵢ² = Σₘ πᵢ,ₘ × Var[rᵢ | regime=m]
   Using individual asset probabilities

4. Blended Correlation: ρ̄ᵢⱼ = Σₘ π̄ₘ × ρᵢⱼ,ₘ
   Using Universe probabilities for the correlation blending

5. Intermediate Covariance: Σ̃ᵢⱼ = σᵢ × σⱼ × ρ̄ᵢⱼ
   Scale blended correlation by individual expected volatilities

6. Law of Total Variance Adjustment:
   Σᵢⱼ = Σ̃ᵢⱼ + Cov(E[rᵢ|m], E[rⱼ|m])
   
   The "Covariance of Means" captures additional covariance from regime switching:
   Cov(E[rᵢ|m], E[rⱼ|m]) = Σₘ π̄ₘ × (μᵢ,ₘ - μᵢ)(μⱼ,ₘ - μⱼ)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

from MODEL_INFO import MODEL_INFO, get_KM_model_dates, get_KM_model_paths, get_asset_names, get_KMRF_prediction_paths

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

TRADING_DAYS = CustomBusinessDay(calendar=USFederalHolidayCalendar())

warnings.filterwarnings('ignore')


class ANALYTICAL_INPUTS:
    """
    Compute portfolio optimization inputs (μ, Σ) analytically using Universe Regime approach.
    
    This class provides a closed-form alternative to Monte Carlo simulation, computing
    expected returns and covariance matrices using:
    1. Pre-computed regime-conditional statistics (mean, variance, correlation)
    2. KMRF-based forward regime probabilities
    3. Law of Total Variance for proper covariance accounting
    
    Parameters
    ----------
    opt_date : str or pd.Timestamp
        Optimization/rebalancing date (YYYYMMDD format)
    asset_list : List[str]
        List of asset names to include in portfolio
    n_days : int, default=21
        Forecast horizon in trading days (for selecting KMRF predictions)
    universe_method : str, default='democracy'
        Method for defining Universe regime:
        - 'democracy': Average regime probabilities across all portfolio assets
        - 'market': Use a single market asset (e.g., S&P 500) as the regime proxy
    market_asset : str, default='SPDR S&P 500 ETF'
        Asset to use as market regime indicator (only used when universe_method='market')
    risk_free_rate : float, default=0.0
        Annual risk-free rate for excess return calculations
    annualize : bool, default=True
        If True, return annualized μ and Σ
        
    Attributes
    ----------
    regime_means : Dict[str, np.ndarray]
        Daily mean returns by regime for each asset, shape (4,)
    regime_variances : Dict[str, np.ndarray]
        Daily variances by regime for each asset, shape (4,)
    regime_correlations : Dict[int, pd.DataFrame]
        Correlation matrices for each market regime
    forward_probs : Dict[str, np.ndarray]
        Forward regime probabilities for each asset
    universe_probs : np.ndarray
        Averaged Universe regime probabilities, shape (4,)
    mu : pd.Series
        Expected returns (annualized if annualize=True)
    Sigma : pd.DataFrame
        Covariance matrix (annualized if annualize=True)
    """
    
    # Regime names for display
    REGIME_NAMES = {0: 'LV_Bull', 1: 'LV_Bear', 2: 'HV_Bull', 3: 'HV_Bear'}
    
    def __init__(
        self, 
        opt_date: str | pd.Timestamp, 
        asset_list: List[str],
        n_days: int = 21,
        universe_method: str = 'democracy',
        market_asset: str = 'SPDR S&P 500 ETF',
        risk_free_rate: float = 0.0,
        annualize: bool = True
    ):
        if opt_date is None:
            raise ValueError("opt_date must be provided")
        if not asset_list:
            raise ValueError("asset_list must be provided and cannot be empty")
        if universe_method not in ['democracy', 'market']:
            raise ValueError("universe_method must be 'democracy' or 'market'")
        
        # Configuration
        self.opt_date = pd.Timestamp(opt_date)
        self.n_days = n_days
        self.universe_method = universe_method
        self.market_asset = market_asset
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize
        
        # Data paths
        self.ALL_KM_MODEL_DATES = get_KM_model_dates()
        self.ALL_ASSET_NAMES = get_asset_names()
        
        # Find nearest available model date
        self._model_date = self._find_nearest_model_date(self.opt_date)
        self.ALL_KM_MODEL_PATHS = get_KM_model_paths(MODEL_DATE=self._model_date)
        self.ALL_KMRF_PREDICTION_PATHS = get_KMRF_prediction_paths()
        
        # Asset configuration
        self.asset_names = asset_list
        self.model_info_objects: Dict[str, MODEL_INFO] = {}
        
        # Pre-computed regime statistics (from historical data)
        self.regime_means: Dict[str, np.ndarray] = {}       # Daily means by regime
        self.regime_variances: Dict[str, np.ndarray] = {}   # Daily variances by regime
        self.regime_correlations: Dict[int, pd.DataFrame] = {}  # Correlations by market regime
        
        # Forward-looking probabilities (from KMRF)
        self.forward_probs: Dict[str, np.ndarray] = {}      # Individual asset probs
        self.universe_probs: Optional[np.ndarray] = None    # Averaged Universe probs
        
        # Optimization outputs
        self.mu: Optional[pd.Series] = None
        self.Sigma: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Intermediate calculations (for debugging/analysis)
        self._expected_volatilities: Optional[pd.Series] = None
        self._blended_correlations: Optional[pd.DataFrame] = None
        self._covariance_of_means: Optional[pd.DataFrame] = None
    
    def _find_nearest_model_date(self, target_date: pd.Timestamp) -> str:
        """Find the nearest available model date <= target_date."""
        available_dates = self.ALL_KM_MODEL_DATES.apply(pd.Timestamp)
        valid_dates = available_dates[available_dates <= target_date]
        
        if len(valid_dates) == 0:
            raise ValueError(f"No models available on or before {target_date}")
        
        nearest_date = valid_dates.max()
        return nearest_date.strftime('%Y%m%d')
    
    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        """Normalize probability array to sum to 1, handling edge cases."""
        probs = np.array(probs, dtype=float)
        probs = np.nan_to_num(probs, nan=1e-10)
        probs = np.maximum(probs, 1e-10)
        
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        return probs / total
    
    def _make_positive_definite(self, matrix: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        """Ensure matrix is positive definite using eigenvalue clipping."""
        arr = matrix.values
        eigenvalues, eigenvectors = np.linalg.eigh(arr)
        
        if eigenvalues.min() >= epsilon:
            return matrix
        
        eigenvalues = np.maximum(eigenvalues, epsilon)
        fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # For correlation matrix, normalize diagonal to 1
        if np.allclose(np.diag(matrix.values), 1.0):
            d = np.sqrt(np.diag(fixed))
            fixed = fixed / np.outer(d, d)
        
        return pd.DataFrame(fixed, index=matrix.index, columns=matrix.columns)
    
    def _compute_democracy_regime_labels(self) -> pd.Series:
        """
        Compute Universe regime labels using majority vote across portfolio assets.
        
        For each date, the Universe regime is the mode (most common) regime
        across all portfolio assets' individual regime labels.
        
        Tiebreaker order (if regimes are tied): HV_Bear (3) > HV_Bull (2) > LV_Bear (1) > LV_Bull (0)
        This prioritizes high-volatility regimes as they represent more significant market states.
        
        Returns
        -------
        pd.Series
            Regime labels (0-3) indexed by date
        """
        if not self.model_info_objects:
            raise ValueError("Must call load_model_info_objects() first")
        
        # Collect regime labels for all assets
        regime_labels_dict = {}
        for asset_name, model_info in self.model_info_objects.items():
            regime_labels_dict[asset_name] = model_info.km_model.regime_labels
        
        regime_labels_df = pd.DataFrame(regime_labels_dict)
        
        # Tiebreaker priority: HV_Bear (3) > HV_Bull (2) > LV_Bear (1) > LV_Bull (0)
        tiebreaker_priority = [3, 2, 1, 0]
        
        # Compute mode (majority vote) for each date with explicit tiebreaker
        def get_mode(row):
            counts = row.dropna().value_counts()
            if len(counts) == 0:
                return np.nan
            max_count = counts.max()
            tied_regimes = counts[counts == max_count].index.tolist()
            # Apply tiebreaker: return first regime in priority order that is tied
            for regime in tiebreaker_priority:
                if regime in tied_regimes:
                    return regime
            return tied_regimes[0]  # Fallback (shouldn't reach here)
        
        democracy_labels = regime_labels_df.apply(get_mode, axis=1)
        democracy_labels.name = 'democracy_regime'
        
        return democracy_labels

    ####################################################################################################################
    # PHASE 1: Load Data and Compute Regime Statistics
    ####################################################################################################################
    
    def load_model_info_objects(self, verbose: bool = True):
        """Load MODEL_INFO objects for all assets."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"LOADING MODEL_INFO FOR {len(self.asset_names)} ASSETS")
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
            
            model_info = MODEL_INFO(km_model_path=km_model_path, kmrf_preds_path=kmrf_preds_path)
            
            self.model_info_objects[asset] = model_info
        
        if verbose:
            print(f"\n✓ Successfully loaded {len(self.model_info_objects)}/{len(self.asset_names)} model info objects")
        
        # Ensure market asset is loaded (for correlation estimation)
        self._load_market_model_info(verbose=verbose)
    
    def _load_market_model_info(self, verbose: bool = True):
        """Load market asset MODEL_INFO if not already in portfolio."""
        if self.market_asset in self.model_info_objects:
            self.market_model_info = self.model_info_objects[self.market_asset]
            return
        
        if verbose:
            print(f"\n  Loading market asset '{self.market_asset}' separately...")
        
        if self.market_asset not in self.ALL_ASSET_NAMES.values:
            raise ValueError(f"Market asset '{self.market_asset}' not found")
        
        asset_idx = self.ALL_ASSET_NAMES[self.ALL_ASSET_NAMES == self.market_asset].index[0]
        km_model_path = self.ALL_KM_MODEL_PATHS[asset_idx]
        kmrf_preds_path = self.ALL_KMRF_PREDICTION_PATHS[asset_idx]
        
        market_model_info = MODEL_INFO(km_model_path=km_model_path, kmrf_preds_path=kmrf_preds_path)
        self.market_model_info = market_model_info
    
    def compute_regime_statistics(self, verbose: bool = True, min_observations: int = 30):
        """
        Compute regime-conditional mean and variance for each asset.
        
        Uses historical returns labeled by regime from KAMA_MSR models.
        
        Parameters
        ----------
        verbose : bool
            Print summary statistics
        min_observations : int
            Minimum observations required per regime
        """
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING REGIME-CONDITIONAL STATISTICS")
            print(f"{'='*80}")
        
        for asset_name, model_info in self.model_info_objects.items():
            returns = model_info.km_model.returns
            regime_labels = model_info.km_model.regime_labels
            
            # Align and remove NaN
            aligned = pd.DataFrame({
                'returns': returns,
                'regime': regime_labels
            }).dropna()
            
            means = np.zeros(4)
            variances = np.zeros(4)
            
            # Overall statistics as fallback
            overall_mean = aligned['returns'].mean()
            overall_var = aligned['returns'].var()
            
            if verbose:
                print(f"\n{asset_name}:")
            
            for regime in range(4):
                regime_data = aligned[aligned['regime'] == regime]['returns']
                n_obs = len(regime_data)
                
                if n_obs < min_observations:
                    means[regime] = overall_mean
                    variances[regime] = overall_var
                    if verbose:
                        print(f"  {self.REGIME_NAMES[regime]}: Using overall stats (n={n_obs})")
                else:
                    means[regime] = regime_data.mean()
                    variances[regime] = regime_data.var()
                    if verbose:
                        ann_ret = means[regime] * 252 * 100
                        ann_vol = np.sqrt(variances[regime]) * np.sqrt(252) * 100
                        print(f"  {self.REGIME_NAMES[regime]}: μ={ann_ret:.1f}%, σ={ann_vol:.1f}% (n={n_obs})")
            
            self.regime_means[asset_name] = means
            self.regime_variances[asset_name] = variances
        
        if verbose:
            print(f"\n✓ Regime statistics computed for {len(self.regime_means)} assets")
    
    def estimate_regime_correlations(self, verbose: bool = True, min_observations: int = 30):
        """
        Estimate correlation matrices for each Universe regime.
        
        When universe_method='market': Uses market asset regime labels
        When universe_method='democracy': Uses majority vote across portfolio assets
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"ESTIMATING REGIME-DEPENDENT CORRELATIONS (method={self.universe_method})")
            print(f"{'='*80}")
        
        # Get Universe regime labels based on method
        if self.universe_method == 'market':
            universe_regime_labels = self.market_model_info.km_model.regime_labels
            if verbose:
                print(f"Using market asset '{self.market_asset}' for regime conditioning")
        else:  # democracy
            universe_regime_labels = self._compute_democracy_regime_labels()
            if verbose:
                print(f"Using democracy (majority vote) across {len(self.model_info_objects)} assets")
        
        # Collect returns for all portfolio assets
        asset_returns = {}
        for asset_name, model_info in self.model_info_objects.items():
            asset_returns[asset_name] = model_info.km_model.returns
        
        returns_df = pd.DataFrame(asset_returns)
        
        # Align with universe regime labels
        common_dates = returns_df.index.intersection(universe_regime_labels.index)
        aligned_returns = returns_df.loc[common_dates]
        aligned_regimes = universe_regime_labels.loc[common_dates]
        
        valid_mask = aligned_regimes.notna()
        aligned_returns = aligned_returns[valid_mask]
        aligned_regimes = aligned_regimes[valid_mask]
        
        if verbose:
            print(f"\nAligned observations: {len(aligned_returns)}")
        
        overall_corr = aligned_returns.corr()
        
        for regime in range(4):
            regime_mask = (aligned_regimes == regime)
            regime_data = aligned_returns[regime_mask]
            n_obs = len(regime_data)
            
            if verbose:
                print(f"\nRegime {regime} ({self.REGIME_NAMES[regime]}): {n_obs} observations")
            
            if n_obs < min_observations:
                if verbose:
                    print(f"  ⚠️ Using overall correlation (insufficient data)")
                self.regime_correlations[regime] = overall_corr.copy()
            else:
                regime_corr = regime_data.corr()
                regime_corr = self._make_positive_definite(regime_corr)
                self.regime_correlations[regime] = regime_corr
                
                if verbose:
                    mask = ~np.eye(regime_corr.shape[0], dtype=bool)
                    if mask.any():
                        avg_corr = regime_corr.values[mask].mean()
                        print(f"  Avg off-diagonal correlation: {avg_corr:.3f}")
        
        if verbose:
            print(f"\n✓ Regime correlations estimated")
    
    ####################################################################################################################
    # PHASE 2: Get Forward Regime Probabilities from KMRF
    ####################################################################################################################
    
    def get_forward_regime_probs(self, verbose: bool = True):
        """
        Retrieve forward regime probabilities from KMRF predictions.
        
        Gets the n_days-ahead regime probability forecast for each asset
        as of the optimization date.
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"RETRIEVING FORWARD REGIME PROBABILITIES (horizon={self.n_days} days)")
            print(f"{'='*80}")
        
        for asset_name, model_info in self.model_info_objects.items():
            # Get forward probs from MODEL_INFO (uses KMRF predictions)
            fwd_probs_df = model_info.get_forward_regime_probs(self.opt_date, self.n_days)
            
            # Average across the horizon to get a single probability vector
            # fwd_probs_df has shape (n_days, 4) with columns ['P(LV_Bull)', 'P(LV_Bear)', ...]
            avg_probs = fwd_probs_df.mean(axis=0).values
            avg_probs = self._normalize_probs(avg_probs)
            
            self.forward_probs[asset_name] = avg_probs
            
            if verbose:
                probs_str = ", ".join([f"{self.REGIME_NAMES[i]}: {avg_probs[i]:.3f}" 
                                       for i in range(4)])
                print(f"  {asset_name}: {probs_str}")
        
        if verbose:
            print(f"\n✓ Forward probabilities retrieved for {len(self.forward_probs)} assets")
    
    def compute_universe_probs(self, verbose: bool = True):
        """
        Compute Universe regime probabilities.
        
        When universe_method='democracy': π̄_m = (1/N) Σᵢ πᵢ,ₘ (average across assets)
        When universe_method='market': π̄_m = π_market,m (use market asset probs)
        """
        if not self.forward_probs:
            raise ValueError("Must call get_forward_regime_probs() first")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"UNIVERSE REGIME PROBABILITIES (method={self.universe_method})")
            print(f"{'='*80}")
        
        if self.universe_method == 'market':
            # Use market asset's forward probabilities
            if self.market_asset in self.forward_probs:
                self.universe_probs = self.forward_probs[self.market_asset].copy()
            else:
                # Market asset not in portfolio, need to get its probs separately
                fwd_probs_df = self.market_model_info.get_forward_regime_probs(self.opt_date, self.n_days)
                avg_probs = fwd_probs_df.mean(axis=0).values
                self.universe_probs = self._normalize_probs(avg_probs)
            if verbose:
                print(f"  Using market asset '{self.market_asset}' probabilities")
        else:  # democracy
            # Stack probabilities: shape (n_assets, 4)
            prob_matrix = np.array([self.forward_probs[asset] for asset in self.asset_names 
                                    if asset in self.forward_probs])
            # Average across assets
            self.universe_probs = self._normalize_probs(prob_matrix.mean(axis=0))
            if verbose:
                print(f"  Averaging probabilities across {len(prob_matrix)} assets")
        
        if verbose:
            for i in range(4):
                print(f"  {self.REGIME_NAMES[i]}: {self.universe_probs[i]:.4f} ({self.universe_probs[i]*100:.2f}%)")
            print(f"\n✓ Universe probabilities computed")
    
    ####################################################################################################################
    # PHASE 3: Compute Expected Returns and Volatilities
    ####################################################################################################################
    
    def compute_expected_returns(self, verbose: bool = True):
        """
        Compute expected returns using individual asset probabilities.
        
        μᵢ = Σₘ πᵢ,ₘ × E[rᵢ | regime=m]
        
        Uses individual asset forward probabilities (not Universe) for returns.
        """
        if not self.regime_means or not self.forward_probs:
            raise ValueError("Must compute regime statistics and forward probs first")
        
        expected_returns = {}
        
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING EXPECTED RETURNS")
            print(f"{'='*80}")
        
        for asset in self.asset_names:
            if asset not in self.forward_probs:
                continue
            
            probs = self.forward_probs[asset]  # Individual asset probabilities
            means = self.regime_means[asset]   # Regime-conditional means
            
            # Daily expected return
            daily_mu = np.sum(probs * means)
            
            # Scale by horizon
            horizon_mu = daily_mu * self.n_days
            
            # Annualize if requested
            if self.annualize:
                expected_returns[asset] = daily_mu * 252
            else:
                expected_returns[asset] = horizon_mu
            
            if verbose:
                ann_ret = daily_mu * 252 * 100
                print(f"  {asset}: {ann_ret:.2f}% (annualized)")
        
        self.mu = pd.Series(expected_returns)
        
        if verbose:
            print(f"\n✓ Expected returns computed for {len(self.mu)} assets")
    
    def compute_expected_volatilities(self, verbose: bool = True):
        """
        Compute expected volatilities using individual asset probabilities.
        
        σᵢ² = Σₘ πᵢ,ₘ × Var[rᵢ | regime=m]
        
        Note: This is the "within-regime" variance component. 
        The full variance also includes the "between-regime" component 
        which is handled via the covariance of means adjustment.
        """
        if not self.regime_variances or not self.forward_probs:
            raise ValueError("Must compute regime statistics and forward probs first")
        
        expected_vols = {}
        
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING EXPECTED VOLATILITIES")
            print(f"{'='*80}")
        
        for asset in self.asset_names:
            if asset not in self.forward_probs:
                continue
            
            probs = self.forward_probs[asset]      # Individual asset probabilities
            variances = self.regime_variances[asset]  # Regime-conditional variances
            
            # Daily expected variance (within-regime component)
            daily_var = np.sum(probs * variances)
            daily_vol = np.sqrt(daily_var)
            
            # Annualize if requested
            if self.annualize:
                expected_vols[asset] = daily_vol * np.sqrt(252)
            else:
                expected_vols[asset] = daily_vol * np.sqrt(self.n_days)
            
            if verbose:
                ann_vol = daily_vol * np.sqrt(252) * 100
                print(f"  {asset}: {ann_vol:.2f}% (annualized)")
        
        self._expected_volatilities = pd.Series(expected_vols)
        
        if verbose:
            print(f"\n✓ Expected volatilities computed for {len(self._expected_volatilities)} assets")
    
    ####################################################################################################################
    # PHASE 4: Compute Covariance Matrix
    ####################################################################################################################
    
    def compute_blended_correlations(self, verbose: bool = True):
        """
        Compute blended correlation matrix using Universe probabilities.
        
        ρ̄ᵢⱼ = Σₘ π̄ₘ × ρᵢⱼ,ₘ
        
        Uses Universe probabilities (average of individual probs) to weight
        the regime-specific correlation matrices.
        """
        if self.universe_probs is None or not self.regime_correlations:
            raise ValueError("Must compute universe probs and regime correlations first")
        
        assets = [a for a in self.asset_names if a in self.forward_probs]
        n_assets = len(assets)
        
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING BLENDED CORRELATION MATRIX")
            print(f"{'='*80}")
        
        # Initialize blended correlation
        blended_corr = np.zeros((n_assets, n_assets))
        
        for regime in range(4):
            weight = self.universe_probs[regime]
            regime_corr = self.regime_correlations[regime].loc[assets, assets].values
            blended_corr += weight * regime_corr
            
            if verbose:
                print(f"  Regime {self.REGIME_NAMES[regime]}: weight = {weight:.4f}")
        
        # Ensure valid correlation matrix
        np.fill_diagonal(blended_corr, 1.0)
        blended_corr = (blended_corr + blended_corr.T) / 2  # Ensure symmetry
        
        self._blended_correlations = pd.DataFrame(
            blended_corr, index=assets, columns=assets
        )
        self._blended_correlations = self._make_positive_definite(self._blended_correlations)
        
        if verbose:
            mask = ~np.eye(n_assets, dtype=bool)
            avg_corr = self._blended_correlations.values[mask].mean()
            print(f"\n  Avg off-diagonal correlation: {avg_corr:.3f}")
            print(f"\n✓ Blended correlations computed")
    
    def compute_covariance_of_means(self, verbose: bool = True):
        """
        Compute the "Covariance of Means" component from Law of Total Variance.
        
        Cov(E[rᵢ|m], E[rⱼ|m]) = Σₘ π̄ₘ × (μᵢ,ₘ - μᵢ)(μⱼ,ₘ - μⱼ)
        
        This captures additional covariance arising from assets moving together
        due to regime transitions.
        """
        if self.universe_probs is None or self.mu is None:
            raise ValueError("Must compute universe probs and expected returns first")
        
        assets = [a for a in self.asset_names if a in self.forward_probs]
        n_assets = len(assets)
        
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING COVARIANCE OF MEANS (Law of Total Variance)")
            print(f"{'='*80}")
        
        cov_of_means = np.zeros((n_assets, n_assets))
        
        # Get daily unconditional means
        daily_means = {}
        for asset in assets:
            probs = self.forward_probs[asset]
            regime_means = self.regime_means[asset]
            daily_means[asset] = np.sum(probs * regime_means)
        
        for regime in range(4):
            weight = self.universe_probs[regime]
            
            # Deviations from unconditional mean for each asset
            deviations = np.array([
                self.regime_means[asset][regime] - daily_means[asset]
                for asset in assets
            ])
            
            # Outer product gives the cross-regime covariance contribution
            cov_of_means += weight * np.outer(deviations, deviations)
        
        # Annualize if requested (daily variance × 252)
        if self.annualize:
            cov_of_means *= 252
        else:
            cov_of_means *= self.n_days
        
        self._covariance_of_means = pd.DataFrame(
            cov_of_means, index=assets, columns=assets
        )
        
        if verbose:
            avg_cov = np.abs(cov_of_means[~np.eye(n_assets, dtype=bool)]).mean()
            print(f"  Avg absolute off-diagonal: {avg_cov:.6f}")
            diag_contribution = np.diag(cov_of_means).mean()
            print(f"  Avg diagonal contribution: {diag_contribution:.6f}")
            print(f"\n✓ Covariance of means computed")
    
    def compute_covariance_matrix(self, verbose: bool = True):
        """
        Compute final covariance matrix combining all components.
        
        Σᵢⱼ = σᵢ × σⱼ × ρ̄ᵢⱼ + Cov(E[rᵢ|m], E[rⱼ|m])
        
        This is the Law of Total Covariance:
        Cov(X,Y) = E[Cov(X,Y|Z)] + Cov(E[X|Z], E[Y|Z])
        
        Where:
        - E[Cov(X,Y|Z)] ≈ σᵢ × σⱼ × ρ̄ᵢⱼ (within-regime covariance)
        - Cov(E[X|Z], E[Y|Z]) = covariance of means (between-regime)
        """
        if self._expected_volatilities is None or self._blended_correlations is None:
            raise ValueError("Must compute volatilities and blended correlations first")
        
        assets = list(self._expected_volatilities.index)
        n_assets = len(assets)
        
        if verbose:
            print(f"\n{'='*80}")
            print("COMPUTING FINAL COVARIANCE MATRIX")
            print(f"{'='*80}")
        
        vols = self._expected_volatilities.values
        corr = self._blended_correlations.values
        
        # Within-regime covariance: Σ̃ᵢⱼ = σᵢ × σⱼ × ρ̄ᵢⱼ
        within_regime_cov = np.outer(vols, vols) * corr
        
        if verbose:
            print(f"  Within-regime covariance: computed")
        
        # Add covariance of means (between-regime component)
        if self._covariance_of_means is not None:
            between_regime_cov = self._covariance_of_means.values
            final_cov = within_regime_cov + between_regime_cov
            if verbose:
                print(f"  Between-regime covariance: added")
        else:
            final_cov = within_regime_cov
            if verbose:
                print(f"  Between-regime covariance: skipped")
        
        # Ensure symmetry and positive definiteness
        final_cov = (final_cov + final_cov.T) / 2
        
        self.Sigma = pd.DataFrame(final_cov, index=assets, columns=assets)
        self.Sigma = self._make_positive_definite(self.Sigma)
        
        # Compute correlation matrix from final covariance
        stds = np.sqrt(np.diag(self.Sigma.values))
        self.correlation_matrix = pd.DataFrame(
            self.Sigma.values / np.outer(stds, stds),
            index=assets, columns=assets
        )
        
        if verbose:
            print(f"\n  Final Volatilities:")
            for asset in assets:
                vol = np.sqrt(self.Sigma.loc[asset, asset]) * 100
                print(f"    {asset}: {vol:.2f}%")
            
            print(f"\n  Final Correlation Matrix:")
            print(self.correlation_matrix.round(3).to_string())
            print(f"\n✓ Covariance matrix computed")
    
    ####################################################################################################################
    # FULL PIPELINE
    ####################################################################################################################
    
    def run_full_pipeline(self, verbose: bool = True):
        """
        Run the complete analytical pipeline.
        
        Steps:
        1. Load SIMULATOR objects
        2. Compute regime-conditional statistics (mean, variance)
        3. Estimate regime-dependent correlations
        4. Get forward regime probabilities from KMRF
        5. Compute Universe probabilities
        6. Compute expected returns (using individual probs)
        7. Compute expected volatilities (using individual probs)
        8. Compute blended correlations (using Universe probs)
        9. Compute covariance of means (Law of Total Variance)
        10. Compute final covariance matrix
        """
        if verbose:
            print(f"\n{'#'*80}")
            print(f"# ANALYTICAL INPUTS PIPELINE (Universe Regime Method)")
            print(f"# Optimization Date: {self.opt_date.strftime('%Y-%m-%d')}")
            print(f"# Assets: {len(self.asset_names)}")
            print(f"# Horizon: {self.n_days} days")
            print(f"# Universe Method: {self.universe_method}")
            print(f"# Annualized: {self.annualize}")
            print(f"{'#'*80}")
        
        # Phase 1: Load and compute historical statistics
        self.load_model_info_objects(verbose=verbose)
        self.compute_regime_statistics(verbose=verbose)
        self.estimate_regime_correlations(verbose=verbose)
        
        # Phase 2: Get forward-looking probabilities
        self.get_forward_regime_probs(verbose=verbose)
        self.compute_universe_probs(verbose=verbose)
        
        # Phase 3: Compute expected returns and volatilities
        self.compute_expected_returns(verbose=verbose)
        self.compute_expected_volatilities(verbose=verbose)
        
        # Phase 4: Compute covariance matrix
        self.compute_blended_correlations(verbose=verbose)
        self.compute_covariance_of_means(verbose=verbose)
        self.compute_covariance_matrix(verbose=verbose)
        
        if verbose:
            print(f"\n{'#'*80}")
            print(f"# PIPELINE COMPLETE")
            print(f"# μ shape: {self.mu.shape}")
            print(f"# Σ shape: {self.Sigma.shape}")
            print(f"{'#'*80}")
    
    ####################################################################################################################
    # ANALYSIS AND COMPARISON UTILITIES
    ####################################################################################################################
    
    def get_sharpe_ratios(self) -> pd.Series:
        """Compute Sharpe ratios for each asset."""
        if self.mu is None or self.Sigma is None:
            raise ValueError("Must run pipeline first")
        
        vols = np.sqrt(np.diag(self.Sigma.values))
        excess_returns = self.mu - self.risk_free_rate
        
        return pd.Series(excess_returns.values / vols, index=self.mu.index, name='Sharpe Ratio')
    
    def summary_df(self) -> pd.DataFrame:
        """Create summary DataFrame of key statistics."""
        if self.mu is None or self.Sigma is None:
            raise ValueError("Must run pipeline first")
        
        vols = pd.Series(np.sqrt(np.diag(self.Sigma.values)), 
                        index=self.Sigma.index, name='Volatility')
        sharpe = self.get_sharpe_ratios()
        
        summary = pd.DataFrame({
            'Expected Return': self.mu,
            'Volatility': vols,
            'Sharpe Ratio': sharpe
        })
        
        return summary
    
