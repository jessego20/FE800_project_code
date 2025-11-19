import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
import contextlib
import io
from scipy import stats as scipy_stats
from scipy.stats import norm, skewnorm, t, norminvgauss

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
        retrain_kmrf: bool = True,
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
        
        # Phase 2: Market regime attributes
        # Always use S&P 500 as market regime indicator
        self.market_regime_asset = 'SPDR S&P 500 ETF'
        self.market_regime_asset_class = 'us_equity'
        self.market_regime_labels: Optional[pd.Series] = None
        self.regime_correlations: Optional[Dict[int, pd.DataFrame]] = None
        self.regime_concordance: Optional[Dict[str, np.ndarray]] = None  # P(asset_regime | market_regime)
        
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
        # For universe asset class, files are named: "{ticker} - {asset_name}_KAMA-MSR_4-regimes.pkl"
        # For other asset classes (us_equity, etc), files are: "{asset_name}_KAMA-MSR_4-regimes.pkl"
        model_dir = self.models_base_path / 'KAMA_MSR' / self.asset_class / self.end_date
        
        # Try exact match first (for us_equity, us_treasury, etc.)
        exact_filename = f"{asset_name}_KAMA-MSR_4-regimes.pkl"
        exact_path = model_dir / exact_filename
        if exact_path.exists():
            return exact_path
        
        # For universe asset class, use glob to find ticker-prefixed file
        pattern = f"*{asset_name}_KAMA-MSR_4-regimes.pkl"
        matching_files = list(model_dir.glob(pattern))
        
        if len(matching_files) == 1:
            return matching_files[0]
        elif len(matching_files) > 1:
            raise ValueError(f"Multiple KAMA+MSR model files found for {asset_name}: {[f.name for f in matching_files]}")
        else:
            raise FileNotFoundError(f"KAMA+MSR model not found for {asset_name} in {model_dir}")
    
    def _get_kmrf_path(self, asset_name: str) -> Path:
        """Get path to KMRF model file."""
        # KMRF models are typically stored by asset class, not date
        # Keep asset name as-is, no space replacement
        filename = f"{asset_name}_KMRF_model.pkl"
        return self.models_base_path / 'KMRF_new' / 'original' / self.asset_class / filename
    
    # ========================================================================
    # Phase 1: Copula Helper Functions
    # ========================================================================
    
    def _inverse_cdf(self, u: float, dist_params: Dict) -> float:
        """
        Sample from distribution using inverse CDF (percent point function).
        
        Used in Gaussian copula to transform uniform samples to distribution-specific
        samples while preserving correlation structure.
        
        Parameters
        ----------
        u : float
            Uniform [0, 1] sample from copula
        dist_params : dict
            Distribution parameters from BayesianForwardSimulator
            Must contain 'distribution' key and 'params' tuple
            
        Returns
        -------
        float
            Sample from the specified distribution
            
        Raises
        ------
        ValueError
            If u not in [0, 1], distribution type unknown, or parameters invalid
            
        Examples
        --------
        >>> dist_params = {'distribution': 'normal', 'params': (0.001, 0.02)}
        >>> sample = self._inverse_cdf(0.5, dist_params)  # Returns median
        """
        # Validate input
        if not (0 <= u <= 1):
            raise ValueError(f"Uniform sample u must be in [0, 1], got {u}")
        
        dist_type = dist_params.get('distribution')
        params = dist_params.get('params')
        
        if dist_type is None or params is None:
            raise ValueError(f"dist_params must contain 'distribution' and 'params' keys")
        
        # Handle edge cases to avoid numerical issues
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
        
        try:
            if dist_type == 'normal':
                loc, scale = params
                return norm.ppf(u_clipped, loc=loc, scale=scale)
            
            elif dist_type == 'skewnorm':
                a, loc, scale = params
                return skewnorm.ppf(u_clipped, a, loc=loc, scale=scale)
            
            elif dist_type == 'student_t':
                df, loc, scale = params
                return t.ppf(u_clipped, df, loc=loc, scale=scale)
            
            elif dist_type == 'norminvgauss':
                a, b, loc, scale = params
                return norminvgauss.ppf(u_clipped, a, b, loc=loc, scale=scale)
            
            elif dist_type == 'empirical':
                # For empirical distributions, use quantile function
                # params should contain the return data
                if isinstance(params, tuple) and len(params) > 0:
                    returns = params[0]
                    return np.quantile(returns, u_clipped)
                else:
                    raise ValueError(f"Empirical distribution params must contain return data")
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
                
        except Exception as e:
            raise ValueError(
                f"Failed to compute inverse CDF for {dist_type} with u={u}: {str(e)}"
            )
    
    def _validate_correlation_matrix(self, corr: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that a matrix is a valid correlation matrix.
        
        Checks:
        1. Matrix is square
        2. Matrix is symmetric
        3. Diagonal elements are all 1.0
        4. All elements in [-1, 1]
        5. Matrix is positive semi-definite
        
        Parameters
        ----------
        corr : pd.DataFrame
            Correlation matrix to validate
            
        Returns
        -------
        tuple
            (is_valid, list_of_issues)
            is_valid: bool indicating if matrix is valid
            list_of_issues: list of strings describing any issues found
            
        Examples
        --------
        >>> valid, issues = self._validate_correlation_matrix(corr_matrix)
        >>> if not valid:
        ...     print(f"Correlation matrix issues: {issues}")
        """
        issues = []
        
        # Check if square
        if corr.shape[0] != corr.shape[1]:
            issues.append(f"Matrix not square: shape {corr.shape}")
            return False, issues
        
        # Check symmetry
        if not np.allclose(corr.values, corr.values.T, atol=1e-8):
            max_diff = np.abs(corr.values - corr.values.T).max()
            issues.append(f"Matrix not symmetric: max difference = {max_diff:.2e}")
        
        # Check diagonal is all 1s
        diag_vals = np.diag(corr.values)
        if not np.allclose(diag_vals, 1.0, atol=1e-6):
            non_one = diag_vals[~np.isclose(diag_vals, 1.0)]
            issues.append(f"Diagonal not all 1.0: found values {non_one}")
        
        # Check all values in [-1, 1]
        if (corr.values < -1).any() or (corr.values > 1).any():
            min_val, max_val = corr.values.min(), corr.values.max()
            issues.append(f"Values outside [-1, 1]: range [{min_val:.4f}, {max_val:.4f}]")
        
        # Check positive semi-definite (all eigenvalues >= 0)
        try:
            eigenvalues = np.linalg.eigvalsh(corr.values)
            min_eigenval = eigenvalues.min()
            
            if min_eigenval < -1e-8:  # Allow small numerical errors
                issues.append(
                    f"Matrix not positive semi-definite: "
                    f"minimum eigenvalue = {min_eigenval:.2e}"
                )
        except np.linalg.LinAlgError as e:
            issues.append(f"Could not compute eigenvalues: {str(e)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _make_positive_definite(
        self, 
        corr: pd.DataFrame, 
        method: str = 'eigenvalue'
    ) -> pd.DataFrame:
        """
        Convert a correlation matrix to the nearest positive definite matrix.
        
        Parameters
        ----------
        corr : pd.DataFrame
            Correlation matrix that may not be positive definite
        method : str, default='eigenvalue'
            Method to use:
            - 'eigenvalue': Clip negative eigenvalues to small positive value
            - 'nearest': Find nearest positive definite matrix (Higham's algorithm)
            
        Returns
        -------
        pd.DataFrame
            Positive definite correlation matrix with same index/columns
            
        Warnings
        --------
        Prints warning if matrix was modified
        
        Examples
        --------
        >>> corr_fixed = self._make_positive_definite(corr_matrix)
        """
        # Check if already valid
        is_valid, issues = self._validate_correlation_matrix(corr)
        
        if is_valid:
            return corr.copy()
        
        # Warn user
        warnings.warn(
            f"Correlation matrix is not positive definite. "
            f"Issues: {'; '.join(issues)}. "
            f"Applying {method} correction."
        )
        
        if method == 'eigenvalue':
            # Eigenvalue clipping method
            eigenvals, eigenvecs = np.linalg.eigh(corr.values)
            
            # Clip negative eigenvalues
            min_eigenval = 1e-8
            eigenvals_clipped = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            corr_fixed = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
            
            # Ensure diagonal is exactly 1
            scaling = np.sqrt(np.diag(corr_fixed))
            corr_fixed = corr_fixed / scaling[:, None] / scaling[None, :]
            
        elif method == 'nearest':
            # Higham's algorithm for nearest correlation matrix
            corr_fixed = self._nearest_correlation_matrix(corr.values)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'eigenvalue' or 'nearest'")
        
        # Convert back to DataFrame
        result = pd.DataFrame(
            corr_fixed,
            index=corr.index,
            columns=corr.columns
        )
        
        # Validate result
        is_valid_result, _ = self._validate_correlation_matrix(result)
        if not is_valid_result:
            warnings.warn(
                f"Correction method '{method}' did not produce valid correlation matrix. "
                "Using identity matrix as fallback."
            )
            result = pd.DataFrame(
                np.eye(len(corr)),
                index=corr.index,
                columns=corr.columns
            )
        
        return result
    
    def _nearest_correlation_matrix(self, A: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        Find the nearest correlation matrix using Higham's algorithm.
        
        Reference: Higham, N. J. (2002). Computing the nearest correlation matrix—
        a problem from finance. IMA Journal of Numerical Analysis, 22(3), 329-343.
        
        Parameters
        ----------
        A : np.ndarray
            Input matrix (may not be positive definite)
        max_iter : int, default=100
            Maximum iterations
            
        Returns
        -------
        np.ndarray
            Nearest positive semi-definite correlation matrix
        """
        n = A.shape[0]
        
        # Initialize
        Y = A.copy()
        Delta_S = np.zeros_like(A)
        
        for _ in range(max_iter):
            # Project onto positive semi-definite matrices
            R = Y - Delta_S
            eigenvals, eigenvecs = np.linalg.eigh(R)
            eigenvals = np.maximum(eigenvals, 0)
            X = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            Delta_S = X - R
            
            # Project onto correlation matrices (diagonal = 1)
            Y = X.copy()
            np.fill_diagonal(Y, 1)
            
            # Check convergence
            if np.linalg.norm(Y - X) < 1e-8:
                break
        
        return Y
    
    # ========================================================================
    # Phase 2: Market Regime Correlation Estimation
    # ========================================================================
    
    def _load_market_regime_labels(self, verbose: bool = True) -> pd.Series:
        """
        Load S&P 500 regime labels to define market state.
        
        Always uses 'SPDR S&P 500 ETF' from 'us_equity' asset class,
        regardless of the portfolio's asset class. This ensures a consistent
        market regime definition across all portfolios.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print diagnostic information
            
        Returns
        -------
        pd.Series
            S&P 500 regime labels (0, 1, 2, 3) indexed by date
            
        Raises
        ------
        FileNotFoundError
            If S&P 500 KAMA_MSR model doesn't exist for end_date
            
        Notes
        -----
        Market regimes defined by S&P 500:
        - Regime 0: Low Volatility Bull
        - Regime 1: Low Volatility Bear
        - Regime 2: High Volatility Bull
        - Regime 3: High Volatility Bear
        """
        if verbose:
            print(f"\nLoading market regime labels from {self.market_regime_asset}...")
            if self.asset_class != self.market_regime_asset_class:
                print(f"  Note: Portfolio asset class is '{self.asset_class}', "
                      f"but using '{self.market_regime_asset_class}' S&P 500 for market regimes")
        
        # Get S&P 500 KAMA_MSR path
        market_kama_msr_path = self._get_kama_msr_path_for_asset(
            self.market_regime_asset,
            self.market_regime_asset_class
        )
        
        if not market_kama_msr_path.exists():
            raise FileNotFoundError(
                f"S&P 500 KAMA_MSR model not found for market regime definition: "
                f"{market_kama_msr_path}\n"
                f"Market regimes require S&P 500 model at end_date={self.end_date}"
            )
        
        # Load KAMA_MSR model
        with open(market_kama_msr_path, 'rb') as f:
            market_kama_msr = pickle.load(f)
        
        regime_labels = market_kama_msr.regime_labels
        
        if verbose:
            print(f"  Loaded {len(regime_labels)} regime labels")
            print(f"  Date range: {regime_labels.index[0]} to {regime_labels.index[-1]}")
            
            # Check for NA values
            n_na = regime_labels.isna().sum()
            if n_na > 0:
                print(f"  Warning: {n_na} NA values in regime labels ({n_na/len(regime_labels)*100:.1f}%)")
            
            print(f"  Regime distribution:")
            # Filter out NA values before getting unique regimes
            valid_regimes = regime_labels.dropna().unique()
            for regime in sorted(valid_regimes):
                count = (regime_labels == regime).sum()
                pct = (count / len(regime_labels)) * 100
                regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
                print(f"    Regime {regime} ({regime_names.get(regime, 'Unknown')}): "
                      f"{count:>5} days ({pct:>5.1f}%)")
        
        self.market_regime_labels = regime_labels
        return regime_labels
    
    def _get_kama_msr_path_for_asset(
        self, 
        asset_name: str, 
        asset_class: str
    ) -> Path:
        """
        Get path to KAMA+MSR model file for a specific asset and class.
        
        Parameters
        ----------
        asset_name : str
            Asset name
        asset_class : str
            Asset class ('us_equity', 'universe', etc.)
            
        Returns
        -------
        Path
            Path to KAMA_MSR model file
        """
        # For universe asset class, files are named: "{ticker} - {asset_name}_KAMA-MSR_4-regimes.pkl"
        # For other asset classes (us_equity, etc), files are: "{asset_name}_KAMA-MSR_4-regimes.pkl"
        model_dir = self.models_base_path / 'KAMA_MSR' / asset_class / self.end_date
        
        # Try exact match first (for us_equity, us_treasury, etc.)
        exact_filename = f"{asset_name}_KAMA-MSR_4-regimes.pkl"
        exact_path = model_dir / exact_filename
        if exact_path.exists():
            return exact_path
        
        # For universe asset class, use glob to find ticker-prefixed file
        pattern = f"*{asset_name}_KAMA-MSR_4-regimes.pkl"
        matching_files = list(model_dir.glob(pattern))
        
        if len(matching_files) == 1:
            return matching_files[0]
        elif len(matching_files) > 1:
            raise ValueError(f"Multiple KAMA+MSR model files found for {asset_name}: {[f.name for f in matching_files]}")
        else:
            raise FileNotFoundError(f"KAMA+MSR model not found for {asset_name} in {model_dir}")
    
    def estimate_regime_correlations(
        self,
        verbose: bool = True,
        min_observations: int = 30
    ) -> Dict[int, pd.DataFrame]:
        """
        Estimate correlation matrices for each market regime.
        
        Uses S&P 500 regime labels to define market state, then computes
        correlation matrices for portfolio assets during each regime.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print diagnostic information
        min_observations : int, default=30
            Minimum observations required per regime. If a regime has fewer
            observations, uses overall correlation as fallback.
            
        Returns
        -------
        Dict[int, pd.DataFrame]
            {regime_id: correlation_matrix} for regimes 0, 1, 2, 3
            
        Raises
        ------
        ValueError
            If market regime labels not loaded or insufficient data
            
        Examples
        --------
        >>> regime_corrs = portfolio_gen.estimate_regime_correlations()
        >>> # Access correlation matrix for HV Bear regime
        >>> hv_bear_corr = regime_corrs[3]
        
        Notes
        -----
        - All correlations are conditional on S&P 500 regime state
        - Works across different asset classes
        - Invalid correlation matrices are automatically corrected
        """
        if verbose:
            print(f"\n{'='*80}")
            print("ESTIMATING REGIME-DEPENDENT CORRELATION MATRICES")
            print(f"{'='*80}")
        
        # Load market regime labels if not already loaded
        if self.market_regime_labels is None:
            self._load_market_regime_labels(verbose=verbose)
        
        market_regime_labels = self.market_regime_labels
        
        if verbose:
            print(f"\nPortfolio assets: {self.asset_names}")
            print(f"Market regime defined by: {self.market_regime_asset}")
        
        # Load returns for all portfolio assets
        asset_returns_dict = {}
        
        for asset_name in self.asset_names:
            if verbose:
                print(f"\nLoading returns for {asset_name}...")
            
            kama_msr, _ = self.load_models(asset_name, verbose=False, kmrf=False)
            asset_returns_dict[asset_name] = kama_msr.returns
            
            if verbose:
                print(f"  Returns: {len(kama_msr.returns)} observations")
                print(f"  Date range: {kama_msr.returns.index[0]} to {kama_msr.returns.index[-1]}")
        
        # Create DataFrame of all asset returns
        returns_df = pd.DataFrame(asset_returns_dict)
        
        # Align with market regime labels
        common_dates = returns_df.index.intersection(market_regime_labels.index)
        
        if len(common_dates) == 0:
            raise ValueError(
                "No overlapping dates between portfolio asset returns and "
                "S&P 500 regime labels. Check data alignment."
            )
        
        aligned_returns = returns_df.loc[common_dates]
        aligned_regimes = market_regime_labels.loc[common_dates]
        
        # Remove rows where regime is NA
        valid_mask = aligned_regimes.notna()
        aligned_returns = aligned_returns[valid_mask]
        aligned_regimes = aligned_regimes[valid_mask]
        
        if verbose:
            print(f"\nAligned data: {len(aligned_returns)} observations (after removing NA regimes)")
            print(f"Date range: {aligned_returns.index[0]} to {aligned_returns.index[-1]}")
        
        # Estimate correlation for each regime
        regime_correlations = {}
        regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        # Also compute overall correlation as fallback
        overall_corr = aligned_returns.corr()
        
        if verbose:
            print(f"\n{'─'*80}")
            print("REGIME-SPECIFIC CORRELATIONS")
            print(f"{'─'*80}")
        
        # Now all regimes should be valid (no NA)
        for regime in sorted(aligned_regimes.unique()):
            regime_mask = (aligned_regimes == regime)
            regime_data = aligned_returns[regime_mask]
            n_obs = len(regime_data)
            
            if verbose:
                print(f"\nRegime {regime} ({regime_names.get(regime, 'Unknown')}):")
                print(f"  Observations: {n_obs}")
            
            if n_obs < min_observations:
                if verbose:
                    print(f"  ⚠️  WARNING: Only {n_obs} observations (< {min_observations} required)")
                    print(f"  Using overall correlation as fallback")
                regime_correlations[regime] = overall_corr.copy()
            else:
                regime_corr = regime_data.corr()
                
                # Validate and correct if needed
                is_valid, issues = self._validate_correlation_matrix(regime_corr)
                
                if not is_valid:
                    if verbose:
                        print(f"  ⚠️  Correlation matrix invalid: {issues}")
                        print(f"  Applying correction...")
                    regime_corr = self._make_positive_definite(regime_corr)
                    is_valid_after, _ = self._validate_correlation_matrix(regime_corr)
                    if verbose and is_valid_after:
                        print(f"  ✓ Correction successful")
                
                regime_correlations[regime] = regime_corr
                
                if verbose:
                    print(f"  Correlation matrix ({regime_corr.shape[0]}x{regime_corr.shape[1]}):")
                    # Show average off-diagonal correlation
                    mask = ~np.eye(regime_corr.shape[0], dtype=bool)
                    avg_corr = regime_corr.values[mask].mean()
                    min_corr = regime_corr.values[mask].min()
                    max_corr = regime_corr.values[mask].max()
                    print(f"    Avg correlation: {avg_corr:.3f}")
                    print(f"    Range: [{min_corr:.3f}, {max_corr:.3f}]")
        
        # Fill in any missing regimes with overall correlation
        for regime in [0, 1, 2, 3]:
            if regime not in regime_correlations:
                if verbose:
                    print(f"\n⚠️  Regime {regime} not found in data, using overall correlation")
                regime_correlations[regime] = overall_corr.copy()
        
        if verbose:
            print(f"\n{'='*80}")
            print("✓ Regime correlation estimation complete")
            print(f"{'='*80}")
        
        self.regime_correlations = regime_correlations
        return regime_correlations
    
    def validate_regime_correlations(
        self,
        min_correlation: float = -1.0,
        max_correlation: float = 1.0,
        check_positive_definite: bool = True,
        verbose: bool = True
    ) -> Dict[int, Dict]:
        """
        Validate estimated regime correlation matrices.
        
        Checks:
        - Correlation bounds (-1 to 1)
        - Symmetry
        - Positive definiteness
        - Eigenvalue distribution
        - Condition number
        
        Parameters
        ----------
        min_correlation : float, default=-1.0
            Minimum allowed correlation
        max_correlation : float, default=1.0
            Maximum allowed correlation
        check_positive_definite : bool, default=True
            Check if matrices are positive definite
        verbose : bool, default=True
            Print validation results
            
        Returns
        -------
        Dict[int, Dict]
            Validation results for each regime
        """
        if self.regime_correlations is None:
            raise ValueError("Must call estimate_regime_correlations() first")
        
        results = {}
        regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
        
        for regime_id, corr_matrix in self.regime_correlations.items():
            corr_array = corr_matrix.values
            n = corr_array.shape[0]
            
            # Check 1: Bounds
            min_val = corr_array.min()
            max_val = corr_array.max()
            bounds_ok = (min_val >= min_correlation) and (max_val <= max_correlation)
            
            # Check 2: Symmetry
            symmetry_error = np.max(np.abs(corr_array - corr_array.T))
            is_symmetric = symmetry_error < 1e-10
            
            # Check 3: Diagonal is 1
            diag_error = np.max(np.abs(np.diag(corr_array) - 1.0))
            diag_ok = diag_error < 1e-10
            
            # Check 4: Positive definite
            eigenvals = np.linalg.eigvalsh(corr_array)
            min_eigenval = eigenvals.min()
            is_positive_definite = min_eigenval > -1e-10
            
            # Check 5: Condition number
            max_eigenval = eigenvals.max()
            condition_number = max_eigenval / max(abs(min_eigenval), 1e-10)
            is_well_conditioned = condition_number < 100
            
            # Check 6: Average correlation
            # Extract upper triangle (excluding diagonal)
            upper_tri_idx = np.triu_indices(n, k=1)
            correlations = corr_array[upper_tri_idx]
            avg_correlation = correlations.mean()
            max_correlation_val = correlations.max()
            min_correlation_val = correlations.min()
            
            # Overall validation
            all_checks_pass = (
                bounds_ok and 
                is_symmetric and 
                diag_ok and 
                is_positive_definite and
                is_well_conditioned
            )
            
            results[regime_id] = {
                'regime_name': regime_names[regime_id],
                'valid': all_checks_pass,
                'bounds_ok': bounds_ok,
                'is_symmetric': is_symmetric,
                'diagonal_ok': diag_ok,
                'is_positive_definite': is_positive_definite,
                'is_well_conditioned': is_well_conditioned,
                'min_eigenval': min_eigenval,
                'max_eigenval': max_eigenval,
                'condition_number': condition_number,
                'avg_correlation': avg_correlation,
                'min_correlation': min_correlation_val,
                'max_correlation': max_correlation_val,
                'symmetry_error': symmetry_error,
                'diagonal_error': diag_error
            }
        
        if verbose:
            print("\n" + "="*80)
            print("REGIME CORRELATION MATRIX VALIDATION")
            print("="*80)
            
            for regime_id, res in results.items():
                print(f"\nRegime {regime_id} ({res['regime_name']}):")
                print(f"  Valid: {'✓' if res['valid'] else '✗'}")
                print(f"  Positive Definite: {'✓' if res['is_positive_definite'] else '✗'}")
                print(f"  Well-Conditioned: {'✓' if res['is_well_conditioned'] else '✗'}")
                print(f"  Eigenvalues: [{res['min_eigenval']:.6f}, {res['max_eigenval']:.6f}]")
                print(f"  Condition Number: {res['condition_number']:.2f}")
                print(f"  Avg Correlation: {res['avg_correlation']:.3f}")
                print(f"  Correlation Range: [{res['min_correlation']:.3f}, {res['max_correlation']:.3f}]")
                
                if not res['valid']:
                    print("  ⚠️  Issues detected:")
                    if not res['is_positive_definite']:
                        print(f"    - Not positive definite (min eigenval: {res['min_eigenval']:.6f})")
                    if not res['is_well_conditioned']:
                        print(f"    - Poorly conditioned (condition number: {res['condition_number']:.2f})")
            
            # Summary
            n_valid = sum(1 for r in results.values() if r['valid'])
            print("\n" + "-"*80)
            print(f"Summary: {n_valid}/{len(results)} regimes have valid correlation matrices")
            
            if n_valid == len(results):
                print("✓ All correlation matrices passed validation")
            else:
                print("⚠️  Some correlation matrices have issues (will be corrected automatically)")
            
            print("="*80)
        
        return results
    
    def validate_regime_concordance(
        self,
        min_probability: float = 0.0,
        max_probability: float = 1.0,
        check_row_sums: bool = True,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Validate estimated regime concordance matrices.
        
        Checks:
        - Probability bounds [0, 1]
        - Row sums to 1 (stochastic matrix)
        - No NaN or Inf values
        - Diagonal dominance (assets follow market regime)
        
        Parameters
        ----------
        min_probability : float, default=0.0
            Minimum probability
        max_probability : float, default=1.0
            Maximum probability
        check_row_sums : bool, default=True
            Check if rows sum to 1
        verbose : bool, default=True
            Print validation results
            
        Returns
        -------
        Dict[str, Dict]
            Validation results for each asset
        """
        if self.regime_concordance is None:
            raise ValueError("Must call estimate_regime_concordance() first")
        
        results = {}
        
        for asset, concordance_df in self.regime_concordance.items():
            concordance_array = concordance_df.values
            
            # Check 1: Bounds [0, 1]
            min_val = concordance_array.min()
            max_val = concordance_array.max()
            bounds_ok = (min_val >= min_probability) and (max_val <= max_probability)
            
            # Check 2: No NaN or Inf
            has_nan = np.isnan(concordance_array).any()
            has_inf = np.isinf(concordance_array).any()
            is_finite = not (has_nan or has_inf)
            
            # Check 3: Row sums to 1 (stochastic matrix)
            row_sums = concordance_array.sum(axis=1)
            row_sum_error = np.max(np.abs(row_sums - 1.0))
            rows_sum_to_one = row_sum_error < 1e-6
            
            # Check 4: Diagonal dominance (asset follows market regime)
            diagonal = np.diag(concordance_array)
            avg_diagonal = diagonal.mean()
            min_diagonal = diagonal.min()
            max_off_diagonal = concordance_array[~np.eye(4, dtype=bool)].max()
            is_diagonal_dominant = avg_diagonal > 0.25  # On average, >25% concordance
            
            # Overall validation
            all_checks_pass = (
                bounds_ok and
                is_finite and
                rows_sum_to_one and
                is_diagonal_dominant
            )
            
            results[asset] = {
                'valid': all_checks_pass,
                'bounds_ok': bounds_ok,
                'is_finite': is_finite,
                'rows_sum_to_one': rows_sum_to_one,
                'is_diagonal_dominant': is_diagonal_dominant,
                'avg_diagonal': avg_diagonal,
                'min_diagonal': min_diagonal,
                'max_off_diagonal': max_off_diagonal,
                'row_sum_error': row_sum_error,
                'min_probability': min_val,
                'max_probability': max_val
            }
        
        if verbose:
            print("\n" + "="*80)
            print("REGIME CONCORDANCE VALIDATION")
            print("="*80)
            
            for asset, res in results.items():
                print(f"\n{asset}:")
                print(f"  Valid: {'✓' if res['valid'] else '✗'}")
                print(f"  Stochastic (rows sum to 1): {'✓' if res['rows_sum_to_one'] else '✗'}")
                print(f"  Diagonal Dominant: {'✓' if res['is_diagonal_dominant'] else '✗'}")
                print(f"  Avg Diagonal: {res['avg_diagonal']:.3f}")
                print(f"  Min Diagonal: {res['min_diagonal']:.3f}")
                print(f"  Max Off-Diagonal: {res['max_off_diagonal']:.3f}")
                
                if not res['valid']:
                    print("  ⚠️  Issues:")
                    if not res['rows_sum_to_one']:
                        print(f"    - Rows don't sum to 1 (max error: {res['row_sum_error']:.6f})")
                    if not res['is_diagonal_dominant']:
                        print(f"    - Weak concordance (avg diagonal: {res['avg_diagonal']:.3f})")
            
            # Summary
            n_valid = sum(1 for r in results.values() if r['valid'])
            print("\n" + "-"*80)
            print(f"Summary: {n_valid}/{len(results)} assets have valid concordance matrices")
            
            if n_valid == len(results):
                print("✓ All concordance matrices passed validation")
            else:
                print("⚠️  Some concordance matrices have issues")
            
            print("="*80)
        
        return results
    
    def diagnostic_summary(self, verbose: bool = True) -> Dict:
        """
        Comprehensive diagnostic summary of all estimations.
        
        Runs all validation methods and provides overview.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print summary
            
        Returns
        -------
        dict
            Complete diagnostic results
        """
        diagnostics = {}
        
        # Validate regime correlations
        if self.regime_correlations is not None:
            diagnostics['correlations'] = self.validate_regime_correlations(verbose=verbose)
        
        # Validate regime concordance
        if self.regime_concordance is not None:
            diagnostics['concordance'] = self.validate_regime_concordance(verbose=verbose)
        
        # Validate individual asset distributions
        if self.simulators:
            diagnostics['distributions'] = {}
            for asset_name, simulator in self.simulators.items():
                if simulator.regime_distributions is not None:
                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"DISTRIBUTION VALIDATION: {asset_name}")
                        print(f"{'='*80}")
                    diagnostics['distributions'][asset_name] = simulator.validate_distributions(verbose=verbose)
        
        if verbose:
            print("\n" + "="*80)
            print("DIAGNOSTIC SUMMARY COMPLETE")
            print("="*80)
        
        return diagnostics
    
    def plot_regime_correlations(
        self,
        figsize: Tuple[int, int] = (16, 12),
        cmap: str = 'RdBu_r',
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrices for all four regimes.
        
        Parameters
        ----------
        figsize : tuple, default=(16, 12)
            Figure size (width, height)
        cmap : str, default='RdBu_r'
            Colormap name (diverging colormaps recommended)
        save_path : str, optional
            Path to save figure. If None, displays interactively.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
            
        Raises
        ------
        ValueError
            If regime correlations have not been estimated
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.regime_correlations is None:
            raise ValueError(
                "Regime correlations not estimated. "
                "Call estimate_regime_correlations() first."
            )
        
        regime_names = {
            0: 'Regime 0: Low Vol Bull',
            1: 'Regime 1: Low Vol Bear',
            2: 'Regime 2: High Vol Bull',
            3: 'Regime 3: High Vol Bear'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for regime in [0, 1, 2, 3]:
            ax = axes[regime]
            corr = self.regime_correlations[regime]
            
            # Plot heatmap
            sns.heatmap(
                corr,
                ax=ax,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                fmt='.2f',
                square=True,
                cbar_kws={'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray'
            )
            
            ax.set_title(
                regime_names[regime],
                fontsize=12,
                fontweight='bold',
                pad=10
            )
            
            # Rotate labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.suptitle(
            f'Correlation Matrices Conditional on S&P 500 Regime\n'
            f'Portfolio: {len(self.asset_names)} assets | End Date: {self.end_date}',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        return fig
    
    def estimate_regime_concordance(
        self,
        verbose: bool = True,
        min_observations: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Estimate conditional regime probabilities: P(asset_regime | market_regime).
        
        This measures how each asset's regime depends on the S&P 500 market regime.
        Instead of assuming independent regime transitions, we model:
        P(IWM in regime j | SPY in regime i) for all i, j combinations.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print diagnostic information
        min_observations : int, default=30
            Minimum observations in market regime to estimate conditional probs
            
        Returns
        -------
        Dict[str, np.ndarray]
            For each asset, a 4x4 matrix where entry [i,j] = P(asset_regime=j | market_regime=i)
            Stored in self.regime_concordance
            
        Examples
        --------
        >>> concordance = portfolio_gen.estimate_regime_concordance()
        >>> # For IWM, what's the probability it's in regime 2 given SPY is in regime 0?
        >>> prob = concordance['iShares Russell 2000 ETF'][0, 2]
        
        Notes
        -----
        This is critical for realistic multi-asset simulation. Without this, we assume
        all assets can be in any regime independently of market conditions, which is
        unrealistic (e.g., small caps rarely in "Bull" regime during market "Bear").
        """
        if self.market_regime_labels is None:
            self._load_market_regime_labels(verbose=False)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ESTIMATING REGIME CONCORDANCE WITH MARKET")
            print(f"{'='*80}\n")
            print(f"Market regime indicator: {self.market_regime_asset}")
            print(f"Portfolio assets: {len(self.asset_names)}")
        
        # Storage for conditional probabilities
        regime_concordance = {}
        
        # Get market regime labels
        market_regimes = self.market_regime_labels
        
        for asset_name in self.asset_names:
            if verbose:
                print(f"\nAnalyzing {asset_name}...")
            
            # Load asset's KAMA_MSR model to get its regime labels
            try:
                kama_msr_path = self._get_kama_msr_path_for_asset(
                    asset_name, 
                    self.asset_class
                )
                
                with open(kama_msr_path, 'rb') as f:
                    asset_kama_msr = pickle.load(f)
                
                asset_regimes = asset_kama_msr.regime_labels
                
                # Align dates
                aligned = pd.DataFrame({
                    'market_regime': market_regimes,
                    'asset_regime': asset_regimes
                }).dropna()
                
                if verbose:
                    print(f"  Aligned observations: {len(aligned)}")
                
                # Estimate P(asset_regime | market_regime) for all combinations
                concordance_matrix = np.zeros((4, 4))  # [market_regime, asset_regime]
                
                for market_regime in range(4):
                    # Get observations where market was in this regime
                    mask = aligned['market_regime'] == market_regime
                    n_obs = mask.sum()
                    
                    if n_obs < min_observations:
                        if verbose:
                            print(f"  Warning: Market regime {market_regime} has only {n_obs} obs "
                                  f"(< {min_observations}), using uniform distribution")
                        concordance_matrix[market_regime, :] = 0.25  # Uniform fallback
                    else:
                        # Count asset regime occurrences conditional on market regime
                        asset_regime_counts = aligned.loc[mask, 'asset_regime'].value_counts()
                        
                        # Convert to probabilities
                        for asset_regime in range(4):
                            count = asset_regime_counts.get(asset_regime, 0)
                            concordance_matrix[market_regime, asset_regime] = count / n_obs
                        
                        if verbose:
                            regime_names = {0: 'LV Bull', 1: 'LV Bear', 2: 'HV Bull', 3: 'HV Bear'}
                            print(f"  Market regime {market_regime} ({regime_names[market_regime]}): {n_obs} obs")
                            for asset_regime in range(4):
                                prob = concordance_matrix[market_regime, asset_regime]
                                if prob > 0.01:  # Only show non-trivial probabilities
                                    print(f"    → Asset regime {asset_regime}: {prob:.3f}")
                
                regime_concordance[asset_name] = concordance_matrix
                
            except FileNotFoundError as e:
                if verbose:
                    print(f"  ⚠️  Could not load model: {e}")
                    print(f"  Using uniform concordance (independent regimes)")
                # Fallback: uniform distribution (independence)
                regime_concordance[asset_name] = np.full((4, 4), 0.25)
        
        self.regime_concordance = regime_concordance
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✓ Regime concordance estimation complete")
            print(f"{'='*80}")
        
        return regime_concordance

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
        # Initialize KMRF with feature selection options and pre-loaded KAMA_MSR model
        kmrf = KMRF(
            asset_name=asset_name,
            asset_class=self.asset_class,
            kama_msr_model=kama_msr,  # Pass the loaded model
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
    
    def simulate_all_assets(
        self,
        verbose: bool = True,
        use_copula: bool = True,
        market_asset: str = 'SPDR S&P 500 ETF'
    ) -> Dict[str, pd.DataFrame]:
        """
        Run Bayesian forward simulations for all assets.
        
        Phase 4 Update: Now uses multi-asset Gaussian copula simulation by default,
        which preserves regime-dependent correlations and regime concordance.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print progress information
        use_copula : bool, default=True
            If True, uses Gaussian copula for multi-asset simulation (recommended)
            If False, uses legacy independent simulation (correlations not preserved)
        market_asset : str, default='SPDR S&P 500 ETF'
            Asset to use as market regime indicator for copula simulation
            
        Returns
        -------
        dict
            {asset_name: simulated_returns} where values are numpy arrays
            of shape (n_simulations, n_days)
            
        Notes
        -----
        The copula method (use_copula=True) is superior because it:
        - Preserves regime-dependent correlation structure
        - Captures regime concordance (assets co-move with market)
        - Uses Bayesian updates for realistic regime evolution
        - Is faster (one simulation vs. N independent simulations)
        
        Legacy independent method (use_copula=False) is retained for:
        - Backward compatibility
        - Single-asset analysis
        - Debugging/comparison purposes
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"PORTFOLIO SIMULATION: {len(self.asset_names)} ASSETS")
            print(f"{'='*80}")
            print(f"Asset Class: {self.asset_class}")
            print(f"End Date: {self.end_date}")
            print(f"Horizon: {self.n_days} days")
            print(f"Simulations: {self.n_simulations:,}")
            print(f"Method: {'Gaussian Copula (correlated)' if use_copula else 'Independent (legacy)'}")
        
        if use_copula:
            # Phase 3 implementation: Multi-asset copula simulation
            self._simulate_all_assets_copula(verbose=verbose, market_asset=market_asset)
        else:
            # Legacy implementation: Independent per-asset simulation
            self._simulate_all_assets_independent(verbose=verbose)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✓ Successfully simulated {len(self.asset_simulations)}/{len(self.asset_names)} assets")
            print(f"{'='*80}")
        
        return self.asset_simulations
    
    def _simulate_all_assets_copula(
        self,
        verbose: bool = True,
        market_asset: str = 'SPDR S&P 500 ETF'
    ) -> None:
        """
        Run multi-asset simulation using Gaussian copula with regime concordance.
        
        This is the Phase 3 implementation that preserves correlations and
        regime dependencies across assets.
        
        Parameters
        ----------
        verbose : bool
            Print progress information
        market_asset : str
            Asset to use as market regime indicator
        """
        if verbose:
            print(f"\n{'='*80}")
            print("MULTI-ASSET COPULA SIMULATION")
            print(f"{'='*80}")
            print(f"Phase 3: Gaussian Copula with Bayesian Updates")
        
        # Step 1: Initialize simulators for all assets (needed for forward probs and distributions)
        if verbose:
            print(f"\n[Step 1/5] Initializing simulators for {len(self.asset_names)} assets...")
        
        assets_forward_probs = {}
        assets_regime_distributions = {}
        
        for i, asset_name in enumerate(self.asset_names, 1):
            if verbose:
                print(f"  [{i}/{len(self.asset_names)}] Loading {asset_name}...")
            
            try:
                # Load models
                kama_msr, kmrf = self.load_models(asset_name, verbose=False, kmrf=True)
                self.kmrf_models[asset_name] = kmrf
                
                # Create simulator (but don't run simulation)
                simulator = BayesianForwardSimulator(
                    kama_msr=kama_msr,
                    kmrf=kmrf,
                    n_days=self.n_days,
                    alpha_confidence=self.alpha,
                    significance_level=self.sig_level
                )
                
                # Compute forward probabilities and fit distributions
                simulator.compute_forward_regime_probs()
                simulator.fit_regime_distributions(verbose=False)
                
                # Store
                self.simulators[asset_name] = simulator
                assets_forward_probs[asset_name] = simulator.forward_probs
                assets_regime_distributions[asset_name] = simulator.regime_distributions
                
            except Exception as e:
                print(f"\n⚠️  ERROR loading {asset_name}: {e}")
                raise
        
        # Step 2: Estimate regime-dependent correlations
        if verbose:
            print(f"\n[Step 2/5] Estimating regime-dependent correlations...")
        
        regime_correlations = self.estimate_regime_correlations(verbose=verbose)
        
        # Step 3: Estimate regime concordance
        if verbose:
            print(f"\n[Step 3/5] Estimating regime concordance...")
        
        regime_concordance = self.estimate_regime_concordance(verbose=verbose)
        
        # Step 4: Get market regime probabilities
        if market_asset not in self.simulators:
            raise ValueError(f"Market asset '{market_asset}' not in portfolio")
        
        market_regime_probs = self.simulators[market_asset].forward_probs
        
        if verbose:
            print(f"\n[Step 4/5] Using {market_asset} as market regime indicator")
            print(f"  Market regime probabilities shape: {market_regime_probs.shape}")
        
        # Step 5: Run multi-asset copula simulation
        if verbose:
            print(f"\n[Step 5/5] Running Gaussian copula simulation...")
        
        simulated_returns = BayesianForwardSimulator.simulate_multiasset_copula(
            assets_forward_probs=assets_forward_probs,
            assets_regime_distributions=assets_regime_distributions,
            regime_correlations=regime_correlations,
            market_regime_probs=market_regime_probs,
            regime_concordance=regime_concordance,
            market_asset=market_asset,
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
            verbose=verbose
        )
        
        # Convert numpy arrays to DataFrames for compatibility with downstream code
        for asset_name, returns_array in simulated_returns.items():
            # returns_array has shape (n_simulations, n_days)
            # Convert to DataFrame: rows=days, columns=simulations
            self.asset_simulations[asset_name] = pd.DataFrame(
                returns_array.T,  # Transpose to (n_days, n_simulations)
                index=range(self.n_days),
                columns=range(self.n_simulations)
            )
        
        if verbose:
            print(f"\n✓ Copula simulation complete")
            print(f"  Output format: DataFrames with shape ({self.n_days} days, {self.n_simulations} simulations)")
    
    def _simulate_all_assets_independent(self, verbose: bool = True) -> None:
        """
        Legacy method: Run independent simulations for each asset.
        
        WARNING: This method does NOT preserve correlations between assets.
        Use simulate_all_assets(use_copula=True) instead for proper multi-asset simulation.
        
        This method is retained for:
        - Backward compatibility
        - Single-asset analysis
        - Debugging purposes
        """
        if verbose:
            print(f"\n⚠️  WARNING: Using legacy independent simulation")
            print(f"   Correlations between assets will NOT be preserved")
            print(f"   Consider using use_copula=True for proper multi-asset simulation\n")
        
        for i, asset_name in enumerate(self.asset_names, 1):
            if verbose:
                print(f"\n[{i}/{len(self.asset_names)}] Processing {asset_name}...")
            
            try:
                self.simulate_asset(asset_name, verbose=verbose)
            except Exception as e:
                print(f"\n⚠️  ERROR simulating {asset_name}: {e}")
                print(f"Skipping {asset_name}...")
                continue
    
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
        alpha_confidence: float = 1.0,
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
            alpha_confidence=alpha_confidence,
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

