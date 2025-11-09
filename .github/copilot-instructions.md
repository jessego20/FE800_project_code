# AI Assistant Instructions for KMRF+MPC Portfolio Optimization

## Overview
Re-implementation of "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework" (Gorse et al.). Combines KAMA+MSR regime detection with Random Forest prediction and Model Predictive Control for long-only portfolio optimization.

## Architecture: Two-Stage Pipeline

### Stage 1: KAMA+MSR (Markov-Switching Regime Detection)
- **File**: `kama_msr.py` - Contains `KAMA`, `MarkovSwitchingModel`, `KAMA_MSR` classes
- **Purpose**: Ex-post regime classification (4 regimes: LV/HV × Bullish/Bearish)
- **Training**: Gibbs sampling with 1000 iterations, 200 burn-in (configured in `fit_kama_msr.ipynb`)
- **Output**: Saved models in `saved_models/KAMA_MSR/{asset_class}/{end_date}/` as pickle files
- **Key Method**: `KAMA_MSR.fit()` performs full KAMA trend detection + MSR volatility regime fitting
- **Naming Convention**: `{Asset Name}_KAMA-MSR_4-regimes.pkl`

### Stage 2: KMRF (Random Forest Regime Prediction)
- **File**: `kmrf.py` - Contains `KMRF` class
- **Purpose**: Ex-ante regime prediction using labeled KAMA+MSR output
- **Label Adaptation**: Converts 4 regimes → 3 classes (Bullish/Bearish/Other) via contrarian logic
  - Bullish (1): LV bullish + extension to peak of next HV bullish
  - Bearish (-1): HV bearish + extension to trough of next LV bearish  
  - Other (0): Remaining periods
- **Feature Selection**: BorutaPy wrapper with 100 max iterations (slow but effective)
- **Output**: Saved in `saved_models/KMRF/{classification_type}_labels/{asset_class}/`

## Critical Workflows

### Batch Training KAMA+MSR Models
```python
# In fit_kama_msr.ipynb - configure these parameters:
ASSET_CLASSES_TO_TRAIN = ['us_equity', 'commodity', 'int_equity']
END_DATE = '2019-01-01'  # Training cutoff
KAMA_ER_PERIOD = 10     # Efficiency ratio period
MIN_DURATION = 5        # Minimum regime duration (periods)
```
**Key**: Training runs asset-by-asset, saves automatically. Models are deterministic given same random seed.

### Batch Training KMRF Models  
```python
# In train_kmrf.ipynb - configure:
ASSET_CLASSES_TO_TRAIN = ['us_equity']
CLASSIFICATION_TYPE = 'adapted'  # 'adapted' (3-class) or 'original' (4-class)
USE_BORUTA = True               # Feature selection - SLOW, ~5-10 min per asset
INCLUDE_MACRO = True            # Add macro features from data/ready/
```
**Key**: KMRF depends on KAMA+MSR outputs. Must train KAMA+MSR first for same assets/dates.

### Feature Engineering Pipeline
- **File**: `derive_features.py` - Class `TimeSeriesDerivedFields`
- **CRITICAL**: All price-derived features are LAGGED by 1 day (suffix `_lag1d`) except `open` and `log_ret_overnight`
- **Reason**: Prevents look-ahead bias - at time t, only know close price through t-1
- **Ready Data**: Pre-computed features in `data/ready/{asset_class}.csv` with MultiIndex (Asset, Feature)

## Data Conventions

### MultiIndex Structure (Asset, Feature)
```python
# Example: data/ready/us_equity.csv
columns = pd.MultiIndex.from_tuples([
    ('SPDR S&P 500 ETF', 'close_lag1d'),
    ('SPDR S&P 500 ETF', 'log_ret_lag1d'),
    ('Invesco QQQ Trust', 'close_lag1d'),
    ...
])
```
**Accessing single asset**: `df.xs('SPDR S&P 500 ETF', level=0, axis=1)`

### Asset Name Mapping
- Stored in dictionaries within notebooks (e.g., `fit_kama_msr.ipynb` cell 2)
- **US Equity**: Ticker → Full name (e.g., 'SPY' → 'SPDR S&P 500 ETF')
- **Commodities**: Uses FMP names from `data/inputs/fmp_commodity_list.csv`
- **Critical**: Models saved with FULL NAMES, not tickers

### Time Period Splits
```python
# Hardcoded in KMRF class initialization:
training_end = '2019-01-01'          # KAMA+MSR training cutoff
validation = '2019-02-01' to '2021-12-31'  # Hyperparameter tuning
test = '2022-02-01' to end of data         # Final evaluation
# 15-day gaps prevent temporal leakage
```

## Project-Specific Patterns

### Model Persistence Strategy
- Both KAMA+MSR and KMRF save entire fitted objects (not just parameters)
- Includes: model weights, regime labels, feature names, training metadata
- **Loading**: `pickle.load()` restores full state including `regime_labels` attribute

### Error Handling in Batch Training
```python
CONTINUE_ON_ERROR = True   # Skip failed assets, continue training
SAVE_SUCCESSFUL_ONLY = True  # Only persist successfully trained models
```
Training summaries saved as JSON with error messages for debugging.

### Feature Selection Logic
- BorutaPy uses Random Forest internally to identify relevant features
- Adapted for time-series via Purged Group Time-Series Split (prevents leakage)
- Selected features stored in `KMRF.selected_features` attribute
- **Without Boruta**: Uses all available features (~200-300 depending on macro inclusion)

## Asset Class Organization
```
data/ready/
├── us_equity.csv      # US ETFs/indices (SPY, QQQ, sectors, size, style)
├── int_equity.csv     # International ETFs (VXUS, VEA, VWO, regional)
├── commodity.csv      # Commodities (energy, metals, agriculture)
└── us_treasury.csv    # Bond ETFs (BIL, SHY, IEF)
```

### Asset Selection for Training
Assets defined via symbol-name dictionaries in notebooks:
- `us_equity_symbol_names`: ~40 ETFs/indices covering overall market, sectors, size, style factors -> 19 are used
- `int_equity_symbol_names`: 8 international equity ETFs  
- `commodity_symbol_names`: 12 commodities from FMP API -> 10 are used

## Dependencies & Environment
- **Core ML**: scikit-learn, boruta (optional but recommended)
- **Optimization**: scipy, numba (JIT compilation for Gibbs sampling)
- **Time Series**: tsfresh (automated feature extraction), ta (technical indicators)
- **Data**: pandas, numpy, fredapi (macro data), python-dotenv (API keys)
- **Viz**: plotly, matplotlib, seaborn

## Common Gotchas

1. **KAMA+MSR requires OHLC data** - uses high/low for volatility estimation
2. **KMRF expects MultiIndex columns** - will fail if reading wrong CSV format
3. **Boruta is slow** - 2-3 min per asset, disable for quick prototyping
4. **Asset names must match exactly** - "SPDR S&P 500 ETF" ≠ "SPY" in model files
5. **Random seeds differ** - KAMA+MSR uses custom seed, KMRF uses 1010 default
6. **Regime labels may have NAs** - first ~50 periods often NA due to moving averages

## Performance Expectations
- **KAMA+MSR**: ~10-20 min per asset (1000 Gibbs iterations)
- **KMRF without Boruta**: ~2-3 sec per asset
- **KMRF with Boruta**: ~1-3 min per asset (100 iterations × cross-validation folds)