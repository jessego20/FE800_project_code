# AI Assistant Instructions for KMRF+MPC Portfolio Optimization

This project implements a regime-switching portfolio optimization framework combining KAMA+MSR+RF for regime prediction with Model Predictive Control for portfolio optimization.

## Project Architecture

### Data Pipeline Structure
- Raw data sources in `/data/inputs/` 
  - Market data: Index and commodity symbols/mappings
  - FMP API integration for price data
  - FRED API for macroeconomic indicators
- Processed data in `/data/processed/`
  - Asset price data normalized to USD
  - Frequency-separated macro indicators (daily/monthly/quarterly)
  - Feature-engineered datasets

### Core Components
1. **Data Collection** (`data_collection.ipynb`)
   - Handles data gathering and initial processing
   - Supports both batch historical and incremental updates
   - Normalizes all equity indices to USD using FX rates

2. **Feature Engineering** (`derive_data.py`)
   - Class `TimeSeriesDerivedFields` generates comprehensive market features
   - Critical features include:
     - Returns (various horizons)
     - Volatility measures (20/60/120/252-day windows) 
     - Momentum indicators (10/20/60/120/252-day)
     - Technical indicators (SMA, EMA, Bollinger Bands)
     - Market microstructure variables

## Development Workflows

### Setting Up Data Pipeline
```python
# 1. Configure API keys in environment
FMP_API_KEY = "your_key"  # For market data
FRED_API_KEY = "your_key" # For macro data

# 2. Run data collection notebook
# Execute cells in order - handles dependencies automatically

# 3. Generate derived features
from derive_data import TimeSeriesDerivedFields
ts_features = TimeSeriesDerivedFields(price_data=asset_data)
derived_data = ts_features.compute_all_derived_fields()
```

### Key Data Conventions
- All timestamps are standardized to daily frequency
- Asset prices converted to USD for consistency
- Missing values handled via forward-fill for prices, interpolation for macro
- MultiIndex column structure: (Asset, Field) for price data
- Date range: 1990-01-01 to 2025-09-26

## Integration Points

### External Data Dependencies
1. Financial Modeling Prep (FMP) API
   - Market prices for indices and commodities
   - Real-time and historical data support
   
2. FRED API 
   - Macroeconomic indicators
   - Mixed frequency data (daily/monthly/quarterly)

### Asset Universe Coverage
- 14 global equity indices including major markets
- 12 commodities across energy, metals, agriculture
- US 3-Month T-Bills as risk-free asset

## Testing & Validation
- Training period: Up to 30/03/2018
- Validation: 02/04/2018 - 12/03/2020  
- Test: 27/03/2020 - 29/04/2022
- 15-day gaps between periods prevent leakage