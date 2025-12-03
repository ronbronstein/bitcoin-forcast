# Bitcoin Conditional Probability Forecast Engine - Claude Context

## Project Overview
Point-in-Time (P.I.T.) forecasting system for Bitcoin using conditional probability analysis, path-dependent Monte Carlo simulation, and regime-based scenario stratification. Research-grade accuracy with methodological rigor prioritized over prediction performance.

## Tech Stack
- **Language**: Python 3.9+
- **Data Processing**: pandas 2.0+, numpy 1.24+
- **Data Sources**: yfinance (Yahoo Finance), Blockchain.com API
- **Visualization**: plotly 5.14+

## Repository Structure
```
src/          - Core modules (data_loader, scenario_engine, backtest_engine)
scripts/      - Executable scripts (fetch data, run backtest, generate forecast)
data/         - Organized CSVs (raw, indicators, features, processed)
outputs/      - Generated reports and visualizations
tests/        - Test suite
```

## Key Architecture Principles

### 1. Point-in-Time (P.I.T.) Processing
**CRITICAL**: All indicators must be calculated ONLY on data available at forecast time.
- Indicators calculated INSIDE backtest loop (not before)
- All features lagged by 1 month (T-1 predictors)
- Training data uses `<= test_date` (inclusive)
- NEVER use future data in historical analysis

### 2. Path-Dependent Monte Carlo
RSI must evolve during simulation, not remain static:
- Store AG/AL components (Average Gain/Loss) for all assets
- Update AG/AL each month using Wilder's formula: `(prev_avg * 13 + current_value) / 14`
- Calculate RSI from evolving AG/AL, not initial conditions

### 3. Data Completeness
- **62 columns** across 119 months (2015-12 to 2025-10)
- **Full parity**: All 6 assets (BTC, SPX, NDX, GLD, TLT, USDT) have Price, RSI, AG/AL, Returns, Lagged
- **99.36% complete** (only USDT missing 23 months pre-2017)

## Code Patterns

### Import Structure (scripts must use)
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import data_loader, scenario_engine, backtest_engine
```

### Wilder's RSI Calculation
Use `calculate_wilder_components()` in data_loader.py - TRUE recursive smoothing:
1. First avg = SMA of first 14 periods
2. Subsequent = `(prev_avg * 13 + current) / 14`
3. NEVER use pandas `.ewm()` or simple rolling

### Variable Naming
- DataFrames: `df`, `train_df`, `raw_df`
- Columns: PascalCase for assets (`BTC`, `SPX`), snake_case for derived (`Ret_BTC`, `hash_rate`)
- Lagged features: `Prev_` prefix (e.g., `Prev_RSI_BTC`)

## Development Commands
```bash
python scripts/fetch_and_organize_data.py           # Use cache
python scripts/fetch_and_organize_data.py --refresh # Force refresh
python scripts/main.py                              # Generate forecast
python scripts/run_backtest.py                      # Run backtest
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Fetch data, calculate RSI (Wilder's), create lagged features |
| `src/scenario_engine.py` | Probability matrix, path-dependent Monte Carlo (2000 sims) |
| `src/backtest_engine.py` | Walk-forward validation, P.I.T. processing per test month |
| `scripts/fetch_and_organize_data.py` | Data fetching + CSV organization (37 files) |

## Known Issues

| Issue | Solution |
|-------|----------|
| Import errors | Scripts need `sys.path.insert(0, ...)` for `src/` modules |
| Yahoo rate limits | Wait 5-10 min between fetches, or use cached data |
| USDT missing pre-2017 | Expected (80.7% coverage) - analysis period has full data |

## Adding New Assets
1. Add ticker to `src/data_loader.py` → `self.tickers` dict
2. Add to RSI loop in `process_indicators()`
3. Add returns calculation: `df['Ret_X'] = df['X'].pct_change() * 100`
4. Add to lagged features dict
5. Update `scripts/fetch_and_organize_data.py` CSV exports
6. Run with `--refresh`

## Critical Rules
1. **NEVER** calculate indicators before backtest loop
2. **ALWAYS** lag features by 1 month for prediction
3. **STORE** AG/AL components for path-dependent simulation
4. **TEST** with `--refresh` after data source changes

## Performance
- Current: 58.3% directional accuracy, 19.8% MAPE
- Target: >60% directional, >80% P10-P90 band capture

---
**Dataset**: V4.1 | 62 columns | 119 months | 99.36% complete
