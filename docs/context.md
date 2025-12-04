# Project Context and Architecture

## Why We Rebuilt (v1 → v2)

The original implementation had:
- **Large monolithic files** (500+ lines each)
- **Suspicious backtest results** (50% direction + 18% MAPE = contradiction)
- **Potential data leakage** we couldn't easily audit
- **USDT gaps** (23 months missing, filled with RSI=100 placeholder)

v2 focuses on: **Simple → Correct → Then Complex**

## Architecture Decisions

### Module Responsibilities

| Module | Purpose | Key Class/Function |
|--------|---------|-------------------|
| `data/loader` | Load CSVs, P.I.T. filtering | `DataLoader.get_data_as_of()` |
| `data/cleaner` | Handle missing values, validate | `validate_rsi_bounds()` |
| `data/indicators` | Technical calculations | `calculate_wilder_rsi()` |
| `features/builder` | Feature matrix construction | `FeatureBuilder.build_training_data()` |
| `features/validator` | Detect data leakage | `PITValidator.run_full_validation()` |
| `models/baseline` | Historical mean model | `HistoricalMeanModel` |
| `models/evaluator` | Metrics calculation | `ModelEvaluator` |
| `backtest/engine` | Walk-forward logic | `BacktestEngine.run()` |
| `backtest/reporter` | Generate reports | `BacktestReporter` |

### Design Constraints

1. **No file > 150 lines** - Forces single responsibility
2. **All features lagged** - `Prev_*` prefix required
3. **Explicit P.I.T.** - `as_of_date` parameter everywhere
4. **USDT dropped** - 8 columns removed (incomplete pre-2017)

## Data Pipeline

```
data/processed/full_dataset.csv (54 cols, 119 months)
         ↓
    DataLoader.get_data_as_of(as_of_date)
         ↓
    FeatureBuilder.build_training_data()
         ↓
    HistoricalMeanModel.fit() → predict()
         ↓
    ModelEvaluator → metrics
```

## Test Coverage

**135 tests** across all modules:

| Area | Tests | Key Validations |
|------|-------|-----------------|
| dates | 15 | P.I.T. filtering works correctly |
| loader | 12 | USDT excluded, no future data |
| cleaner | 14 | RSI bounds, gap detection |
| indicators | 17 | Wilder's RSI ≠ ewm() |
| builder | 11 | All features lagged |
| validator | 13 | Detects unlagged features |
| baseline | 15 | P.I.T. in fit() |
| evaluator | 18 | Correct metric calculation |
| engine | 10 | Walk-forward P.I.T. |
| reporter | 10 | Output generation |

## Baseline Results Interpretation

| Metric | Value | Meaning |
|--------|-------|---------|
| Dir. Accuracy | 56.7% | Slightly above random (BTC positive bias) |
| MAPE | 16.0% | Reasonable for BTC volatility |
| P10-P90 | 85.0% | Well-calibrated uncertainty bands |

**Why 56.7% and not 50%?**
BTC has historically positive mean return (~9%/month in training data). Predicting "up" is correct slightly more often than random.

## Known Limitations

1. **Monthly only** - No daily/weekly granularity
2. **Limited macro data** - No VIX or yield curve yet

## Phase 6 Status

| Feature | Status | Command |
|---------|--------|---------|
| Conditional Model | ✅ Done | `python run_backtest.py --model conditional` |
| Scenario Matching | ✅ Done | `python run_scenarios.py` |
| Path-Dependent Monte Carlo | ✅ Done | `python run_forecast.py` |
| Additional Features | ⏳ TODO | - |

## Remaining TODO

1. **Additional Features**
   - VIX (volatility index) from Yahoo Finance
   - Yield curve (2Y/10Y spread)
   - More on-chain metrics (exchange flows, whale activity)

2. **Integration Tests** - End-to-end pipeline tests (tests/integration/ is empty)

3. **Daily/Weekly Granularity** - Currently monthly only

## Archived Code Reference

Original implementation in `archive/v1/`:
- `src/data_loader.py` - Wilder's RSI reference (lines 213-247)
- `src/scenario_engine.py` - Monte Carlo reference (lines 274-413)
- `src/backtest_engine.py` - Original P.I.T. loop
