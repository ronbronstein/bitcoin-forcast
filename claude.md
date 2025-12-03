# Bitcoin Forecast Engine v2.0

Point-in-Time (P.I.T.) Bitcoin forecasting with walk-forward backtesting.

## Stack
Python 3.9+ | pandas | numpy | pytest

## Structure
```
src/
├── data/       loader, cleaner, indicators
├── features/   builder, validator
├── models/     baseline, evaluator
├── backtest/   engine, reporter
└── utils/      dates
tests/unit/     135 tests
data/           54 cols, 119 months (2015-12 to 2025-10)
```

## Commands
```bash
python run_backtest.py              # Run baseline backtest
python run_backtest.py --detailed   # Show month-by-month
pytest tests/ -v                    # Run all tests
```

## Critical Rules

### 1. Point-in-Time (P.I.T.)
ALL data access must use `as_of_date` parameter. Never use future data.
```python
train_data = loader.get_data_as_of(as_of_date)  # CORRECT
train_data = loader.load_full_dataset()          # WRONG in backtest
```

### 2. Features Must Be Lagged
All features start with `Prev_` (shifted 1 month). Unlagged features = data leakage.
```python
feature_cols = ['Prev_RSI_BTC', 'Prev_RSI_SPX']  # CORRECT
feature_cols = ['RSI_BTC', 'RSI_SPX']            # WRONG - leakage!
```

### 3. Wilder's RSI (NOT ewm)
```python
avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14  # CORRECT
avg_gain = gain.ewm(span=14).mean()                 # WRONG
```

## Current Baseline (2020-2024)
| Metric | Value |
|--------|-------|
| Directional Accuracy | 56.7% |
| MAPE | 16.0% |
| P10-P90 Capture | 85.0% |

## Key Files
- `src/data/loader.py` - DataLoader with P.I.T. filtering
- `src/features/builder.py` - FeatureBuilder (enforces Prev_* columns)
- `src/backtest/engine.py` - Walk-forward BacktestEngine
- `data/processed/full_dataset.csv` - Master dataset (no USDT)
