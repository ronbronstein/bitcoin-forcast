# Bitcoin Conditional Probability Forecast Engine

A sophisticated **Point-in-Time (P.I.T.) Forecasting System** that predicts Bitcoin's price path using conditional probability analysis, path-dependent Monte Carlo simulation, and regime-based scenario stratification.

## 🎯 Core Innovation

Unlike simple historical averages, this engine asks:
> *"What typically happens next month when Bitcoin's RSI is **overbought**, stocks are **bullish**, the dollar is **strengthening**, and rates are **rising**?"*

It constructs conditional probability distributions from historical scenarios that match current market conditions, then simulates thousands of price paths where **RSI evolves dynamically** throughout the forecast period.

---

## 🔬 Key Features

### 1. **Point-in-Time (P.I.T.) Processing**
- All indicators calculated only on data available at forecast time
- Eliminates lookahead bias in backtesting
- Walk-forward validation from 2020-2025

### 2. **Path-Dependent Monte Carlo Simulation**
- RSI components (Average Gain/Loss) evolve month-by-month using Wilder's formula
- Each simulation path maintains internal consistency
- Returns sampled based on **simulated** RSI state, not static conditions

### 3. **Comprehensive Data Coverage** (62 columns, 119 months)
- **Assets**: BTC, SPX, NDX, GLD, TLT, USDT (all with full metrics)
- **Indicators**: RSI, AG/AL components, Returns, Trends
- **Network**: Active addresses, hash rate, transactions (+ growth rates)
- **Coverage**: 99.36% complete (2015-12 to 2025-10)

### 4. **Time-Weighted Sampling**
- 4-year half-life exponential decay
- Recent data weighted more heavily
- Prevents stale historical patterns from dominating

---

## 📂 Project Structure

```
bitcoin-forecast/
├── src/                          # Core modules
│   ├── data_loader.py           # Data fetching & P.I.T. processing
│   ├── scenario_engine.py       # Monte Carlo simulation engine
│   └── backtest_engine.py       # Walk-forward validation
│
├── scripts/                      # Executable scripts
│   ├── fetch_and_organize_data.py   # Data collection & CSV organization
│   ├── run_backtest.py              # Run backtest suite
│   ├── main.py                      # Generate forecast
│   ├── backtest_viz.py              # Visualization generator
│   └── dashboard_view.py            # Dashboard renderer
│
├── data/                         # Organized data storage
│   ├── raw/                      # 12 CSV files (prices, volume, network)
│   ├── indicators/               # 12 CSV files (RSI + components)
│   ├── features/                 # 12 CSV files (returns, trends, z-scores)
│   └── processed/                # full_dataset.csv (62 columns)
│
├── outputs/                      # Generated results
│   ├── backtest_results.csv     # Backtest performance data
│   ├── backtest_report.html     # Interactive backtest report
│   └── btc_matrix_v4.html       # Forecast visualization
│
├── tests/                        # Test suite
│   └── test_suite.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Internet connection (for data fetching)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd bitcoin-forecast

# Install dependencies
pip install -r requirements.txt

# Fetch and organize data
python scripts/fetch_and_organize_data.py --refresh
```

### Usage

#### Generate Forecast
```bash
python scripts/main.py
```
Opens interactive HTML dashboard with:
- Current market regime
- 12-month forecast (P10/P50/P90 percentiles)
- Scenario probability matrix

#### Run Backtest
```bash
python scripts/run_backtest.py
```
Generates:
- `outputs/backtest_results.csv` - Raw performance data
- `outputs/backtest_report.html` - Visual analysis report

#### Update Data
```bash
# Use cached data (if < 1 day old)
python scripts/fetch_and_organize_data.py

# Force refresh all data
python scripts/fetch_and_organize_data.py --refresh
```

---

## 📊 Data Overview

### Assets with Full Parity
All 6 assets have: Price, RSI, AG/AL, Returns, Lagged features

| Asset | Description | Coverage |
|-------|-------------|----------|
| **BTC** | Bitcoin | 100% (119 months) |
| **SPX** | S&P 500 | 100% |
| **NDX** | Nasdaq 100 (QQQ) | 100% |
| **GLD** | Gold ETF | 100% |
| **TLT** | 20Y Treasury Bonds | 100% |
| **USDT** | Tether Stablecoin | 80.7% (from 2017-11) |

### Network Metrics
- Active addresses (raw + Z-score normalized)
- Hash rate (raw + growth rate)
- Daily transactions (raw + growth rate)

### Data Sources
- **Yahoo Finance**: Price, volume data
- **Blockchain.com**: On-chain metrics (full history)

---

## 🧠 Methodology

### Fixed Flaws (V4.1)
1. **T-2 Lag Bug** - Training data now uses T-1 correctly
2. **Lookahead Bias** - All features lagged by 1 month
3. **Non-P.I.T. Processing** - Indicators calculated inside backtest loop
4. **Path-Dependent RSI** - AG/AL evolve during simulation
5. **Missing AG/AL** - Full components stored for Monte Carlo
6. **Hardcoded Scenarios** - Dynamic scenario generation
7. **Incomplete Month Handling** - Auto-detected and excluded
8. **Invalid Rebalancing** - Removed mid-simulation rebalancing
9. **NaN Rate Data** - Linear interpolation for missing values
10. **Regime Persistence** - Fixed 3-month minimum for regime changes

### Scenario Conditions
- **RSI Bins**: Oversold (<45), Neutral (45-65), Overbought (>65)
- **SPX/NDX**: Bearish/Bullish (RSI <50 / >50)
- **DXY Trend**: Falling/Rising
- **Rate Trend**: Cutting/Hiking

---

## 📈 Performance

Backtest Results (2020-2025, 48 tests):
- **Directional Accuracy**: 58.3%
- **Mean Absolute % Error**: 19.8%
- **P10-P90 Band Capture**: 43.8% (target: ~80%)

*Note: Model calibration ongoing. Current version emphasizes methodological rigor over prediction accuracy.*

---

## 🔧 Configuration

### Data Cache
- Default freshness: 1 day
- Cached in: `data/processed/full_dataset.csv`
- Modify: `--cache-days` parameter

### Backtest Period
- Edit `scripts/run_backtest.py`:
```python
engine = BacktestEngine(start_year=2020, end_year=2024)
```

---

## 📝 Development

### Adding New Data Sources
1. Add ticker to `src/data_loader.py` (tickers dictionary)
2. Update `process_indicators()` to calculate RSI/returns
3. Add to `scripts/fetch_and_organize_data.py` CSV exports
4. Run: `python scripts/fetch_and_organize_data.py --refresh`

### Running Tests
```bash
python -m pytest tests/
```

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Market data: Yahoo Finance API
- On-chain data: Blockchain.com API
- Methodology: Wilder's RSI (1978), Monte Carlo simulation principles

---

## ⚠️ Disclaimer

**This is a research tool, not financial advice.** Bitcoin is highly volatile. Past performance does not predict future results. Always do your own research and consult financial professionals before making investment decisions.
