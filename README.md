# Bitcoin Scenario Matrix (V4)

A sophisticated **Conditional Probability Engine** that forecasts Bitcoin's future price path by analyzing historical scenarios rather than simple averages.

## 🧠 The Logic: "Market Memory"
Most forecasts just average the last 10 years. This model is smarter. It asks:
*"What usually happens in January... when the **RSI is High** and the **Dollar is Strong**?"*

It builds a massive **Scenario Matrix** of 12 Months x 10 Conditions to find the true historical probabilities for the current market environment.

### Key Factors Analyzed:
1.  **RSI State (14-Month):** Is Bitcoin Overheated (>65) or Oversold (<45)?
2.  **Equities Correlation:** Is the S&P 500 Bullish or Bearish?
3.  **Macro Environment:** Is the Dollar Index (DXY) Rising or Falling?
4.  **Fed Policy:** Are Rates Hiking (Tightening) or Cutting (Easing)?

## 📊 The Dashboard
The tool generates an interactive HTML dashboard (`btc_matrix_v4.html`) containing:
1.  **Real-Time Status:** Shows exactly which "Regime" we are in today.
2.  **The Matrix:** A color-coded probability table for every month under every condition.
3.  **Projected Path:** A Monte Carlo simulation that biases the *immediate* future (Next Month) based on today's specific conditions, then reverts to baseline probability for the distant future.

## 🚀 Quick Start

### Prerequisites
*   Python 3.9+
*   Internet connection (to fetch live Yahoo Finance data)

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python main.py
```

## 📂 Project Structure
*   `main.py` - Entry point. Orchestrates data flow.
*   `data_loader.py` - Fetches 12y history and calculates RSI/Trends.
*   `scenario_engine.py` - The core logic. Calculates the Matrix and runs the simulation.
*   `dashboard_view.py` - Renders the HTML visualization.
