import pandas as pd
import numpy as np

class ScenarioEngine:
    def __init__(self, history_df):
        self.df = history_df
        self.matrix = []
        
    def get_current_conditions(self):
        """
        Returns the conditions of the VERY LAST row in the dataset.
        This represents the 'Now' state used to forecast next month.
        """
        if self.df.empty: return {}
        
        # Get the last COMPLETED month for trend comparison
        # We treat the 'current' partial candle as the state we are in, 
        # but we must be careful about comparing it to fully closed months.
        last = self.df.iloc[-1] 
        prev = self.df.iloc[-2]
        
        # Re-calculate trends manually for the dashboard to ensure no NaN leakage
        dxy_trend_val = last['DXY'] - prev['DXY'] if 'DXY' in last else 0
        rate_trend_val = last['Rates'] - prev['Rates'] if 'Rates' in last else 0

        return {
            "RSI_BTC": last['RSI_BTC'],
            "RSI_SPX": last['RSI_SPX'],
            "RSI_NDX": last['RSI_NDX'],
            "DXY_Trend": dxy_trend_val, 
            "Rate_Trend": rate_trend_val,
            "Date": last.name
        }

    def run_matrix_analysis(self):
        """
        Generates the Matrix Table:
        12 Months x N Scenarios
        """
        print("🧮 Running Scenario Matrix Analysis...")
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        full_results = []
        
        # Define Scenarios using Lambda functions on the dataframe
        # df is filtered by Month first, then these conditions apply
        scenarios = [
            {
                "name": "1. Baseline (All History)",
                "cond": lambda d: [True] * len(d),
                "desc": "Average of last 10 years"
            },
            # --- BTC RSI ---
            {
                "name": "2. High RSI (>65)",
                "cond": lambda d: d['Prev_RSI_BTC'] > 65,
                "desc": "Momentum Overheated"
            },
            {
                "name": "3. Low RSI (<45)",
                "cond": lambda d: d['Prev_RSI_BTC'] < 45,
                "desc": "Oversold / Bottoming"
            },
            # --- Equities ---
            {
                "name": "4. Risk-On (SPX RSI > 60)",
                "cond": lambda d: d['Prev_RSI_SPX'] > 60,
                "desc": "Strong Stock Market"
            },
            {
                "name": "5. Risk-Off (SPX RSI < 40)",
                "cond": lambda d: d['Prev_RSI_SPX'] < 40,
                "desc": "Weak Stock Market"
            },
            # --- Macro ---
            {
                "name": "6. Strong Dollar (Rising)",
                "cond": lambda d: d['Prev_DXY_Trend'] > 0,
                "desc": "DXY Trending Up"
            },
            {
                "name": "7. Weak Dollar (Falling)",
                "cond": lambda d: d['Prev_DXY_Trend'] <= 0,
                "desc": "DXY Trending Down"
            },
            # --- Fed ---
            {
                "name": "8. Hiking/Tight (Rates Up)",
                "cond": lambda d: d['Prev_Rate_Trend'] > 0,
                "desc": "Rates Rising"
            },
            {
                "name": "9. Cutting/Loose (Rates Flat/Down)",
                "cond": lambda d: d['Prev_Rate_Trend'] <= 0,
                "desc": "Rates Falling or Stable"
            },
            # --- Combos ---
            {
                "name": "10. Golden Pocket (Weak DXY + Bull RSI)",
                "cond": lambda d: (d['Prev_DXY_Trend'] <= 0) & (d['Prev_RSI_BTC'] > 50),
                "desc": "Best Case Macro"
            },
            {
                "name": "11. Doom Scenario (Strong DXY + Hiking)",
                "cond": lambda d: (d['Prev_DXY_Trend'] > 0) & (d['Prev_Rate_Trend'] > 0),
                "desc": "Liquidity Crunch"
            }
        ]

        for m_idx in range(1, 13):
            m_name = month_names[m_idx-1]
            month_data = self.df[self.df.index.month == m_idx]
            
            for sc in scenarios:
                # Filter
                mask = sc['cond'](month_data)
                subset = month_data[mask]
                
                if len(subset) > 0:
                    stats = {
                        "Month": m_name,
                        "Scenario": sc['name'],
                        "Desc": sc['desc'],
                        "Count": len(subset),
                        "Win_Rate": (subset['Ret_BTC'] > 0).mean() * 100,
                        "Avg_Return": subset['Ret_BTC'].mean(),
                        "Median_Return": subset['Ret_BTC'].median(),
                        "Best": subset['Ret_BTC'].max(),
                        "Worst": subset['Ret_BTC'].min(),
                        "Matching_Years": subset.index.year.tolist()
                    }
                    full_results.append(stats)
                else:
                    # Empty case
                    full_results.append({
                        "Month": m_name,
                        "Scenario": sc['name'],
                        "Desc": sc['desc'],
                        "Count": 0,
                        "Win_Rate": 0,
                        "Avg_Return": 0,
                        "Median_Return": 0,
                        "Best": 0,
                        "Worst": 0,
                        "Matching_Years": []
                    })

        return pd.DataFrame(full_results)

    def generate_forecast(self, current_price, start_date, months=12, simulations=2000):
        """
        Generates a forecast path.
        Logic:
        - Month 1: Use samples from Historical Months that match CURRENT conditions.
        - Month 2-12: Use samples from 'Baseline' (All Years) for that month.
        """
        print(f"🔮 Generating Forecast Paths... (Price: {current_price})")
        
        # 1. Identify Current State
        current = self.get_current_conditions()
        
        # Create Logic for Matching
        # We match broadly to avoid sample size 0.
        # Default Logic: Match RSI State AND DXY Trend.
        
        is_high_rsi = current['RSI_BTC'] > 60
        is_low_rsi = current['RSI_BTC'] < 45
        is_strong_dxy = current['DXY_Trend'] > 0
        
        # 2. Simulation Setup
        paths = np.zeros((simulations, months))
        price_paths = np.zeros((simulations, months + 1))
        price_paths[:, 0] = current_price
        
        # Time
        # We need dates matching price_paths (Start + 12 months)
        future_dates = [start_date] + [start_date + pd.DateOffset(months=i+1) for i in range(months)]
        
        # Track sample size for transparency
        sample_sizes = []
        
        for t in range(months):
            # Target month is the one we are predicting (t+1)
            target_month = future_dates[t+1].month
            
            # Get pool of historical returns for this month
            month_data = self.df[self.df.index.month == target_month]
            
            # LOGIC CHANGE: Apply Conditional Logic for the first 3 months (t=0, 1, 2)
            # This assumes regimes last at least a quarter.
            if t < 3:
                # FIRST 3 MONTHS: Use Smart Matching
                # Try exact match: RSI State + DXY Trend
                if is_high_rsi:
                    subset = month_data[month_data['Prev_RSI_BTC'] > 60]
                elif is_low_rsi:
                    subset = month_data[month_data['Prev_RSI_BTC'] < 45]
                else:
                    subset = month_data
                
                # Overlay DXY if we still have samples
                if is_strong_dxy:
                    dxy_subset = subset[subset['Prev_DXY_Trend'] > 0]
                else:
                    dxy_subset = subset[subset['Prev_DXY_Trend'] <= 0]
                    
                # Fallback if too strict
                if len(dxy_subset) >= 3:
                    samples = dxy_subset['Ret_BTC'].values
                else:
                    samples = subset['Ret_BTC'].values
                    
                # CRITICAL: Add Sample Size Guard
                # If filtered subset is too small (< 5), force fallback to Baseline immediately
                if len(samples) < 5:
                    samples = month_data['Ret_BTC'].values  # Fallback
                    match_logic = f"Baseline (Insufficient Samples for Month {t+1})"
                else:
                    match_logic = "Smart Match (3-Month Decay)"
            else:
                # Months 4-12: Pure Baseline
                samples = month_data['Ret_BTC'].values
                match_logic = "Baseline"
            
            sample_sizes.append(len(samples))
                
            # Bootstrap Simulation
            if len(samples) > 0:
                # Convert pct to log returns for summation? 
                # Or just use simple compounding: Price * (1 + r/100)
                # Let's use simple returns here as we have monthly discrete chunks
                drawn_returns = np.random.choice(samples, size=simulations)
                paths[:, t] = drawn_returns
                
                # Update Price
                price_paths[:, t+1] = price_paths[:, t] * (1 + drawn_returns/100)
            else:
                price_paths[:, t+1] = price_paths[:, t] # No change if no data

        return future_dates, price_paths, match_logic, sample_sizes[0]  # Return first month sample size

