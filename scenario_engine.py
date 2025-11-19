import pandas as pd
import numpy as np

def get_halving_phase(date):
    """
    Returns the number of months since the most recent Bitcoin halving.
    Returns None if date is before first halving.
    
    Halving dates:
    - 2012-11-28: First halving (50 -> 25 BTC)
    - 2016-07-09: Second halving (25 -> 12.5 BTC)
    - 2020-05-11: Third halving (12.5 -> 6.25 BTC)
    - 2024-04-19: Fourth halving (6.25 -> 3.125 BTC)
    - ~2028-04: Fifth halving (projected)
    """
    halvings = [
        pd.Timestamp('2012-11-28'),
        pd.Timestamp('2016-07-09'),
        pd.Timestamp('2020-05-11'),
        pd.Timestamp('2024-04-19')
    ]
    
    for h in reversed(halvings):
        if date >= h:
            months_since = (date - h).days / 30.44
            return months_since
    
    return None  # Before first halving

class ScenarioEngine:
    def __init__(self, history_df):
        self.df = history_df
        self.matrix = []
        
    def get_current_conditions(self):
        """
        Returns the conditions of the VERY LAST row in the dataset.
        This represents the 'Now' state used to forecast next month.
        
        CRITICAL: Uses the same calculated columns that historical scenarios use
        to ensure apples-to-apples comparison.
        """
        if self.df.empty: return {}
        
        last = self.df.iloc[-1]
        
        # Use the actual pre-calculated trend columns (not manual recalc)
        # This ensures consistency with how historical scenarios are matched
        conditions = {
            "RSI_BTC": last['RSI_BTC'],
            "RSI_SPX": last['RSI_SPX'],
            "RSI_NDX": last['RSI_NDX'],
            "DXY_Trend": last['Trend_DXY'],  # FIXED: Use pre-calculated column
            "Rate_Trend": last['Trend_Rates'],  # FIXED: Use pre-calculated column
            "Date": last.name
        }
        
        # Add active addresses if available
        if 'active_addresses' in last.index:
            conditions['Active_Addresses'] = last['active_addresses']
        
        return conditions

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
            {
                "name": "1b. Recent Regime (2020+)",
                "cond": lambda d: d.index.year >= 2020,
                "desc": "Post-institutional adoption era only"
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
            },
            # --- Halving Cycle ---
            {
                "name": "12. Post-Halving Year 1 (0-12mo)",
                "cond": lambda d: d.apply(lambda row: (
                    get_halving_phase(row.name) is not None and 
                    0 <= get_halving_phase(row.name) < 12
                ), axis=1),
                "desc": "First year after halving - historically strongest"
            },
            {
                "name": "13. Post-Halving Year 2 (12-24mo)",
                "cond": lambda d: d.apply(lambda row: (
                    get_halving_phase(row.name) is not None and 
                    12 <= get_halving_phase(row.name) < 24
                ), axis=1),
                "desc": "Second year - bull market continuation"
            },
            {
                "name": "14. Pre-Halving Year (36-48mo)",
                "cond": lambda d: d.apply(lambda row: (
                    get_halving_phase(row.name) is not None and 
                    36 <= get_halving_phase(row.name) < 48
                ), axis=1),
                "desc": "Year before halving - accumulation phase"
            },
            # --- Network Health ---
            {
                "name": "15. High Network Activity",
                "cond": lambda d: (
                    d['active_addresses'] > d['active_addresses'].quantile(0.7)
                ) if 'active_addresses' in d.columns else [False] * len(d),
                "desc": "Top 30% network usage - strong demand"
            },
            {
                "name": "16. Low Network Activity",
                "cond": lambda d: (
                    d['active_addresses'] < d['active_addresses'].quantile(0.3)
                ) if 'active_addresses' in d.columns else [False] * len(d),
                "desc": "Bottom 30% network usage - weak demand"
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
                    # Calculate bootstrap confidence intervals
                    ci_lower, ci_upper = np.nan, np.nan
                    if len(subset) >= 3:
                        # Bootstrap 90% confidence interval for mean return
                        bootstrap_means = []
                        for _ in range(1000):
                            resample = np.random.choice(subset['Ret_BTC'].values, 
                                                       size=len(subset), 
                                                       replace=True)
                            bootstrap_means.append(resample.mean())
                        
                        ci_lower = np.percentile(bootstrap_means, 5)
                        ci_upper = np.percentile(bootstrap_means, 95)
                    
                    stats = {
                        "Month": m_name,
                        "Scenario": sc['name'],
                        "Desc": sc['desc'],
                        "Count": len(subset),
                        "Win_Rate": (subset['Ret_BTC'] > 0).mean() * 100,
                        "Avg_Return": subset['Ret_BTC'].mean(),
                        "Median_Return": subset['Ret_BTC'].median(),
                        "CI_Lower_90": ci_lower,  # NEW
                        "CI_Upper_90": ci_upper,  # NEW
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
                        "CI_Lower_90": np.nan,
                        "CI_Upper_90": np.nan,
                        "Best": 0,
                        "Worst": 0,
                        "Matching_Years": []
                    })

        results_df = pd.DataFrame(full_results)
        
        # Add sample size quality flags
        def get_quality_flag(count):
            if count < 5:
                return "⚠️ Unreliable"
            elif count < 10:
                return "⚠ Small Sample"
            else:
                return "✓ Sufficient"
        
        results_df['Quality'] = results_df['Count'].apply(get_quality_flag)
        
        # Print summary statistics
        total_cells = len(results_df)
        unreliable = len(results_df[results_df['Count'] < 5])
        small = len(results_df[(results_df['Count'] >= 5) & (results_df['Count'] < 10)])
        
        print(f"\n📊 Matrix Quality Summary:")
        print(f"   Total Scenario-Month Cells: {total_cells}")
        print(f"   ⚠️ Unreliable (<5 samples): {unreliable} ({unreliable/total_cells*100:.1f}%)")
        print(f"   ⚠ Small Sample (5-9): {small} ({small/total_cells*100:.1f}%)")
        print(f"   ✓ Sufficient (10+): {total_cells - unreliable - small}")
        
        if unreliable > total_cells * 0.3:
            print(f"   🚨 WARNING: >30% of cells have insufficient data for reliable forecasting")
        
        return results_df

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
        
        # Match scenario definitions exactly (65/45 thresholds)
        is_high_rsi = current['RSI_BTC'] > 65  # FIXED: was 60, now 65
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
        
        # Track the FIRST month's matching logic for display
        first_month_logic = None
        first_month_sample_size = 0
        
        for t in range(months):
            # Target month is the one we are predicting (t+1)
            target_month = future_dates[t+1].month
            
            # Get pool of historical returns for this month
            month_data = self.df[self.df.index.month == target_month]
            
            # 6-MONTH EXPONENTIAL DECAY: Gradually blend from conditioned -> baseline
            # Month 0: 100% conditioned
            # Month 1: 80% conditioned (0.8^1)
            # Month 2: 64% conditioned (0.8^2)
            # ...
            # Month 6: 26% conditioned (0.8^6)
            # Month 7+: Pure baseline
            
            decay_rate = 0.8  # 20% decay per month
            conditioning_weight = decay_rate ** t
            
            if t < 6 and conditioning_weight > 0.25:
                # Apply smart matching with decay weight
                
                # Filter by RSI state
                if is_high_rsi:
                    subset = month_data[month_data['Prev_RSI_BTC'] > 65]
                elif is_low_rsi:
                    subset = month_data[month_data['Prev_RSI_BTC'] < 45]
                else:
                    subset = month_data
                
                # Overlay DXY filter if sufficient samples
                if is_strong_dxy:
                    dxy_subset = subset[subset['Prev_DXY_Trend'] > 0]
                else:
                    dxy_subset = subset[subset['Prev_DXY_Trend'] <= 0]
                
                # Use DXY filter only if we have enough samples
                if len(dxy_subset) >= 3:
                    conditioned_samples = dxy_subset['Ret_BTC'].values
                else:
                    conditioned_samples = subset['Ret_BTC'].values
                
                # Baseline samples
                baseline_samples = month_data['Ret_BTC'].values
                
                # Blend conditioned and baseline based on decay weight
                if len(conditioned_samples) >= 5:
                    # Enough samples: blend
                    n_conditioned = int(simulations * conditioning_weight)
                    n_baseline = simulations - n_conditioned
                    
                    drawn_cond = np.random.choice(conditioned_samples, size=n_conditioned)
                    drawn_base = np.random.choice(baseline_samples, size=n_baseline)
                    samples = np.concatenate([drawn_cond, drawn_base])
                    
                    current_logic = f"Blended (Month {t+1}, {conditioning_weight*100:.0f}% conditioned, {len(conditioned_samples)}y)"
                else:
                    # Not enough conditioned samples: use baseline
                    samples = baseline_samples
                    current_logic = f"Baseline (Insufficient samples for conditioning)"
            else:
                # Month 6+: Pure baseline
                samples = month_data['Ret_BTC'].values
                current_logic = "Baseline (Beyond conditioning window)"
            
            # Store first month's logic
            if t == 0:
                first_month_logic = current_logic
                first_month_sample_size = len(samples)
            
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

        return future_dates, price_paths, first_month_logic, first_month_sample_size

