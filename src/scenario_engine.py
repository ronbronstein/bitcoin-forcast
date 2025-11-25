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
            "Date": last.name,
            # FLAW 4 PREP: Add RSI components (use .get() for safety)
            "BTC_AG": last.get('BTC_AG'),
            "BTC_AL": last.get('BTC_AL'),
        }

        # FLAW 2/10: Add active addresses with corrected column names
        if 'Prev_Active_Addresses' in last.index and pd.notna(last['Prev_Active_Addresses']):
            conditions['Active_Addresses (Lagged)'] = last['Prev_Active_Addresses']
        if 'Prev_Active_Addresses_Z' in last.index and pd.notna(last['Prev_Active_Addresses_Z']):
            conditions['Active_Addresses_Z'] = last['Prev_Active_Addresses_Z']

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
            # --- Network Health (FLAW 10 FIX: Use P.I.T. Z-score) ---
            {
                "name": "15. High Network Activity (Z>1.0)",
                "cond": lambda d: (
                    d['Prev_Active_Addresses_Z'] > 1.0
                ) if 'Prev_Active_Addresses_Z' in d.columns else [False] * len(d),
                "desc": "Activity significantly above 12-month rolling average (P.I.T.)"
            },
            {
                "name": "16. Low Network Activity (Z<-1.0)",
                "cond": lambda d: (
                    d['Prev_Active_Addresses_Z'] < -1.0
                ) if 'Prev_Active_Addresses_Z' in d.columns else [False] * len(d),
                "desc": "Activity significantly below 12-month rolling average (P.I.T.)"
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
        FLAW 4 FIX: Generates a forecast path using Path-Dependent Monte Carlo.
        RSI evolves efficiently by tracking AG/AL components.
        FLAW 6 FIX: Uses time-weighted sampling to address stationarity.
        """
        print(f"🔮 Generating Path-Dependent Forecast Paths... (Price: {current_price})")

        # 1. Setup
        RSI_PERIOD = 14
        RSI_HIGH = 65  # Flaw 8: Acknowledged threshold
        RSI_LOW = 45
        MIN_SAMPLES = 5  # Flaw 5: Minimum sample size threshold

        price_paths = np.zeros((simulations, months + 1))
        price_paths[:, 0] = current_price
        future_dates = [start_date] + [start_date + pd.DateOffset(months=i+1) for i in range(months)]

        # 2. Initialize Simulation States (Vectorized)
        initial_conditions = self.get_current_conditions()

        # Check if components are available
        if initial_conditions.get('BTC_AG') is None or initial_conditions.get('BTC_AL') is None or pd.isna(initial_conditions['BTC_AG']):
            print("❌ Error: Missing or NaN RSI components (AG/AL). Cannot run simulation.")
            return future_dates, price_paths, "Error: Insufficient Data (AG/AL)", 0

        states = {
            'price': np.full(simulations, current_price),
            'BTC_AG': np.full(simulations, initial_conditions['BTC_AG']),
            'BTC_AL': np.full(simulations, initial_conditions['BTC_AL']),
        }

        # Get initial macro conditions (Exogenous/Static assumption)
        is_strong_dxy = initial_conditions.get('DXY_Trend', 0) > 0

        # FLAW 6 FIX (Stationarity): Pre-calculate time-decay weights
        # Half-life of 4 years (prioritizes recent data)
        half_life_years = 4
        decay_lambda = np.log(2) / half_life_years
        # Ensure start_date and self.df.index have compatible types (timezone naive)
        start_date_naive = start_date.tz_localize(None) if start_date.tzinfo is not None else start_date
        df_index_naive = self.df.index.tz_localize(None) if self.df.index.tzinfo is not None else self.df.index

        history_ages_years = (start_date_naive - df_index_naive).days / 365.25
        history_weights = np.exp(-decay_lambda * np.maximum(0, history_ages_years))
        weight_series = pd.Series(history_weights, index=self.df.index)

        match_logic = "Path-Dependent (RSI) + Static (DXY) + Time-Weighted Sampling"
        sample_size_context = len(self.df)  # Report history length

        # 3. Simulation Loop
        for t in range(months):
            target_month = future_dates[t+1].month
            month_data = self.df[self.df.index.month == target_month]

            if month_data.empty:
                price_paths[:, t+1] = price_paths[:, t]
                continue

            # 3.1 Calculate Current Simulated Conditions (RSI)
            # Handle division by zero safely (if AL is 0, RS is effectively infinite, RSI=100)
            rs = np.divide(states['BTC_AG'], states['BTC_AL'], out=np.full_like(states['BTC_AG'], 1000.0), where=states['BTC_AL']!=0)
            sim_rsi = 100 - (100 / (1 + rs))

            # 3.2 Sample Returns based on Simulated Conditions
            drawn_returns = np.zeros(simulations)

            # Optimization: Group simulations by RSI state
            rsi_bins = pd.cut(sim_rsi, bins=[0, RSI_LOW, RSI_HIGH, 100], labels=['Low', 'Mid', 'High'])

            for state in ['Low', 'Mid', 'High']:
                mask = (rsi_bins == state)
                if mask.sum() == 0:
                    continue

                # Define the filter based on the simulated state
                if state == 'Low':
                    subset = month_data[month_data['Prev_RSI_BTC'] < RSI_LOW]
                elif state == 'High':
                    subset = month_data[month_data['Prev_RSI_BTC'] > RSI_HIGH]
                else:
                    subset = month_data  # Neutral RSI

                # Apply Exogenous Macro filter
                if is_strong_dxy:
                    macro_subset = subset[subset['Prev_DXY_Trend'] > 0]
                else:
                    macro_subset = subset[subset['Prev_DXY_Trend'] <= 0]

                # Flaw 5: Robust Fallback Logic
                if len(macro_subset) >= MIN_SAMPLES:
                    samples_df = macro_subset
                elif len(subset) >= MIN_SAMPLES:
                    # Fallback 1: RSI only
                    samples_df = subset
                else:
                    # Fallback 2: Baseline (All data for this month)
                    samples_df = month_data

                # FLAW 8: The previous decay-based blending (decay_rate=0.8) is removed

                # --- Weighted Sampling (FLAW 6) ---
                conditioned_samples = samples_df['Ret_BTC'].values
                sample_indices = samples_df.index

                # Get weights corresponding to these samples
                weights = weight_series.loc[sample_indices].values

                # Normalize weights
                weights_sum = weights.sum()
                if weights_sum > 0 and not np.isclose(weights_sum, 0):
                    normalized_weights = weights / weights_sum
                else:
                    normalized_weights = None  # Fallback to uniform if weights are zero/invalid

                # Draw samples
                if len(conditioned_samples) > 0:
                    drawn_returns[mask] = np.random.choice(conditioned_samples, size=mask.sum(), p=normalized_weights)

            # 3.3 Update States (Price and RSI components)

            # Store previous price before update
            prev_price = states['price'].copy()

            # Update Price
            states['price'] *= (1 + drawn_returns/100)
            price_paths[:, t+1] = states['price']

            # FLAW 4: Efficiently Update RSI Components (Wilder's Smoothing)
            delta = states['price'] - prev_price

            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            # Wilder's update formula (vectorized)
            states['BTC_AG'] = (states['BTC_AG'] * (RSI_PERIOD-1) + gain) / RSI_PERIOD
            states['BTC_AL'] = (states['BTC_AL'] * (RSI_PERIOD-1) + loss) / RSI_PERIOD

        return future_dates, price_paths, match_logic, sample_size_context

