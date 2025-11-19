import yfinance as yf
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        self.tickers = {
            'BTC': 'BTC-USD',
            'SPX': '^GSPC',
            'NDX': 'QQQ',
            'DXY': 'DX-Y.NYB',
            'Rates': '^IRX'  # 13 Week Treasury Bill (primary)
        }
        self.rates_fallback = '^FVX'  # 5-Year Treasury (backup if ^IRX fails)
        self.data = pd.DataFrame()

    def fetch_data(self):
        """
        Fetches 13 years of monthly data (to ensure 12 full years + indicators).
        Uses explicit date range instead of period="max" for reliability.
        """
        print("📡 Fetching Data from Yahoo Finance...")
        
        try:
            # Calculate explicit date range
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=13)
            
            print(f"   Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Download bulk data with explicit dates
            raw = yf.download(
                list(self.tickers.values()), 
                start=start_date,
                end=end_date,
                interval="1mo", 
                auto_adjust=True, 
                progress=False
            )
            
            if raw.empty:
                print("❌ ERROR: Yahoo Finance returned no data")
                return pd.DataFrame()
            
            # Handle yfinance MultiIndex structure
            df = pd.DataFrame()
            
            # Extract Close prices
            if isinstance(raw.columns, pd.MultiIndex):
                # Check if 'Close' is in level 0 or level 1
                if 'Close' in raw.columns.get_level_values(0):
                    df = raw['Close']
                elif 'Close' in raw.columns.get_level_values(1):
                    df = raw.xs('Close', level=1, axis=1)
                else:
                    # Fallback to first level
                    print("⚠️ WARNING: Unexpected column structure, using fallback")
                    df = raw.iloc[:, :len(self.tickers)]
            else:
                df = raw['Close'] if 'Close' in raw.columns else raw
            
            # Rename columns based on our mapping
            inv_map = {v: k for k, v in self.tickers.items()}
            
            # Filter and Rename
            new_cols = {}
            for col in df.columns:
                if col in inv_map:
                    new_cols[col] = inv_map[col]
            
            df = df.rename(columns=new_cols)
            
            # Ensure we have all critical columns
            required = ['BTC', 'SPX', 'NDX', 'DXY', 'Rates']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"⚠️ Warning: Missing data for {missing}")
            
            # Check data freshness
            last_date = df.index[-1]
            days_old = (pd.Timestamp.now() - last_date).days
            print(f"   Last Data Point: {last_date.strftime('%Y-%m-%d')} ({days_old} days old)")
            
            if days_old > 45:
                print(f"   🚨 CRITICAL: Data is {days_old} days old - Yahoo Finance may be down!")
            
            # Check if Rates data is mostly NaN
            if 'Rates' in df.columns:
                rates_nulls = df['Rates'].isna().sum()
                rates_pct_null = rates_nulls / len(df) * 100
                
                if rates_pct_null > 30:
                    print(f"   ⚠️ ^IRX has {rates_pct_null:.0f}% missing data, trying fallback...")
                    
                    # Try fallback ticker
                    fallback_data = yf.download(
                        self.rates_fallback,
                        start=start_date,
                        end=end_date,
                        interval="1mo",
                        progress=False
                    )
                    
                    if not fallback_data.empty:
                        if isinstance(fallback_data.columns, pd.MultiIndex):
                            fallback_rates = fallback_data['Close']
                        else:
                            fallback_rates = fallback_data['Close'] if 'Close' in fallback_data.columns else fallback_data
                        
                        # Use fallback where original is NaN
                        df['Rates'] = df['Rates'].fillna(fallback_rates)
                        print(f"   ✅ Filled {rates_nulls} missing Rates values with {self.rates_fallback}")
            
            # Forward Fill Macro Data - but ONLY 1 period max
            macro_cols = ['DXY', 'Rates', 'SPX', 'NDX']
            
            for col in macro_cols:
                if col in df.columns:
                    # Fill only 1 period forward
                    original_nulls = df[col].isna().sum()
                    df[col] = df[col].ffill(limit=1)
                    filled_nulls = original_nulls - df[col].isna().sum()
                    
                    if filled_nulls > 0:
                        print(f"   ⚠️ Forward-filled {filled_nulls} missing value(s) in {col}")
                    
                    # Check if last value is stale (forward-filled)
                    if len(df) >= 2 and df[col].iloc[-1] == df[col].iloc[-2]:
                        print(f"   ⚠️ WARNING: {col} last value may be forward-filled (stale data)")
            
            # Drop rows where BTC is NaN (Crypto history is the constraint)
            before_drop = len(df)
            df = df.dropna(subset=['BTC'])
            after_drop = len(df)
            
            if before_drop > after_drop:
                print(f"   Dropped {before_drop - after_drop} rows with missing BTC data")
            
            self.data = df
            print(f"✅ Raw Data Loaded: {len(df)} months. Last Date: {df.index[-1].strftime('%Y-%m-%d')}")
            
            # Fetch Active Addresses (optional, non-blocking)
            active_addr = self.fetch_active_addresses()
            if active_addr is not None:
                # Convert active addresses index to month-start to match main dataframe
                # The resample('ME') gives month-end, but we need month-start
                active_addr.index = active_addr.index.to_period('M').to_timestamp()
                
                # Merge with main data (left join - keep all BTC rows)
                df = df.join(active_addr, how='left')
                
                # Forward fill addresses (up to 2 months) and backward fill recent gaps
                df['active_addresses'] = df['active_addresses'].ffill(limit=2).bfill(limit=1)
                
                filled_count = df['active_addresses'].notna().sum()
                print(f"   ✅ Active Addresses merged: {filled_count}/{len(df)} rows populated")
                
                self.data = df
            
            self.process_indicators()
            self.validate_data()
            
            return self.data

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def calculate_monthly_rsi(self, series, period=14):
        """
        Calculates RSI using TRUE Wilder's Smoothing method.
        This is more accurate than EWMA approximation.
        
        Wilder's formula:
        - First avg = SMA of first 'period' values
        - Subsequent avgs = (prev_avg * (period-1) + current_value) / period
        """
        delta = series.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Initialize series for Wilder's smoothing
        avg_gain = pd.Series(index=series.index, dtype=float)
        avg_loss = pd.Series(index=series.index, dtype=float)
        
        # First values: Simple Moving Average
        avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
        avg_loss.iloc[period] = loss.iloc[1:period+1].mean()
        
        # Subsequent values: Wilder's recursive smoothing
        for i in range(period + 1, len(series)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def process_indicators(self):
        """
        Calculates RSI, Trends, and Lagged States for the Scenario Matrix.
        """
        df = self.data.copy()
        
        # 1. Calculate RSI (14 Month) for Assets
        for asset in ['BTC', 'SPX', 'NDX']:
            if asset in df.columns:
                df[f'RSI_{asset}'] = self.calculate_monthly_rsi(df[asset], 14)

        # 2. Calculate Macro Trends (Month-over-Month change)
        if 'DXY' in df.columns:
            df['Trend_DXY'] = df['DXY'].diff()
            # Fill NaN for first row (no previous value)
            df.loc[df.index[0], 'Trend_DXY'] = 0
        
        if 'Rates' in df.columns:
            # Calculate trend, but handle gaps by using last known value
            df['Trend_Rates'] = df['Rates'].diff()
            
            # For rows where previous Rates was NaN, calculate from last known Rates
            rates_series = df['Rates'].copy()
            for i in range(1, len(df)):
                if pd.isna(df.loc[df.index[i], 'Trend_Rates']):
                    # Find last non-NaN Rates value
                    prev_rates = rates_series.iloc[:i].dropna()
                    if len(prev_rates) > 0:
                        last_known_rate = prev_rates.iloc[-1]
                        current_rate = df.loc[df.index[i], 'Rates']
                        if not pd.isna(current_rate):
                            df.loc[df.index[i], 'Trend_Rates'] = current_rate - last_known_rate
            
            # Fill first row NaN with 0
            if pd.isna(df.loc[df.index[0], 'Trend_Rates']):
                df.loc[df.index[0], 'Trend_Rates'] = 0

        # 3. Create "Lagged State" Columns
        # Crucial: We want to know the condition at the END of the Previous Month
        # to predict the Return of the Current Month.
        
        # Future Returns (The target we want to predict)
        df['Ret_BTC'] = df['BTC'].pct_change() * 100
        df['Ret_SPX'] = df['SPX'].pct_change() * 100
        df['Ret_NDX'] = df['NDX'].pct_change() * 100
        
        # Conditions (Shifted by 1 so row 'Jan' has 'Dec' conditions)
        shift_cols = {
            'RSI_BTC': 'Prev_RSI_BTC',
            'RSI_SPX': 'Prev_RSI_SPX',
            'RSI_NDX': 'Prev_RSI_NDX',
            'Trend_DXY': 'Prev_DXY_Trend',
            'Trend_Rates': 'Prev_Rate_Trend'
        }
        
        for src, dest in shift_cols.items():
            if src in df.columns:
                df[dest] = df[src].shift(1)
        
        # Be careful with dropna here too. 
        # We need the last row (current month) even if it doesn't have lagged values.
        # We dropna mainly to remove the beginning of history where Lagged vars are NaN.
        # But we must NOT drop the last row - we need it for current conditions!
        
        cols_to_check = list(shift_cols.values())
        
        # Save the last row before dropping NaN
        last_row = df.iloc[[-1]].copy()
        
        # Drop rows with missing lagged variables (removes beginning of history)
        df = df.dropna(subset=cols_to_check)
        
        # Re-append the last row even if it has NaN lagged values
        # This ensures we have current conditions available
        if len(df) == 0 or df.index[-1] != last_row.index[0]:
            df = pd.concat([df, last_row])
        
        self.data = df
        print(f"🧠 Processed Indicators. Analysis History: {len(self.data)} months. Last: {self.data.index[-1]}")

    def validate_data(self):
        """
        Validates data quality to catch API issues, missing data, or extreme outliers.
        Prints warnings but doesn't stop execution.
        """
        print("🔍 Validating Data Quality...")
        
        df = self.data
        
        # 1. Check for data completeness
        expected_months = (df.index[-1] - df.index[0]).days / 30.44
        actual_months = len(df)
        missing_pct = (expected_months - actual_months) / expected_months * 100
        
        if missing_pct > 5:
            print(f"⚠️ WARNING: Missing ~{int(expected_months - actual_months)} months of data ({missing_pct:.1f}% gaps)")
        else:
            print(f"✅ Data Completeness: {actual_months} months, minimal gaps")
        
        # 2. Check for extreme returns (potential data errors)
        for col in ['BTC', 'SPX', 'NDX']:
            if col in df.columns and f'Ret_{col}' in df.columns:
                extreme = df[f'Ret_{col}'].abs() > 80  # 80% monthly move
                if extreme.any():
                    extreme_dates = df[extreme].index.tolist()
                    print(f"⚠️ WARNING: Extreme monthly returns in {col}: {extreme_dates}")
        
        # 3. Validate RSI bounds
        for col in ['RSI_BTC', 'RSI_SPX', 'RSI_NDX']:
            if col in df.columns:
                invalid = (df[col] < 0) | (df[col] > 100)
                if invalid.any():
                    print(f"❌ ERROR: Invalid RSI values in {col} (outside 0-100 range)")
                else:
                    print(f"✅ {col}: Valid range")
        
        # 4. Check for recent data
        days_since_last = (pd.Timestamp.now() - df.index[-1]).days
        if days_since_last > 45:
            print(f"⚠️ WARNING: Last data point is {days_since_last} days old (stale)")
        else:
            print(f"✅ Data Freshness: Last update {days_since_last} days ago")
        
        print("")

    def fetch_active_addresses(self):
        """
        Fetches Bitcoin active addresses from Blockchain.com's free API.
        Resamples to monthly frequency to match our data.
        
        Returns: Series with monthly average active addresses, or None if fails
        """
        print("🌐 Fetching Active Addresses from Blockchain.com...")
        
        try:
            import requests
            
            url = "https://api.blockchain.info/charts/n-unique-addresses"
            params = {
                'timespan': '5years',
                'rollingAverage': '7days',
                'format': 'json'
            }
            
            print(f"   Requesting: {url}")
            response = requests.get(url, params=params, timeout=15)
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' not in data:
                    print(f"   ❌ API response missing 'values' key")
                    print(f"   Response keys: {list(data.keys())}")
                    return None
                
                # Parse into DataFrame
                values = pd.DataFrame(data['values'])
                
                if values.empty:
                    print(f"   ❌ API returned empty values array")
                    return None
                
                values['date'] = pd.to_datetime(values['x'], unit='s')
                values = values.rename(columns={'y': 'active_addresses'})
                values = values.set_index('date')
                
                # Resample to monthly (end of month to match our data)
                monthly_addresses = values['active_addresses'].resample('ME').mean()
                
                print(f"✅ Active Addresses: {len(monthly_addresses)} months fetched")
                print(f"   Date Range: {monthly_addresses.index[0].strftime('%Y-%m-%d')} to {monthly_addresses.index[-1].strftime('%Y-%m-%d')}")
                print(f"   Latest: {monthly_addresses.iloc[-1]:,.0f} addresses/day")
                
                return monthly_addresses
                
            else:
                print(f"⚠️ Blockchain.com API returned status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"⚠️ Failed to fetch active addresses: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch_data()
    print(df.tail())

