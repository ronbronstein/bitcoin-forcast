import yfinance as yf
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        self.tickers = {
            # Core Assets
            'BTC': 'BTC-USD',
            'SPX': '^GSPC',
            'NDX': 'QQQ',
            'DXY': 'DX-Y.NYB',
            'Rates': '^IRX',  # 13 Week Treasury Bill (primary)
            # Additional Assets (Expansion)
            'GLD': 'GLD',      # Gold ETF (safe haven)
            'TLT': 'TLT',      # 20Y Treasury Bonds (duration risk)
            'USDT': 'USDT-USD' # Tether (stablecoin/liquidity proxy)
        }
        self.rates_fallback = '^FVX'  # 5-Year Treasury (backup if ^IRX fails)
        self.data = pd.DataFrame()

    def fetch_raw_data(self):
        """
        Fetches 13 years of monthly data, ending at the start of the current month.
        FLAW 7 FIX: Excludes incomplete current month to ensure data integrity.
        """
        print("📡 Fetching Data from Yahoo Finance...")

        try:
            # FLAW 7 FIX: Calculate explicit date range ending at START of current month
            # This ensures we only include complete months
            end_date = pd.Timestamp.now()
            start_of_current_month = end_date.to_period('M').start_time
            start_date = start_of_current_month - pd.DateOffset(years=13)

            print(f"   Date Range: {start_date.strftime('%Y-%m-%d')} to {start_of_current_month.strftime('%Y-%m-%d')} (Exclusive)")
            
            # Download bulk data with the exclusive end date
            raw = yf.download(
                list(self.tickers.values()),
                start=start_date,
                end=start_of_current_month,  # Use start_of_current_month (exclusive)
                interval="1mo",
                auto_adjust=True,
                progress=False
            )
            
            if raw.empty:
                print("❌ ERROR: Yahoo Finance returned no data")
                return pd.DataFrame()
            
            # Handle yfinance MultiIndex structure
            df = pd.DataFrame()
            volume_df = pd.DataFrame()

            # Extract Close prices and Volume
            if isinstance(raw.columns, pd.MultiIndex):
                # Check if 'Close' is in level 0 or level 1
                if 'Close' in raw.columns.get_level_values(0):
                    df = raw['Close'].copy()
                    if 'Volume' in raw.columns.get_level_values(0):
                        volume_df = raw['Volume'].copy()
                elif 'Close' in raw.columns.get_level_values(1):
                    df = raw.xs('Close', level=1, axis=1).copy()
                    if 'Volume' in raw.columns.get_level_values(1):
                        volume_df = raw.xs('Volume', level=1, axis=1).copy()
                else:
                    # Fallback to first level
                    print("⚠️ WARNING: Unexpected column structure, using fallback")
                    df = raw.iloc[:, :len(self.tickers)].copy()
            else:
                df = raw['Close'].copy() if 'Close' in raw.columns else raw.copy()
                if 'Volume' in raw.columns:
                    volume_df = raw['Volume'].copy()
            
            # Rename columns based on our mapping
            inv_map = {v: k for k, v in self.tickers.items()}
            
            # Filter and Rename
            new_cols = {}
            for col in df.columns:
                if col in inv_map:
                    new_cols[col] = inv_map[col]
            
            df = df.rename(columns=new_cols)

            # Add BTC Volume if available
            if not volume_df.empty and 'BTC-USD' in volume_df.columns:
                df['BTC_Volume'] = volume_df['BTC-USD']
                print(f"   ✅ BTC Volume data captured")

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
            
            # FLAW 9 FIX: Use Interpolation for Macro Data Gaps
            # This is more robust than simple forward-fill
            macro_cols_interpolate = ['DXY', 'Rates', 'SPX', 'NDX']

            for col in macro_cols_interpolate:
                if col in df.columns:
                    original_nans = df[col].isna().sum()
                    # Use linear interpolation
                    df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
                    filled_nans = original_nans - df[col].isna().sum()
                    if filled_nans > 0:
                        print(f"   🧠 Interpolated {filled_nans} missing values in {col} (Flaw 9 Fix).")
            
            # Drop rows where BTC is NaN (Crypto history is the constraint)
            before_drop = len(df)
            df = df.dropna(subset=['BTC'])
            after_drop = len(df)
            
            if before_drop > after_drop:
                print(f"   Dropped {before_drop - after_drop} rows with missing BTC data")
            
            self.data = df
            if not df.empty:
                print(f"✅ Raw Data Loaded: {len(df)} complete months. Last Date: {df.index[-1].strftime('%Y-%m-%d')}")

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

            # Fetch Hash Rate (optional, non-blocking)
            hash_rate = self.fetch_blockchain_metric('hash-rate', 'Hash Rate')
            if hash_rate is not None:
                hash_rate.index = hash_rate.index.to_period('M').to_timestamp()
                df = df.join(hash_rate, how='left')
                # Rename column from 'hash-rate' to 'hash_rate'
                if 'hash-rate' in df.columns:
                    df['hash_rate'] = df['hash-rate'].ffill(limit=2).bfill(limit=1)
                    df.drop(columns=['hash-rate'], inplace=True)
                    filled_count = df['hash_rate'].notna().sum()
                    print(f"   ✅ Hash Rate merged: {filled_count}/{len(df)} rows populated")
                self.data = df

            # Fetch Transaction Count (optional, non-blocking)
            transactions = self.fetch_blockchain_metric('n-transactions', 'Transaction Count')
            if transactions is not None:
                transactions.index = transactions.index.to_period('M').to_timestamp()
                df = df.join(transactions, how='left')
                # Rename column from 'n-transactions' to 'n_transactions'
                if 'n-transactions' in df.columns:
                    df['n_transactions'] = df['n-transactions'].ffill(limit=2).bfill(limit=1)
                    df.drop(columns=['n-transactions'], inplace=True)
                    filled_count = df['n_transactions'].notna().sum()
                    print(f"   ✅ Transaction Count merged: {filled_count}/{len(df)} rows populated")
                self.data = df

            # FLAW 3 FIX: DO NOT call process_indicators or validate_data here
            # They will be called by the public fetch_data() wrapper or P.I.T. in backtest

            return self.data

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def calculate_wilder_components(self, series, period=14):
        """
        Calculates Average Gain (AG) and Average Loss (AL) using TRUE Wilder's Smoothing.
        Required for efficient path-dependent simulation (Flaw 4).

        Wilder's formula:
        - First avg = SMA of first 'period' values
        - Subsequent avgs = (prev_avg * (period-1) + current_value) / period

        Returns: (avg_gain, avg_loss) - the components needed for RSI calculation
        """
        delta = series.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Initialize series
        avg_gain = pd.Series(index=series.index, dtype=float)
        avg_loss = pd.Series(index=series.index, dtype=float)

        # Handle short series
        if len(series) <= period:
            return avg_gain, avg_loss

        # First values: Simple Moving Average
        avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
        avg_loss.iloc[period] = loss.iloc[1:period+1].mean()

        # Subsequent values: Wilder's recursive smoothing
        for i in range(period + 1, len(series)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period

        return avg_gain, avg_loss

    def process_indicators(self, data):
        """
        Calculates RSI, Trends, and Lagged States. Must be run P.I.T. (Flaw 3).

        Args:
            data: DataFrame with raw price data to process

        Returns:
            DataFrame with all indicators and lagged features
        """
        df = data.copy()

        # 1. Calculate RSI and Components (Flaw 4 Prep)
        # Expand to all price-based assets (not just BTC, SPX, NDX)
        for asset in ['BTC', 'SPX', 'NDX', 'GLD', 'TLT', 'USDT']:
            if asset in df.columns:
                # Use the new components function
                ag, al = self.calculate_wilder_components(df[asset], 14)
                df[f'{asset}_AG'] = ag
                df[f'{asset}_AL'] = al

                # Calculate RSI from components
                rs = ag / al
                # Handle division by zero (if AL is 0, RSI is 100)
                df[f'RSI_{asset}'] = np.where(al == 0, 100, 100 - (100 / (1 + rs)))

        # 2. Calculate Macro Trends
        # FLAW 9 FIX: Since we interpolated in fetch_raw_data, trend calculation is simple
        if 'DXY' in df.columns:
            df['Trend_DXY'] = df['DXY'].diff()
            df['Trend_DXY'] = df['Trend_DXY'].fillna(0)  # Fill first row NaN

        if 'Rates' in df.columns:
            df['Trend_Rates'] = df['Rates'].diff()
            # REMOVED: complex gap handling loop (obsolete due to interpolation)
            df['Trend_Rates'] = df['Trend_Rates'].fillna(0)  # Fill first row NaN

        # 3. Future Returns (Target variable + Additional Assets)
        # Core assets
        df['Ret_BTC'] = df['BTC'].pct_change() * 100
        df['Ret_SPX'] = df['SPX'].pct_change() * 100
        df['Ret_NDX'] = df['NDX'].pct_change() * 100

        # Additional assets (GLD, TLT, USDT)
        if 'GLD' in df.columns:
            df['Ret_GLD'] = df['GLD'].pct_change() * 100
        if 'TLT' in df.columns:
            df['Ret_TLT'] = df['TLT'].pct_change() * 100
        if 'USDT' in df.columns:
            df['Ret_USDT'] = df['USDT'].pct_change() * 100

        # Network metrics (growth rates)
        if 'hash_rate' in df.columns:
            df['Ret_HashRate'] = df['hash_rate'].pct_change() * 100
        if 'n_transactions' in df.columns:
            df['Ret_Transactions'] = df['n_transactions'].pct_change() * 100

        # 4. Create "Lagged State" Columns (Predictors)
        shift_cols = {
            # Core assets
            'RSI_BTC': 'Prev_RSI_BTC',
            'RSI_SPX': 'Prev_RSI_SPX',
            'RSI_NDX': 'Prev_RSI_NDX',
            'Trend_DXY': 'Prev_DXY_Trend',
            'Trend_Rates': 'Prev_Rate_Trend',
            # Flaw 4 Prep: Lag components for core assets
            'BTC_AG': 'Prev_BTC_AG',
            'BTC_AL': 'Prev_BTC_AL',
            'SPX_AG': 'Prev_SPX_AG',
            'SPX_AL': 'Prev_SPX_AL',
            'NDX_AG': 'Prev_NDX_AG',
            'NDX_AL': 'Prev_NDX_AL',
        }

        # Additional assets - RSI and components
        for asset in ['GLD', 'TLT', 'USDT']:
            if f'RSI_{asset}' in df.columns:
                shift_cols[f'RSI_{asset}'] = f'Prev_RSI_{asset}'
            if f'{asset}_AG' in df.columns:
                shift_cols[f'{asset}_AG'] = f'Prev_{asset}_AG'
            if f'{asset}_AL' in df.columns:
                shift_cols[f'{asset}_AL'] = f'Prev_{asset}_AL'

        # FLAW 2 FIX: Lag Active Addresses
        if 'active_addresses' in df.columns:
            shift_cols['active_addresses'] = 'Prev_Active_Addresses'

        for src, dest in shift_cols.items():
            if src in df.columns:
                df[dest] = df[src].shift(1)

        # FLAW 10 FIX: Calculate Point-in-Time Relative Features (Rolling Z-Score)
        if 'Prev_Active_Addresses' in df.columns:
            # Calculate rolling statistics (12 months window) on the lagged data P.I.T.
            rolling_mean = df['Prev_Active_Addresses'].rolling(window=12, min_periods=6).mean()
            rolling_std = df['Prev_Active_Addresses'].rolling(window=12, min_periods=6).std()

            # Calculate Z-score
            df['Prev_Active_Addresses_Z'] = (df['Prev_Active_Addresses'] - rolling_mean) / rolling_std
            # Handle division by zero and inf/nan
            df['Prev_Active_Addresses_Z'] = df['Prev_Active_Addresses_Z'].replace([np.inf, -np.inf], 0).fillna(0)

        # Data Cleaning: Drop rows where CORE lagged variables (predictors) are NaN
        # Network data is optional - don't require it for all history
        core_required_cols = [
            'Prev_RSI_BTC', 'Prev_RSI_SPX', 'Prev_RSI_NDX',
            'Prev_DXY_Trend', 'Prev_Rate_Trend',
            'Prev_BTC_AG', 'Prev_BTC_AL'
        ]
        # Only check columns that actually exist
        cols_to_check = [col for col in core_required_cols if col in df.columns]
        df = df.dropna(subset=cols_to_check)

        if not df.empty:
            # Ensure index is clean after processing
            print(f"🧠 Processed Indicators. Analysis History: {len(df)} months. Last: {df.index[-1].strftime('%Y-%m-%d')}")

        return df  # Return the processed dataframe

    def validate_data(self, data):
        """
        Validates data quality to catch API issues, missing data, or extreme outliers.
        Prints warnings but doesn't stop execution.

        Args:
            data: DataFrame with processed indicators to validate
        """
        print("🔍 Validating Data Quality...")

        df = data
        
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

    def fetch_data(self):
        """
        Public API for main.py: Fetches raw data, processes indicators, and validates for live use.

        Returns:
            DataFrame with all indicators processed and validated
        """
        raw_df = self.fetch_raw_data()
        if not raw_df.empty:
            processed_df = self.process_indicators(raw_df)
            if not processed_df.empty:
                self.validate_data(processed_df)
                self.data = processed_df  # Store in instance for backward compatibility
            return processed_df
        return pd.DataFrame()

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
                'timespan': 'all',  # Get full history back to 2009
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

    def fetch_blockchain_metric(self, metric_name, display_name):
        """
        Generic method to fetch any Blockchain.com metric.

        Args:
            metric_name: API metric name (e.g., 'hash-rate', 'n-transactions')
            display_name: Human-readable name for logging

        Returns: Series with monthly data, or None if fails
        """
        print(f"🌐 Fetching {display_name} from Blockchain.com...")

        try:
            import requests

            url = f"https://api.blockchain.info/charts/{metric_name}"
            params = {
                'timespan': 'all',  # Get full history
                'rollingAverage': '7days',
                'format': 'json'
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                if 'values' not in data:
                    print(f"   ❌ API response missing 'values' key")
                    return None

                # Parse into DataFrame
                values = pd.DataFrame(data['values'])

                if values.empty:
                    print(f"   ❌ API returned empty values array")
                    return None

                values['date'] = pd.to_datetime(values['x'], unit='s')
                values = values.rename(columns={'y': metric_name})
                values = values.set_index('date')

                # Resample to monthly (end of month to match our data)
                monthly_data = values[metric_name].resample('ME').mean()

                print(f"✅ {display_name}: {len(monthly_data)} months fetched")
                print(f"   Date Range: {monthly_data.index[0].strftime('%Y-%m-%d')} to {monthly_data.index[-1].strftime('%Y-%m-%d')}")

                return monthly_data

            else:
                print(f"⚠️ Blockchain.com API returned status {response.status_code}")
                return None

        except Exception as e:
            print(f"⚠️ Failed to fetch {display_name}: {e}")
            return None

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch_data()
    print(df.tail())

