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
            'Rates': '^IRX'  # 13 Week Treasury Bill
        }
        self.data = pd.DataFrame()

    def fetch_data(self):
        """
        Fetches 12 years of monthly data to ensure 10 full years of analysis 
        plus lag buffers for indicators.
        """
        print("📡 Fetching Data from Yahoo Finance...")
        
        try:
            # Download bulk data
            # period="12y" captures enough history
            raw = yf.download(
                list(self.tickers.values()), 
                start="2012-01-01", # Explicit start is safer
                interval="1mo", 
                auto_adjust=True, 
                progress=False
            )
            
            # Handle yfinance MultiIndex structure
            df = pd.DataFrame()
            
            # Extract Close prices
            if isinstance(raw.columns, pd.MultiIndex):
                # Usually Level 0 is 'Close' or 'Price'
                if 'Close' in raw.columns.get_level_values(0):
                    df = raw['Close']
                elif 'Close' in raw.columns.get_level_values(1):
                    # Inverted case
                    df = raw.xs('Close', level=1, axis=1)
                else:
                    # Fallback
                    df = raw
            else:
                df = raw['Close'] if 'Close' in raw.columns else raw

            # Rename columns based on our mapping
            # Invert the dictionary to map Ticker -> Name
            inv_map = {v: k for k, v in self.tickers.items()}
            
            # Filter and Rename
            # Note: yfinance columns might be slightly different (e.g. missing symbols)
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
            
            # Forward Fill Macro Data (if DXY/Rates are lagging/missing for last month)
            # This prevents dropping the latest BTC candle just because DXY is a day late
            df = df.ffill()
            
            # Drop rows where BTC is NaN (Crypto history is the constraint)
            df = df.dropna(subset=['BTC'])
            
            self.data = df
            print(f"✅ Raw Data Loaded: {len(df)} months. Last Date: {df.index[-1]}")
            
            self.process_indicators()
            return self.data

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()

    def calculate_monthly_rsi(self, series, period=14):
        """
        Calculates RSI on Monthly data.
        Formula: RSI = 100 - (100 / (1 + RS))
        RS = Avg Gain / Avg Loss
        """
        delta = series.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain/loss
        # First average is simple mean, subsequent are smoothed
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
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
        
        if 'Rates' in df.columns:
            df['Trend_Rates'] = df['Rates'].diff()

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
        # We need the last row (current month) even if it doesn't have "Ret_BTC" (Next Month Return)
        # Actually Ret_BTC is Current Month Return. 
        # We dropna mainly to remove the beginning of history where Lagged vars are NaN.
        # But we must NOT drop the last row just because something minor is missing.
        # We will dropna only if the LAGGED variables are missing (start of array).
        
        cols_to_check = list(shift_cols.values())
        df = df.dropna(subset=cols_to_check)
        
        self.data = df
        print(f"🧠 Processed Indicators. Analysis History: {len(self.data)} months. Last: {self.data.index[-1]}")

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch_data()
    print(df.tail())

