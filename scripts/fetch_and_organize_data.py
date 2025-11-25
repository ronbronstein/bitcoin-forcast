"""
Bitcoin Forecast - Data Fetcher & Organizer

Fetches all data sources and organizes them into clean CSV files.
Run this periodically to update the data cache.

Usage:
    python3 fetch_and_organize_data.py              # Normal run (uses cache if fresh)
    python3 fetch_and_organize_data.py --refresh    # Force re-fetch all data
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
import pandas as pd
from src import data_loader

def check_cache_freshness(cache_file, max_age_days=1):
    """Check if cache file exists and is fresh enough"""
    if not os.path.exists(cache_file):
        return False

    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
    return file_age < timedelta(days=max_age_days)

def save_metadata(output_dir, df, source):
    """Save metadata about the dataset"""
    metadata = {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': source,
        'date_range_start': df.index[0].strftime('%Y-%m-%d'),
        'date_range_end': df.index[-1].strftime('%Y-%m-%d'),
        'total_months': len(df),
        'columns': list(df.columns)
    }

    metadata_file = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"   📄 Metadata saved: {metadata_file}")

def save_csv(df, filepath, columns, description):
    """Helper to save a CSV with specified columns"""
    if all(col in df.columns for col in columns):
        subset = df[columns].copy()
        subset.to_csv(filepath, index=True)
        print(f"   ✅ {description}: {filepath} ({len(subset)} rows)")
        return True
    else:
        missing = [col for col in columns if col not in df.columns]
        print(f"   ⚠️  Skipped {description}: missing columns {missing}")
        return False

def organize_data(force_refresh=False):
    """Main function to fetch and organize all data"""
    print("="*70)
    print("🗂️  BITCOIN FORECAST - DATA ORGANIZATION")
    print("="*70)

    cache_file = 'data/processed/full_dataset.csv'

    # Check if we can use cached data
    if not force_refresh and check_cache_freshness(cache_file):
        print("\n📦 Using cached data (fresh)...")
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"   Loaded {len(df)} months from cache")
        except Exception as e:
            print(f"   ⚠️  Cache load failed: {e}")
            print("   Fetching fresh data...")
            df = None
    else:
        df = None

    # Fetch fresh data if needed
    if df is None:
        if force_refresh:
            print("\n🔄 Force refresh requested...")
        else:
            print("\n⏰ Cache stale or missing...")

        print("📡 Fetching fresh data from sources...")
        loader = data_loader.DataLoader()

        # Fetch raw data
        raw_df = loader.fetch_raw_data()
        if raw_df.empty:
            print("❌ Failed to fetch data")
            return False

        # Process indicators
        df = loader.process_indicators(raw_df)
        if df.empty:
            print("❌ Failed to process indicators")
            return False

        print(f"✅ Data ready: {len(df)} months ({df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')})")

    print("\n📂 Organizing data into CSV files...")

    # ===== 1. RAW PRICES =====
    print("\n1️⃣  Raw Prices:")
    raw_dir = 'data/raw'

    # Core Assets
    save_csv(df, f'{raw_dir}/prices_btc.csv', ['BTC'], 'Bitcoin Price')
    save_csv(df, f'{raw_dir}/prices_spx.csv', ['SPX'], 'S&P 500 Index')
    save_csv(df, f'{raw_dir}/prices_ndx.csv', ['NDX'], 'Nasdaq 100 (QQQ)')
    save_csv(df, f'{raw_dir}/prices_dxy.csv', ['DXY'], 'US Dollar Index')
    save_csv(df, f'{raw_dir}/prices_rates.csv', ['Rates'], 'Interest Rates (13W Treasury)')

    # Additional Assets
    save_csv(df, f'{raw_dir}/prices_gold.csv', ['GLD'], 'Gold ETF (GLD)')
    save_csv(df, f'{raw_dir}/prices_bonds.csv', ['TLT'], '20Y Treasury Bonds (TLT)')
    save_csv(df, f'{raw_dir}/prices_stablecoin.csv', ['USDT'], 'Tether Stablecoin (USDT)')

    # Volume Data
    if 'BTC_Volume' in df.columns:
        save_csv(df, f'{raw_dir}/volume_btc.csv', ['BTC_Volume'], 'Bitcoin Trading Volume')

    # Network Data (Blockchain.com)
    if 'active_addresses' in df.columns:
        save_csv(df, f'{raw_dir}/network_active_addresses.csv',
                ['active_addresses'], 'Bitcoin Active Addresses')

    if 'hash_rate' in df.columns:
        save_csv(df, f'{raw_dir}/network_hashrate.csv',
                ['hash_rate'], 'Bitcoin Network Hash Rate')

    if 'n_transactions' in df.columns:
        save_csv(df, f'{raw_dir}/network_transactions.csv',
                ['n_transactions'], 'Bitcoin Daily Transaction Count')

    save_metadata(raw_dir, df[['BTC', 'SPX', 'NDX', 'DXY', 'Rates']], 'Yahoo Finance + Blockchain.com')

    # ===== 2. TECHNICAL INDICATORS =====
    print("\n2️⃣  Technical Indicators (RSI):")
    ind_dir = 'data/indicators'

    # Core Assets RSI
    save_csv(df, f'{ind_dir}/rsi_btc.csv', ['RSI_BTC'], 'Bitcoin RSI (14-month)')
    save_csv(df, f'{ind_dir}/rsi_spx.csv', ['RSI_SPX'], 'S&P 500 RSI (14-month)')
    save_csv(df, f'{ind_dir}/rsi_ndx.csv', ['RSI_NDX'], 'Nasdaq RSI (14-month)')

    # Additional Assets RSI
    save_csv(df, f'{ind_dir}/rsi_gld.csv', ['RSI_GLD'], 'Gold RSI (14-month)')
    save_csv(df, f'{ind_dir}/rsi_tlt.csv', ['RSI_TLT'], 'Bonds RSI (14-month)')
    save_csv(df, f'{ind_dir}/rsi_usdt.csv', ['RSI_USDT'], 'Tether RSI (14-month)')

    # RSI Components (for path-dependent simulation)
    if all(col in df.columns for col in ['BTC_AG', 'BTC_AL']):
        save_csv(df, f'{ind_dir}/rsi_components_btc.csv',
                ['BTC_AG', 'BTC_AL'], "Bitcoin RSI Components (AG/AL)")

    if all(col in df.columns for col in ['SPX_AG', 'SPX_AL']):
        save_csv(df, f'{ind_dir}/rsi_components_spx.csv',
                ['SPX_AG', 'SPX_AL'], "S&P 500 RSI Components (AG/AL)")

    if all(col in df.columns for col in ['NDX_AG', 'NDX_AL']):
        save_csv(df, f'{ind_dir}/rsi_components_ndx.csv',
                ['NDX_AG', 'NDX_AL'], "Nasdaq RSI Components (AG/AL)")

    if all(col in df.columns for col in ['GLD_AG', 'GLD_AL']):
        save_csv(df, f'{ind_dir}/rsi_components_gld.csv',
                ['GLD_AG', 'GLD_AL'], "Gold RSI Components (AG/AL)")

    if all(col in df.columns for col in ['TLT_AG', 'TLT_AL']):
        save_csv(df, f'{ind_dir}/rsi_components_tlt.csv',
                ['TLT_AG', 'TLT_AL'], "Bonds RSI Components (AG/AL)")

    if all(col in df.columns for col in ['USDT_AG', 'USDT_AL']):
        save_csv(df, f'{ind_dir}/rsi_components_usdt.csv',
                ['USDT_AG', 'USDT_AL'], "Tether RSI Components (AG/AL)")

    save_metadata(ind_dir, df[['RSI_BTC', 'RSI_SPX', 'RSI_NDX']], 'Calculated (Wilder\'s Method)')

    # ===== 3. DERIVED FEATURES =====
    print("\n3️⃣  Derived Features:")
    feat_dir = 'data/features'

    # Returns - Core Assets
    save_csv(df, f'{feat_dir}/returns_btc.csv', ['Ret_BTC'], 'Bitcoin Monthly Returns (%)')
    save_csv(df, f'{feat_dir}/returns_spx.csv', ['Ret_SPX'], 'S&P 500 Monthly Returns (%)')
    save_csv(df, f'{feat_dir}/returns_ndx.csv', ['Ret_NDX'], 'Nasdaq Monthly Returns (%)')

    # Returns - Additional Assets
    save_csv(df, f'{feat_dir}/returns_gld.csv', ['Ret_GLD'], 'Gold Monthly Returns (%)')
    save_csv(df, f'{feat_dir}/returns_tlt.csv', ['Ret_TLT'], 'Bonds Monthly Returns (%)')
    save_csv(df, f'{feat_dir}/returns_usdt.csv', ['Ret_USDT'], 'Tether Monthly Returns (%)')

    # Returns - Network Metrics (Growth Rates)
    save_csv(df, f'{feat_dir}/returns_hashrate.csv', ['Ret_HashRate'], 'Hash Rate Growth (%)')
    save_csv(df, f'{feat_dir}/returns_transactions.csv', ['Ret_Transactions'], 'Transaction Count Growth (%)')

    # Macro Trends
    save_csv(df, f'{feat_dir}/trend_dxy.csv', ['Trend_DXY'], 'Dollar Index Trend (MoM change)')
    save_csv(df, f'{feat_dir}/trend_rates.csv', ['Trend_Rates'], 'Interest Rate Trend (MoM change)')

    # Network Data (normalized)
    if 'Prev_Active_Addresses_Z' in df.columns:
        save_csv(df, f'{feat_dir}/network_zscore.csv',
                ['Prev_Active_Addresses_Z'], 'Network Activity Z-Score (12mo rolling)')

    # Lagged Features (predictors) - Expanded
    lagged_cols = [
        # Core assets
        'Prev_RSI_BTC', 'Prev_RSI_SPX', 'Prev_RSI_NDX',
        'Prev_DXY_Trend', 'Prev_Rate_Trend',
        'Prev_BTC_AG', 'Prev_BTC_AL',
        'Prev_SPX_AG', 'Prev_SPX_AL',
        'Prev_NDX_AG', 'Prev_NDX_AL',
        # Additional assets
        'Prev_RSI_GLD', 'Prev_RSI_TLT', 'Prev_RSI_USDT',
        'Prev_GLD_AG', 'Prev_GLD_AL',
        'Prev_TLT_AG', 'Prev_TLT_AL',
        'Prev_USDT_AG', 'Prev_USDT_AL',
        # Network
        'Prev_Active_Addresses'
    ]
    available_lagged = [col for col in lagged_cols if col in df.columns]

    if available_lagged:
        save_csv(df, f'{feat_dir}/lagged_features.csv',
                available_lagged, 'Lagged Features (T-1 predictors)')

    save_metadata(feat_dir, df[['Ret_BTC', 'Trend_DXY', 'Trend_Rates']], 'Calculated (derived from raw)')

    # ===== 4. FULL DATASET (Processed) =====
    print("\n4️⃣  Full Processed Dataset:")
    proc_dir = 'data/processed'

    # Save complete dataset
    full_path = f'{proc_dir}/full_dataset.csv'
    df.to_csv(full_path, index=True)
    print(f"   ✅ Complete Dataset: {full_path}")
    print(f"      Columns: {len(df.columns)}, Rows: {len(df)}")
    print(f"      Size: {os.path.getsize(full_path) / 1024:.1f} KB")

    save_metadata(proc_dir, df, 'All sources combined & processed')

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("✨ DATA ORGANIZATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Total Months: {len(df)}")
    print(f"   Total Columns: {len(df.columns)}")

    # Count files per directory
    for subdir in ['raw', 'indicators', 'features', 'processed']:
        dir_path = f'data/{subdir}'
        csv_count = len([f for f in os.listdir(dir_path) if f.endswith('.csv')])
        print(f"   {subdir.capitalize()}: {csv_count} CSV files")

    print(f"\n📁 Data stored in: ./data/")
    print(f"   └── Use these CSVs for analysis, visualization, or other projects")

    return True

def main():
    parser = argparse.ArgumentParser(description='Fetch and organize Bitcoin forecast data')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh data (ignore cache)')
    parser.add_argument('--cache-days', type=int, default=1,
                       help='Cache freshness in days (default: 1)')

    args = parser.parse_args()

    success = organize_data(force_refresh=args.refresh)

    if success:
        print("\n✅ Success! Data is organized and ready to use.")
    else:
        print("\n❌ Failed to organize data. Check errors above.")
        exit(1)

if __name__ == "__main__":
    main()
