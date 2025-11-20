"""
Walk-Forward Backtesting Engine
Tests the forecast model month-by-month from 2020-2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import data_loader
import scenario_engine

class BacktestEngine:
    def __init__(self, start_year=2020, end_year=2024):
        """
        Args:
            start_year: First year to test (2020)
            end_year: Last year to test (2024)
        """
        self.start_year = start_year
        self.end_year = end_year
        self.results = []
        
    def run_backtest(self):
        """
        Main backtest loop:
        1. For each month from Jan 2020 to Nov 2024
        2. Train on all data BEFORE that month
        3. Generate 1-month forecast
        4. Compare to actual outcome
        5. Record all metrics
        """
        print("="*70)
        print("🔬 STARTING WALK-FORWARD BACKTEST")
        print("="*70)
        
        # Load full dataset
        loader = data_loader.DataLoader()
        full_df = loader.fetch_data()
        
        if full_df.empty:
            print("❌ No data loaded")
            return pd.DataFrame()
        
        # Get test period dates
        test_start = pd.Timestamp(f'{self.start_year}-01-01')
        test_end = pd.Timestamp.now()
        
        # Filter to months in test period that have actual data
        test_months = full_df[
            (full_df.index >= test_start) & 
            (full_df.index <= test_end)
        ].index.tolist()
        
        print(f"\n📅 Testing Period: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
        print(f"   Total Months to Test: {len(test_months)}")
        print(f"   This will take ~{len(test_months) * 2} seconds...\n")
        
        # Loop through each test month
        for i, test_date in enumerate(test_months):
            # Skip if this is the last available month (no future data to compare)
            if i >= len(test_months) - 1:
                print(f"\n⏭️  Skipping {test_date.strftime('%Y-%m')} (current month, no actual outcome yet)")
                break

            # Get actual outcome (next month's return)
            actual_next_month = test_months[i + 1]

            # Validate that next month exists and is within reasonable timeframe (max 45 days)
            days_diff = (actual_next_month - test_date).days
            if days_diff > 45 or days_diff < 20:
                print(f"   ⏭️  Invalid time gap ({days_diff} days), skipping {test_date.strftime('%Y-%m')}...")
                continue

            actual_return = full_df.loc[actual_next_month, 'Ret_BTC']
            actual_price_start = full_df.loc[test_date, 'BTC']
            actual_price_end = full_df.loc[actual_next_month, 'BTC']

            # Validate actual data quality
            if pd.isna(actual_return) or pd.isna(actual_price_start) or pd.isna(actual_price_end):
                print(f"   ⏭️  Missing price/return data, skipping {test_date.strftime('%Y-%m')}...")
                continue

            print(f"\n{'='*70}")
            print(f"📊 Testing: {test_date.strftime('%B %Y')} → {actual_next_month.strftime('%B %Y')}")
            print(f"   Actual: ${actual_price_start:,.0f} → ${actual_price_end:,.0f} ({actual_return:+.1f}%)")
            
            # Train on data UP TO (but not including) the test month
            train_df = full_df[full_df.index < test_date].copy()
            
            if len(train_df) < 24:  # Need at least 2 years to train
                print(f"   ⏭️  Insufficient training data ({len(train_df)} months), skipping...")
                continue
            
            # Create engine with training data only
            engine = scenario_engine.ScenarioEngine(train_df)
            
            # Get conditions at the test date (end of month we're predicting FROM)
            current_conditions = self._get_conditions_at_date(full_df, test_date)

            # Validate that all required conditions exist (no NaN values)
            required_conditions = ['RSI_BTC', 'RSI_SPX', 'DXY_Trend', 'Rate_Trend']
            if any(pd.isna(current_conditions.get(key)) for key in required_conditions):
                print(f"   ⏭️  Missing market conditions, skipping...")
                continue

            # Generate 1-month forecast
            try:
                dates, price_paths, match_logic, sample_size = engine.generate_forecast(
                    current_price=actual_price_start,
                    start_date=test_date,
                    months=1,  # Only forecast 1 month ahead
                    simulations=2000
                )
                
                # Extract forecast statistics
                forecast_prices = price_paths[:, 1]  # Next month's prices
                forecast_returns = ((forecast_prices / actual_price_start) - 1) * 100
                
                p10 = np.percentile(forecast_prices, 10)
                p25 = np.percentile(forecast_prices, 25)
                p50 = np.percentile(forecast_prices, 50)
                p75 = np.percentile(forecast_prices, 75)
                p90 = np.percentile(forecast_prices, 90)
                mean_forecast = np.mean(forecast_prices)
                
                # Calculate accuracy metrics
                direction_correct = (
                    (actual_return > 0 and p50 > actual_price_start) or
                    (actual_return < 0 and p50 < actual_price_start)
                )
                
                within_p10_p90 = p10 <= actual_price_end <= p90
                within_p25_p75 = p25 <= actual_price_end <= p75
                
                price_error = actual_price_end - p50
                price_error_pct = (price_error / actual_price_start) * 100
                
                # Record results
                result = {
                    'test_date': test_date,
                    'target_date': actual_next_month,
                    'train_months': len(train_df),
                    'match_logic': match_logic,
                    'sample_size': sample_size,
                    
                    # Actual values
                    'actual_price_start': actual_price_start,
                    'actual_price_end': actual_price_end,
                    'actual_return': actual_return,
                    
                    # Forecast values
                    'forecast_p10': p10,
                    'forecast_p25': p25,
                    'forecast_p50': p50,
                    'forecast_p75': p75,
                    'forecast_p90': p90,
                    'forecast_mean': mean_forecast,
                    
                    # Accuracy metrics
                    'direction_correct': direction_correct,
                    'within_p10_p90': within_p10_p90,
                    'within_p25_p75': within_p25_p75,
                    'price_error': price_error,
                    'price_error_pct': price_error_pct,
                    'abs_price_error': abs(price_error),
                    'abs_price_error_pct': abs(price_error_pct),
                    
                    # Conditions at forecast time
                    'condition_rsi_btc': current_conditions.get('RSI_BTC'),
                    'condition_rsi_spx': current_conditions.get('RSI_SPX'),
                    'condition_dxy_trend': current_conditions.get('DXY_Trend'),
                    'condition_rate_trend': current_conditions.get('Rate_Trend'),
                }
                
                self.results.append(result)
                
                # Print summary
                print(f"   Forecast P50: ${p50:,.0f} (Error: {price_error:+,.0f} / {price_error_pct:+.1f}%)")
                print(f"   Forecast Range: ${p10:,.0f} - ${p90:,.0f}")
                print(f"   Direction: {'✅' if direction_correct else '❌'}")
                print(f"   Within Bands: {'✅ P10-P90' if within_p10_p90 else '❌ P10-P90'}")
                print(f"   Logic: {match_logic} (N={sample_size})")
                
            except Exception as e:
                print(f"   ❌ Error generating forecast: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        if not results_df.empty:
            self._print_summary(results_df)
        
        return results_df
    
    def _get_conditions_at_date(self, df, date):
        """Extract market conditions at a specific date"""
        row = df.loc[date]
        
        return {
            'RSI_BTC': row['RSI_BTC'],
            'RSI_SPX': row['RSI_SPX'],
            'RSI_NDX': row['RSI_NDX'],
            'DXY_Trend': row['Trend_DXY'],
            'Rate_Trend': row['Trend_Rates'],
            'Date': date
        }
    
    def _print_summary(self, results_df):
        """Print overall backtest summary statistics"""
        print("\n" + "="*70)
        print("📈 BACKTEST SUMMARY")
        print("="*70)
        
        total_tests = len(results_df)
        
        # Directional Accuracy
        dir_acc = results_df['direction_correct'].mean() * 100
        print(f"\n🎯 Directional Accuracy: {dir_acc:.1f}% ({results_df['direction_correct'].sum()}/{total_tests})")
        
        # Band Capture Rates
        p10_p90_rate = results_df['within_p10_p90'].mean() * 100
        p25_p75_rate = results_df['within_p25_p75'].mean() * 100
        print(f"\n📊 Band Capture Rates:")
        print(f"   P10-P90 (Should be ~80%): {p10_p90_rate:.1f}%")
        print(f"   P25-P75 (Should be ~50%): {p25_p75_rate:.1f}%")
        
        # Price Errors
        mae = results_df['abs_price_error'].mean()
        mape = results_df['abs_price_error_pct'].mean()
        print(f"\n💰 Price Accuracy:")
        print(f"   Mean Absolute Error: ${mae:,.0f}")
        print(f"   Mean Absolute % Error: {mape:.1f}%")
        
        # By Sample Size
        print(f"\n📏 Performance by Sample Size:")
        for threshold in [5, 10, 15]:
            subset = results_df[results_df['sample_size'] >= threshold]
            if len(subset) > 0:
                dir_acc_subset = subset['direction_correct'].mean() * 100
                mape_subset = subset['abs_price_error_pct'].mean()
                print(f"   N≥{threshold}: {len(subset)} tests, {dir_acc_subset:.1f}% dir. acc, {mape_subset:.1f}% MAPE")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    engine = BacktestEngine(start_year=2020, end_year=2024)
    results = engine.run_backtest()
    
    if not results.empty:
        # Save results
        results.to_csv('backtest_results.csv', index=False)
        print(f"💾 Results saved to: backtest_results.csv")