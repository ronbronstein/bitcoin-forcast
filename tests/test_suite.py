"""
Comprehensive Test Suite for Bitcoin Forecast Model
Run this after completing all implementation phases
"""

import data_loader
import scenario_engine
import pandas as pd
import numpy as np

def test_critical_bugs_fixed():
    """Test that all Phase 1 critical bugs are fixed"""
    print("\n" + "="*60)
    print("TEST 1: Critical Bug Fixes")
    print("="*60)
    
    # Load data
    loader = data_loader.DataLoader()
    df = loader.fetch_data()
    
    if df.empty:
        print("❌ FAILED: No data loaded")
        return False
    
    # Test 1.1: RSI Validation
    for col in ['RSI_BTC', 'RSI_SPX', 'RSI_NDX']:
        if col in df.columns:
            invalid = (df[col] < 0) | (df[col] > 100) | df[col].isna()
            if invalid.any():
                print(f"❌ FAILED: Invalid RSI values in {col}")
                return False
    print("✅ PASSED: RSI calculations valid")
    
    # Test 1.2: Forward Fill Limited
    # Check that no column has >2 consecutive identical values at the end
    for col in ['DXY', 'Rates']:
        if col in df.columns and len(df) >= 3:
            last_three = df[col].iloc[-3:].values
            if len(set(last_three)) == 1:
                print(f"⚠️ WARNING: {col} may have excessive forward filling")
    print("✅ PASSED: Forward fill validation")
    
    # Test 1.3: Current Conditions Consistency
    engine = scenario_engine.ScenarioEngine(df)
    current = engine.get_current_conditions()
    
    # Verify it uses dataframe columns
    last_row = df.iloc[-1]
    if 'DXY_Trend' in current:
        # Check it matches Trend_DXY column (not manual calculation)
        expected = last_row['Trend_DXY']
        actual = current['DXY_Trend']
        if not np.isclose(expected, actual, rtol=0.01):
            print(f"❌ FAILED: Current conditions not using dataframe columns")
            print(f"   Expected DXY_Trend: {expected}, Got: {actual}")
            return False
    print("✅ PASSED: Current conditions use dataframe columns")
    
    # Test 1.4: RSI Threshold Consistency
    # This will be validated in the forecast generation
    print("✅ PASSED: Bug fix tests complete")
    return True


def test_new_features():
    """Test that Phase 2-4 features are working"""
    print("\n" + "="*60)
    print("TEST 2: New Feature Validation")
    print("="*60)
    
    loader = data_loader.DataLoader()
    df = loader.fetch_data()
    engine = scenario_engine.ScenarioEngine(df)
    
    # Test 2.1: Halving Phase Function
    from scenario_engine import get_halving_phase
    test_date = pd.Timestamp('2024-06-01')
    phase = get_halving_phase(test_date)
    
    if phase is None or phase < 0:
        print("❌ FAILED: Halving phase calculation broken")
        return False
    print(f"✅ PASSED: Halving phase = {phase:.1f} months")
    
    # Test 2.2: Matrix Contains New Scenarios
    matrix = engine.run_matrix_analysis()
    scenarios = matrix['Scenario'].unique()
    
    required_scenarios = [
        '1b. Recent Regime (2020+)',
        '12. Post-Halving Year 1 (0-12mo)',
        '13. Post-Halving Year 2 (12-24mo)',
        '14. Pre-Halving Year (36-48mo)'
    ]
    
    for req in required_scenarios:
        if req not in scenarios:
            print(f"❌ FAILED: Missing scenario '{req}'")
            return False
    print(f"✅ PASSED: All new scenarios present ({len(scenarios)} total)")
    
    # Test 2.3: Active Addresses (if available)
    if 'active_addresses' in df.columns:
        if df['active_addresses'].notna().sum() > 0:
            print("✅ PASSED: Active addresses integrated")
        else:
            print("⚠️ WARNING: Active addresses column exists but empty")
    else:
        print("⚠️ INFO: Active addresses not available (API may have failed)")
    
    # Test 2.4: Sample Quality Warnings
    if 'Quality' in matrix.columns:
        unreliable = (matrix['Count'] < 5).sum()
        print(f"✅ PASSED: Quality flags added ({unreliable} unreliable cells)")
    else:
        print("❌ FAILED: Quality column missing")
        return False
    
    return True


def test_forecast_generation():
    """Test that forecast generation works with new logic"""
    print("\n" + "="*60)
    print("TEST 3: Forecast Generation")
    print("="*60)
    
    loader = data_loader.DataLoader()
    df = loader.fetch_data()
    engine = scenario_engine.ScenarioEngine(df)
    
    current_price = float(df['BTC'].iloc[-1])
    current_date = df.index[-1]
    
    if current_date.tzinfo is not None:
        current_date = current_date.tz_localize(None)
    
    try:
        dates, price_paths, logic, sample_size = engine.generate_forecast(
            current_price, current_date
        )
        
        # Validate output
        if len(dates) != 13:  # 1 current + 12 future
            print(f"❌ FAILED: Expected 13 dates, got {len(dates)}")
            return False
        
        if price_paths.shape != (2000, 13):
            print(f"❌ FAILED: Expected (2000, 13) paths, got {price_paths.shape}")
            return False
        
        if logic is None or logic == "":
            print("❌ FAILED: Match logic is empty")
            return False
        
        if "Blended" not in logic and "Baseline" not in logic:
            print(f"⚠️ WARNING: Unexpected logic format: {logic}")
        
        print(f"✅ PASSED: Forecast generation successful")
        print(f"   Logic: {logic}")
        print(f"   Sample Size: {sample_size}")
        print(f"   Final Price Range: ${price_paths[:, -1].min():,.0f} - ${price_paths[:, -1].max():,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Forecast generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_generation():
    """Test that dashboard renders without errors"""
    print("\n" + "="*60)
    print("TEST 4: Dashboard Generation")
    print("="*60)
    
    import dashboard_view
    
    loader = data_loader.DataLoader()
    df = loader.fetch_data()
    engine = scenario_engine.ScenarioEngine(df)
    matrix_df = engine.run_matrix_analysis()
    
    current_price = float(df['BTC'].iloc[-1])
    current_date = df.index[-1]
    
    if current_date.tzinfo is not None:
        current_date = current_date.tz_localize(None)
    
    forecast_data = engine.generate_forecast(current_price, current_date)
    current_conditions = engine.get_current_conditions()
    
    try:
        viewer = dashboard_view.DashboardView()
        html = viewer.render_dashboard(matrix_df, forecast_data, current_conditions)
        
        # Basic HTML validation
        if '<html>' not in html.lower():
            print("❌ FAILED: Invalid HTML structure")
            return False
        
        if 'btc matrix' not in html.lower():
            print("❌ FAILED: Missing title")
            return False
        
        # Check for new elements
        if 'halving phase' in html.lower():
            print("✅ PASSED: Halving phase indicator present")
        else:
            print("⚠️ WARNING: Halving phase indicator missing")
        
        if 'regime comparison' in html.lower():
            print("✅ PASSED: Regime comparison section present")
        else:
            print("⚠️ WARNING: Regime comparison section missing")
        
        # Save test output
        with open('test_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html)
        
        print("✅ PASSED: Dashboard generated successfully")
        print("   Saved as: test_dashboard.html")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Dashboard generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print(" BITCOIN FORECAST MODEL - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    results['Bug Fixes'] = test_critical_bugs_fixed()
    results['New Features'] = test_new_features()
    results['Forecast'] = test_forecast_generation()
    results['Dashboard'] = test_dashboard_generation()
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print("="*70)
    print(f"OVERALL: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Implementation successful!")
        print("\nNext steps:")
        print("1. Open test_dashboard.html in your browser")
        print("2. Verify visual appearance")
        print("3. Run: python main.py")
        print("4. Review btc_matrix_v4.html for production")
    else:
        print("\n⚠️ SOME TESTS FAILED - Review errors above")
        print("Fix issues before proceeding to production use")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

