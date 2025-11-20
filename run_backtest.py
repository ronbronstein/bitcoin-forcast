"""
Simple script to run the full backtest and generate report
"""

import backtest_engine
import backtest_viz
import pandas as pd
import webbrowser
import os

def main():
    print("="*70)
    print("🚀 BITCOIN FORECAST MODEL - BACKTEST SUITE")
    print("="*70)
    
    # Step 1: Run Backtest
    print("\nStep 1/3: Running walk-forward validation...")
    engine = backtest_engine.BacktestEngine(start_year=2020, end_year=2024)
    results = engine.run_backtest()
    
    if results.empty:
        print("❌ No results generated. Aborting.")
        return
    
    # Step 2: Save CSV
    print("\nStep 2/3: Saving results...")
    results.to_csv('backtest_results.csv', index=False)
    print(f"💾 CSV saved: backtest_results.csv")
    
    # Step 3: Generate Visual Report
    print("\nStep 3/3: Generating visual report...")
    viz = backtest_viz.BacktestViz(results)
    html = viz.generate_report()
    
    with open('backtest_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report saved: backtest_report.html")
    
    # Open in browser
    print("\n🌐 Opening report in browser...")
    webbrowser.open('file://' + os.path.abspath('backtest_report.html'))
    
    print("\n" + "="*70)
    print("✨ BACKTEST COMPLETE!")
    print("="*70)
    print("\nFiles generated:")
    print("  • backtest_results.csv - Raw data")
    print("  • backtest_report.html - Interactive dashboard")
    print("\nNext steps:")
    print("  1. Review the report to see where the model succeeds/fails")
    print("  2. Check which scenarios have predictive power")
    print("  3. Decide whether to simplify, add features, or change approach")

if __name__ == "__main__":
    main()