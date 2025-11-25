"""
Simple script to run the full backtest and generate report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import backtest_engine
from scripts import backtest_viz
import pandas as pd
import webbrowser

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
    output_csv = os.path.join('outputs', 'backtest_results.csv')
    results.to_csv(output_csv, index=False)
    print(f"💾 CSV saved: {output_csv}")

    # Step 3: Generate Visual Report
    print("\nStep 3/3: Generating visual report...")
    viz = backtest_viz.BacktestViz(results)
    html = viz.generate_report()

    output_html = os.path.join('outputs', 'backtest_report.html')
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Report saved: {output_html}")

    # Open in browser
    print("\n🌐 Opening report in browser...")
    webbrowser.open('file://' + os.path.abspath(output_html))
    
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