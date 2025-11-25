import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import data_loader, scenario_engine
from scripts import dashboard_view
import webbrowser
import datetime

def main():
    print("🚀 Starting Bitcoin Scenario Matrix (V4)...")
    
    # 1. Fetch & Process Data
    loader = data_loader.DataLoader()
    df = loader.fetch_data()
    
    if df.empty:
        print("❌ Aborting: No data available.")
        return

    # 2. Run Matrix Analysis
    engine = scenario_engine.ScenarioEngine(df)
    matrix_df = engine.run_matrix_analysis()
    
    # 3. Generate Forecast for Next 12 Months
    # We need current price from the dataframe
    current_price = float(df['BTC'].iloc[-1])  # Ensure float
    print(f"DEBUG: Current Price is {current_price}")
    current_date = df.index[-1]

    # Clean up timezone for logic
    if current_date.tzinfo is not None:
        current_date = current_date.tz_localize(None)

    # FLAW 7 FIX: Incomplete month handling is now done in data_loader.py
    # No warning needed here as data_loader excludes incomplete months

    forecast_data = engine.generate_forecast(current_price, current_date)
    current_conditions = engine.get_current_conditions()
    
    # 4. Render View
    viewer = dashboard_view.DashboardView()
    html_content = viewer.render_dashboard(matrix_df, forecast_data, current_conditions)
    
    # Save
    filename = os.path.join('outputs', 'btc_matrix_v4.html')
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✨ Dashboard Ready: {os.path.abspath(filename)}")
    webbrowser.open('file://' + os.path.abspath(filename))

if __name__ == "__main__":
    main()

