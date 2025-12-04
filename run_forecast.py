#!/usr/bin/env python3
"""
Run Monte Carlo simulation for Bitcoin price forecasting.

Usage:
    python run_forecast.py                    # Default 12-month forecast
    python run_forecast.py --months 6         # 6-month forecast
    python run_forecast.py --sims 5000        # More simulations
    python run_forecast.py --price 95000      # Custom starting price
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.data import DataLoader
from src.simulation import MonteCarloEngine, SimulationConfig


def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo forecast')
    parser.add_argument('--price', type=float, default=None,
                        help='Starting price (default: latest BTC price)')
    parser.add_argument('--months', type=int, default=12,
                        help='Forecast horizon in months')
    parser.add_argument('--sims', type=int, default=2000,
                        help='Number of simulations')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for summary')

    args = parser.parse_args()

    # Load data
    print('Loading data...')
    loader = DataLoader(Path('data'))
    df = loader.load_full_dataset(exclude_usdt=True)
    print(f'Data loaded: {len(df)} months')

    # Get as_of_date and current price
    as_of_date = df.index.max()
    current_row = df.iloc[-1]

    if args.price:
        current_price = args.price
    else:
        current_price = current_row['BTC']

    print(f'\nForecast Configuration:')
    print(f'  As of date: {as_of_date.date()}')
    print(f'  Starting price: ${current_price:,.0f}')
    print(f'  Forecast months: {args.months}')
    print(f'  Simulations: {args.sims}')

    # Show current conditions
    print(f'\nCurrent Conditions:')
    print(f'  RSI: {current_row["RSI_BTC"]:.1f}')
    print(f'  AG/AL: {current_row["BTC_AG"]:.2f} / {current_row["BTC_AL"]:.2f}')
    if 'Prev_DXY_Trend' in current_row:
        dxy_status = "Rising" if current_row['Prev_DXY_Trend'] > 0 else "Falling"
        print(f'  DXY Trend: {dxy_status}')

    # Configure simulation
    config = SimulationConfig(
        n_simulations=args.sims,
        n_months=args.months,
        random_seed=args.seed,
    )

    # Run simulation
    print('\nRunning Monte Carlo simulation...')
    engine = MonteCarloEngine(df, config)
    result = engine.run(current_price, as_of_date)
    print(f'Simulation complete. Match logic: {result.match_logic}')

    # Print results
    print('\n' + '=' * 70)
    print('FORECAST RESULTS')
    print('=' * 70)

    print('\nPrice Distribution by Month:')
    print('-' * 70)
    print(f'{"Month":<12} {"P10":>12} {"P25":>12} {"P50":>12} {"P75":>12} {"P90":>12}')
    print('-' * 70)

    for i, date in enumerate(result.dates):
        if i == 0:
            continue  # Skip initial price
        pcts = result.get_percentiles(i)
        month_str = date.strftime('%b %Y')
        print(f'{month_str:<12} '
              f'${pcts["p10"]:>10,.0f} '
              f'${pcts["p25"]:>10,.0f} '
              f'${pcts["p50"]:>10,.0f} '
              f'${pcts["p75"]:>10,.0f} '
              f'${pcts["p90"]:>10,.0f}')

    # Final distribution
    final = result.get_final_distribution()
    final_ret = result.get_return_percentiles(result.n_months)

    print('\n' + '=' * 70)
    print(f'FINAL MONTH ({result.dates[-1].strftime("%b %Y")})')
    print('=' * 70)

    print(f'\nPrice Distribution:')
    print(f'  P10:  ${final["p10"]:>12,.0f}  ({final_ret["p10"]:>+6.1f}%)')
    print(f'  P25:  ${final["p25"]:>12,.0f}  ({final_ret["p25"]:>+6.1f}%)')
    print(f'  P50:  ${final["p50"]:>12,.0f}  ({final_ret["p50"]:>+6.1f}%)')
    print(f'  P75:  ${final["p75"]:>12,.0f}  ({final_ret["p75"]:>+6.1f}%)')
    print(f'  P90:  ${final["p90"]:>12,.0f}  ({final_ret["p90"]:>+6.1f}%)')
    print(f'\n  Mean: ${final["mean"]:>12,.0f}  ({final_ret["mean"]:>+6.1f}%)')
    print(f'  Std:  ${final["std"]:>12,.0f}')

    # Probability analysis
    print('\n' + '=' * 70)
    print('PROBABILITY ANALYSIS')
    print('=' * 70)

    final_prices = result.price_paths[:, -1]
    targets = [50000, 75000, 100000, 150000, 200000]

    print(f'\nProbability of reaching price targets:')
    for target in targets:
        prob = (final_prices >= target).mean() * 100
        print(f'  >=${target:>7,}: {prob:>5.1f}%')

    # Probability of positive return
    returns = (final_prices / current_price - 1) * 100
    prob_positive = (returns > 0).mean() * 100
    prob_double = (returns >= 100).mean() * 100

    print(f'\n  Positive return: {prob_positive:.1f}%')
    print(f'  >100% gain:      {prob_double:.1f}%')

    # Save summary
    if args.output:
        summary = result.get_summary()
        summary.to_csv(args.output, index=False)
        print(f'\nSummary saved to {args.output}')


if __name__ == '__main__':
    main()
