#!/usr/bin/env python3
"""
Run scenario matrix analysis for Bitcoin forecasting.

Usage:
    python run_scenarios.py                    # Full matrix
    python run_scenarios.py --current          # Show current scenarios
    python run_scenarios.py --month Jan        # Filter by month
    python run_scenarios.py --output matrix.csv
"""
import argparse
from pathlib import Path
import pandas as pd

from src.data import DataLoader
from src.scenarios import ScenarioMatcher, ScenarioType


def main():
    parser = argparse.ArgumentParser(description='Run scenario matrix analysis')
    parser.add_argument('--as-of', type=str, default=None,
                        help='P.I.T. cutoff date (YYYY-MM-DD)')
    parser.add_argument('--current', action='store_true',
                        help='Show current matching scenarios')
    parser.add_argument('--month', type=str, default=None,
                        help='Filter to specific month (e.g., Jan)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Filter to specific scenario type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')

    args = parser.parse_args()

    # Load data
    print('Loading data...')
    loader = DataLoader(Path('data'))
    df = loader.load_full_dataset(exclude_usdt=True)
    print(f'Data loaded: {len(df)} months')

    # Parse as_of date
    as_of_date = None
    if args.as_of:
        as_of_date = pd.Timestamp(args.as_of)
    else:
        as_of_date = df.index.max()

    print(f'Analysis as of: {as_of_date.date()}')

    # Create matcher
    matcher = ScenarioMatcher(df)

    # Show current scenarios
    if args.current:
        print('\n' + '=' * 60)
        print('CURRENT MATCHING SCENARIOS')
        print('=' * 60)

        current = matcher.get_current_scenarios(as_of_date)
        current_month = as_of_date.strftime('%B')

        print(f'\nCurrent month: {current_month}')
        print(f'Matching scenarios: {len(current)}')
        print()

        for scenario in current:
            from src.scenarios.definitions import SCENARIOS
            config = SCENARIOS[scenario]
            print(f'  - {config["name"]}')
            print(f'    {config["desc"]}')

        return

    # Run matrix analysis
    print('\nGenerating scenario matrix...')
    matrix = matcher.run_matrix_analysis(as_of_date=as_of_date)

    # Apply filters
    if args.month:
        matrix = matrix[matrix['Month'] == args.month]

    if args.scenario:
        matrix = matrix[matrix['Scenario'].str.contains(args.scenario, case=False)]

    # Print quality summary
    matcher.print_summary(matrix)

    # Print top scenarios by win rate
    print('\n' + '=' * 60)
    print('TOP SCENARIOS BY WIN RATE (Min 5 samples)')
    print('=' * 60)

    reliable = matrix[matrix['Count'] >= 5].copy()
    top_10 = reliable.nlargest(10, 'Win_Rate')

    for _, row in top_10.iterrows():
        print(f"\n{row['Month']} - {row['Scenario']}")
        print(f"  Win Rate: {row['Win_Rate']:.1f}% | Avg: {row['Avg_Return']:.1f}%")
        print(f"  Count: {row['Count']} | Years: {row['Matching_Years'][:5]}...")

    # Print worst scenarios
    print('\n' + '=' * 60)
    print('WORST SCENARIOS BY WIN RATE (Min 5 samples)')
    print('=' * 60)

    bottom_10 = reliable.nsmallest(10, 'Win_Rate')

    for _, row in bottom_10.iterrows():
        print(f"\n{row['Month']} - {row['Scenario']}")
        print(f"  Win Rate: {row['Win_Rate']:.1f}% | Avg: {row['Avg_Return']:.1f}%")
        print(f"  Count: {row['Count']} | Years: {row['Matching_Years'][:5]}...")

    # Print current month outlook
    current_month = as_of_date.strftime('%b')
    print('\n' + '=' * 60)
    print(f'CURRENT MONTH OUTLOOK ({current_month})')
    print('=' * 60)

    current_scenarios = matcher.get_current_scenarios(as_of_date)
    current_month_data = matrix[matrix['Month'] == current_month]

    for scenario in current_scenarios:
        from src.scenarios.definitions import SCENARIOS
        config = SCENARIOS[scenario]
        row = current_month_data[
            current_month_data['Scenario'] == config['name']
        ]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"\n{row['Scenario']}")
            print(f"  Win Rate: {row['Win_Rate']:.1f}% | Avg: {row['Avg_Return']:.1f}%")
            print(f"  CI 90%: [{row['CI_Lower_90']:.1f}%, {row['CI_Upper_90']:.1f}%]")
            print(f"  Count: {row['Count']} samples")

    # Save to CSV
    if args.output:
        matrix.to_csv(args.output, index=False)
        print(f'\nMatrix saved to {args.output}')


if __name__ == '__main__':
    main()
