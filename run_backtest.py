#!/usr/bin/env python3
"""
Run baseline backtest for Bitcoin forecast model.

Usage:
    python run_backtest.py
    python run_backtest.py --start 2021-01-01 --end 2024-12-01
"""
import argparse
from pathlib import Path
import pandas as pd

from src.data import DataLoader
from src.models import HistoricalMeanModel
from src.backtest import BacktestEngine, BacktestConfig, BacktestReporter


def main():
    parser = argparse.ArgumentParser(description='Run baseline backtest')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-01',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--min-train', type=int, default=36,
                        help='Minimum training months')
    parser.add_argument('--output', type=str, default='outputs/backtest_results.csv',
                        help='Output CSV path')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed month-by-month results')

    args = parser.parse_args()

    # Load data
    print('Loading data...')
    loader = DataLoader(Path('data'))
    df = loader.load_full_dataset(exclude_usdt=True)
    print(f'Data loaded: {len(df)} months ({df.index.min().date()} to {df.index.max().date()})')

    # Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end),
        min_training_months=args.min_train,
        target_col='Ret_BTC'
    )

    # Run backtest
    print('\nRunning backtest...')
    engine = BacktestEngine(HistoricalMeanModel, config)
    results = engine.run(df, verbose=True)

    # Generate report
    print()
    reporter = BacktestReporter(results['results'], results['metrics'])
    reporter.print_summary()

    if args.detailed:
        reporter.print_detailed()

    # Annual breakdown
    print('\nANNUAL BREAKDOWN:')
    annual = reporter.get_annual_breakdown()
    print(annual.to_string())

    # Save results
    reporter.to_csv(Path(args.output))


if __name__ == '__main__':
    main()
