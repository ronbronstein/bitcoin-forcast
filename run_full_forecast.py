#!/usr/bin/env python3
"""
Run complete unified Bitcoin forecast with all models.

Combines:
- Regime classification
- Scenario matching
- Path-dependent Monte Carlo

Usage:
    python run_full_forecast.py                # Full 12-month forecast
    python run_full_forecast.py --months 6     # 6-month forecast
    python run_full_forecast.py --json         # Output as JSON
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.data import DataLoader
from src.forecast import UnifiedForecastEngine


def print_header(text: str, char: str = '='):
    """Print formatted header."""
    print(f'\n{char * 70}')
    print(f' {text}')
    print(f'{char * 70}')


def main():
    parser = argparse.ArgumentParser(
        description='Run complete unified Bitcoin forecast'
    )
    parser.add_argument('--price', type=float, default=None,
                        help='Starting price (default: latest)')
    parser.add_argument('--months', type=int, default=12,
                        help='Forecast horizon')
    parser.add_argument('--sims', type=int, default=2000,
                        help='Number of simulations')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to file')

    args = parser.parse_args()

    # Load data
    print('Loading data...')
    loader = DataLoader(Path('data'))
    df = loader.load_full_dataset(exclude_usdt=True)

    # Run unified forecast
    print('Running unified forecast...')
    engine = UnifiedForecastEngine(
        data=df,
        n_simulations=args.sims,
        n_months=args.months,
        random_seed=args.seed,
    )

    result = engine.run(current_price=args.price)

    # JSON output
    if args.json:
        summary = result.to_summary_dict()
        print(json.dumps(summary, indent=2, default=str))
        return

    # Pretty print results
    cond = result.conditions

    print_header('CURRENT CONDITIONS')
    print(f'''
  Date:           {cond.date.date()}
  Price:          ${cond.price:,.0f}
  RSI:            {cond.rsi:.1f} ({cond.rsi_state})
  DXY Trend:      {cond.dxy_trend}
  Regime:         {cond.regime.value}
  Halving Phase:  {cond.halving_phase:.1f} months since halving''' if cond.halving_phase else f'''
  Date:           {cond.date.date()}
  Price:          ${cond.price:,.0f}
  RSI:            {cond.rsi:.1f} ({cond.rsi_state})
  DXY Trend:      {cond.dxy_trend}
  Regime:         {cond.regime.value}''')

    print_header('MATCHING SCENARIOS', '-')
    print(f'\n  {len(cond.matching_scenarios)} scenarios match current conditions:\n')

    for sf in result.scenario_forecasts:
        quality_icon = '✓' if sf.quality == 'Sufficient' else '⚠' if sf.quality == 'Small Sample' else '?'
        print(f'  {quality_icon} {sf.name}')
        print(f'    Win Rate: {sf.win_rate:.0f}% | Avg Return: {sf.avg_return:+.1f}% | Samples: {sf.count}')

    # Weighted forecast
    wf = result.weighted_forecast
    print_header('WEIGHTED SCENARIO FORECAST', '-')
    print(f'''
  Expected Return: {wf["expected_return"]:+.1f}%
  Win Rate:        {wf["win_rate"]:.0f}%
  Confidence:      {wf["confidence"]}
  Total Samples:   {wf["total_samples"]}''')

    # Monte Carlo results
    print_header('MONTE CARLO SIMULATION')
    print(f'''
  Simulations:     {result.simulation.n_simulations:,}
  Horizon:         {result.simulation.n_months} months
  Method:          {result.simulation.match_logic}''')

    print('\n  Price Distribution by Month:')
    print('  ' + '-' * 66)
    print(f'  {"Month":<10} {"P10":>12} {"P25":>12} {"P50":>12} {"P75":>12} {"P90":>12}')
    print('  ' + '-' * 66)

    for i, date in enumerate(result.simulation.dates):
        if i == 0:
            continue
        pcts = result.simulation.get_percentiles(i)
        print(f'  {date.strftime("%b %Y"):<10} '
              f'${pcts["p10"]:>10,.0f} '
              f'${pcts["p25"]:>10,.0f} '
              f'${pcts["p50"]:>10,.0f} '
              f'${pcts["p75"]:>10,.0f} '
              f'${pcts["p90"]:>10,.0f}')

    # Final distribution
    final = result.simulation.get_final_distribution()
    final_ret = result.simulation.get_return_percentiles(result.simulation.n_months)
    final_date = result.simulation.dates[-1].strftime('%b %Y')

    print_header(f'FINAL FORECAST ({final_date})')
    print(f'''
  Price Distribution:
    P10:  ${final["p10"]:>12,.0f}  ({final_ret["p10"]:>+6.1f}%)
    P25:  ${final["p25"]:>12,.0f}  ({final_ret["p25"]:>+6.1f}%)
    P50:  ${final["p50"]:>12,.0f}  ({final_ret["p50"]:>+6.1f}%)
    P75:  ${final["p75"]:>12,.0f}  ({final_ret["p75"]:>+6.1f}%)
    P90:  ${final["p90"]:>12,.0f}  ({final_ret["p90"]:>+6.1f}%)

    Mean: ${final["mean"]:>12,.0f}  ({final_ret["mean"]:>+6.1f}%)
    Std:  ${final["std"]:>12,.0f}''')

    # Probability analysis
    print_header('PROBABILITY ANALYSIS')

    targets = result.get_price_targets()
    print('\n  Price Targets:')
    for target, prob in targets.items():
        if prob > 1:  # Only show targets with >1% probability
            print(f'    >=${target:>7,}: {prob:>5.1f}%')

    probs = result.get_return_probabilities()
    print(f'''
  Return Probabilities:
    Positive return:   {probs["positive"]:.1f}%
    >20% gain:         {probs["above_20pct"]:.1f}%
    >50% gain:         {probs["above_50pct"]:.1f}%
    >100% gain:        {probs["above_100pct"]:.1f}%
    <20% loss:         {probs["below_minus20pct"]:.1f}%
    <50% loss:         {probs["below_minus50pct"]:.1f}%''')

    # Combined signal
    print_header('COMBINED SIGNAL')

    # Determine overall signal
    mc_bullish = final_ret["p50"] > 20
    scenario_bullish = wf["win_rate"] > 55 and wf["expected_return"] > 5

    if mc_bullish and scenario_bullish:
        signal = "BULLISH"
        signal_desc = "Both Monte Carlo and historical scenarios favor upside"
    elif mc_bullish or scenario_bullish:
        signal = "NEUTRAL-BULLISH"
        signal_desc = "Mixed signals, slight bullish lean"
    elif wf["win_rate"] < 45 or final_ret["p50"] < -10:
        signal = "BEARISH"
        signal_desc = "Both models suggest caution"
    else:
        signal = "NEUTRAL"
        signal_desc = "No strong directional signal"

    print(f'''
  Signal:      {signal}
  Rationale:   {signal_desc}

  Monte Carlo P50:        {final_ret["p50"]:+.1f}%
  Scenario Win Rate:      {wf["win_rate"]:.0f}%
  Scenario Exp. Return:   {wf["expected_return"]:+.1f}%''')

    # Save if requested
    if args.output:
        summary = result.to_summary_dict()
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
