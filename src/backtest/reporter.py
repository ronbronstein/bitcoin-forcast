"""
Backtest reporting and visualization.

Generates summaries and exports for backtest results.
"""
import pandas as pd
from typing import Dict
from pathlib import Path


class BacktestReporter:
    """Generates backtest reports and exports."""

    def __init__(self, results: pd.DataFrame, metrics: Dict[str, float]):
        """
        Initialize reporter.

        Args:
            results: DataFrame with individual predictions
            metrics: Dictionary of aggregate metrics
        """
        self.results = results
        self.metrics = metrics

    def print_summary(self) -> None:
        """Print formatted backtest summary."""
        m = self.metrics

        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Predictions:      {m['n_predictions']}")
        print()
        print("ACCURACY METRICS:")
        print(f"  Directional Accuracy: {m['directional_accuracy']:.1%}")
        print(f"  MAPE:                 {m['mape']:.1f}%")
        print(f"  RMSE:                 {m['rmse']:.1f}%")
        print()
        print("CALIBRATION:")
        print(f"  P10-P90 Capture:      {m['p10_p90_capture']:.1%}")
        print(f"  P25-P75 Capture:      {m['p25_p75_capture']:.1%}")
        print("=" * 60)

        # Performance interpretation
        print("\nINTERPRETATION:")
        if m['directional_accuracy'] > 0.55:
            print("  Directional: ABOVE random chance (>55%)")
        elif m['directional_accuracy'] < 0.45:
            print("  Directional: BELOW random chance (<45%) - investigate!")
        else:
            print("  Directional: Near random chance (45-55%)")

        if m['p10_p90_capture'] > 0.75 and m['p10_p90_capture'] < 0.85:
            print("  Calibration: Well calibrated P10-P90 bands")
        elif m['p10_p90_capture'] > 0.90:
            print("  Calibration: Bands too wide (overconfident)")
        elif m['p10_p90_capture'] < 0.70:
            print("  Calibration: Bands too narrow (underconfident)")

    def print_detailed(self) -> None:
        """Print detailed month-by-month results."""
        if self.results.empty:
            print("No results to display")
            return

        print("\nDETAILED RESULTS:")
        print("-" * 80)

        for _, row in self.results.iterrows():
            date_str = row['date'].strftime('%Y-%m')
            pred = row['predicted_return']
            actual = row['actual_return']
            correct = "✓" if row['direction_correct'] else "✗"
            in_band = "✓" if row['within_p10_p90'] else "✗"

            print(
                f"{date_str}: Pred={pred:+6.1f}%, Actual={actual:+6.1f}%, "
                f"Dir={correct}, Band={in_band}"
            )

    def to_csv(self, path: Path) -> None:
        """
        Export results to CSV.

        Args:
            path: Output file path
        """
        self.results.to_csv(path, index=False)
        print(f"Results saved to {path}")

    def get_annual_breakdown(self) -> pd.DataFrame:
        """Get metrics broken down by year."""
        if self.results.empty:
            return pd.DataFrame()

        df = self.results.copy()
        df['year'] = pd.to_datetime(df['date']).dt.year

        annual = df.groupby('year').agg({
            'direction_correct': ['sum', 'count', 'mean'],
            'abs_error': 'mean',
            'within_p10_p90': 'mean',
        })

        annual.columns = [
            'correct_count', 'total', 'directional_acc',
            'mape', 'p10_p90_capture'
        ]

        return annual

    def get_streak_analysis(self) -> Dict:
        """Analyze prediction streaks."""
        if self.results.empty:
            return {}

        correct = self.results['direction_correct'].tolist()

        # Find longest winning and losing streaks
        max_win = max_lose = current_win = current_lose = 0

        for c in correct:
            if c:
                current_win += 1
                current_lose = 0
                max_win = max(max_win, current_win)
            else:
                current_lose += 1
                current_win = 0
                max_lose = max(max_lose, current_lose)

        return {
            'longest_winning_streak': max_win,
            'longest_losing_streak': max_lose,
            'total_correct': sum(correct),
            'total_incorrect': len(correct) - sum(correct),
        }
