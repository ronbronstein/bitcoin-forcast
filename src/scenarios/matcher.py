"""
Scenario matching for conditional probability analysis.

Generates a 12 months × N scenarios matrix with historical statistics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .definitions import ScenarioType, SCENARIOS, MONTH_NAMES
from ..utils.dates import get_available_history


@dataclass
class ScenarioStats:
    """Statistics for a scenario-month cell."""
    month: str
    scenario: ScenarioType
    scenario_name: str
    description: str
    count: int
    win_rate: float
    avg_return: float
    median_return: float
    ci_lower: float
    ci_upper: float
    best: float
    worst: float
    matching_years: List[int]
    quality: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'Month': self.month,
            'Scenario': self.scenario_name,
            'Description': self.description,
            'Count': self.count,
            'Win_Rate': self.win_rate,
            'Avg_Return': self.avg_return,
            'Median_Return': self.median_return,
            'CI_Lower_90': self.ci_lower,
            'CI_Upper_90': self.ci_upper,
            'Best': self.best,
            'Worst': self.worst,
            'Matching_Years': self.matching_years,
            'Quality': self.quality,
        }


class ScenarioMatcher:
    """
    Matches current conditions to historical scenarios.

    Generates a matrix of 12 months × N scenarios with statistics.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = 'Ret_BTC',
        bootstrap_samples: int = 1000
    ):
        """
        Initialize matcher.

        Args:
            data: DataFrame with Prev_* features and target
            target_col: Column with returns to analyze
            bootstrap_samples: Number of bootstrap samples for CI
        """
        self.data = data
        self.target_col = target_col
        self.bootstrap_samples = bootstrap_samples

    def run_matrix_analysis(
        self,
        as_of_date: Optional[pd.Timestamp] = None,
        scenarios: Optional[List[ScenarioType]] = None
    ) -> pd.DataFrame:
        """
        Generate scenario-month matrix.

        Args:
            as_of_date: P.I.T. cutoff date (None = use all data)
            scenarios: Scenarios to analyze (None = all)

        Returns:
            DataFrame with scenario statistics
        """
        # P.I.T. filter
        if as_of_date is not None:
            df = get_available_history(self.data, as_of_date)
        else:
            df = self.data

        if scenarios is None:
            scenarios = list(ScenarioType)

        results = []

        for month_idx in range(1, 13):
            month_name = MONTH_NAMES[month_idx - 1]
            month_data = df[df.index.month == month_idx]

            for scenario in scenarios:
                stats = self._analyze_cell(
                    month_data, month_name, scenario
                )
                results.append(stats.to_dict())

        return pd.DataFrame(results)

    def _analyze_cell(
        self,
        month_data: pd.DataFrame,
        month_name: str,
        scenario: ScenarioType
    ) -> ScenarioStats:
        """Analyze single scenario-month cell."""
        config = SCENARIOS[scenario]

        try:
            mask = config['filter'](month_data)
            subset = month_data[mask]
        except (KeyError, TypeError):
            subset = pd.DataFrame()

        if len(subset) == 0:
            return self._empty_stats(month_name, scenario, config)

        returns = subset[self.target_col].dropna()

        if len(returns) == 0:
            return self._empty_stats(month_name, scenario, config)

        # Calculate statistics
        ci_lower, ci_upper = self._bootstrap_ci(returns)

        return ScenarioStats(
            month=month_name,
            scenario=scenario,
            scenario_name=config['name'],
            description=config['desc'],
            count=len(returns),
            win_rate=(returns > 0).mean() * 100,
            avg_return=returns.mean(),
            median_return=returns.median(),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            best=returns.max(),
            worst=returns.min(),
            matching_years=subset.index.year.tolist(),
            quality=self._get_quality(len(returns)),
        )

    def _empty_stats(
        self,
        month_name: str,
        scenario: ScenarioType,
        config: Dict
    ) -> ScenarioStats:
        """Create empty stats for scenario with no matches."""
        return ScenarioStats(
            month=month_name,
            scenario=scenario,
            scenario_name=config['name'],
            description=config['desc'],
            count=0,
            win_rate=0.0,
            avg_return=0.0,
            median_return=0.0,
            ci_lower=np.nan,
            ci_upper=np.nan,
            best=0.0,
            worst=0.0,
            matching_years=[],
            quality='No Data',
        )

    def _bootstrap_ci(self, returns: pd.Series) -> tuple:
        """Calculate bootstrap 90% confidence interval for mean."""
        if len(returns) < 3:
            return np.nan, np.nan

        bootstrap_means = []
        values = returns.values

        for _ in range(self.bootstrap_samples):
            resample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(resample.mean())

        return (
            np.percentile(bootstrap_means, 5),
            np.percentile(bootstrap_means, 95)
        )

    def _get_quality(self, count: int) -> str:
        """Get quality flag based on sample size."""
        if count < 5:
            return 'Unreliable'
        elif count < 10:
            return 'Small Sample'
        return 'Sufficient'

    def get_current_scenarios(
        self,
        as_of_date: pd.Timestamp
    ) -> List[ScenarioType]:
        """
        Identify which scenarios match current conditions.

        Args:
            as_of_date: Date to check conditions for

        Returns:
            List of matching scenario types
        """
        df = get_available_history(self.data, as_of_date)
        if df.empty:
            return [ScenarioType.BASELINE]

        current = df.iloc[[-1]]  # Last row as DataFrame
        matching = []

        for scenario, config in SCENARIOS.items():
            try:
                mask = config['filter'](current)
                # Handle both Series and ndarray returns
                if isinstance(mask, pd.Series):
                    match = mask.iloc[0]
                elif hasattr(mask, '__iter__'):
                    match = list(mask)[0]
                else:
                    match = bool(mask)

                if match:
                    matching.append(scenario)
            except (KeyError, TypeError, IndexError):
                continue

        return matching if matching else [ScenarioType.BASELINE]

    def get_quality_summary(self, matrix: pd.DataFrame) -> Dict:
        """Get quality summary of matrix."""
        total = len(matrix)
        no_data = len(matrix[matrix['Count'] == 0])
        unreliable = len(matrix[matrix['Quality'] == 'Unreliable'])
        small = len(matrix[matrix['Quality'] == 'Small Sample'])
        sufficient = len(matrix[matrix['Quality'] == 'Sufficient'])

        return {
            'total_cells': total,
            'no_data': no_data,
            'unreliable': unreliable,
            'small_sample': small,
            'sufficient': sufficient,
            'pct_reliable': (sufficient + small) / total * 100 if total > 0 else 0,
        }

    def print_summary(self, matrix: pd.DataFrame) -> None:
        """Print matrix quality summary."""
        summary = self.get_quality_summary(matrix)

        print("\nScenario Matrix Quality Summary:")
        print(f"  Total Cells: {summary['total_cells']}")
        print(f"  Sufficient (10+): {summary['sufficient']}")
        print(f"  Small Sample (5-9): {summary['small_sample']}")
        print(f"  Unreliable (<5): {summary['unreliable']}")
        print(f"  No Data: {summary['no_data']}")
        print(f"  Reliable Rate: {summary['pct_reliable']:.1f}%")
