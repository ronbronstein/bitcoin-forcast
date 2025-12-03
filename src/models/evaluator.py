"""
Model evaluation metrics and reporting.

Provides standardized metrics for comparing forecasting models.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class PredictionResult:
    """Single prediction with actual outcome."""
    date: pd.Timestamp
    predicted_return: float
    actual_return: float
    predicted_direction: int
    actual_direction: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float

    def direction_correct(self) -> bool:
        """Check if direction prediction was correct."""
        return self.predicted_direction == self.actual_direction

    def within_p10_p90(self) -> bool:
        """Check if actual was within P10-P90 band."""
        return self.p10 <= self.actual_return <= self.p90

    def within_p25_p75(self) -> bool:
        """Check if actual was within P25-P75 band."""
        return self.p25 <= self.actual_return <= self.p75

    def absolute_error(self) -> float:
        """Get absolute error between predicted and actual."""
        return abs(self.actual_return - self.predicted_return)


class ModelEvaluator:
    """
    Calculates forecasting performance metrics.

    Tracks individual predictions and computes aggregate statistics.
    """

    def __init__(self):
        self.results: List[PredictionResult] = []

    def add_result(self, result: PredictionResult) -> None:
        """Add a single prediction result."""
        self.results.append(result)

    def clear(self) -> None:
        """Clear all stored results."""
        self.results = []

    @property
    def n_predictions(self) -> int:
        """Return number of predictions made."""
        return len(self.results)

    def directional_accuracy(self) -> float:
        """Calculate percentage of correct directional predictions."""
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.direction_correct())
        return correct / len(self.results)

    def mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Note: This is absolute error in percentage points, not relative MAPE.
        """
        if not self.results:
            return 0.0
        errors = [r.absolute_error() for r in self.results]
        return np.mean(errors)

    def rmse(self) -> float:
        """Calculate Root Mean Squared Error."""
        if not self.results:
            return 0.0
        squared_errors = [
            (r.actual_return - r.predicted_return) ** 2
            for r in self.results
        ]
        return np.sqrt(np.mean(squared_errors))

    def band_capture_rate(self, band: str = 'p10_p90') -> float:
        """
        Calculate percentage of actuals within prediction bands.

        Args:
            band: 'p10_p90' or 'p25_p75'
        """
        if not self.results:
            return 0.0

        if band == 'p10_p90':
            within = sum(1 for r in self.results if r.within_p10_p90())
        elif band == 'p25_p75':
            within = sum(1 for r in self.results if r.within_p25_p75())
        else:
            raise ValueError(f"Unknown band: {band}")

        return within / len(self.results)

    def generate_report(self) -> Dict[str, float]:
        """Generate comprehensive evaluation report."""
        return {
            'n_predictions': self.n_predictions,
            'directional_accuracy': self.directional_accuracy(),
            'mape': self.mape(),
            'rmse': self.rmse(),
            'p10_p90_capture': self.band_capture_rate('p10_p90'),
            'p25_p75_capture': self.band_capture_rate('p25_p75'),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()

        records = []
        for r in self.results:
            record = asdict(r)
            record['direction_correct'] = r.direction_correct()
            record['within_p10_p90'] = r.within_p10_p90()
            record['abs_error'] = r.absolute_error()
            records.append(record)

        return pd.DataFrame(records)

    def print_summary(self) -> None:
        """Print formatted evaluation summary."""
        report = self.generate_report()
        print("=" * 50)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Predictions:      {report['n_predictions']}")
        print(f"Directional Accuracy:   {report['directional_accuracy']:.1%}")
        print(f"MAPE:                   {report['mape']:.1f}%")
        print(f"RMSE:                   {report['rmse']:.1f}%")
        print(f"P10-P90 Capture:        {report['p10_p90_capture']:.1%}")
        print(f"P25-P75 Capture:        {report['p25_p75_capture']:.1%}")
        print("=" * 50)
