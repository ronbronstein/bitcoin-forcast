"""
Baseline model: Historical mean return prediction.

This establishes the minimum performance bar for more sophisticated models.
If we can't beat this, more complex models are likely overfitting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from ..utils.dates import get_available_history


@dataclass
class Prediction:
    """Container for model predictions with uncertainty."""
    mean: float
    std: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    direction: int  # 1 = up, -1 = down, 0 = neutral


class HistoricalMeanModel:
    """
    Baseline forecasting model using historical mean returns.

    Predicts: next month's return = historical average return.
    This is the simplest reasonable forecast and establishes baseline.
    """

    def __init__(self, neutral_threshold: float = 0.5):
        """
        Initialize the model.

        Args:
            neutral_threshold: Return threshold for neutral prediction (%)
        """
        self.neutral_threshold = neutral_threshold
        self.mean_return: Optional[float] = None
        self.std_return: Optional[float] = None
        self.n_samples: int = 0
        self.training_end_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        returns: pd.Series,
        as_of_date: pd.Timestamp
    ) -> 'HistoricalMeanModel':
        """
        Fit model on historical returns up to as_of_date.

        Args:
            returns: Series of monthly returns (%)
            as_of_date: Maximum date for training (P.I.T.)

        Returns:
            Self for method chaining
        """
        # P.I.T. filter
        pit_returns = returns[returns.index <= as_of_date].dropna()

        self.mean_return = pit_returns.mean()
        self.std_return = pit_returns.std()
        self.n_samples = len(pit_returns)
        self.training_end_date = as_of_date

        return self

    def predict(self) -> Prediction:
        """
        Generate prediction with uncertainty bounds.

        Returns:
            Prediction object with mean, percentiles, and direction
        """
        if self.mean_return is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Calculate percentiles assuming normal distribution
        z_scores = {
            'p10': -1.28,
            'p25': -0.67,
            'p50': 0.0,
            'p75': 0.67,
            'p90': 1.28,
        }

        direction = self._get_direction(self.mean_return)

        return Prediction(
            mean=self.mean_return,
            std=self.std_return,
            p10=self.mean_return + z_scores['p10'] * self.std_return,
            p25=self.mean_return + z_scores['p25'] * self.std_return,
            p50=self.mean_return,  # Median = Mean for symmetric
            p75=self.mean_return + z_scores['p75'] * self.std_return,
            p90=self.mean_return + z_scores['p90'] * self.std_return,
            direction=direction,
        )

    def _get_direction(self, return_value: float) -> int:
        """Convert return to direction."""
        if return_value > self.neutral_threshold:
            return 1
        elif return_value < -self.neutral_threshold:
            return -1
        return 0

    def get_model_info(self) -> Dict:
        """Return model metadata for logging."""
        return {
            'model_type': 'HistoricalMeanModel',
            'n_samples': self.n_samples,
            'training_end_date': str(self.training_end_date),
            'mean_return': self.mean_return,
            'std_return': self.std_return,
        }

    def __repr__(self) -> str:
        if self.mean_return is None:
            return "HistoricalMeanModel(not fitted)"
        return (
            f"HistoricalMeanModel(mean={self.mean_return:.2f}%, "
            f"std={self.std_return:.2f}%, n={self.n_samples})"
        )
