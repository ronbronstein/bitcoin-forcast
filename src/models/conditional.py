"""
Conditional mean model: predicts based on current market regime.

Uses regime-specific historical returns rather than global mean.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .baseline import Prediction
from .regimes import RegimeClassifier, RegimeType
from ..utils.dates import get_available_history


class ConditionalMeanModel:
    """
    Forecasts using historical returns from similar market regimes.

    Unlike HistoricalMeanModel which uses all history, this model:
    1. Classifies current market regime from features
    2. Filters historical data to matching regime periods
    3. Predicts based on regime-specific mean

    Falls back to baseline (all history) if insufficient regime samples.
    """

    def __init__(
        self,
        min_samples: int = 5,
        neutral_threshold: float = 0.5
    ):
        """
        Initialize the model.

        Args:
            min_samples: Minimum samples for regime-specific prediction
            neutral_threshold: Return threshold for neutral direction (%)
        """
        self.min_samples = min_samples
        self.neutral_threshold = neutral_threshold
        self.classifier = RegimeClassifier(min_samples=min_samples)

        # State after fit()
        self.current_regime: Optional[RegimeType] = None
        self.regime_mean: Optional[float] = None
        self.regime_std: Optional[float] = None
        self.n_samples: int = 0
        self.used_fallback: bool = False
        self.training_end_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        returns: pd.Series,
        as_of_date: pd.Timestamp,
        features: Optional[pd.DataFrame] = None
    ) -> 'ConditionalMeanModel':
        """
        Fit model by identifying regime and computing regime-specific stats.

        Args:
            returns: Series of monthly returns (%)
            as_of_date: Maximum date for training (P.I.T.)
            features: DataFrame with Prev_* columns for classification

        Returns:
            Self for method chaining
        """
        # P.I.T. filter
        pit_returns = returns[returns.index <= as_of_date].dropna()

        if features is None:
            # Fall back to baseline if no features provided
            self._fit_baseline(pit_returns, as_of_date)
            return self

        # P.I.T. filter features
        pit_features = get_available_history(features, as_of_date)

        if pit_features.empty:
            self._fit_baseline(pit_returns, as_of_date)
            return self

        # Get current features for classification (most recent row)
        current_features = pit_features.iloc[-1]

        # Classify current regime
        self.current_regime = self.classifier.classify(current_features)

        # Filter history for regime
        regime_data = self.classifier.filter_history(
            pit_features,
            self.current_regime
        )

        # Get returns for regime periods
        regime_indices = regime_data.index.intersection(pit_returns.index)
        regime_returns = pit_returns.loc[regime_indices]

        # Fallback if insufficient samples
        if len(regime_returns) < self.min_samples:
            self.used_fallback = True
            self.current_regime = RegimeType.BASELINE
            regime_returns = pit_returns

        # Compute statistics
        self.regime_mean = regime_returns.mean()
        self.regime_std = regime_returns.std()
        self.n_samples = len(regime_returns)
        self.training_end_date = as_of_date

        return self

    def _fit_baseline(
        self,
        returns: pd.Series,
        as_of_date: pd.Timestamp
    ) -> None:
        """Fall back to baseline (all history) prediction."""
        self.current_regime = RegimeType.BASELINE
        self.used_fallback = True
        self.regime_mean = returns.mean()
        self.regime_std = returns.std()
        self.n_samples = len(returns)
        self.training_end_date = as_of_date

    def predict(self) -> Prediction:
        """
        Generate regime-conditioned prediction.

        Returns:
            Prediction object with mean, percentiles, and direction
        """
        if self.regime_mean is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Calculate percentiles assuming normal distribution
        z_scores = {
            'p10': -1.28,
            'p25': -0.67,
            'p50': 0.0,
            'p75': 0.67,
            'p90': 1.28,
        }

        direction = self._get_direction(self.regime_mean)

        return Prediction(
            mean=self.regime_mean,
            std=self.regime_std,
            p10=self.regime_mean + z_scores['p10'] * self.regime_std,
            p25=self.regime_mean + z_scores['p25'] * self.regime_std,
            p50=self.regime_mean,
            p75=self.regime_mean + z_scores['p75'] * self.regime_std,
            p90=self.regime_mean + z_scores['p90'] * self.regime_std,
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
        """Return regime-aware model metadata."""
        return {
            'model_type': 'ConditionalMeanModel',
            'current_regime': self.current_regime.value if self.current_regime else None,
            'n_samples': self.n_samples,
            'used_fallback': self.used_fallback,
            'training_end_date': str(self.training_end_date),
            'regime_mean': self.regime_mean,
            'regime_std': self.regime_std,
        }

    def __repr__(self) -> str:
        if self.regime_mean is None:
            return "ConditionalMeanModel(not fitted)"
        fallback_str = " [fallback]" if self.used_fallback else ""
        return (
            f"ConditionalMeanModel(regime={self.current_regime.value}, "
            f"mean={self.regime_mean:.2f}%, n={self.n_samples}){fallback_str}"
        )
