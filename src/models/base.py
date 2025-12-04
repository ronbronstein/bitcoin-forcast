"""Base model protocol for type safety and consistent interface."""
from typing import Protocol, Dict, Optional, runtime_checkable
import pandas as pd

from .baseline import Prediction


@runtime_checkable
class ForecastModel(Protocol):
    """
    Protocol defining the forecast model interface.

    All forecast models must implement these methods to work with
    the BacktestEngine and evaluation framework.
    """

    def fit(
        self,
        returns: pd.Series,
        as_of_date: pd.Timestamp,
        features: Optional[pd.DataFrame] = None
    ) -> 'ForecastModel':
        """
        Fit model on historical data (P.I.T. compliant).

        Args:
            returns: Series of monthly returns (%)
            as_of_date: Maximum date for training (P.I.T.)
            features: Optional DataFrame with Prev_* columns

        Returns:
            Self for method chaining
        """
        ...

    def predict(self) -> Prediction:
        """
        Generate prediction with uncertainty bounds.

        Returns:
            Prediction object with mean, percentiles, direction
        """
        ...

    def get_model_info(self) -> Dict:
        """
        Return model metadata for logging.

        Returns:
            Dictionary with model type, parameters, training info
        """
        ...
