"""
Walk-forward backtesting with strict Point-in-Time (P.I.T.) enforcement.

Simulates real-world forecasting by training only on past data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Type
from dataclasses import dataclass

from ..utils.dates import get_available_history, get_next_month
from ..models.evaluator import ModelEvaluator, PredictionResult
from ..features.validator import PITValidator


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    min_training_months: int = 36
    target_col: str = 'Ret_BTC'


class BacktestEngine:
    """
    Walk-forward backtesting engine with P.I.T. validation.

    For each test month:
    1. Filter data to only include history up to that month
    2. Train model on filtered data
    3. Predict next month
    4. Record actual vs predicted
    """

    def __init__(self, model_class: Type, config: BacktestConfig):
        """
        Initialize the backtest engine.

        Args:
            model_class: Model class to instantiate for each test
            config: Backtest configuration
        """
        self.model_class = model_class
        self.config = config
        self.evaluator = ModelEvaluator()
        self.pit_validator = None

    def run(self, data: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Execute walk-forward backtest.

        Args:
            data: Full dataset with DatetimeIndex
            verbose: Print progress

        Returns:
            Dictionary with results DataFrame and metrics
        """
        self.evaluator.clear()
        self.pit_validator = PITValidator(data)

        test_months = self._get_test_months(data)

        if verbose:
            print(f"Running backtest: {len(test_months)} months")
            print(f"Period: {test_months[0].date()} to {test_months[-1].date()}")

        for i, forecast_date in enumerate(test_months):
            # Get the actual outcome date (next month)
            actual_date = get_next_month(forecast_date)

            # Skip if actual date not in data
            if actual_date not in data.index:
                continue

            result = self._run_single_step(data, forecast_date, actual_date)

            if result is not None:
                self.evaluator.add_result(result)

                if verbose and (i + 1) % 12 == 0:
                    print(f"  Processed {i + 1}/{len(test_months)} months...")

        if verbose:
            print(f"Completed: {self.evaluator.n_predictions} valid predictions")

        return {
            'results': self.evaluator.to_dataframe(),
            'metrics': self.evaluator.generate_report(),
        }

    def _get_test_months(self, data: pd.DataFrame) -> List[pd.Timestamp]:
        """Generate list of months to test."""
        mask = (
            (data.index >= self.config.start_date) &
            (data.index <= self.config.end_date)
        )
        return data[mask].index.tolist()

    def _run_single_step(
        self,
        data: pd.DataFrame,
        forecast_date: pd.Timestamp,
        actual_date: pd.Timestamp
    ) -> Optional[PredictionResult]:
        """
        Run single backtest step with P.I.T. validation.

        Args:
            data: Full dataset
            forecast_date: Date we're making prediction FROM
            actual_date: Date we're predicting FOR

        Returns:
            PredictionResult or None if invalid
        """
        # CRITICAL P.I.T. FILTER
        train_data = get_available_history(data, forecast_date)

        # Validate P.I.T. compliance
        if not self.pit_validator.validate_training_split(train_data, forecast_date):
            raise ValueError(f"P.I.T. violation at {forecast_date}")

        # Check minimum training samples
        train_returns = train_data[self.config.target_col].dropna()
        if len(train_returns) < self.config.min_training_months:
            return None

        # Train model (P.I.T. compliant)
        model = self.model_class()
        model.fit(train_returns, forecast_date)

        # Generate prediction
        prediction = model.predict()

        # Get actual outcome
        try:
            actual_return = data.loc[actual_date, self.config.target_col]
        except KeyError:
            return None

        if pd.isna(actual_return):
            return None

        # Determine actual direction
        actual_direction = self._get_direction(actual_return)

        return PredictionResult(
            date=actual_date,
            predicted_return=prediction.mean,
            actual_return=actual_return,
            predicted_direction=prediction.direction,
            actual_direction=actual_direction,
            p10=prediction.p10,
            p25=prediction.p25,
            p50=prediction.p50,
            p75=prediction.p75,
            p90=prediction.p90,
        )

    def _get_direction(self, return_value: float) -> int:
        """Convert return to direction."""
        if return_value > 0.5:
            return 1
        elif return_value < -0.5:
            return -1
        return 0
