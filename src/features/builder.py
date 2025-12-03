"""
Feature matrix construction with strict Point-in-Time (P.I.T.) enforcement.

All methods require explicit as_of_date parameter to prevent data leakage.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from ..utils.dates import get_available_history


# Default features for prediction (all must be Prev_* lagged)
DEFAULT_FEATURES = [
    'Prev_RSI_BTC',
    'Prev_RSI_SPX',
    'Prev_RSI_NDX',
    'Prev_RSI_GLD',
    'Prev_RSI_TLT',
    'Prev_DXY_Trend',
    'Prev_Rate_Trend',
    'Prev_Active_Addresses_Z',
]


class FeatureBuilder:
    """
    Builds feature matrices for model training and prediction.

    CRITICAL: All methods require explicit as_of_date parameter.
    Features are ALWAYS lagged by 1 month (T-1 predictors).
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with full dataset.

        Args:
            data: DataFrame with DatetimeIndex containing all features and targets
        """
        self.data = data
        self._validate_required_columns()

    def _validate_required_columns(self) -> None:
        """Check that essential columns exist."""
        required = ['Ret_BTC', 'Prev_RSI_BTC']
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_available_features(self) -> List[str]:
        """Return list of available lagged features (Prev_* columns)."""
        return [c for c in self.data.columns if c.startswith('Prev_')]

    def build_training_data(
        self,
        as_of_date: pd.Timestamp,
        target_col: str = 'Ret_BTC',
        feature_cols: Optional[List[str]] = None,
        min_samples: int = 24
    ) -> pd.DataFrame:
        """
        Build feature matrix for training, strictly P.I.T. compliant.

        Args:
            as_of_date: Maximum date to include (inclusive)
            target_col: Target variable column name
            feature_cols: List of feature columns (default: DEFAULT_FEATURES)
            min_samples: Minimum required samples

        Returns:
            DataFrame with features and target, NaN rows dropped

        Raises:
            ValueError: If data leakage detected or insufficient samples
        """
        if feature_cols is None:
            feature_cols = [c for c in DEFAULT_FEATURES if c in self.data.columns]

        # P.I.T. filter: Only data up to and including as_of_date
        pit_data = get_available_history(self.data, as_of_date)

        # Select features + target
        cols_to_use = feature_cols + [target_col]
        cols_available = [c for c in cols_to_use if c in pit_data.columns]
        result = pit_data[cols_available].dropna()

        # Validation
        self._validate_no_future_data(result, as_of_date)
        self._validate_all_features_lagged(feature_cols)

        if len(result) < min_samples:
            raise ValueError(
                f"Insufficient samples: {len(result)} < {min_samples} required"
            )

        return result

    def get_current_features(
        self,
        as_of_date: pd.Timestamp,
        feature_cols: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Get feature values for the most recent complete month.

        Used for making predictions.

        Args:
            as_of_date: Date to get features for
            feature_cols: Specific features to return

        Returns:
            Series with feature values for as_of_date
        """
        if feature_cols is None:
            feature_cols = [c for c in DEFAULT_FEATURES if c in self.data.columns]

        pit_data = get_available_history(self.data, as_of_date)

        if pit_data.empty:
            raise ValueError(f"No data available as of {as_of_date}")

        current = pit_data.iloc[-1][feature_cols]

        if current.isna().any():
            missing = current[current.isna()].index.tolist()
            raise ValueError(f"Missing feature values: {missing}")

        return current

    def _validate_no_future_data(
        self,
        features: pd.DataFrame,
        as_of_date: pd.Timestamp
    ) -> None:
        """CRITICAL: Ensure no data leakage."""
        if features.index.max() > as_of_date:
            raise ValueError(
                f"Data leakage! Max date {features.index.max()} > as_of_date {as_of_date}"
            )

    def _validate_all_features_lagged(self, feature_cols: List[str]) -> None:
        """Ensure all feature columns are properly lagged."""
        unlagged = [c for c in feature_cols if not c.startswith('Prev_')]
        if unlagged:
            raise ValueError(
                f"Unlagged features detected (must start with 'Prev_'): {unlagged}"
            )

    def get_feature_stats(self, as_of_date: pd.Timestamp) -> Dict:
        """Get statistics about available features."""
        train_data = self.build_training_data(
            as_of_date,
            min_samples=1  # Allow small for stats
        )
        return {
            'n_samples': len(train_data),
            'n_features': len(train_data.columns) - 1,  # Exclude target
            'date_range': (train_data.index.min(), train_data.index.max()),
            'feature_means': train_data.mean().to_dict(),
        }
