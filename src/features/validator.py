"""
Look-ahead bias detection and Point-in-Time (P.I.T.) validation.

Provides tools to detect data leakage in feature matrices.
"""
import pandas as pd
from typing import List, Tuple, Dict


class PITValidator:
    """
    Validates Point-in-Time compliance of feature matrices.

    Use this to audit feature construction and catch data leakage.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with dataset to validate.

        Args:
            data: DataFrame with features and targets
        """
        self.data = data

    def check_feature_lag(
        self,
        feature_df: pd.DataFrame,
        target_col: str = 'Ret_BTC'
    ) -> Tuple[bool, List[str]]:
        """
        Verify all feature columns are lagged (start with 'Prev_').

        Args:
            feature_df: DataFrame to check
            target_col: Target column name (allowed to not be lagged)

        Returns:
            Tuple of (is_valid, list of unlagged columns)
        """
        unlagged = []
        for col in feature_df.columns:
            if col == target_col:
                continue
            if not col.startswith('Prev_'):
                unlagged.append(col)

        return len(unlagged) == 0, unlagged

    def detect_target_leakage(
        self,
        feature_df: pd.DataFrame,
        target_col: str = 'Ret_BTC'
    ) -> bool:
        """
        Check if target column appears in features (not as target).

        Args:
            feature_df: DataFrame with features
            target_col: Target column name

        Returns:
            True if leakage detected, False otherwise
        """
        feature_cols = [c for c in feature_df.columns if c != target_col]

        # Check if target or derived target in features
        for col in feature_cols:
            # Direct match
            if col == target_col:
                return True
            # Partial match (e.g., 'Ret_BTC_lag' would be suspicious)
            if target_col in col and col != f'Prev_{target_col}':
                return True

        return False

    def validate_training_split(
        self,
        train_df: pd.DataFrame,
        test_date: pd.Timestamp
    ) -> bool:
        """
        Ensure training data does not extend past test_date.

        Args:
            train_df: Training DataFrame
            test_date: Maximum allowed date

        Returns:
            True if valid, False if data leakage detected
        """
        if train_df.empty:
            return True
        return train_df.index.max() <= test_date

    def check_date_alignment(self, df: pd.DataFrame) -> Dict:
        """
        Check for date alignment issues.

        Returns:
            Dictionary with alignment analysis
        """
        issues = []

        # Check for gaps in dates
        expected_months = pd.date_range(
            df.index.min(), df.index.max(), freq='MS'
        )
        actual_months = df.index
        missing = expected_months.difference(actual_months)

        if len(missing) > 0:
            issues.append(f"Missing {len(missing)} months")

        # Check for duplicates
        if df.index.duplicated().any():
            issues.append("Duplicate dates found")

        return {
            'is_aligned': len(issues) == 0,
            'issues': issues,
            'expected_months': len(expected_months),
            'actual_months': len(actual_months),
        }

    def run_full_validation(
        self,
        feature_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
        target_col: str = 'Ret_BTC'
    ) -> Dict:
        """
        Run all validation checks and return detailed report.

        Args:
            feature_df: DataFrame to validate
            as_of_date: Maximum allowed date
            target_col: Target column name

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        # Check 1: Feature lag
        lag_valid, unlagged = self.check_feature_lag(feature_df, target_col)
        if not lag_valid:
            issues.append(f"Unlagged features: {unlagged}")

        # Check 2: Target leakage
        if self.detect_target_leakage(feature_df, target_col):
            issues.append(f"Target '{target_col}' leakage detected in features")

        # Check 3: Future data
        if not self.validate_training_split(feature_df, as_of_date):
            issues.append(
                f"Data extends past as_of_date ({as_of_date}): "
                f"max date is {feature_df.index.max()}"
            )

        # Check 4: Date alignment
        alignment = self.check_date_alignment(feature_df)
        if not alignment['is_aligned']:
            warnings.extend(alignment['issues'])

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'n_samples': len(feature_df),
            'date_range': (feature_df.index.min(), feature_df.index.max()),
        }
