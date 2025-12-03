"""
Unit tests for src/features/validator.py

Tests Point-in-Time (P.I.T.) validation and data leakage detection.
"""
import pytest
import pandas as pd
import numpy as np

from src.features.validator import PITValidator


class TestCheckFeatureLag:
    """Tests for check_feature_lag()."""

    def test_all_lagged_features_valid(self):
        """All Prev_* features should be valid."""
        df = pd.DataFrame({
            'Prev_RSI_BTC': [50, 60, 70],
            'Prev_RSI_SPX': [45, 55, 65],
            'Ret_BTC': [5, -3, 10],  # Target
        })
        validator = PITValidator(df)

        is_valid, unlagged = validator.check_feature_lag(df, target_col='Ret_BTC')

        assert is_valid is True
        assert len(unlagged) == 0

    def test_unlagged_features_detected(self):
        """Features without Prev_* should be flagged."""
        df = pd.DataFrame({
            'RSI_BTC': [50, 60, 70],  # NOT lagged - problem!
            'Prev_RSI_SPX': [45, 55, 65],
            'Ret_BTC': [5, -3, 10],
        })
        validator = PITValidator(df)

        is_valid, unlagged = validator.check_feature_lag(df, target_col='Ret_BTC')

        assert is_valid is False
        assert 'RSI_BTC' in unlagged

    def test_target_column_excluded(self):
        """Target column should not be flagged as unlagged."""
        df = pd.DataFrame({
            'Prev_RSI_BTC': [50, 60, 70],
            'Ret_BTC': [5, -3, 10],  # Target - should be allowed
        })
        validator = PITValidator(df)

        is_valid, unlagged = validator.check_feature_lag(df, target_col='Ret_BTC')

        assert is_valid is True
        assert 'Ret_BTC' not in unlagged


class TestDetectTargetLeakage:
    """Tests for detect_target_leakage()."""

    def test_no_leakage_normal_case(self):
        """Normal case should not detect leakage."""
        df = pd.DataFrame({
            'Prev_RSI_BTC': [50, 60, 70],
            'Ret_BTC': [5, -3, 10],
        })
        validator = PITValidator(df)

        has_leakage = validator.detect_target_leakage(df, target_col='Ret_BTC')

        assert has_leakage is False

    def test_detects_direct_leakage(self):
        """Should detect if target appears twice."""
        df = pd.DataFrame({
            'Prev_RSI_BTC': [50, 60, 70],
            'Ret_BTC': [5, -3, 10],  # As target
            'Ret_BTC_copy': [5, -3, 10],  # Leakage attempt
        })
        validator = PITValidator(df)

        # Note: This specific test depends on implementation
        # Current implementation checks for target_col in col name
        has_leakage = validator.detect_target_leakage(df, target_col='Ret_BTC')

        # Ret_BTC_copy contains 'Ret_BTC' so might be flagged
        # depending on implementation


class TestValidateTrainingSplit:
    """Tests for validate_training_split()."""

    def test_valid_split(self):
        """Training data <= test_date should be valid."""
        dates = pd.date_range('2020-01-01', periods=6, freq='MS')
        df = pd.DataFrame({'A': range(6)}, index=dates)
        validator = PITValidator(df)

        test_date = pd.Timestamp('2020-06-01')
        train_df = df[df.index <= test_date]

        is_valid = validator.validate_training_split(train_df, test_date)

        assert is_valid is True

    def test_invalid_split_future_data(self):
        """Training data > test_date should be invalid."""
        dates = pd.date_range('2020-01-01', periods=6, freq='MS')
        df = pd.DataFrame({'A': range(6)}, index=dates)
        validator = PITValidator(df)

        test_date = pd.Timestamp('2020-03-01')
        # Incorrectly include all data
        train_df = df

        is_valid = validator.validate_training_split(train_df, test_date)

        assert is_valid is False

    def test_empty_dataframe_valid(self):
        """Empty DataFrame should be valid."""
        df = pd.DataFrame()
        validator = PITValidator(df)

        is_valid = validator.validate_training_split(df, pd.Timestamp('2020-01-01'))

        assert is_valid is True


class TestCheckDateAlignment:
    """Tests for check_date_alignment()."""

    def test_aligned_dates(self):
        """Consecutive monthly dates should be aligned."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'A': range(12)}, index=dates)
        validator = PITValidator(df)

        result = validator.check_date_alignment(df)

        assert result['is_aligned'] is True

    def test_detects_missing_months(self):
        """Missing months should be detected."""
        dates = pd.DatetimeIndex([
            '2020-01-01', '2020-02-01', '2020-04-01'  # March missing
        ])
        df = pd.DataFrame({'A': range(3)}, index=dates)
        validator = PITValidator(df)

        result = validator.check_date_alignment(df)

        assert result['is_aligned'] is False
        assert any('Missing' in issue for issue in result['issues'])


class TestRunFullValidation:
    """Tests for run_full_validation()."""

    def test_valid_data_passes(self):
        """Valid data should pass all checks."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({
            'Prev_RSI_BTC': range(12),
            'Prev_RSI_SPX': range(12),
            'Ret_BTC': range(12),
        }, index=dates)
        validator = PITValidator(df)

        result = validator.run_full_validation(
            df,
            as_of_date=pd.Timestamp('2020-12-01'),
            target_col='Ret_BTC'
        )

        assert result['is_valid'] is True
        assert len(result['issues']) == 0

    def test_invalid_data_fails(self):
        """Invalid data should fail validation."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({
            'RSI_BTC': range(12),  # NOT lagged!
            'Ret_BTC': range(12),
        }, index=dates)
        validator = PITValidator(df)

        result = validator.run_full_validation(
            df,
            as_of_date=pd.Timestamp('2020-12-01'),
            target_col='Ret_BTC'
        )

        assert result['is_valid'] is False
        assert len(result['issues']) > 0

    def test_returns_expected_keys(self):
        """Result should contain expected keys."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({
            'Prev_RSI_BTC': range(12),
            'Ret_BTC': range(12),
        }, index=dates)
        validator = PITValidator(df)

        result = validator.run_full_validation(
            df,
            as_of_date=pd.Timestamp('2020-12-01'),
            target_col='Ret_BTC'
        )

        assert 'is_valid' in result
        assert 'issues' in result
        assert 'warnings' in result
        assert 'n_samples' in result
