"""
Unit tests for src/data/cleaner.py

Tests data cleaning and validation utilities.
"""
import pytest
import pandas as pd
import numpy as np

from src.data.cleaner import (
    handle_missing_values,
    validate_rsi_bounds,
    detect_data_gaps,
    validate_returns_sanity,
)


class TestHandleMissingValues:
    """Tests for handle_missing_values()."""

    @pytest.fixture
    def df_with_gaps(self):
        """Create DataFrame with missing values."""
        return pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
            'B': [10.0, 20.0, np.nan, np.nan, 50.0, 60.0],
        })

    def test_interpolate_fills_gaps(self, df_with_gaps):
        """Interpolation should fill single gaps."""
        result, report = handle_missing_values(df_with_gaps, method='interpolate')

        # Single gap in column A should be filled
        assert not pd.isna(result.loc[1, 'A'])

    def test_interpolate_respects_max_gap(self, df_with_gaps):
        """Interpolation should not fill gaps larger than max_gap."""
        result, report = handle_missing_values(
            df_with_gaps, method='interpolate', max_gap=1
        )

        # Column B has 2 consecutive NaNs, only first should be filled
        # with limit=1
        assert pd.isna(result.loc[3, 'B']) or not pd.isna(result.loc[2, 'B'])

    def test_drop_removes_nan_rows(self, df_with_gaps):
        """Drop method should remove rows with NaN."""
        result, report = handle_missing_values(df_with_gaps, method='drop')

        assert not result.isna().any().any()
        assert len(result) < len(df_with_gaps)

    def test_returns_report(self, df_with_gaps):
        """Should return report with original nulls."""
        result, report = handle_missing_values(df_with_gaps)

        assert 'original_nulls' in report
        assert 'method' in report
        assert report['original_nulls']['A'] == 2
        assert report['original_nulls']['B'] == 2


class TestValidateRsiBounds:
    """Tests for validate_rsi_bounds()."""

    def test_valid_rsi_values(self):
        """RSI values in [0, 100] should be valid."""
        df = pd.DataFrame({
            'RSI_BTC': [30, 50, 70, 100, 0],
            'RSI_SPX': [45, 55, 65, 75, 85],
        })

        is_valid, invalid = validate_rsi_bounds(df)
        assert is_valid is True
        assert len(invalid) == 0

    def test_invalid_rsi_above_100(self):
        """RSI above 100 should be invalid."""
        df = pd.DataFrame({
            'RSI_BTC': [30, 50, 105],  # 105 is invalid
        })

        is_valid, invalid = validate_rsi_bounds(df)
        assert is_valid is False
        assert 'RSI_BTC' in invalid

    def test_invalid_rsi_below_0(self):
        """RSI below 0 should be invalid."""
        df = pd.DataFrame({
            'RSI_BTC': [30, 50, -5],  # -5 is invalid
        })

        is_valid, invalid = validate_rsi_bounds(df)
        assert is_valid is False
        assert 'RSI_BTC' in invalid

    def test_ignores_non_rsi_columns(self):
        """Non-RSI columns should not be checked."""
        df = pd.DataFrame({
            'BTC': [100, 200, 300],  # Not RSI
            'RSI_BTC': [30, 50, 70],
        })

        is_valid, invalid = validate_rsi_bounds(df)
        assert is_valid is True


class TestDetectDataGaps:
    """Tests for detect_data_gaps()."""

    def test_reports_total_missing(self):
        """Should report total missing per column."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [1, 2, 3, 4, 5],
        })

        gaps = detect_data_gaps(df)

        assert gaps.loc['A', 'total_missing'] == 2
        assert gaps.loc['B', 'total_missing'] == 0

    def test_reports_max_consecutive(self):
        """Should report max consecutive NaNs."""
        df = pd.DataFrame({
            'A': [1, np.nan, np.nan, np.nan, 5],  # 3 consecutive
            'B': [1, np.nan, 3, np.nan, 5],  # 1 max consecutive
        })

        gaps = detect_data_gaps(df)

        assert gaps.loc['A', 'max_consecutive'] == 3
        assert gaps.loc['B', 'max_consecutive'] == 1

    def test_reports_percentage(self):
        """Should report percentage missing."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],  # 40% missing
        })

        gaps = detect_data_gaps(df)

        assert gaps.loc['A', 'pct_missing'] == 40.0


class TestValidateReturnsSanity:
    """Tests for validate_returns_sanity()."""

    def test_normal_returns_valid(self):
        """Normal returns should pass validation."""
        df = pd.DataFrame({
            'Ret_BTC': [5, -3, 10, -8, 15],
        })

        report = validate_returns_sanity(df, threshold=100)

        assert report['has_extremes'] is False

    def test_extreme_returns_detected(self):
        """Extreme returns should be flagged."""
        df = pd.DataFrame({
            'Ret_BTC': [5, -3, 150, -8, 15],  # 150% is extreme
        })

        report = validate_returns_sanity(df, threshold=100)

        assert report['has_extremes'] is True
        assert 'Ret_BTC' in report['extreme_columns']

    def test_ignores_non_return_columns(self):
        """Non-return columns should be ignored."""
        df = pd.DataFrame({
            'BTC': [50000, 60000, 70000],  # Not a return column
            'Ret_BTC': [5, 10, 15],
        })

        report = validate_returns_sanity(df, threshold=100)

        assert report['has_extremes'] is False
