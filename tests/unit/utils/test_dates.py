"""
Unit tests for src/utils/dates.py

Tests Point-in-Time (P.I.T.) date utilities.
"""
import pytest
import pandas as pd
import numpy as np

from src.utils.dates import (
    get_month_start,
    validate_as_of_date,
    get_available_history,
    months_between,
    get_next_month,
)


class TestGetMonthStart:
    """Tests for get_month_start()."""

    def test_mid_month_date(self):
        """Mid-month date should return first of month."""
        date = pd.Timestamp('2023-06-15')
        result = get_month_start(date)
        assert result == pd.Timestamp('2023-06-01')

    def test_first_of_month(self):
        """First of month should remain unchanged."""
        date = pd.Timestamp('2023-06-01')
        result = get_month_start(date)
        assert result == pd.Timestamp('2023-06-01')

    def test_last_of_month(self):
        """Last of month should return first of same month."""
        date = pd.Timestamp('2023-06-30')
        result = get_month_start(date)
        assert result == pd.Timestamp('2023-06-01')


class TestValidateAsOfDate:
    """Tests for validate_as_of_date()."""

    def test_valid_date_in_index(self):
        """Date in index and not future should be valid."""
        index = pd.DatetimeIndex(['2020-01-01', '2020-02-01', '2020-03-01'])
        result = validate_as_of_date(pd.Timestamp('2020-02-01'), index)
        assert result is True

    def test_date_not_in_index(self):
        """Date not in index should be invalid."""
        index = pd.DatetimeIndex(['2020-01-01', '2020-03-01'])
        result = validate_as_of_date(pd.Timestamp('2020-02-01'), index)
        assert result is False

    def test_future_date_invalid(self):
        """Future date should be invalid."""
        future = pd.Timestamp.now() + pd.DateOffset(years=1)
        index = pd.DatetimeIndex([future])
        result = validate_as_of_date(future, index)
        assert result is False


class TestGetAvailableHistory:
    """Tests for get_available_history() - CRITICAL P.I.T. function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        return pd.DataFrame(
            {'value': range(12)},
            index=dates
        )

    def test_excludes_future_data(self, sample_df):
        """Data after as_of_date must be excluded."""
        as_of = pd.Timestamp('2020-06-01')
        result = get_available_history(sample_df, as_of)

        assert result.index.max() <= as_of
        assert len(result) == 6  # Jan-Jun

    def test_includes_as_of_date(self, sample_df):
        """as_of_date itself should be included."""
        as_of = pd.Timestamp('2020-06-01')
        result = get_available_history(sample_df, as_of)

        assert as_of in result.index

    def test_returns_copy(self, sample_df):
        """Should return a copy, not a view."""
        as_of = pd.Timestamp('2020-06-01')
        result = get_available_history(sample_df, as_of)

        # Modify result and verify original unchanged
        result.loc[result.index[0], 'value'] = 999
        assert sample_df.iloc[0]['value'] != 999

    def test_empty_result_for_early_date(self, sample_df):
        """Date before data start should return empty."""
        as_of = pd.Timestamp('2019-01-01')
        result = get_available_history(sample_df, as_of)

        assert len(result) == 0


class TestMonthsBetween:
    """Tests for months_between()."""

    def test_same_month(self):
        """Same month should return 0."""
        start = pd.Timestamp('2020-06-01')
        end = pd.Timestamp('2020-06-15')
        assert months_between(start, end) == 0

    def test_one_month_apart(self):
        """One month apart should return 1."""
        start = pd.Timestamp('2020-06-01')
        end = pd.Timestamp('2020-07-01')
        assert months_between(start, end) == 1

    def test_one_year_apart(self):
        """One year apart should return 12."""
        start = pd.Timestamp('2020-06-01')
        end = pd.Timestamp('2021-06-01')
        assert months_between(start, end) == 12

    def test_multi_year(self):
        """Multiple years should calculate correctly."""
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2023-06-01')
        assert months_between(start, end) == 41  # 3*12 + 5


class TestGetNextMonth:
    """Tests for get_next_month()."""

    def test_standard_month(self):
        """Standard month should return first of next month."""
        date = pd.Timestamp('2020-06-15')
        result = get_next_month(date)
        assert result == pd.Timestamp('2020-07-01')

    def test_year_end(self):
        """December should return January of next year."""
        date = pd.Timestamp('2020-12-15')
        result = get_next_month(date)
        assert result == pd.Timestamp('2021-01-01')

    def test_first_of_month(self):
        """First of month should return first of next month."""
        date = pd.Timestamp('2020-06-01')
        result = get_next_month(date)
        assert result == pd.Timestamp('2020-07-01')
