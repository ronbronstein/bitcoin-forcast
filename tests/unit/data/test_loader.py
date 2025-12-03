"""
Unit tests for src/data/loader.py

Tests data loading and P.I.T. filtering.
"""
import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader class."""

    @pytest.fixture
    def loader(self):
        """Create DataLoader instance."""
        return DataLoader(Path('data'))

    def test_load_full_dataset(self, loader):
        """Should load full dataset successfully."""
        df = loader.load_full_dataset(exclude_usdt=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_exclude_usdt_columns(self, loader):
        """USDT columns should be excluded when specified."""
        df = loader.load_full_dataset(exclude_usdt=True)

        usdt_cols = [c for c in df.columns if 'USDT' in c]
        assert len(usdt_cols) == 0

    def test_include_usdt_columns(self, loader):
        """USDT columns should be included when not excluded."""
        df = loader.load_full_dataset(exclude_usdt=False)

        usdt_cols = [c for c in df.columns if 'USDT' in c]
        assert len(usdt_cols) > 0

    def test_has_required_columns(self, loader):
        """Dataset should have required columns."""
        df = loader.load_full_dataset(exclude_usdt=True)

        required = ['BTC', 'Ret_BTC', 'RSI_BTC', 'Prev_RSI_BTC']
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_datetime_index(self, loader):
        """Index should be DatetimeIndex."""
        df = loader.load_full_dataset(exclude_usdt=True)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == 'Date'


class TestGetDataAsOf:
    """Tests for get_data_as_of() - P.I.T. method."""

    @pytest.fixture
    def loader(self):
        """Create DataLoader instance."""
        return DataLoader(Path('data'))

    def test_pit_filter_excludes_future(self, loader):
        """Data after as_of_date must be excluded."""
        as_of = pd.Timestamp('2022-06-01')
        df = loader.get_data_as_of(as_of)

        assert df.index.max() <= as_of

    def test_pit_filter_includes_as_of(self, loader):
        """as_of_date itself should be included."""
        as_of = pd.Timestamp('2022-06-01')
        df = loader.get_data_as_of(as_of)

        assert as_of in df.index

    def test_pit_returns_less_data(self, loader):
        """P.I.T. filtered data should have fewer rows than full."""
        full_df = loader.load_full_dataset(exclude_usdt=True)
        as_of = pd.Timestamp('2022-06-01')
        pit_df = loader.get_data_as_of(as_of)

        assert len(pit_df) < len(full_df)

    def test_pit_data_is_copy(self, loader):
        """P.I.T. data should be a copy, not view."""
        as_of = pd.Timestamp('2022-06-01')
        df1 = loader.get_data_as_of(as_of)
        df2 = loader.get_data_as_of(as_of)

        # Modify df1 and verify df2 unchanged
        df1.iloc[0, 0] = -999999
        assert df2.iloc[0, 0] != -999999


class TestValidateDataCompleteness:
    """Tests for validate_data_completeness()."""

    @pytest.fixture
    def loader(self):
        """Create DataLoader instance."""
        return DataLoader(Path('data'))

    def test_returns_report_dict(self, loader):
        """Should return dictionary with expected keys."""
        report = loader.validate_data_completeness()

        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'completeness_pct' in report
        assert 'date_range' in report

    def test_completeness_percentage(self, loader):
        """Completeness should be between 0 and 100."""
        report = loader.validate_data_completeness()

        assert 0 <= report['completeness_pct'] <= 100

    def test_high_completeness_after_usdt_exclusion(self, loader):
        """After excluding USDT, completeness should be ~100%."""
        df = loader.load_full_dataset(exclude_usdt=True)
        report = loader.validate_data_completeness(df)

        # Should be very close to 100% after USDT exclusion
        assert report['completeness_pct'] > 99.0
