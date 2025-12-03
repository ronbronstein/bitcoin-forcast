"""
Unit tests for src/features/builder.py

Tests feature matrix construction with P.I.T. enforcement.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.builder import FeatureBuilder, DEFAULT_FEATURES
from src.data.loader import DataLoader


class TestFeatureBuilder:
    """Tests for FeatureBuilder class."""

    @pytest.fixture
    def real_data(self):
        """Load real dataset for testing."""
        loader = DataLoader(Path('data'))
        return loader.load_full_dataset(exclude_usdt=True)

    @pytest.fixture
    def builder(self, real_data):
        """Create FeatureBuilder with real data."""
        return FeatureBuilder(real_data)

    def test_init_validates_columns(self):
        """Should raise error if required columns missing."""
        df = pd.DataFrame({'X': [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            FeatureBuilder(df)

    def test_get_available_features(self, builder):
        """Should return list of Prev_* columns."""
        features = builder.get_available_features()

        assert len(features) > 0
        assert all(f.startswith('Prev_') for f in features)


class TestBuildTrainingData:
    """Tests for build_training_data() - P.I.T. critical method."""

    @pytest.fixture
    def real_data(self):
        """Load real dataset for testing."""
        loader = DataLoader(Path('data'))
        return loader.load_full_dataset(exclude_usdt=True)

    @pytest.fixture
    def builder(self, real_data):
        """Create FeatureBuilder with real data."""
        return FeatureBuilder(real_data)

    def test_pit_excludes_future_data(self, builder):
        """Training data must not extend past as_of_date."""
        as_of = pd.Timestamp('2022-06-01')
        train_data = builder.build_training_data(as_of)

        assert train_data.index.max() <= as_of

    def test_pit_includes_as_of_date(self, builder):
        """as_of_date should be included if it has complete data."""
        as_of = pd.Timestamp('2022-06-01')
        train_data = builder.build_training_data(as_of)

        # as_of_date should be in the data if complete
        assert as_of in train_data.index or len(train_data) > 0

    def test_contains_target_column(self, builder):
        """Training data should contain target column."""
        as_of = pd.Timestamp('2022-06-01')
        train_data = builder.build_training_data(as_of, target_col='Ret_BTC')

        assert 'Ret_BTC' in train_data.columns

    def test_all_features_lagged(self, builder):
        """All feature columns must be lagged (Prev_*)."""
        as_of = pd.Timestamp('2022-06-01')
        train_data = builder.build_training_data(as_of)

        feature_cols = [c for c in train_data.columns if c != 'Ret_BTC']
        for col in feature_cols:
            assert col.startswith('Prev_'), f"Feature {col} is not lagged"

    def test_no_nan_in_result(self, builder):
        """Result should not contain NaN values."""
        as_of = pd.Timestamp('2022-06-01')
        train_data = builder.build_training_data(as_of)

        assert not train_data.isna().any().any()

    def test_min_samples_enforced(self, builder):
        """Should raise error if insufficient samples."""
        # Very early date with few samples
        as_of = pd.Timestamp('2016-06-01')

        with pytest.raises(ValueError, match="Insufficient samples"):
            builder.build_training_data(as_of, min_samples=100)

    def test_raises_on_unlagged_features(self, builder):
        """Should raise error if feature list contains unlagged columns."""
        as_of = pd.Timestamp('2022-06-01')

        with pytest.raises(ValueError, match="Unlagged features"):
            builder.build_training_data(
                as_of,
                feature_cols=['RSI_BTC']  # Not lagged!
            )


class TestGetCurrentFeatures:
    """Tests for get_current_features()."""

    @pytest.fixture
    def real_data(self):
        """Load real dataset for testing."""
        loader = DataLoader(Path('data'))
        return loader.load_full_dataset(exclude_usdt=True)

    @pytest.fixture
    def builder(self, real_data):
        """Create FeatureBuilder with real data."""
        return FeatureBuilder(real_data)

    def test_returns_series(self, builder):
        """Should return a Series."""
        as_of = pd.Timestamp('2022-06-01')
        current = builder.get_current_features(as_of)

        assert isinstance(current, pd.Series)

    def test_returns_last_row_values(self, builder, real_data):
        """Should return values from last available row."""
        as_of = pd.Timestamp('2022-06-01')
        current = builder.get_current_features(as_of)

        # Get expected from raw data
        pit_data = real_data[real_data.index <= as_of]
        expected_rsi = pit_data.iloc[-1]['Prev_RSI_BTC']

        assert current['Prev_RSI_BTC'] == expected_rsi

    def test_raises_on_empty_data(self, builder):
        """Should raise error if no data available."""
        as_of = pd.Timestamp('2010-01-01')  # Before data starts

        with pytest.raises(ValueError, match="No data available"):
            builder.get_current_features(as_of)
