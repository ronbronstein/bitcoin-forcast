"""
Unit tests for src/data/indicators.py

Tests technical indicator calculations, especially Wilder's RSI.
"""
import pytest
import pandas as pd
import numpy as np

from src.data.indicators import (
    calculate_wilder_rsi,
    calculate_returns,
    calculate_trend,
    calculate_zscore,
    lag_features,
)


class TestCalculateWilderRsi:
    """Tests for calculate_wilder_rsi() - CRITICAL function."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        # 20 months of prices with some ups and downs
        np.random.seed(42)
        base = 100
        changes = np.random.randn(20) * 5
        prices = base + np.cumsum(changes)
        return pd.Series(prices, index=pd.date_range('2020-01-01', periods=20, freq='MS'))

    def test_returns_three_series(self, sample_prices):
        """Should return RSI, AG, and AL series."""
        rsi, ag, al = calculate_wilder_rsi(sample_prices)

        assert isinstance(rsi, pd.Series)
        assert isinstance(ag, pd.Series)
        assert isinstance(al, pd.Series)

    def test_same_length_as_input(self, sample_prices):
        """Output series should have same length as input."""
        rsi, ag, al = calculate_wilder_rsi(sample_prices)

        assert len(rsi) == len(sample_prices)
        assert len(ag) == len(sample_prices)
        assert len(al) == len(sample_prices)

    def test_rsi_bounds(self, sample_prices):
        """RSI values should be between 0 and 100."""
        rsi, _, _ = calculate_wilder_rsi(sample_prices)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_first_values_are_nan(self, sample_prices):
        """First 14 values should be NaN (need 14 periods)."""
        rsi, ag, al = calculate_wilder_rsi(sample_prices, period=14)

        # First 14 should be NaN
        assert rsi.iloc[:14].isna().all()

    def test_wilder_smoothing_not_ewm(self, sample_prices):
        """Wilder's RSI should NOT equal pandas ewm RSI."""
        rsi_wilder, _, _ = calculate_wilder_rsi(sample_prices, period=14)

        # Calculate using ewm for comparison
        delta = sample_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        ewm_ag = gain.ewm(span=14, adjust=False).mean()
        ewm_al = loss.ewm(span=14, adjust=False).mean()
        rs_ewm = ewm_ag / ewm_al
        rsi_ewm = 100 - (100 / (1 + rs_ewm))

        # Get valid values for comparison
        valid_idx = rsi_wilder.dropna().index.intersection(rsi_ewm.dropna().index)

        if len(valid_idx) > 5:
            # Values should NOT be exactly equal
            assert not np.allclose(
                rsi_wilder[valid_idx].values,
                rsi_ewm[valid_idx].values,
                rtol=0.001
            )

    def test_ag_al_components_stored(self, sample_prices):
        """AG and AL components must be stored (for path-dependent simulation)."""
        rsi, ag, al = calculate_wilder_rsi(sample_prices)

        # Should have non-NaN values after period
        valid_ag = ag.dropna()
        valid_al = al.dropna()

        assert len(valid_ag) > 0
        assert len(valid_al) > 0

    def test_consistent_results(self, sample_prices):
        """Same input should give same output (deterministic)."""
        rsi1, ag1, al1 = calculate_wilder_rsi(sample_prices)
        rsi2, ag2, al2 = calculate_wilder_rsi(sample_prices)

        pd.testing.assert_series_equal(rsi1, rsi2)
        pd.testing.assert_series_equal(ag1, ag2)


class TestCalculateReturns:
    """Tests for calculate_returns()."""

    def test_percentage_returns(self):
        """Should calculate percentage returns correctly."""
        prices = pd.Series([100, 110, 99])
        returns = calculate_returns(prices, as_percentage=True)

        assert np.isnan(returns.iloc[0])  # First is NaN
        assert np.isclose(returns.iloc[1], 10.0)  # 10% gain
        assert np.isclose(returns.iloc[2], -10.0)  # 10% loss

    def test_decimal_returns(self):
        """Should calculate decimal returns when specified."""
        prices = pd.Series([100, 110, 99])
        returns = calculate_returns(prices, as_percentage=False)

        assert np.isclose(returns.iloc[1], 0.1)  # 0.1 = 10%


class TestCalculateTrend:
    """Tests for calculate_trend()."""

    def test_positive_trend(self):
        """Rising values should show positive trend."""
        series = pd.Series([100, 110, 120])
        trend = calculate_trend(series)

        assert np.isnan(trend.iloc[0])
        assert trend.iloc[1] > 0
        assert trend.iloc[2] > 0

    def test_negative_trend(self):
        """Falling values should show negative trend."""
        series = pd.Series([120, 110, 100])
        trend = calculate_trend(series)

        assert trend.iloc[1] < 0
        assert trend.iloc[2] < 0


class TestCalculateZscore:
    """Tests for calculate_zscore()."""

    def test_zscore_centered(self):
        """Z-scores should be roughly centered around 0."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(100) * 10 + 50)
        zscore = calculate_zscore(series, window=12)

        valid_z = zscore.dropna()
        assert abs(valid_z.mean()) < 0.5  # Roughly centered

    def test_handles_infinities(self):
        """Should handle division by zero (inf)."""
        series = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])  # Constant then change
        zscore = calculate_zscore(series, window=12)

        # Should not have infinities
        assert not np.isinf(zscore).any()


class TestLagFeatures:
    """Tests for lag_features()."""

    def test_creates_prev_columns(self):
        """Should create Prev_ prefixed columns."""
        df = pd.DataFrame({
            'RSI_BTC': [50, 60, 70],
            'RSI_SPX': [45, 55, 65],
        })

        result = lag_features(df, ['RSI_BTC', 'RSI_SPX'])

        assert 'Prev_RSI_BTC' in result.columns
        assert 'Prev_RSI_SPX' in result.columns

    def test_lagged_values_correct(self):
        """Lagged values should be shifted by 1 period."""
        df = pd.DataFrame({
            'RSI_BTC': [50, 60, 70],
        })

        result = lag_features(df, ['RSI_BTC'])

        assert pd.isna(result['Prev_RSI_BTC'].iloc[0])  # First is NaN
        assert result['Prev_RSI_BTC'].iloc[1] == 50  # Second = first original
        assert result['Prev_RSI_BTC'].iloc[2] == 60  # Third = second original

    def test_preserves_original_columns(self):
        """Original columns should be preserved."""
        df = pd.DataFrame({
            'RSI_BTC': [50, 60, 70],
        })

        result = lag_features(df, ['RSI_BTC'])

        assert 'RSI_BTC' in result.columns
