"""
Unit tests for src/models/conditional.py

Tests the ConditionalMeanModel regime-based forecasting model.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.conditional import ConditionalMeanModel
from src.models.regimes import RegimeType
from src.models.baseline import Prediction


class TestConditionalMeanModel:
    """Tests for ConditionalMeanModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with returns and features."""
        dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        np.random.seed(42)

        # Create features with varying regimes
        rsi_btc = np.concatenate([
            np.random.uniform(70, 80, 12),  # First year: high RSI
            np.random.uniform(30, 40, 12),  # Second year: low RSI
            np.random.uniform(45, 55, 12),  # Third year: neutral
        ])

        # Returns that correlate with regime
        returns = np.concatenate([
            np.random.uniform(-5, 15, 12),   # High RSI: mixed
            np.random.uniform(-15, 5, 12),   # Low RSI: bearish
            np.random.uniform(-10, 10, 12),  # Neutral: random
        ])

        features = pd.DataFrame({
            'Ret_BTC': returns,
            'Prev_RSI_BTC': rsi_btc,
            'Prev_RSI_SPX': np.random.uniform(40, 60, 36),
            'Prev_DXY_Trend': np.random.choice([-1, 0, 1], 36),
        }, index=dates)

        returns_series = pd.Series(returns, index=dates, name='Ret_BTC')
        return features, returns_series

    def test_fit_identifies_regime(self, sample_data):
        """Fit should identify current market regime."""
        features, returns = sample_data
        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        assert model.current_regime is not None
        assert isinstance(model.current_regime, RegimeType)

    def test_fit_calculates_regime_stats(self, sample_data):
        """Fit should calculate regime-specific statistics."""
        features, returns = sample_data
        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        assert model.regime_mean is not None
        assert model.regime_std is not None
        assert model.n_samples > 0

    def test_fit_respects_as_of_date(self, sample_data):
        """Fit should only use data up to as_of_date."""
        features, returns = sample_data
        model = ConditionalMeanModel()

        as_of = pd.Timestamp('2020-06-01')
        model.fit(returns, as_of, features=features)

        # Should only have used 6 months of data max
        assert model.n_samples <= 6

    def test_fit_different_dates_different_results(self, sample_data):
        """Fit with different as_of dates should give different regimes."""
        features, returns = sample_data

        model1 = ConditionalMeanModel()
        model1.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        model2 = ConditionalMeanModel()
        model2.fit(returns, pd.Timestamp('2021-12-01'), features=features)

        # Regimes or sample counts should differ
        assert (model1.current_regime != model2.current_regime or
                model1.n_samples != model2.n_samples)

    def test_fit_without_features_uses_baseline(self, sample_data):
        """Fit without features should fall back to baseline."""
        _, returns = sample_data
        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=None)

        assert model.current_regime == RegimeType.BASELINE
        assert model.used_fallback is True


class TestConditionalPredict:
    """Tests for predict() method."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        returns = pd.Series([5, -3, 10, -8, 15, -2, 8, -5, 12, -7, 6, -4,
                            7, -3, 9, -6, 11, -1, 4, -8, 13, -5, 6, -2],
                           index=dates, name='Ret_BTC')
        features = pd.DataFrame({
            'Prev_RSI_BTC': [70] * 24,  # All high RSI
            'Prev_RSI_SPX': [50] * 24,
            'Prev_DXY_Trend': [0] * 24,
        }, index=dates)

        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)
        return model

    def test_returns_prediction_object(self, fitted_model):
        """Predict should return Prediction object."""
        pred = fitted_model.predict()
        assert isinstance(pred, Prediction)

    def test_prediction_has_percentiles(self, fitted_model):
        """Prediction should have all percentiles."""
        pred = fitted_model.predict()

        assert hasattr(pred, 'p10')
        assert hasattr(pred, 'p25')
        assert hasattr(pred, 'p50')
        assert hasattr(pred, 'p75')
        assert hasattr(pred, 'p90')

    def test_percentiles_ordered(self, fitted_model):
        """Percentiles should be in ascending order."""
        pred = fitted_model.predict()
        assert pred.p10 < pred.p25 < pred.p50 < pred.p75 < pred.p90

    def test_raises_if_not_fitted(self):
        """Predict should raise error if model not fitted."""
        model = ConditionalMeanModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict()


class TestFallbackBehavior:
    """Tests for fallback to baseline."""

    def test_fallback_when_insufficient_samples(self):
        """Should fall back to baseline with few regime samples."""
        dates = pd.date_range('2020-01-01', periods=10, freq='MS')
        returns = pd.Series([5, 10, -3, 8, -2, 6, -4, 9, -1, 7],
                           index=dates, name='Ret_BTC')

        # Create data where high RSI only appears once
        features = pd.DataFrame({
            'Prev_RSI_BTC': [70] + [50] * 9,  # Only first month is high RSI
            'Prev_RSI_SPX': [50] * 10,
            'Prev_DXY_Trend': [0] * 10,
        }, index=dates)

        model = ConditionalMeanModel(min_samples=5)
        # Fit as of first month (high RSI) - won't have enough samples
        model.fit(returns, pd.Timestamp('2020-01-01'), features=features)

        assert model.used_fallback is True
        assert model.current_regime == RegimeType.BASELINE

    def test_no_fallback_with_sufficient_samples(self):
        """Should not fall back when enough regime samples."""
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        returns = pd.Series([5] * 20, index=dates, name='Ret_BTC')

        # All high RSI
        features = pd.DataFrame({
            'Prev_RSI_BTC': [70] * 20,
            'Prev_RSI_SPX': [50] * 20,
            'Prev_DXY_Trend': [0] * 20,
        }, index=dates)

        model = ConditionalMeanModel(min_samples=5)
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        assert model.used_fallback is False
        assert model.current_regime == RegimeType.HIGH_RSI


class TestModelInfo:
    """Tests for get_model_info() method."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        returns = pd.Series([5, 10, -3, 8, -2, 6, -4, 9, -1, 7, 3, 5],
                           index=dates, name='Ret_BTC')
        features = pd.DataFrame({
            'Prev_RSI_BTC': [70] * 12,
            'Prev_RSI_SPX': [50] * 12,
            'Prev_DXY_Trend': [0] * 12,
        }, index=dates)

        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)
        return model

    def test_returns_dict(self, fitted_model):
        """get_model_info should return dictionary."""
        info = fitted_model.get_model_info()
        assert isinstance(info, dict)

    def test_contains_regime_info(self, fitted_model):
        """Info dict should have regime-specific keys."""
        info = fitted_model.get_model_info()

        assert 'model_type' in info
        assert 'current_regime' in info
        assert 'used_fallback' in info
        assert info['model_type'] == 'ConditionalMeanModel'

    def test_regime_value_is_string(self, fitted_model):
        """Regime should be serialized as string."""
        info = fitted_model.get_model_info()
        assert isinstance(info['current_regime'], str)


class TestPITCompliance:
    """P.I.T. compliance tests for ConditionalMeanModel."""

    def test_no_future_data_in_classification(self):
        """Classification must only use data <= as_of_date."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        returns = pd.Series([5] * 24, index=dates, name='Ret_BTC')

        # First 12 months: low RSI, Last 12 months: high RSI
        features = pd.DataFrame({
            'Prev_RSI_BTC': [40] * 12 + [70] * 12,
            'Prev_RSI_SPX': [50] * 24,
            'Prev_DXY_Trend': [0] * 24,
        }, index=dates)

        model = ConditionalMeanModel()
        # Fit as of month 12 - should see LOW RSI, not HIGH
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        assert model.current_regime == RegimeType.LOW_RSI

    def test_no_future_data_in_regime_filtering(self):
        """Regime filter must only use data <= as_of_date."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')

        # Returns differ by half
        returns = pd.Series([5] * 12 + [15] * 12, index=dates, name='Ret_BTC')

        # All same regime
        features = pd.DataFrame({
            'Prev_RSI_BTC': [70] * 24,
            'Prev_RSI_SPX': [50] * 24,
            'Prev_DXY_Trend': [0] * 24,
        }, index=dates)

        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        # Mean should be ~5 (first half only), not ~10 (all data)
        assert model.regime_mean < 10


class TestRepr:
    """Tests for __repr__ method."""

    def test_not_fitted_repr(self):
        """Unfitted model should have informative repr."""
        model = ConditionalMeanModel()
        assert "not fitted" in repr(model)

    def test_fitted_repr(self):
        """Fitted model should show regime and stats."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        returns = pd.Series([5] * 12, index=dates, name='Ret_BTC')
        features = pd.DataFrame({
            'Prev_RSI_BTC': [70] * 12,
            'Prev_RSI_SPX': [50] * 12,
            'Prev_DXY_Trend': [0] * 12,
        }, index=dates)

        model = ConditionalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'), features=features)

        repr_str = repr(model)
        assert "ConditionalMeanModel" in repr_str
        assert "regime=" in repr_str
        assert "mean=" in repr_str
