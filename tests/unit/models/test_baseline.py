"""
Unit tests for src/models/baseline.py

Tests the HistoricalMeanModel baseline forecasting model.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.baseline import HistoricalMeanModel, Prediction


class TestHistoricalMeanModel:
    """Tests for HistoricalMeanModel class."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        # Mix of positive and negative returns
        returns = [5, -3, 10, -8, 15, -2, 8, -5, 12, -7, 6, -4,
                   7, -3, 9, -6, 11, -1, 4, -8, 13, -5, 6, -2]
        return pd.Series(returns, index=dates)

    def test_fit_calculates_mean(self, sample_returns):
        """Fit should calculate mean return."""
        model = HistoricalMeanModel()
        model.fit(sample_returns, pd.Timestamp('2020-12-01'))

        assert model.mean_return is not None
        assert isinstance(model.mean_return, float)

    def test_fit_calculates_std(self, sample_returns):
        """Fit should calculate standard deviation."""
        model = HistoricalMeanModel()
        model.fit(sample_returns, pd.Timestamp('2020-12-01'))

        assert model.std_return is not None
        assert model.std_return > 0

    def test_fit_counts_samples(self, sample_returns):
        """Fit should count training samples."""
        model = HistoricalMeanModel()
        as_of = pd.Timestamp('2020-06-01')
        model.fit(sample_returns, as_of)

        # Should have 6 samples (Jan-Jun 2020)
        assert model.n_samples == 6

    def test_fit_respects_as_of_date(self, sample_returns):
        """Fit should only use data up to as_of_date."""
        model = HistoricalMeanModel()
        as_of = pd.Timestamp('2020-06-01')
        model.fit(sample_returns, as_of)

        # Calculate expected mean manually
        pit_returns = sample_returns[sample_returns.index <= as_of]
        expected_mean = pit_returns.mean()

        assert np.isclose(model.mean_return, expected_mean)

    def test_fit_pit_excludes_future(self, sample_returns):
        """Fit with different as_of dates should give different results."""
        model1 = HistoricalMeanModel()
        model1.fit(sample_returns, pd.Timestamp('2020-06-01'))

        model2 = HistoricalMeanModel()
        model2.fit(sample_returns, pd.Timestamp('2020-12-01'))

        assert model1.n_samples != model2.n_samples


class TestPredict:
    """Tests for predict() method."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        returns = pd.Series([5, -3, 10, -8, 15, -2, 8, -5, 12, -7, 6, -4,
                            7, -3, 9, -6, 11, -1, 4, -8, 13, -5, 6, -2],
                           index=dates)
        model = HistoricalMeanModel()
        model.fit(returns, pd.Timestamp('2020-12-01'))
        return model

    def test_returns_prediction_object(self, fitted_model):
        """Predict should return Prediction object."""
        pred = fitted_model.predict()

        assert isinstance(pred, Prediction)

    def test_prediction_has_percentiles(self, fitted_model):
        """Prediction should have p10, p50, p90."""
        pred = fitted_model.predict()

        assert hasattr(pred, 'p10')
        assert hasattr(pred, 'p50')
        assert hasattr(pred, 'p90')

    def test_percentiles_ordered(self, fitted_model):
        """Percentiles should be in ascending order."""
        pred = fitted_model.predict()

        assert pred.p10 < pred.p25 < pred.p50 < pred.p75 < pred.p90

    def test_p50_equals_mean(self, fitted_model):
        """P50 should equal mean for normal distribution."""
        pred = fitted_model.predict()

        assert np.isclose(pred.p50, pred.mean)

    def test_raises_if_not_fitted(self):
        """Predict should raise error if model not fitted."""
        model = HistoricalMeanModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.predict()


class TestPredictDirection:
    """Tests for predict_direction() method."""

    def test_positive_mean_predicts_up(self):
        """Positive mean return should predict up (1)."""
        dates = pd.date_range('2020-01-01', periods=5, freq='MS')
        returns = pd.Series([10, 15, 20, 8, 12], index=dates)  # All positive
        model = HistoricalMeanModel()
        model.fit(returns, pd.Timestamp('2020-05-01'))

        assert model.predict().direction == 1

    def test_negative_mean_predicts_down(self):
        """Negative mean return should predict down (-1)."""
        dates = pd.date_range('2020-01-01', periods=5, freq='MS')
        returns = pd.Series([-10, -15, -20, -8, -12], index=dates)  # All negative
        model = HistoricalMeanModel()
        model.fit(returns, pd.Timestamp('2020-05-01'))

        assert model.predict().direction == -1

    def test_neutral_near_zero(self):
        """Near-zero mean should predict neutral (0)."""
        dates = pd.date_range('2020-01-01', periods=5, freq='MS')
        returns = pd.Series([0.1, -0.1, 0.2, -0.2, 0.0], index=dates)  # Near zero
        model = HistoricalMeanModel(neutral_threshold=0.5)
        model.fit(returns, pd.Timestamp('2020-05-01'))

        assert model.predict().direction == 0


class TestModelInfo:
    """Tests for get_model_info() method."""

    def test_returns_dict(self):
        """get_model_info should return dictionary."""
        returns = pd.Series([5, 10, 15], index=pd.date_range('2020-01-01', periods=3, freq='MS'))
        model = HistoricalMeanModel()
        model.fit(returns, pd.Timestamp('2020-03-01'))

        info = model.get_model_info()

        assert isinstance(info, dict)

    def test_contains_expected_keys(self):
        """Info dict should have expected keys."""
        returns = pd.Series([5, 10, 15], index=pd.date_range('2020-01-01', periods=3, freq='MS'))
        model = HistoricalMeanModel()
        model.fit(returns, pd.Timestamp('2020-03-01'))

        info = model.get_model_info()

        assert 'model_type' in info
        assert 'n_samples' in info
        assert 'mean_return' in info
        assert 'std_return' in info
