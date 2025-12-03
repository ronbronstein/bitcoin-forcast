"""
Unit tests for src/models/evaluator.py

Tests model evaluation metrics and reporting.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.evaluator import ModelEvaluator, PredictionResult


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            date=pd.Timestamp('2020-01-01'),
            predicted_return=10.0,
            actual_return=8.0,
            predicted_direction=1,
            actual_direction=1,
            p10=-5.0,
            p25=2.0,
            p50=10.0,
            p75=18.0,
            p90=25.0,
        )

    def test_direction_correct_true(self, sample_result):
        """Should return True when directions match."""
        assert sample_result.direction_correct() is True

    def test_direction_correct_false(self):
        """Should return False when directions don't match."""
        result = PredictionResult(
            date=pd.Timestamp('2020-01-01'),
            predicted_return=10.0,
            actual_return=-5.0,
            predicted_direction=1,
            actual_direction=-1,
            p10=-5.0, p25=2.0, p50=10.0, p75=18.0, p90=25.0,
        )
        assert result.direction_correct() is False

    def test_within_p10_p90_true(self, sample_result):
        """Should return True when actual in P10-P90 band."""
        assert sample_result.within_p10_p90() is True

    def test_within_p10_p90_false(self):
        """Should return False when actual outside P10-P90 band."""
        result = PredictionResult(
            date=pd.Timestamp('2020-01-01'),
            predicted_return=10.0,
            actual_return=50.0,  # Way outside band
            predicted_direction=1,
            actual_direction=1,
            p10=-5.0, p25=2.0, p50=10.0, p75=18.0, p90=25.0,
        )
        assert result.within_p10_p90() is False

    def test_absolute_error(self, sample_result):
        """Should calculate correct absolute error."""
        # predicted=10, actual=8, error=2
        assert sample_result.absolute_error() == 2.0


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def evaluator_with_results(self):
        """Create evaluator with sample results."""
        evaluator = ModelEvaluator()

        results = [
            # Correct direction, within band
            PredictionResult(
                pd.Timestamp('2020-01-01'), 10.0, 8.0, 1, 1,
                -5.0, 2.0, 10.0, 18.0, 25.0
            ),
            # Correct direction, within band
            PredictionResult(
                pd.Timestamp('2020-02-01'), -5.0, -3.0, -1, -1,
                -15.0, -10.0, -5.0, 0.0, 5.0
            ),
            # Wrong direction, outside band
            PredictionResult(
                pd.Timestamp('2020-03-01'), 5.0, -10.0, 1, -1,
                -5.0, 0.0, 5.0, 10.0, 15.0
            ),
            # Correct direction, within band
            PredictionResult(
                pd.Timestamp('2020-04-01'), 8.0, 12.0, 1, 1,
                -2.0, 3.0, 8.0, 13.0, 18.0
            ),
        ]

        for r in results:
            evaluator.add_result(r)

        return evaluator

    def test_n_predictions(self, evaluator_with_results):
        """Should count predictions correctly."""
        assert evaluator_with_results.n_predictions == 4

    def test_directional_accuracy(self, evaluator_with_results):
        """Should calculate directional accuracy correctly."""
        # 3 out of 4 correct
        assert evaluator_with_results.directional_accuracy() == 0.75

    def test_mape(self, evaluator_with_results):
        """Should calculate MAPE correctly."""
        # Errors: |10-8|=2, |-5-(-3)|=2, |5-(-10)|=15, |8-12|=4
        # Mean: (2+2+15+4)/4 = 5.75
        assert np.isclose(evaluator_with_results.mape(), 5.75)

    def test_band_capture_rate(self, evaluator_with_results):
        """Should calculate band capture rate correctly."""
        # 3 out of 4 within P10-P90
        assert evaluator_with_results.band_capture_rate('p10_p90') == 0.75

    def test_empty_evaluator(self):
        """Empty evaluator should return 0 for all metrics."""
        evaluator = ModelEvaluator()

        assert evaluator.n_predictions == 0
        assert evaluator.directional_accuracy() == 0.0
        assert evaluator.mape() == 0.0
        assert evaluator.band_capture_rate('p10_p90') == 0.0

    def test_clear(self, evaluator_with_results):
        """Clear should remove all results."""
        evaluator_with_results.clear()

        assert evaluator_with_results.n_predictions == 0


class TestGenerateReport:
    """Tests for generate_report() method."""

    @pytest.fixture
    def evaluator_with_results(self):
        """Create evaluator with sample results."""
        evaluator = ModelEvaluator()

        results = [
            PredictionResult(
                pd.Timestamp('2020-01-01'), 10.0, 8.0, 1, 1,
                -5.0, 2.0, 10.0, 18.0, 25.0
            ),
            PredictionResult(
                pd.Timestamp('2020-02-01'), -5.0, -3.0, -1, -1,
                -15.0, -10.0, -5.0, 0.0, 5.0
            ),
        ]

        for r in results:
            evaluator.add_result(r)

        return evaluator

    def test_returns_dict(self, evaluator_with_results):
        """Should return dictionary."""
        report = evaluator_with_results.generate_report()

        assert isinstance(report, dict)

    def test_contains_expected_keys(self, evaluator_with_results):
        """Report should have all expected metrics."""
        report = evaluator_with_results.generate_report()

        assert 'n_predictions' in report
        assert 'directional_accuracy' in report
        assert 'mape' in report
        assert 'rmse' in report
        assert 'p10_p90_capture' in report


class TestToDataFrame:
    """Tests for to_dataframe() method."""

    def test_returns_dataframe(self):
        """Should return DataFrame."""
        evaluator = ModelEvaluator()
        evaluator.add_result(PredictionResult(
            pd.Timestamp('2020-01-01'), 10.0, 8.0, 1, 1,
            -5.0, 2.0, 10.0, 18.0, 25.0
        ))

        df = evaluator.to_dataframe()

        assert isinstance(df, pd.DataFrame)

    def test_empty_returns_empty_df(self):
        """Empty evaluator should return empty DataFrame."""
        evaluator = ModelEvaluator()

        df = evaluator.to_dataframe()

        assert len(df) == 0

    def test_contains_computed_columns(self):
        """DataFrame should have computed columns."""
        evaluator = ModelEvaluator()
        evaluator.add_result(PredictionResult(
            pd.Timestamp('2020-01-01'), 10.0, 8.0, 1, 1,
            -5.0, 2.0, 10.0, 18.0, 25.0
        ))

        df = evaluator.to_dataframe()

        assert 'direction_correct' in df.columns
        assert 'within_p10_p90' in df.columns
        assert 'abs_error' in df.columns
