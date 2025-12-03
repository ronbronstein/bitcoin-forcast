"""
Unit tests for src/backtest/reporter.py

Tests backtest reporting and visualization.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.backtest.reporter import BacktestReporter


class TestBacktestReporter:
    """Tests for BacktestReporter class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results DataFrame."""
        return pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=12, freq='MS'),
            'predicted_return': [5, -3, 10, 8, -2, 6, 12, -5, 3, 7, -4, 9],
            'actual_return': [8, -5, 12, 3, -8, 10, 8, -2, 6, 4, -6, 11],
            'predicted_direction': [1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1],
            'actual_direction': [1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1],
            'direction_correct': [True, True, True, True, True, True, True, True, True, True, True, True],
            'within_p10_p90': [True, True, True, False, True, True, False, True, True, True, True, True],
            'abs_error': [3, 2, 2, 5, 6, 4, 4, 3, 3, 3, 2, 2],
        })

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics dict."""
        return {
            'n_predictions': 12,
            'directional_accuracy': 1.0,
            'mape': 3.25,
            'rmse': 3.5,
            'p10_p90_capture': 0.833,
            'p25_p75_capture': 0.667,
        }

    @pytest.fixture
    def reporter(self, sample_results, sample_metrics):
        """Create reporter with sample data."""
        return BacktestReporter(sample_results, sample_metrics)

    def test_init(self, sample_results, sample_metrics):
        """Should initialize correctly."""
        reporter = BacktestReporter(sample_results, sample_metrics)

        assert reporter.results is not None
        assert reporter.metrics is not None

    def test_print_summary_no_error(self, reporter, capsys):
        """print_summary should run without error."""
        reporter.print_summary()

        captured = capsys.readouterr()
        assert 'BACKTEST RESULTS' in captured.out
        assert 'Directional Accuracy' in captured.out

    def test_to_csv(self, reporter, sample_results):
        """Should save results to CSV."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)

        reporter.to_csv(path)

        # Verify file was created and has content
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == len(sample_results)

        # Cleanup
        path.unlink()


class TestGetAnnualBreakdown:
    """Tests for get_annual_breakdown() method."""

    @pytest.fixture
    def multi_year_results(self):
        """Create results spanning multiple years."""
        dates = pd.date_range('2020-01-01', periods=36, freq='MS')  # 3 years
        return pd.DataFrame({
            'date': dates,
            'predicted_return': np.random.randn(36) * 10,
            'actual_return': np.random.randn(36) * 10,
            'direction_correct': np.random.choice([True, False], 36),
            'within_p10_p90': np.random.choice([True, False], 36),
            'abs_error': np.abs(np.random.randn(36) * 5),
        })

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics."""
        return {'n_predictions': 36, 'directional_accuracy': 0.5}

    def test_returns_dataframe(self, multi_year_results, sample_metrics):
        """Should return DataFrame."""
        reporter = BacktestReporter(multi_year_results, sample_metrics)
        annual = reporter.get_annual_breakdown()

        assert isinstance(annual, pd.DataFrame)

    def test_has_year_index(self, multi_year_results, sample_metrics):
        """Should have year as index."""
        reporter = BacktestReporter(multi_year_results, sample_metrics)
        annual = reporter.get_annual_breakdown()

        assert annual.index.name == 'year'
        assert 2020 in annual.index
        assert 2021 in annual.index
        assert 2022 in annual.index

    def test_empty_results(self):
        """Empty results should return empty DataFrame."""
        reporter = BacktestReporter(pd.DataFrame(), {})
        annual = reporter.get_annual_breakdown()

        assert len(annual) == 0


class TestGetStreakAnalysis:
    """Tests for get_streak_analysis() method."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics."""
        return {'n_predictions': 10}

    def test_returns_dict(self, sample_metrics):
        """Should return dictionary."""
        results = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='MS'),
            'direction_correct': [True, True, True, False, True, True, False, False, True, True],
        })
        reporter = BacktestReporter(results, sample_metrics)
        streaks = reporter.get_streak_analysis()

        assert isinstance(streaks, dict)

    def test_finds_longest_winning(self, sample_metrics):
        """Should find longest winning streak."""
        results = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='MS'),
            'direction_correct': [True, True, True, False, True, True, False, False, True, True],
        })
        reporter = BacktestReporter(results, sample_metrics)
        streaks = reporter.get_streak_analysis()

        # Longest winning: first 3
        assert streaks['longest_winning_streak'] == 3

    def test_finds_longest_losing(self, sample_metrics):
        """Should find longest losing streak."""
        results = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='MS'),
            'direction_correct': [True, True, True, False, True, True, False, False, True, True],
        })
        reporter = BacktestReporter(results, sample_metrics)
        streaks = reporter.get_streak_analysis()

        # Longest losing: 2 consecutive False
        assert streaks['longest_losing_streak'] == 2

    def test_empty_results(self):
        """Empty results should return empty dict."""
        reporter = BacktestReporter(pd.DataFrame(), {})
        streaks = reporter.get_streak_analysis()

        assert streaks == {}
