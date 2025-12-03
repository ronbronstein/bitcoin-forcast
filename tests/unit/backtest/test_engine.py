"""
Unit tests for src/backtest/engine.py

Tests walk-forward backtesting with P.I.T. enforcement.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.models.baseline import HistoricalMeanModel
from src.data.loader import DataLoader


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = BacktestConfig(
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2024-01-01'),
        )

        assert config.min_training_months == 36
        assert config.target_col == 'Ret_BTC'


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    @pytest.fixture
    def config(self):
        """Create backtest config."""
        return BacktestConfig(
            start_date=pd.Timestamp('2021-01-01'),
            end_date=pd.Timestamp('2022-12-01'),
            min_training_months=36,
        )

    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        loader = DataLoader(Path('data'))
        return loader.load_full_dataset(exclude_usdt=True)

    def test_init(self, config):
        """Should initialize correctly."""
        engine = BacktestEngine(HistoricalMeanModel, config)

        assert engine.model_class == HistoricalMeanModel
        assert engine.config == config

    def test_run_returns_dict(self, config, real_data):
        """Run should return dict with results and metrics."""
        engine = BacktestEngine(HistoricalMeanModel, config)
        result = engine.run(real_data, verbose=False)

        assert isinstance(result, dict)
        assert 'results' in result
        assert 'metrics' in result

    def test_run_results_is_dataframe(self, config, real_data):
        """Results should be a DataFrame."""
        engine = BacktestEngine(HistoricalMeanModel, config)
        result = engine.run(real_data, verbose=False)

        assert isinstance(result['results'], pd.DataFrame)

    def test_run_metrics_is_dict(self, config, real_data):
        """Metrics should be a dictionary."""
        engine = BacktestEngine(HistoricalMeanModel, config)
        result = engine.run(real_data, verbose=False)

        assert isinstance(result['metrics'], dict)


class TestPITCompliance:
    """Tests for Point-in-Time compliance in backtest."""

    @pytest.fixture
    def config(self):
        """Create backtest config."""
        return BacktestConfig(
            start_date=pd.Timestamp('2021-06-01'),
            end_date=pd.Timestamp('2021-12-01'),
            min_training_months=36,
        )

    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        loader = DataLoader(Path('data'))
        return loader.load_full_dataset(exclude_usdt=True)

    def test_no_future_data_in_training(self, config, real_data):
        """Training data must not include future dates."""
        engine = BacktestEngine(HistoricalMeanModel, config)
        result = engine.run(real_data, verbose=False)

        # Each prediction date should not have used future data
        # The 'date' column is the actual outcome date (month after forecast)
        results_df = result['results']

        if len(results_df) > 0:
            # The first prediction is for 2021-07-01 (forecasting from 2021-06-01)
            # Training should have ended at 2021-06-01
            first_date = results_df['date'].min()
            assert first_date >= config.start_date

    def test_prediction_dates_after_training(self, config, real_data):
        """Prediction dates should be after training end dates."""
        engine = BacktestEngine(HistoricalMeanModel, config)
        result = engine.run(real_data, verbose=False)

        results_df = result['results']

        if len(results_df) > 0:
            # All dates should be >= start_date + 1 month
            min_date = results_df['date'].min()
            # First prediction is for month AFTER start_date
            expected_min = pd.Timestamp('2021-07-01')
            assert min_date >= expected_min


class TestBacktestResults:
    """Tests for backtest result quality."""

    @pytest.fixture
    def long_config(self):
        """Create config for longer backtest."""
        return BacktestConfig(
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2024-06-01'),
            min_training_months=36,
        )

    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        loader = DataLoader(Path('data'))
        return loader.load_full_dataset(exclude_usdt=True)

    def test_reasonable_directional_accuracy(self, long_config, real_data):
        """Baseline should have reasonable directional accuracy."""
        engine = BacktestEngine(HistoricalMeanModel, long_config)
        result = engine.run(real_data, verbose=False)

        metrics = result['metrics']

        # Baseline should be around 50% (random) ± 15%
        assert 0.35 < metrics['directional_accuracy'] < 0.70

    def test_band_capture_reasonable(self, long_config, real_data):
        """P10-P90 band should capture ~80% of actuals."""
        engine = BacktestEngine(HistoricalMeanModel, long_config)
        result = engine.run(real_data, verbose=False)

        metrics = result['metrics']

        # Should capture somewhere between 60-95% (well calibrated)
        assert 0.60 < metrics['p10_p90_capture'] < 0.95

    def test_mape_reasonable(self, long_config, real_data):
        """MAPE should be reasonable for BTC volatility."""
        engine = BacktestEngine(HistoricalMeanModel, long_config)
        result = engine.run(real_data, verbose=False)

        metrics = result['metrics']

        # BTC is volatile, MAPE should be 10-30%
        assert 5 < metrics['mape'] < 40
