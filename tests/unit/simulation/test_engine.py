"""
Unit tests for src/simulation/engine.py

Tests Monte Carlo simulation engine.
"""
import pytest
import pandas as pd
import numpy as np

from src.simulation.engine import MonteCarloEngine
from src.simulation.config import SimulationConfig, SimulationResult


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        cfg = SimulationConfig()
        assert cfg.n_simulations == 2000
        assert cfg.n_months == 12
        assert cfg.rsi_period == 14
        assert cfg.half_life_years == 4.0

    def test_custom_values(self):
        """Config should accept custom values."""
        cfg = SimulationConfig(n_simulations=100, n_months=6)
        assert cfg.n_simulations == 100
        assert cfg.n_months == 6


class TestMonteCarloEngine:
    """Tests for MonteCarloEngine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with AG/AL components."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        np.random.seed(42)

        return pd.DataFrame({
            'Ret_BTC': np.random.randn(48) * 10 + 5,
            'RSI_BTC': np.random.uniform(30, 70, 48),
            'Prev_RSI_BTC': np.random.uniform(30, 70, 48),
            'BTC_AG': np.random.uniform(1, 10, 48),
            'BTC_AL': np.random.uniform(1, 10, 48),
            'Prev_DXY_Trend': np.random.choice([-1, 0, 1], 48),
        }, index=dates)

    @pytest.fixture
    def engine(self, sample_data):
        """Create engine instance."""
        cfg = SimulationConfig(n_simulations=100, n_months=3, random_seed=42)
        return MonteCarloEngine(sample_data, cfg)

    def test_run_returns_result(self, engine):
        """Run should return SimulationResult."""
        result = engine.run(
            current_price=50000,
            as_of_date=pd.Timestamp('2023-12-01')
        )
        assert isinstance(result, SimulationResult)

    def test_result_has_correct_shape(self, engine):
        """Result should have correct dimensions."""
        result = engine.run(
            current_price=50000,
            as_of_date=pd.Timestamp('2023-12-01')
        )
        assert result.n_simulations == 100
        assert result.n_months == 3
        assert result.price_paths.shape == (100, 4)  # n_months + 1

    def test_initial_price_preserved(self, engine):
        """First column should be initial price."""
        result = engine.run(
            current_price=50000,
            as_of_date=pd.Timestamp('2023-12-01')
        )
        assert np.all(result.price_paths[:, 0] == 50000)

    def test_prices_positive(self, engine):
        """All prices should remain positive."""
        result = engine.run(
            current_price=50000,
            as_of_date=pd.Timestamp('2023-12-01')
        )
        assert np.all(result.price_paths > 0)

    def test_dates_correct_length(self, engine):
        """Dates list should match months + 1."""
        result = engine.run(
            current_price=50000,
            as_of_date=pd.Timestamp('2023-12-01')
        )
        assert len(result.dates) == 4  # n_months + 1

    def test_reproducibility_with_seed(self, sample_data):
        """Same seed should give same results."""
        cfg = SimulationConfig(n_simulations=50, n_months=2, random_seed=123)

        engine1 = MonteCarloEngine(sample_data, cfg)
        result1 = engine1.run(50000, pd.Timestamp('2023-12-01'))

        engine2 = MonteCarloEngine(sample_data, cfg)
        result2 = engine2.run(50000, pd.Timestamp('2023-12-01'))

        np.testing.assert_array_almost_equal(
            result1.price_paths, result2.price_paths
        )


class TestSimulationResult:
    """Tests for SimulationResult."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result."""
        dates = [pd.Timestamp('2024-01-01') + pd.DateOffset(months=i) for i in range(4)]
        np.random.seed(42)
        price_paths = np.random.uniform(40000, 60000, (100, 4))
        price_paths[:, 0] = 50000  # Fix initial price

        return SimulationResult(
            dates=dates,
            price_paths=price_paths,
            initial_price=50000,
            initial_rsi=55.0,
            match_logic="Test",
            config=SimulationConfig(n_simulations=100, n_months=3),
        )

    def test_get_percentiles(self, sample_result):
        """Should return percentile dict."""
        pcts = sample_result.get_percentiles(1)
        assert 'p10' in pcts
        assert 'p50' in pcts
        assert 'p90' in pcts
        assert 'mean' in pcts

    def test_percentiles_ordered(self, sample_result):
        """Percentiles should be in order."""
        pcts = sample_result.get_percentiles(3)
        assert pcts['p10'] <= pcts['p25'] <= pcts['p50']
        assert pcts['p50'] <= pcts['p75'] <= pcts['p90']

    def test_get_return_percentiles(self, sample_result):
        """Should return return percentiles."""
        ret_pcts = sample_result.get_return_percentiles(3)
        assert 'p10' in ret_pcts
        assert 'p50' in ret_pcts
        assert 'mean' in ret_pcts

    def test_to_dataframe(self, sample_result):
        """Should convert to DataFrame."""
        df = sample_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # n_months + 1
        assert len(df.columns) == 100  # n_simulations

    def test_get_summary(self, sample_result):
        """Should return summary DataFrame."""
        summary = sample_result.get_summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'Price_P50' in summary.columns
        assert 'Return_P50' in summary.columns


class TestRSIEvolution:
    """Tests for RSI evolution during simulation."""

    @pytest.fixture
    def engine_for_rsi(self):
        """Create engine with controlled RSI data."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        np.random.seed(42)

        # High AG, low AL = high RSI
        data = pd.DataFrame({
            'Ret_BTC': np.random.randn(48) * 10,
            'RSI_BTC': [70] * 48,
            'Prev_RSI_BTC': [70] * 48,
            'BTC_AG': [8.0] * 48,  # High average gain
            'BTC_AL': [2.0] * 48,  # Low average loss
            'Prev_DXY_Trend': [0] * 48,
        }, index=dates)

        cfg = SimulationConfig(n_simulations=10, n_months=3, random_seed=42)
        return MonteCarloEngine(data, cfg)

    def test_rsi_calculation(self, engine_for_rsi):
        """RSI should be calculated from AG/AL."""
        ag = np.array([8.0, 8.0, 2.0])
        al = np.array([2.0, 8.0, 8.0])

        rsi = engine_for_rsi._calculate_rsi(ag, al)

        # RSI = 100 - 100/(1 + RS), where RS = AG/AL
        # For AG=8, AL=2: RS=4, RSI = 100 - 100/5 = 80
        assert np.isclose(rsi[0], 80.0)
        # For AG=8, AL=8: RS=1, RSI = 100 - 100/2 = 50
        assert np.isclose(rsi[1], 50.0)
        # For AG=2, AL=8: RS=0.25, RSI = 100 - 100/1.25 = 20
        assert np.isclose(rsi[2], 20.0)


class TestTimeWeighting:
    """Tests for time-weighted sampling."""

    def test_recent_data_weighted_higher(self):
        """Recent data should have higher weights."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        data = pd.DataFrame({
            'Ret_BTC': [5] * 48,
            'RSI_BTC': [50] * 48,
            'Prev_RSI_BTC': [50] * 48,
            'BTC_AG': [5] * 48,
            'BTC_AL': [5] * 48,
            'Prev_DXY_Trend': [0] * 48,
        }, index=dates)

        cfg = SimulationConfig(half_life_years=4.0)
        engine = MonteCarloEngine(data, cfg)

        weights = engine._calculate_time_weights(
            data, pd.Timestamp('2023-12-01')
        )

        # More recent dates should have higher weights
        assert weights.iloc[-1] > weights.iloc[0]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_missing_ag_al_raises(self):
        """Should raise if AG/AL missing."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'Ret_BTC': [5] * 12,
            'Prev_RSI_BTC': [50] * 12,
            'Prev_DXY_Trend': [0] * 12,
            # Missing BTC_AG and BTC_AL
        }, index=dates)

        engine = MonteCarloEngine(data)

        with pytest.raises(ValueError, match="Missing AG/AL"):
            engine.run(50000, pd.Timestamp('2020-12-01'))

    def test_empty_data_raises(self):
        """Should raise if no data available."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'Ret_BTC': [5] * 12,
            'BTC_AG': [5] * 12,
            'BTC_AL': [5] * 12,
            'Prev_RSI_BTC': [50] * 12,
            'Prev_DXY_Trend': [0] * 12,
        }, index=dates)

        engine = MonteCarloEngine(data)

        with pytest.raises(ValueError, match="No data available"):
            engine.run(50000, pd.Timestamp('2019-01-01'))
