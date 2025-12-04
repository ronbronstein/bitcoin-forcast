"""
Unit tests for src/scenarios/matcher.py

Tests scenario matching and matrix generation.
"""
import pytest
import pandas as pd
import numpy as np

from src.scenarios.matcher import ScenarioMatcher, ScenarioStats
from src.scenarios.definitions import (
    ScenarioType, SCENARIOS, MONTH_NAMES, get_halving_phase
)


class TestHalvingPhase:
    """Tests for halving phase detection."""

    def test_before_first_halving(self):
        """Dates before first halving return None."""
        date = pd.Timestamp('2010-01-01')
        assert get_halving_phase(date) is None

    def test_after_first_halving(self):
        """Dates after first halving return positive months."""
        date = pd.Timestamp('2013-11-28')  # 1 year after first
        phase = get_halving_phase(date)
        assert phase is not None
        assert 11 < phase < 13  # ~12 months

    def test_after_latest_halving(self):
        """Recent dates show phase from 2024 halving."""
        date = pd.Timestamp('2024-10-01')
        phase = get_halving_phase(date)
        assert phase is not None
        assert 5 < phase < 7  # ~5-6 months


class TestScenarioDefinitions:
    """Tests for scenario definitions."""

    def test_all_scenarios_have_required_keys(self):
        """Each scenario must have name, desc, filter."""
        for scenario, config in SCENARIOS.items():
            assert 'name' in config
            assert 'desc' in config
            assert 'filter' in config
            assert callable(config['filter'])

    def test_scenario_count(self):
        """Should have 17 scenarios defined."""
        assert len(SCENARIOS) == 17

    def test_month_names(self):
        """Should have 12 month names."""
        assert len(MONTH_NAMES) == 12
        assert MONTH_NAMES[0] == 'Jan'
        assert MONTH_NAMES[11] == 'Dec'


class TestScenarioMatcher:
    """Tests for ScenarioMatcher class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data spanning multiple years and months."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        np.random.seed(42)

        return pd.DataFrame({
            'Ret_BTC': np.random.randn(48) * 10 + 5,
            'Prev_RSI_BTC': np.random.uniform(30, 80, 48),
            'Prev_RSI_SPX': np.random.uniform(35, 65, 48),
            'Prev_DXY_Trend': np.random.choice([-1, 0, 1], 48),
            'Prev_Rate_Trend': np.random.choice([-1, 0, 1], 48),
            'Prev_Active_Addresses_Z': np.random.randn(48),
        }, index=dates)

    @pytest.fixture
    def matcher(self, sample_data):
        """Create matcher instance."""
        return ScenarioMatcher(sample_data)

    def test_run_matrix_analysis_returns_dataframe(self, matcher):
        """Matrix analysis should return DataFrame."""
        result = matcher.run_matrix_analysis()
        assert isinstance(result, pd.DataFrame)

    def test_matrix_has_expected_columns(self, matcher):
        """Matrix should have all expected columns."""
        result = matcher.run_matrix_analysis()
        expected_cols = [
            'Month', 'Scenario', 'Description', 'Count',
            'Win_Rate', 'Avg_Return', 'Median_Return',
            'CI_Lower_90', 'CI_Upper_90', 'Best', 'Worst',
            'Matching_Years', 'Quality'
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_matrix_size(self, matcher):
        """Matrix should have 12 months × 17 scenarios rows."""
        result = matcher.run_matrix_analysis()
        assert len(result) == 12 * 17

    def test_subset_scenarios(self, matcher):
        """Can analyze subset of scenarios."""
        scenarios = [ScenarioType.BASELINE, ScenarioType.HIGH_RSI]
        result = matcher.run_matrix_analysis(scenarios=scenarios)
        assert len(result) == 12 * 2

    def test_pit_filtering(self, sample_data):
        """P.I.T. date should limit data used."""
        matcher = ScenarioMatcher(sample_data)

        # Full data
        result1 = matcher.run_matrix_analysis()

        # Limited to first year
        result2 = matcher.run_matrix_analysis(
            as_of_date=pd.Timestamp('2020-12-01')
        )

        # First year should have fewer samples
        baseline1 = result1[result1['Scenario'] == 'Baseline (All History)']
        baseline2 = result2[result2['Scenario'] == 'Baseline (All History)']

        assert baseline2['Count'].sum() < baseline1['Count'].sum()


class TestScenarioStats:
    """Tests for ScenarioStats dataclass."""

    def test_to_dict(self):
        """to_dict should return all fields."""
        stats = ScenarioStats(
            month='Jan',
            scenario=ScenarioType.BASELINE,
            scenario_name='Baseline',
            description='Test',
            count=10,
            win_rate=60.0,
            avg_return=5.0,
            median_return=4.0,
            ci_lower=2.0,
            ci_upper=8.0,
            best=15.0,
            worst=-10.0,
            matching_years=[2020, 2021],
            quality='Sufficient'
        )

        d = stats.to_dict()
        assert d['Month'] == 'Jan'
        assert d['Count'] == 10
        assert d['Win_Rate'] == 60.0


class TestGetCurrentScenarios:
    """Tests for identifying current scenarios."""

    @pytest.fixture
    def matcher_with_data(self):
        """Create matcher with controlled data."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'Ret_BTC': [5] * 12,
            'Prev_RSI_BTC': [70] * 12,  # High RSI
            'Prev_RSI_SPX': [50] * 12,
            'Prev_DXY_Trend': [-1] * 12,  # Weak dollar
            'Prev_Rate_Trend': [0] * 12,
        }, index=dates)
        return ScenarioMatcher(data)

    def test_identifies_matching_scenarios(self, matcher_with_data):
        """Should identify HIGH_RSI and WEAK_DOLLAR."""
        matches = matcher_with_data.get_current_scenarios(
            pd.Timestamp('2020-12-01')
        )

        assert ScenarioType.HIGH_RSI in matches
        assert ScenarioType.WEAK_DOLLAR in matches

    def test_empty_data_returns_baseline(self):
        """Empty data should return baseline."""
        dates = pd.date_range('2020-01-01', periods=1, freq='MS')
        data = pd.DataFrame({'Ret_BTC': [5]}, index=dates)
        matcher = ScenarioMatcher(data)

        matches = matcher.get_current_scenarios(pd.Timestamp('2019-01-01'))
        assert matches == [ScenarioType.BASELINE]


class TestQualitySummary:
    """Tests for quality summary."""

    def test_quality_summary_structure(self):
        """Quality summary should have expected keys."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        data = pd.DataFrame({
            'Ret_BTC': np.random.randn(48),
            'Prev_RSI_BTC': [50] * 48,
            'Prev_RSI_SPX': [50] * 48,
            'Prev_DXY_Trend': [0] * 48,
            'Prev_Rate_Trend': [0] * 48,
        }, index=dates)

        matcher = ScenarioMatcher(data)
        matrix = matcher.run_matrix_analysis(
            scenarios=[ScenarioType.BASELINE]
        )
        summary = matcher.get_quality_summary(matrix)

        assert 'total_cells' in summary
        assert 'sufficient' in summary
        assert 'small_sample' in summary
        assert 'unreliable' in summary
        assert 'pct_reliable' in summary


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_with_sufficient_samples(self):
        """Should compute CI with enough samples."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        # All January data
        data = pd.DataFrame({
            'Ret_BTC': [5] * 24,
            'Prev_RSI_BTC': [50] * 24,
            'Prev_RSI_SPX': [50] * 24,
            'Prev_DXY_Trend': [0] * 24,
            'Prev_Rate_Trend': [0] * 24,
        }, index=dates)

        matcher = ScenarioMatcher(data, bootstrap_samples=100)
        matrix = matcher.run_matrix_analysis(
            scenarios=[ScenarioType.BASELINE]
        )

        jan_row = matrix[matrix['Month'] == 'Jan'].iloc[0]
        # Should have valid CI for months with data
        if jan_row['Count'] >= 3:
            assert not np.isnan(jan_row['CI_Lower_90'])
            assert not np.isnan(jan_row['CI_Upper_90'])
            assert jan_row['CI_Lower_90'] <= jan_row['CI_Upper_90']
