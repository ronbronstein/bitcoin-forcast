"""
Unit tests for src/models/regimes.py

Tests regime classification and filtering logic.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.regimes import (
    RegimeType,
    RegimeClassifier,
    REGIME_FILTERS,
    REGIME_PRIORITY,
)


class TestRegimeType:
    """Tests for RegimeType enum."""

    def test_all_regimes_defined(self):
        """All expected regimes should exist."""
        expected = [
            'BASELINE', 'HIGH_RSI', 'LOW_RSI', 'RISK_ON', 'RISK_OFF',
            'STRONG_DOLLAR', 'WEAK_DOLLAR', 'GOLDEN_POCKET'
        ]
        for name in expected:
            assert hasattr(RegimeType, name)

    def test_regime_values_are_strings(self):
        """Regime values should be lowercase strings."""
        for regime in RegimeType:
            assert isinstance(regime.value, str)
            assert regime.value == regime.value.lower()


class TestRegimeFilters:
    """Tests for regime filter functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with various conditions."""
        dates = pd.date_range('2020-01-01', periods=10, freq='MS')
        return pd.DataFrame({
            'Prev_RSI_BTC': [70, 40, 55, 75, 30, 60, 50, 80, 35, 45],
            'Prev_RSI_SPX': [65, 35, 50, 70, 30, 55, 45, 75, 25, 40],
            'Prev_DXY_Trend': [1, -1, 0, 1, -1, 0, -1, 1, -1, 0],
        }, index=dates)

    def test_all_regimes_have_filters(self):
        """Every RegimeType should have a filter function."""
        for regime in RegimeType:
            assert regime in REGIME_FILTERS

    def test_filter_returns_boolean_series(self, sample_data):
        """Filter functions should return boolean Series."""
        for regime, filter_func in REGIME_FILTERS.items():
            result = filter_func(sample_data)
            assert isinstance(result, pd.Series)
            assert result.dtype == bool

    def test_high_rsi_filter(self, sample_data):
        """HIGH_RSI filter: Prev_RSI_BTC > 65."""
        mask = REGIME_FILTERS[RegimeType.HIGH_RSI](sample_data)
        # Indices 0, 3, 7 have RSI > 65
        assert mask.iloc[0] == True
        assert mask.iloc[3] == True
        assert mask.iloc[7] == True
        assert mask.iloc[2] == False

    def test_low_rsi_filter(self, sample_data):
        """LOW_RSI filter: Prev_RSI_BTC < 45."""
        mask = REGIME_FILTERS[RegimeType.LOW_RSI](sample_data)
        # Indices 1, 4, 8 have RSI < 45
        assert mask.iloc[1] == True
        assert mask.iloc[4] == True
        assert mask.iloc[8] == True
        assert mask.iloc[0] == False

    def test_golden_pocket_filter(self, sample_data):
        """GOLDEN_POCKET: weak DXY AND RSI > 50."""
        mask = REGIME_FILTERS[RegimeType.GOLDEN_POCKET](sample_data)
        # Need DXY_Trend <= 0 AND RSI_BTC > 50
        # Idx 2: DXY=0, RSI=55 -> True
        # Idx 6: DXY=-1, RSI=50 -> False (not > 50)
        assert mask.iloc[2] == True
        assert mask.iloc[6] == False

    def test_baseline_matches_all(self, sample_data):
        """BASELINE filter should match all rows."""
        mask = REGIME_FILTERS[RegimeType.BASELINE](sample_data)
        assert mask.all()


class TestRegimeClassifier:
    """Tests for RegimeClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return RegimeClassifier(min_samples=5)

    def test_classify_high_rsi(self, classifier):
        """High RSI should be classified correctly."""
        features = pd.Series({
            'Prev_RSI_BTC': 70,
            'Prev_RSI_SPX': 50,
            'Prev_DXY_Trend': 0,
        })
        assert classifier.classify(features) == RegimeType.HIGH_RSI

    def test_classify_low_rsi(self, classifier):
        """Low RSI should be classified correctly."""
        features = pd.Series({
            'Prev_RSI_BTC': 40,
            'Prev_RSI_SPX': 50,
            'Prev_DXY_Trend': 0,
        })
        assert classifier.classify(features) == RegimeType.LOW_RSI

    def test_classify_risk_on(self, classifier):
        """Risk-on should be classified when SPX RSI high."""
        features = pd.Series({
            'Prev_RSI_BTC': 48,  # < 50 so not GOLDEN_POCKET
            'Prev_RSI_SPX': 65,  # > 60
            'Prev_DXY_Trend': 1,  # Strong dollar (not weak)
        })
        assert classifier.classify(features) == RegimeType.RISK_ON

    def test_classify_risk_off(self, classifier):
        """Risk-off should be classified when SPX RSI low."""
        features = pd.Series({
            'Prev_RSI_BTC': 48,  # < 50 so not GOLDEN_POCKET
            'Prev_RSI_SPX': 35,  # < 40
            'Prev_DXY_Trend': 1,  # Strong dollar (not weak)
        })
        assert classifier.classify(features) == RegimeType.RISK_OFF

    def test_classify_golden_pocket(self, classifier):
        """Golden pocket needs weak DXY + bull RSI."""
        features = pd.Series({
            'Prev_RSI_BTC': 55,  # > 50 but < 65
            'Prev_RSI_SPX': 50,  # Neutral
            'Prev_DXY_Trend': -1,  # Weak dollar
        })
        assert classifier.classify(features) == RegimeType.GOLDEN_POCKET

    def test_classify_fallback_to_baseline(self, classifier):
        """Should fall back to baseline when no regime matches."""
        features = pd.Series({
            'Prev_RSI_BTC': 50,  # Neutral
            'Prev_RSI_SPX': 50,  # Neutral
            'Prev_DXY_Trend': 1,  # Strong dollar (not golden pocket)
        })
        # No specific regime matches, should be STRONG_DOLLAR or lower priority
        result = classifier.classify(features)
        assert result in [RegimeType.STRONG_DOLLAR, RegimeType.BASELINE]

    def test_classify_missing_feature_fallback(self, classifier):
        """Missing features should fall back gracefully."""
        features = pd.Series({'Prev_RSI_BTC': 70})  # Missing SPX and DXY
        # Should still classify based on available feature
        result = classifier.classify(features)
        assert result == RegimeType.HIGH_RSI

    def test_priority_rsi_over_macro(self, classifier):
        """RSI regimes should have priority over macro regimes."""
        features = pd.Series({
            'Prev_RSI_BTC': 70,  # HIGH_RSI
            'Prev_RSI_SPX': 65,  # Also RISK_ON
            'Prev_DXY_Trend': 1,  # Also STRONG_DOLLAR
        })
        # HIGH_RSI should win due to priority
        assert classifier.classify(features) == RegimeType.HIGH_RSI


class TestFilterHistory:
    """Tests for filter_history method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        np.random.seed(42)
        return pd.DataFrame({
            'Prev_RSI_BTC': np.random.uniform(30, 80, 20),
            'Prev_RSI_SPX': np.random.uniform(30, 70, 20),
            'Prev_DXY_Trend': np.random.choice([-1, 0, 1], 20),
            'Ret_BTC': np.random.randn(20) * 10,
        }, index=dates)

    @pytest.fixture
    def classifier(self):
        return RegimeClassifier(min_samples=5)

    def test_filter_reduces_data(self, classifier, sample_data):
        """Filtering should reduce dataset size."""
        filtered = classifier.filter_history(sample_data, RegimeType.HIGH_RSI)
        assert len(filtered) <= len(sample_data)

    def test_filter_baseline_returns_all(self, classifier, sample_data):
        """Baseline filter should return all data."""
        filtered = classifier.filter_history(sample_data, RegimeType.BASELINE)
        assert len(filtered) == len(sample_data)

    def test_filtered_data_matches_condition(self, classifier, sample_data):
        """Filtered rows should match regime condition."""
        filtered = classifier.filter_history(sample_data, RegimeType.HIGH_RSI)
        assert (filtered['Prev_RSI_BTC'] > 65).all()

    def test_get_regime_samples_count(self, classifier, sample_data):
        """get_regime_samples should return correct count."""
        count = classifier.get_regime_samples(sample_data, RegimeType.HIGH_RSI)
        expected = (sample_data['Prev_RSI_BTC'] > 65).sum()
        assert count == expected


class TestRegimePriority:
    """Tests for regime priority ordering."""

    def test_priority_list_contains_all_regimes(self):
        """Priority list should contain all regime types."""
        for regime in RegimeType:
            assert regime in REGIME_PRIORITY

    def test_baseline_is_last(self):
        """Baseline should be last in priority."""
        assert REGIME_PRIORITY[-1] == RegimeType.BASELINE

    def test_rsi_regimes_first(self):
        """RSI regimes should be early in priority."""
        high_idx = REGIME_PRIORITY.index(RegimeType.HIGH_RSI)
        low_idx = REGIME_PRIORITY.index(RegimeType.LOW_RSI)
        baseline_idx = REGIME_PRIORITY.index(RegimeType.BASELINE)

        assert high_idx < baseline_idx
        assert low_idx < baseline_idx
