"""
Market regime classification for conditional forecasting.

Identifies market conditions from technical and macro indicators.
All classification uses lagged (Prev_*) features for P.I.T. compliance.
"""
from enum import Enum
from typing import Callable, Dict, Optional
import pandas as pd


class RegimeType(Enum):
    """Supported market regime categories."""
    BASELINE = "baseline"
    HIGH_RSI = "high_rsi"
    LOW_RSI = "low_rsi"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    STRONG_DOLLAR = "strong_dollar"
    WEAK_DOLLAR = "weak_dollar"
    GOLDEN_POCKET = "golden_pocket"


# Regime filter functions: each returns a boolean mask for DataFrame
REGIME_FILTERS: Dict[RegimeType, Callable[[pd.DataFrame], pd.Series]] = {
    RegimeType.BASELINE: lambda df: pd.Series(True, index=df.index),
    RegimeType.HIGH_RSI: lambda df: df['Prev_RSI_BTC'] > 65,
    RegimeType.LOW_RSI: lambda df: df['Prev_RSI_BTC'] < 45,
    RegimeType.RISK_ON: lambda df: df['Prev_RSI_SPX'] > 60,
    RegimeType.RISK_OFF: lambda df: df['Prev_RSI_SPX'] < 40,
    RegimeType.STRONG_DOLLAR: lambda df: df['Prev_DXY_Trend'] > 0,
    RegimeType.WEAK_DOLLAR: lambda df: df['Prev_DXY_Trend'] <= 0,
    RegimeType.GOLDEN_POCKET: lambda df: (
        (df['Prev_DXY_Trend'] <= 0) & (df['Prev_RSI_BTC'] > 50)
    ),
}

# Classification priority order (first match wins)
REGIME_PRIORITY = [
    RegimeType.HIGH_RSI,
    RegimeType.LOW_RSI,
    RegimeType.GOLDEN_POCKET,
    RegimeType.RISK_ON,
    RegimeType.RISK_OFF,
    RegimeType.STRONG_DOLLAR,
    RegimeType.WEAK_DOLLAR,
    RegimeType.BASELINE,
]


class RegimeClassifier:
    """
    Classifies market conditions into regimes.

    Uses lagged features (Prev_*) to identify current market regime
    for conditional forecasting.
    """

    def __init__(self, min_samples: int = 5):
        """
        Initialize classifier.

        Args:
            min_samples: Minimum samples for regime to be valid
        """
        self.min_samples = min_samples

    def classify(self, features: pd.Series) -> RegimeType:
        """
        Determine current regime from feature values.

        Args:
            features: Series with Prev_* column values

        Returns:
            RegimeType matching current conditions
        """
        # Check regimes in priority order
        for regime in REGIME_PRIORITY:
            if regime == RegimeType.BASELINE:
                return RegimeType.BASELINE

            try:
                if self._check_regime_condition(features, regime):
                    return regime
            except KeyError:
                continue

        return RegimeType.BASELINE

    def _check_regime_condition(
        self,
        features: pd.Series,
        regime: RegimeType
    ) -> bool:
        """Check if features match regime condition."""
        if regime == RegimeType.HIGH_RSI:
            return features['Prev_RSI_BTC'] > 65
        elif regime == RegimeType.LOW_RSI:
            return features['Prev_RSI_BTC'] < 45
        elif regime == RegimeType.RISK_ON:
            return features['Prev_RSI_SPX'] > 60
        elif regime == RegimeType.RISK_OFF:
            return features['Prev_RSI_SPX'] < 40
        elif regime == RegimeType.STRONG_DOLLAR:
            return features['Prev_DXY_Trend'] > 0
        elif regime == RegimeType.WEAK_DOLLAR:
            return features['Prev_DXY_Trend'] <= 0
        elif regime == RegimeType.GOLDEN_POCKET:
            return (
                features['Prev_DXY_Trend'] <= 0 and
                features['Prev_RSI_BTC'] > 50
            )
        return False

    def filter_history(
        self,
        data: pd.DataFrame,
        regime: RegimeType
    ) -> pd.DataFrame:
        """
        Filter historical data matching regime criteria.

        Args:
            data: DataFrame with Prev_* columns
            regime: Regime to filter for

        Returns:
            Filtered DataFrame
        """
        if regime not in REGIME_FILTERS:
            return data

        filter_func = REGIME_FILTERS[regime]

        try:
            mask = filter_func(data)
            return data[mask]
        except KeyError:
            return data

    def get_regime_samples(
        self,
        data: pd.DataFrame,
        regime: RegimeType
    ) -> int:
        """Get count of samples matching regime."""
        filtered = self.filter_history(data, regime)
        return len(filtered)
