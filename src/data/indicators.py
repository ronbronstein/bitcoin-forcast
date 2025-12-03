"""
Technical indicator calculations.

Includes Wilder's RSI with TRUE recursive smoothing (not ewm).
"""
import pandas as pd
import numpy as np
from typing import Tuple


def calculate_wilder_rsi(
    prices: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate RSI using TRUE Wilder's smoothing.

    CRITICAL: Uses recursive formula, NOT pandas ewm().
    Formula: avg_new = (avg_prev * (period-1) + current) / period

    Args:
        prices: Price series (e.g., BTC closing prices)
        period: RSI period (default 14)

    Returns:
        Tuple of (RSI series, Average Gain series, Average Loss series)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = pd.Series(index=prices.index, dtype=float)
    avg_loss = pd.Series(index=prices.index, dtype=float)
    rsi = pd.Series(index=prices.index, dtype=float)

    if len(prices) <= period:
        return rsi, avg_gain, avg_loss

    # First value: Simple Moving Average of first 'period' changes
    first_avg_gain = gain.iloc[1:period + 1].mean()
    first_avg_loss = loss.iloc[1:period + 1].mean()

    avg_gain.iloc[period] = first_avg_gain
    avg_loss.iloc[period] = first_avg_loss

    # Calculate first RSI
    if first_avg_loss == 0:
        rsi.iloc[period] = 100.0
    else:
        rs = first_avg_gain / first_avg_loss
        rsi.iloc[period] = 100.0 - (100.0 / (1.0 + rs))

    # Subsequent values: Wilder's recursive smoothing
    for i in range(period + 1, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

        if avg_loss.iloc[i] == 0:
            rsi.iloc[i] = 100.0
        else:
            rs = avg_gain.iloc[i] / avg_loss.iloc[i]
            rsi.iloc[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi, avg_gain, avg_loss


def calculate_returns(prices: pd.Series, as_percentage: bool = True) -> pd.Series:
    """
    Calculate period-over-period returns.

    Args:
        prices: Price series
        as_percentage: If True, multiply by 100

    Returns:
        Returns series
    """
    returns = prices.pct_change()
    if as_percentage:
        returns = returns * 100
    return returns


def calculate_trend(series: pd.Series) -> pd.Series:
    """
    Calculate first difference (trend direction).

    Positive = rising, Negative = falling.

    Args:
        series: Any numeric series

    Returns:
        First difference series
    """
    return series.diff()


def calculate_zscore(
    series: pd.Series,
    window: int = 12,
    min_periods: int = 6
) -> pd.Series:
    """
    Calculate rolling z-score for relative comparisons.

    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum observations required

    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    zscore = (series - rolling_mean) / rolling_std

    # Handle infinities and NaN
    zscore = zscore.replace([np.inf, -np.inf], 0.0)
    zscore = zscore.fillna(0.0)

    return zscore


def lag_features(df: pd.DataFrame, columns: list, periods: int = 1) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to lag
        periods: Number of periods to lag (default 1)

    Returns:
        DataFrame with lagged columns (Prev_* naming)
    """
    result = df.copy()

    for col in columns:
        if col in df.columns:
            new_col = f'Prev_{col}' if not col.startswith('Prev_') else col
            result[new_col] = df[col].shift(periods)

    return result
