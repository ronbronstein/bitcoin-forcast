"""
Data cleaning and validation utilities.

Handles missing values, validates data bounds, and detects gaps.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'interpolate',
    max_gap: int = 2
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values with explicit tracking.

    Args:
        df: DataFrame to clean
        method: 'interpolate', 'drop', or 'ffill'
        max_gap: Maximum consecutive NaNs to fill (for interpolate/ffill)

    Returns:
        Tuple of (cleaned DataFrame, report dict)
    """
    report = {
        'original_nulls': df.isna().sum().to_dict(),
        'filled': {},
        'method': method
    }
    result = df.copy()

    if method == 'interpolate':
        for col in result.columns:
            if result[col].dtype in ['float64', 'int64']:
                before = result[col].isna().sum()
                result[col] = result[col].interpolate(method='linear', limit=max_gap)
                after = result[col].isna().sum()
                if before > after:
                    report['filled'][col] = before - after

    elif method == 'ffill':
        for col in result.columns:
            if result[col].dtype in ['float64', 'int64']:
                before = result[col].isna().sum()
                result[col] = result[col].ffill(limit=max_gap)
                after = result[col].isna().sum()
                if before > after:
                    report['filled'][col] = before - after

    elif method == 'drop':
        result = result.dropna()

    report['remaining_nulls'] = result.isna().sum().sum()
    return result, report


def validate_rsi_bounds(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Ensure all RSI columns are within [0, 100].

    Args:
        df: DataFrame with RSI columns

    Returns:
        Tuple of (is_valid, list of invalid columns)
    """
    rsi_cols = [c for c in df.columns if 'RSI' in c]
    invalid_cols = []

    for col in rsi_cols:
        series = df[col].dropna()
        if len(series) > 0:
            if (series < 0).any() or (series > 100).any():
                invalid_cols.append(col)

    return len(invalid_cols) == 0, invalid_cols


def detect_data_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame showing gaps in each column.

    Args:
        df: DataFrame to analyze

    Returns:
        DataFrame with gap statistics per column
    """
    gaps = pd.DataFrame(
        index=df.columns,
        columns=['total_missing', 'max_consecutive', 'pct_missing']
    )

    for col in df.columns:
        total_missing = df[col].isna().sum()
        gaps.loc[col, 'total_missing'] = total_missing
        gaps.loc[col, 'pct_missing'] = (total_missing / len(df)) * 100

        # Calculate max consecutive NaNs
        mask = df[col].isna()
        if mask.any():
            groups = (~mask).cumsum()
            max_consec = mask.groupby(groups).sum().max()
            gaps.loc[col, 'max_consecutive'] = max_consec
        else:
            gaps.loc[col, 'max_consecutive'] = 0

    return gaps.astype({'total_missing': int, 'max_consecutive': int})


def validate_returns_sanity(df: pd.DataFrame, threshold: float = 100.0) -> Dict:
    """
    Check for extreme returns that might indicate data errors.

    Args:
        df: DataFrame with return columns
        threshold: Maximum reasonable monthly return (%)

    Returns:
        Dictionary with extreme value report
    """
    ret_cols = [c for c in df.columns if c.startswith('Ret_')]
    extreme_values = {}

    for col in ret_cols:
        series = df[col].dropna()
        extreme = series[series.abs() > threshold]
        if len(extreme) > 0:
            extreme_values[col] = {
                'count': len(extreme),
                'max': series.max(),
                'min': series.min(),
                'dates': extreme.index.tolist()
            }

    return {
        'has_extremes': len(extreme_values) > 0,
        'threshold': threshold,
        'extreme_columns': extreme_values
    }
