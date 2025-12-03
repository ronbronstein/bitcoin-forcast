"""
Date handling utilities with strict Point-in-Time (P.I.T.) enforcement.

All date operations ensure no future data leakage.
"""
from datetime import datetime
from typing import Optional
import pandas as pd


def get_month_start(date: pd.Timestamp) -> pd.Timestamp:
    """Convert any date to the first day of its month."""
    return date.to_period('M').to_timestamp()


def validate_as_of_date(as_of_date: pd.Timestamp, data_index: pd.DatetimeIndex) -> bool:
    """
    Ensure as_of_date exists in data and is not in the future.

    Args:
        as_of_date: The date to validate
        data_index: DatetimeIndex of the dataset

    Returns:
        True if valid, False otherwise
    """
    today = pd.Timestamp.now().normalize()
    return as_of_date <= today and as_of_date in data_index


def get_available_history(data: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Return data strictly <= as_of_date (P.I.T. compliant).

    CRITICAL: This is the core P.I.T. filtering function.
    Always use this when building training data.

    Args:
        data: DataFrame with DatetimeIndex
        as_of_date: Maximum date to include (inclusive)

    Returns:
        Copy of data filtered to <= as_of_date
    """
    return data[data.index <= as_of_date].copy()


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Calculate months between two dates."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def get_next_month(date: pd.Timestamp) -> pd.Timestamp:
    """Get the first day of the next month."""
    return (date + pd.DateOffset(months=1)).to_period('M').to_timestamp()
