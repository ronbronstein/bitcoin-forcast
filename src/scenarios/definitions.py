"""
Scenario definitions for conditional probability analysis.

16 scenarios covering RSI states, macro conditions, and halving cycles.
"""
from enum import Enum
from typing import Callable, Dict
import pandas as pd


class ScenarioType(Enum):
    """All supported scenario types."""
    BASELINE = "baseline"
    RECENT_REGIME = "recent_regime"
    HIGH_RSI = "high_rsi"
    LOW_RSI = "low_rsi"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    STRONG_DOLLAR = "strong_dollar"
    WEAK_DOLLAR = "weak_dollar"
    HIKING = "hiking"
    CUTTING = "cutting"
    GOLDEN_POCKET = "golden_pocket"
    DOOM = "doom"
    POST_HALVING_Y1 = "post_halving_y1"
    POST_HALVING_Y2 = "post_halving_y2"
    PRE_HALVING = "pre_halving"
    HIGH_NETWORK = "high_network"
    LOW_NETWORK = "low_network"


# Bitcoin halving dates
HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-19'),
]


def get_halving_phase(date: pd.Timestamp) -> float:
    """
    Get months since most recent halving.

    Returns None if before first halving.
    """
    for h in reversed(HALVING_DATES):
        if date >= h:
            return (date - h).days / 30.44
    return None


def _halving_filter(min_months: float, max_months: float):
    """Create filter for halving cycle phase."""
    def filter_func(df: pd.DataFrame) -> pd.Series:
        def check_row(row_date):
            phase = get_halving_phase(row_date)
            if phase is None:
                return False
            return min_months <= phase < max_months
        return pd.Series([check_row(d) for d in df.index], index=df.index)
    return filter_func


# Scenario definitions with filter functions
SCENARIOS: Dict[ScenarioType, Dict] = {
    ScenarioType.BASELINE: {
        "name": "Baseline (All History)",
        "desc": "Average of all available history",
        "filter": lambda df: pd.Series(True, index=df.index),
    },
    ScenarioType.RECENT_REGIME: {
        "name": "Recent Regime (2020+)",
        "desc": "Post-institutional adoption era",
        "filter": lambda df: df.index.year >= 2020,
    },
    ScenarioType.HIGH_RSI: {
        "name": "High RSI (>65)",
        "desc": "Momentum overheated",
        "filter": lambda df: df['Prev_RSI_BTC'] > 65,
    },
    ScenarioType.LOW_RSI: {
        "name": "Low RSI (<45)",
        "desc": "Oversold / bottoming",
        "filter": lambda df: df['Prev_RSI_BTC'] < 45,
    },
    ScenarioType.RISK_ON: {
        "name": "Risk-On (SPX RSI > 60)",
        "desc": "Strong stock market",
        "filter": lambda df: df['Prev_RSI_SPX'] > 60,
    },
    ScenarioType.RISK_OFF: {
        "name": "Risk-Off (SPX RSI < 40)",
        "desc": "Weak stock market",
        "filter": lambda df: df['Prev_RSI_SPX'] < 40,
    },
    ScenarioType.STRONG_DOLLAR: {
        "name": "Strong Dollar (Rising)",
        "desc": "DXY trending up",
        "filter": lambda df: df['Prev_DXY_Trend'] > 0,
    },
    ScenarioType.WEAK_DOLLAR: {
        "name": "Weak Dollar (Falling)",
        "desc": "DXY trending down",
        "filter": lambda df: df['Prev_DXY_Trend'] <= 0,
    },
    ScenarioType.HIKING: {
        "name": "Hiking/Tight (Rates Up)",
        "desc": "Rates rising",
        "filter": lambda df: df['Prev_Rate_Trend'] > 0,
    },
    ScenarioType.CUTTING: {
        "name": "Cutting/Loose (Rates Down)",
        "desc": "Rates falling or stable",
        "filter": lambda df: df['Prev_Rate_Trend'] <= 0,
    },
    ScenarioType.GOLDEN_POCKET: {
        "name": "Golden Pocket",
        "desc": "Weak DXY + Bull RSI (best case)",
        "filter": lambda df: (df['Prev_DXY_Trend'] <= 0) & (df['Prev_RSI_BTC'] > 50),
    },
    ScenarioType.DOOM: {
        "name": "Doom Scenario",
        "desc": "Strong DXY + Hiking (liquidity crunch)",
        "filter": lambda df: (df['Prev_DXY_Trend'] > 0) & (df['Prev_Rate_Trend'] > 0),
    },
    ScenarioType.POST_HALVING_Y1: {
        "name": "Post-Halving Year 1",
        "desc": "0-12 months after halving",
        "filter": _halving_filter(0, 12),
    },
    ScenarioType.POST_HALVING_Y2: {
        "name": "Post-Halving Year 2",
        "desc": "12-24 months after halving",
        "filter": _halving_filter(12, 24),
    },
    ScenarioType.PRE_HALVING: {
        "name": "Pre-Halving Year",
        "desc": "36-48 months after halving",
        "filter": _halving_filter(36, 48),
    },
    ScenarioType.HIGH_NETWORK: {
        "name": "High Network Activity",
        "desc": "Active addresses Z > 1.0",
        "filter": lambda df: (
            df['Prev_Active_Addresses_Z'] > 1.0
            if 'Prev_Active_Addresses_Z' in df.columns
            else pd.Series(False, index=df.index)
        ),
    },
    ScenarioType.LOW_NETWORK: {
        "name": "Low Network Activity",
        "desc": "Active addresses Z < -1.0",
        "filter": lambda df: (
            df['Prev_Active_Addresses_Z'] < -1.0
            if 'Prev_Active_Addresses_Z' in df.columns
            else pd.Series(False, index=df.index)
        ),
    },
}


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
