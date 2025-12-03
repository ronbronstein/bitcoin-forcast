"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import DataLoader


@pytest.fixture(scope='session')
def data_dir():
    """Return path to data directory."""
    return Path('data')


@pytest.fixture(scope='session')
def loader(data_dir):
    """Create DataLoader instance (session-scoped for efficiency)."""
    return DataLoader(data_dir)


@pytest.fixture(scope='session')
def full_dataset(loader):
    """Load full dataset (session-scoped for efficiency)."""
    return loader.load_full_dataset(exclude_usdt=True)
