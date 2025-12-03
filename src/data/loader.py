"""
Data loader with Point-in-Time (P.I.T.) filtering.

Loads pre-processed data from CSV files and provides strict P.I.T. access.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from ..utils.dates import get_available_history


class DataLoader:
    """
    Loads pre-processed data from CSV files.

    Key features:
    - Automatic USDT column exclusion (incomplete data pre-2017)
    - Strict P.I.T. filtering via get_data_as_of()
    - Data completeness validation
    """

    # USDT columns to exclude (missing data pre-2017)
    USDT_COLUMNS = [
        'USDT', 'RSI_USDT', 'USDT_AG', 'USDT_AL',
        'Ret_USDT', 'Prev_RSI_USDT', 'Prev_USDT_AG', 'Prev_USDT_AL'
    ]

    def __init__(self, data_dir: Path):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data/ directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self._cached_data: Optional[pd.DataFrame] = None

    def load_full_dataset(self, exclude_usdt: bool = True) -> pd.DataFrame:
        """
        Load the master dataset from full_dataset.csv.

        Args:
            exclude_usdt: If True, remove USDT-related columns

        Returns:
            DataFrame with Date as index
        """
        csv_path = self.processed_dir / 'full_dataset.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df = df.set_index('Date')

        if exclude_usdt:
            cols_to_drop = [c for c in self.USDT_COLUMNS if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        return df

    def get_data_as_of(
        self,
        as_of_date: pd.Timestamp,
        exclude_usdt: bool = True
    ) -> pd.DataFrame:
        """
        CRITICAL P.I.T. METHOD: Returns data available as of a specific date.

        This is the primary method for getting training data in backtests.
        Ensures no future data leakage.

        Args:
            as_of_date: Maximum date to include (inclusive)
            exclude_usdt: If True, remove USDT-related columns

        Returns:
            Copy of data filtered to <= as_of_date
        """
        if self._cached_data is None:
            self._cached_data = self.load_full_dataset(exclude_usdt=exclude_usdt)

        return get_available_history(self._cached_data, as_of_date)

    def get_column_info(self) -> Dict[str, List[str]]:
        """Return categorized column names."""
        df = self.load_full_dataset(exclude_usdt=True)
        cols = df.columns.tolist()

        return {
            'prices': [c for c in cols if c in ['BTC', 'SPX', 'NDX', 'GLD', 'TLT', 'DXY', 'Rates']],
            'rsi': [c for c in cols if 'RSI_' in c],
            'rsi_components': [c for c in cols if '_AG' in c or '_AL' in c],
            'returns': [c for c in cols if c.startswith('Ret_')],
            'lagged': [c for c in cols if c.startswith('Prev_')],
            'network': [c for c in cols if c in ['hash_rate', 'active_addresses', 'n_transactions', 'BTC_Volume']],
            'trends': [c for c in cols if 'Trend' in c],
        }

    def validate_data_completeness(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Check for missing values and data gaps.

        Args:
            df: DataFrame to validate (or load fresh if None)

        Returns:
            Dictionary with completeness report
        """
        if df is None:
            df = self.load_full_dataset(exclude_usdt=True)

        missing = df.isna().sum()
        missing_cols = missing[missing > 0].to_dict()

        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_by_column': missing_cols,
            'complete_rows': df.dropna().shape[0],
            'completeness_pct': (1 - df.isna().sum().sum() / df.size) * 100,
            'date_range': (df.index.min(), df.index.max())
        }
