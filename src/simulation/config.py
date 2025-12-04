"""Configuration and result types for Monte Carlo simulation."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 2000
    n_months: int = 12
    rsi_period: int = 14
    rsi_high: float = 65.0
    rsi_low: float = 45.0
    min_samples: int = 5
    half_life_years: float = 4.0
    random_seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    dates: List[pd.Timestamp]
    price_paths: np.ndarray  # Shape: (n_simulations, n_months + 1)
    initial_price: float
    initial_rsi: float
    match_logic: str
    config: SimulationConfig

    @property
    def n_simulations(self) -> int:
        return self.price_paths.shape[0]

    @property
    def n_months(self) -> int:
        return self.price_paths.shape[1] - 1

    def get_percentiles(self, month: int) -> Dict[str, float]:
        """Get price percentiles for a specific month."""
        prices = self.price_paths[:, month]
        return {
            'p5': np.percentile(prices, 5),
            'p10': np.percentile(prices, 10),
            'p25': np.percentile(prices, 25),
            'p50': np.percentile(prices, 50),
            'p75': np.percentile(prices, 75),
            'p90': np.percentile(prices, 90),
            'p95': np.percentile(prices, 95),
            'mean': prices.mean(),
            'std': prices.std(),
        }

    def get_return_percentiles(self, month: int) -> Dict[str, float]:
        """Get return percentiles for a specific month."""
        returns = (self.price_paths[:, month] / self.initial_price - 1) * 100
        return {
            'p5': np.percentile(returns, 5),
            'p10': np.percentile(returns, 10),
            'p25': np.percentile(returns, 25),
            'p50': np.percentile(returns, 50),
            'p75': np.percentile(returns, 75),
            'p90': np.percentile(returns, 90),
            'p95': np.percentile(returns, 95),
            'mean': returns.mean(),
            'std': returns.std(),
        }

    def get_final_distribution(self) -> Dict[str, float]:
        """Get final month price distribution."""
        return self.get_percentiles(self.n_months)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert paths to DataFrame."""
        return pd.DataFrame(
            self.price_paths.T,
            index=self.dates,
            columns=[f'sim_{i}' for i in range(self.n_simulations)]
        )

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics by month."""
        rows = []
        for i, date in enumerate(self.dates):
            stats = self.get_percentiles(i)
            ret_stats = self.get_return_percentiles(i) if i > 0 else {}
            rows.append({
                'Date': date,
                'Month': i,
                'Price_P10': stats['p10'],
                'Price_P50': stats['p50'],
                'Price_P90': stats['p90'],
                'Price_Mean': stats['mean'],
                'Return_P10': ret_stats.get('p10', 0),
                'Return_P50': ret_stats.get('p50', 0),
                'Return_P90': ret_stats.get('p90', 0),
            })
        return pd.DataFrame(rows)
