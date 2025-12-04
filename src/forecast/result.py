"""Unified forecast result container."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from ..models.regimes import RegimeType
from ..scenarios.definitions import ScenarioType
from ..simulation.config import SimulationResult


@dataclass
class CurrentConditions:
    """Current market conditions."""
    date: pd.Timestamp
    price: float
    rsi: float
    rsi_state: str  # High/Mid/Low
    dxy_trend: str  # Rising/Falling
    halving_phase: Optional[float]  # Months since halving
    regime: RegimeType
    matching_scenarios: List[ScenarioType]


@dataclass
class ScenarioForecast:
    """Forecast for a specific scenario."""
    scenario: ScenarioType
    name: str
    count: int
    win_rate: float
    avg_return: float
    ci_lower: float
    ci_upper: float
    quality: str


@dataclass
class ForecastResult:
    """Complete unified forecast result."""
    conditions: CurrentConditions
    simulation: SimulationResult
    scenario_forecasts: List[ScenarioForecast]
    weighted_forecast: Dict[str, float]  # Combined weighted prediction

    def get_price_targets(self) -> Dict[int, float]:
        """Get probability of reaching price targets."""
        final_prices = self.simulation.price_paths[:, -1]
        targets = [50000, 75000, 100000, 150000, 200000, 300000, 500000]

        return {
            target: (final_prices >= target).mean() * 100
            for target in targets
        }

    def get_return_probabilities(self) -> Dict[str, float]:
        """Get probability of various return outcomes."""
        returns = (
            self.simulation.price_paths[:, -1] /
            self.conditions.price - 1
        ) * 100

        return {
            'positive': (returns > 0).mean() * 100,
            'above_20pct': (returns > 20).mean() * 100,
            'above_50pct': (returns > 50).mean() * 100,
            'above_100pct': (returns > 100).mean() * 100,
            'below_minus20pct': (returns < -20).mean() * 100,
            'below_minus50pct': (returns < -50).mean() * 100,
        }

    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for display/JSON."""
        final_dist = self.simulation.get_final_distribution()
        final_ret = self.simulation.get_return_percentiles(
            self.simulation.n_months
        )

        return {
            'date': str(self.conditions.date.date()),
            'current_price': self.conditions.price,
            'current_rsi': self.conditions.rsi,
            'rsi_state': self.conditions.rsi_state,
            'dxy_trend': self.conditions.dxy_trend,
            'regime': self.conditions.regime.value,
            'n_matching_scenarios': len(self.conditions.matching_scenarios),
            'forecast_months': self.simulation.n_months,
            'n_simulations': self.simulation.n_simulations,
            'price_p10': final_dist['p10'],
            'price_p50': final_dist['p50'],
            'price_p90': final_dist['p90'],
            'price_mean': final_dist['mean'],
            'return_p10': final_ret['p10'],
            'return_p50': final_ret['p50'],
            'return_p90': final_ret['p90'],
            'return_mean': final_ret['mean'],
            'prob_positive': self.get_return_probabilities()['positive'],
            'prob_double': self.get_return_probabilities()['above_100pct'],
            'weighted_return': self.weighted_forecast.get('expected_return'),
            'weighted_win_rate': self.weighted_forecast.get('win_rate'),
        }
