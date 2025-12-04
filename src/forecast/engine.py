"""
Unified forecast engine combining regime classification, scenario matching,
and path-dependent Monte Carlo simulation.
"""
import pandas as pd
import numpy as np
from typing import Optional, List

from .result import (
    ForecastResult,
    CurrentConditions,
    ScenarioForecast,
)
from ..models.regimes import RegimeClassifier, RegimeType
from ..scenarios import ScenarioMatcher, ScenarioType, get_halving_phase
from ..scenarios.definitions import SCENARIOS
from ..simulation import MonteCarloEngine, SimulationConfig
from ..utils.dates import get_available_history


class UnifiedForecastEngine:
    """
    Unified forecast engine that combines all forecasting components.

    Workflow:
    1. Analyze current conditions (RSI, DXY, halving phase)
    2. Classify regime and find matching scenarios
    3. Get scenario-specific historical statistics
    4. Run Monte Carlo with regime-aware sampling
    5. Combine into weighted forecast
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_simulations: int = 2000,
        n_months: int = 12,
        random_seed: Optional[int] = None
    ):
        """
        Initialize unified engine.

        Args:
            data: Full dataset with features and returns
            n_simulations: Number of Monte Carlo simulations
            n_months: Forecast horizon in months
            random_seed: For reproducibility
        """
        self.data = data
        self.n_simulations = n_simulations
        self.n_months = n_months
        self.random_seed = random_seed

        # Initialize sub-components
        self.regime_classifier = RegimeClassifier()
        self.scenario_matcher = ScenarioMatcher(data)

    def run(
        self,
        current_price: Optional[float] = None,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> ForecastResult:
        """
        Run complete unified forecast.

        Args:
            current_price: Starting price (default: latest from data)
            as_of_date: P.I.T. date (default: latest in data)

        Returns:
            ForecastResult with all forecast components
        """
        # Default to latest data
        if as_of_date is None:
            as_of_date = self.data.index.max()

        df = get_available_history(self.data, as_of_date)
        current_row = df.iloc[-1]

        if current_price is None:
            current_price = current_row['BTC']

        # 1. Analyze current conditions
        conditions = self._analyze_conditions(
            current_row, current_price, as_of_date
        )

        # 2. Get scenario forecasts for current month
        scenario_forecasts = self._get_scenario_forecasts(
            conditions, as_of_date
        )

        # 3. Run Monte Carlo simulation
        sim_config = SimulationConfig(
            n_simulations=self.n_simulations,
            n_months=self.n_months,
            random_seed=self.random_seed,
        )
        mc_engine = MonteCarloEngine(self.data, sim_config)
        simulation = mc_engine.run(current_price, as_of_date)

        # 4. Calculate weighted forecast from scenarios
        weighted = self._calculate_weighted_forecast(scenario_forecasts)

        return ForecastResult(
            conditions=conditions,
            simulation=simulation,
            scenario_forecasts=scenario_forecasts,
            weighted_forecast=weighted,
        )

    def _analyze_conditions(
        self,
        current_row: pd.Series,
        current_price: float,
        as_of_date: pd.Timestamp
    ) -> CurrentConditions:
        """Analyze current market conditions."""
        # Get RSI state
        rsi = current_row.get('RSI_BTC', current_row.get('Prev_RSI_BTC', 50))
        if rsi > 65:
            rsi_state = 'High'
        elif rsi < 45:
            rsi_state = 'Low'
        else:
            rsi_state = 'Mid'

        # Get DXY trend
        dxy_trend_val = current_row.get('Prev_DXY_Trend', 0)
        dxy_trend = 'Rising' if dxy_trend_val > 0 else 'Falling'

        # Get halving phase
        halving_phase = get_halving_phase(as_of_date)

        # Classify regime
        features = pd.Series({
            'Prev_RSI_BTC': current_row.get('Prev_RSI_BTC', rsi),
            'Prev_RSI_SPX': current_row.get('Prev_RSI_SPX', 50),
            'Prev_DXY_Trend': dxy_trend_val,
            'Prev_Rate_Trend': current_row.get('Prev_Rate_Trend', 0),
        })
        regime = self.regime_classifier.classify(features)

        # Get matching scenarios
        matching = self.scenario_matcher.get_current_scenarios(as_of_date)

        return CurrentConditions(
            date=as_of_date,
            price=current_price,
            rsi=rsi,
            rsi_state=rsi_state,
            dxy_trend=dxy_trend,
            halving_phase=halving_phase,
            regime=regime,
            matching_scenarios=matching,
        )

    def _get_scenario_forecasts(
        self,
        conditions: CurrentConditions,
        as_of_date: pd.Timestamp
    ) -> List[ScenarioForecast]:
        """Get forecasts for matching scenarios."""
        # Run scenario analysis for current month only
        matrix = self.scenario_matcher.run_matrix_analysis(
            as_of_date=as_of_date,
            scenarios=conditions.matching_scenarios
        )

        # Filter to current month
        current_month = as_of_date.strftime('%b')
        month_data = matrix[matrix['Month'] == current_month]

        forecasts = []
        for _, row in month_data.iterrows():
            # Find scenario type from name
            scenario_type = None
            for st, config in SCENARIOS.items():
                if config['name'] == row['Scenario']:
                    scenario_type = st
                    break

            if scenario_type is None:
                continue

            forecasts.append(ScenarioForecast(
                scenario=scenario_type,
                name=row['Scenario'],
                count=row['Count'],
                win_rate=row['Win_Rate'],
                avg_return=row['Avg_Return'],
                ci_lower=row['CI_Lower_90'],
                ci_upper=row['CI_Upper_90'],
                quality=row['Quality'],
            ))

        return forecasts

    def _calculate_weighted_forecast(
        self,
        scenario_forecasts: List[ScenarioForecast]
    ) -> dict:
        """Calculate weighted forecast from matching scenarios."""
        if not scenario_forecasts:
            return {'expected_return': 0, 'win_rate': 50, 'confidence': 'Low'}

        # Weight by sample count (more samples = more reliable)
        total_count = sum(sf.count for sf in scenario_forecasts)

        if total_count == 0:
            return {'expected_return': 0, 'win_rate': 50, 'confidence': 'Low'}

        weighted_return = sum(
            sf.avg_return * sf.count for sf in scenario_forecasts
        ) / total_count

        weighted_win_rate = sum(
            sf.win_rate * sf.count for sf in scenario_forecasts
        ) / total_count

        # Determine confidence level
        reliable_count = sum(
            sf.count for sf in scenario_forecasts if sf.count >= 5
        )
        if reliable_count >= total_count * 0.7:
            confidence = 'High'
        elif reliable_count >= total_count * 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        return {
            'expected_return': weighted_return,
            'win_rate': weighted_win_rate,
            'confidence': confidence,
            'total_samples': total_count,
        }
