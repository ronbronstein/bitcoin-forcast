"""
Path-dependent Monte Carlo simulation engine.

RSI evolves during simulation using Wilder's smoothing.
Uses time-weighted sampling to prioritize recent data.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from .config import SimulationConfig, SimulationResult
from ..utils.dates import get_available_history


class MonteCarloEngine:
    """
    Path-dependent Monte Carlo simulation for Bitcoin price forecasting.

    Key features:
    - RSI evolves during simulation (not static)
    - AG/AL components tracked with Wilder's smoothing
    - Time-weighted sampling (4-year half-life decay)
    - Regime-based return sampling (High/Mid/Low RSI)
    """

    def __init__(self, data: pd.DataFrame, config: Optional[SimulationConfig] = None):
        """
        Initialize engine.

        Args:
            data: DataFrame with Prev_* features and Ret_BTC
            config: Simulation configuration
        """
        self.data = data
        self.config = config or SimulationConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def run(
        self,
        current_price: float,
        as_of_date: pd.Timestamp
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            current_price: Starting price
            as_of_date: P.I.T. date for data cutoff

        Returns:
            SimulationResult with price paths
        """
        cfg = self.config

        # P.I.T. filter
        df = get_available_history(self.data, as_of_date)

        if df.empty:
            raise ValueError(f"No data available as of {as_of_date}")

        # Get initial conditions
        initial = self._get_initial_conditions(df)

        if initial['BTC_AG'] is None or initial['BTC_AL'] is None:
            raise ValueError("Missing AG/AL components for RSI evolution")

        # Initialize simulation states (vectorized)
        n_sims = cfg.n_simulations
        states = {
            'price': np.full(n_sims, float(current_price)),
            'BTC_AG': np.full(n_sims, float(initial['BTC_AG'])),
            'BTC_AL': np.full(n_sims, float(initial['BTC_AL'])),
        }

        # Generate future dates
        dates = [as_of_date]
        for i in range(cfg.n_months):
            dates.append(as_of_date + pd.DateOffset(months=i + 1))

        # Initialize price paths
        price_paths = np.zeros((n_sims, cfg.n_months + 1))
        price_paths[:, 0] = current_price

        # Pre-calculate time-decay weights
        weights = self._calculate_time_weights(df, as_of_date)

        # Get macro condition (static assumption)
        is_strong_dxy = initial.get('DXY_Trend', 0) > 0

        # Simulation loop
        for t in range(cfg.n_months):
            target_month = dates[t + 1].month
            month_data = df[df.index.month == target_month]

            if month_data.empty:
                price_paths[:, t + 1] = price_paths[:, t]
                continue

            # Calculate current RSI from AG/AL
            sim_rsi = self._calculate_rsi(states['BTC_AG'], states['BTC_AL'])

            # Draw returns based on RSI state
            drawn_returns = self._draw_returns(
                sim_rsi, month_data, weights, is_strong_dxy
            )

            # Update price
            prev_price = states['price'].copy()
            states['price'] *= (1 + drawn_returns / 100)
            price_paths[:, t + 1] = states['price']

            # Update RSI components (Wilder's smoothing)
            delta = states['price'] - prev_price
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            states['BTC_AG'] = (
                states['BTC_AG'] * (cfg.rsi_period - 1) + gain
            ) / cfg.rsi_period
            states['BTC_AL'] = (
                states['BTC_AL'] * (cfg.rsi_period - 1) + loss
            ) / cfg.rsi_period

        return SimulationResult(
            dates=dates,
            price_paths=price_paths,
            initial_price=current_price,
            initial_rsi=initial['RSI_BTC'],
            match_logic="Path-Dependent (RSI) + Static (DXY) + Time-Weighted",
            config=cfg,
        )

    def _get_initial_conditions(self, df: pd.DataFrame) -> dict:
        """Extract initial conditions from most recent data."""
        last = df.iloc[-1]

        return {
            'RSI_BTC': last.get('RSI_BTC', last.get('Prev_RSI_BTC', 50)),
            'BTC_AG': last.get('BTC_AG'),
            'BTC_AL': last.get('BTC_AL'),
            'DXY_Trend': last.get('Prev_DXY_Trend', 0),
        }

    def _calculate_time_weights(
        self,
        df: pd.DataFrame,
        as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Calculate time-decay weights for historical data."""
        decay_lambda = np.log(2) / self.config.half_life_years

        # Handle timezone
        as_of_naive = (
            as_of_date.tz_localize(None)
            if as_of_date.tzinfo else as_of_date
        )
        idx_naive = (
            df.index.tz_localize(None)
            if df.index.tzinfo else df.index
        )

        ages_years = (as_of_naive - idx_naive).days / 365.25
        weights = np.exp(-decay_lambda * np.maximum(0, ages_years))

        return pd.Series(weights, index=df.index)

    def _calculate_rsi(self, ag: np.ndarray, al: np.ndarray) -> np.ndarray:
        """Calculate RSI from AG/AL components."""
        # Handle division by zero
        rs = np.divide(
            ag, al,
            out=np.full_like(ag, 1000.0),
            where=al != 0
        )
        return 100 - (100 / (1 + rs))

    def _draw_returns(
        self,
        sim_rsi: np.ndarray,
        month_data: pd.DataFrame,
        weights: pd.Series,
        is_strong_dxy: bool
    ) -> np.ndarray:
        """Draw returns based on simulated RSI state."""
        cfg = self.config
        n_sims = len(sim_rsi)
        drawn_returns = np.zeros(n_sims)

        # Bin simulations by RSI state
        rsi_bins = pd.cut(
            sim_rsi,
            bins=[0, cfg.rsi_low, cfg.rsi_high, 100],
            labels=['Low', 'Mid', 'High']
        )

        for state in ['Low', 'Mid', 'High']:
            mask = (rsi_bins == state)
            if mask.sum() == 0:
                continue

            # Filter historical data by RSI state
            if state == 'Low':
                subset = month_data[month_data['Prev_RSI_BTC'] < cfg.rsi_low]
            elif state == 'High':
                subset = month_data[month_data['Prev_RSI_BTC'] > cfg.rsi_high]
            else:
                subset = month_data

            # Apply macro filter
            if is_strong_dxy:
                macro_subset = subset[subset['Prev_DXY_Trend'] > 0]
            else:
                macro_subset = subset[subset['Prev_DXY_Trend'] <= 0]

            # Fallback logic
            if len(macro_subset) >= cfg.min_samples:
                samples_df = macro_subset
            elif len(subset) >= cfg.min_samples:
                samples_df = subset
            else:
                samples_df = month_data

            if len(samples_df) == 0:
                continue

            # Time-weighted sampling
            sample_returns = samples_df['Ret_BTC'].values
            sample_weights = weights.loc[samples_df.index].values

            # Normalize weights
            weight_sum = sample_weights.sum()
            if weight_sum > 0:
                norm_weights = sample_weights / weight_sum
            else:
                norm_weights = None

            # Draw samples
            drawn_returns[mask] = np.random.choice(
                sample_returns,
                size=mask.sum(),
                p=norm_weights
            )

        return drawn_returns
