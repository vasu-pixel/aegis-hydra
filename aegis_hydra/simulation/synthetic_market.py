"""
aegis_hydra.simulation.synthetic_market — Fake Data Generator

Generates synthetic order book data with configurable properties
(volatility, mean-reversion, fat tails, regime switches) to
stress-test the physics engine under controlled conditions.

Dependencies: jax, numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..market.tensor_field import OrderBookSnapshot


@dataclass
class MarketRegime:
    """Configuration for a synthetic market regime."""
    name: str
    volatility: float          # annualized vol
    drift: float               # annualized drift
    mean_reversion: float      # speed of reversion to equilibrium
    spread_bps: float          # spread in basis points
    depth_base: float          # base volume at each level
    depth_decay: float         # exponential decay of depth away from mid
    fat_tail_alpha: float      # Pareto tail index (lower = fatter)
    jump_intensity: float      # Poisson rate of price jumps per minute
    jump_size_std: float       # std of jump sizes


# Pre-built regimes
REGIME_CALM = MarketRegime(
    name="calm", volatility=0.20, drift=0.0, mean_reversion=0.5,
    spread_bps=5, depth_base=100, depth_decay=0.95,
    fat_tail_alpha=3.0, jump_intensity=0.01, jump_size_std=0.005,
)

REGIME_VOLATILE = MarketRegime(
    name="volatile", volatility=0.80, drift=0.0, mean_reversion=0.1,
    spread_bps=20, depth_base=50, depth_decay=0.90,
    fat_tail_alpha=1.5, jump_intensity=0.1, jump_size_std=0.02,
)

REGIME_TRENDING = MarketRegime(
    name="trending", volatility=0.40, drift=0.30, mean_reversion=0.05,
    spread_bps=10, depth_base=80, depth_decay=0.93,
    fat_tail_alpha=2.5, jump_intensity=0.05, jump_size_std=0.01,
)

REGIME_CRASH = MarketRegime(
    name="crash", volatility=1.50, drift=-1.0, mean_reversion=0.0,
    spread_bps=100, depth_base=20, depth_decay=0.80,
    fat_tail_alpha=1.2, jump_intensity=0.5, jump_size_std=0.05,
)


class SyntheticMarket:
    """
    Generates synthetic order book snapshots for testing.

    Parameters
    ----------
    initial_price : float
        Starting mid price.
    n_levels : int
        Number of order book levels per side.
    tick_size : float
        Minimum price increment.
    symbol : str
        Trading pair name.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        initial_price: float = 50000.0,
        n_levels: int = 50,
        tick_size: float = 0.01,
        symbol: str = "SYN/USD",
        seed: int = 42,
    ):
        self.initial_price = initial_price
        self.n_levels = n_levels
        self.tick_size = tick_size
        self.symbol = symbol
        self.rng = np.random.default_rng(seed)
        self._current_price = initial_price
        self._time_us = 0

    def _generate_book(
        self,
        mid: float,
        regime: MarketRegime,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Generate synthetic bid/ask prices and volumes."""
        spread = mid * regime.spread_bps / 10000
        best_bid = mid - spread / 2
        best_ask = mid + spread / 2

        bid_prices = [best_bid - i * self.tick_size for i in range(self.n_levels)]
        ask_prices = [best_ask + i * self.tick_size for i in range(self.n_levels)]

        # Volume with exponential decay + noise
        bid_volumes = [
            max(0.1, regime.depth_base * (regime.depth_decay ** i)
                + self.rng.normal(0, regime.depth_base * 0.1))
            for i in range(self.n_levels)
        ]
        ask_volumes = [
            max(0.1, regime.depth_base * (regime.depth_decay ** i)
                + self.rng.normal(0, regime.depth_base * 0.1))
            for i in range(self.n_levels)
        ]

        return bid_prices, bid_volumes, ask_prices, ask_volumes

    def generate(
        self,
        n_snapshots: int,
        regime: MarketRegime = REGIME_CALM,
        dt_minutes: float = 1.0,
    ) -> List[OrderBookSnapshot]:
        """
        Generate a sequence of synthetic order book snapshots.

        Parameters
        ----------
        n_snapshots : int
            Number of snapshots to generate.
        regime : MarketRegime
            Market regime configuration.
        dt_minutes : float
            Time between snapshots in minutes.

        Returns
        -------
        List[OrderBookSnapshot]
        """
        snapshots: List[OrderBookSnapshot] = []
        dt_years = dt_minutes / (252 * 1440)  # convert to annualized

        for _ in range(n_snapshots):
            # Price dynamics: GBM + mean reversion + jumps
            dW = self.rng.normal()
            drift = regime.drift * dt_years
            diffusion = regime.volatility * np.sqrt(dt_years) * dW
            reversion = -regime.mean_reversion * (
                self._current_price - self.initial_price
            ) / self.initial_price * dt_years

            # Jump component (Poisson)
            n_jumps = self.rng.poisson(regime.jump_intensity * dt_minutes)
            jump = sum(
                self.rng.normal(0, regime.jump_size_std)
                for _ in range(n_jumps)
            )

            # Fat tails via Pareto mixture
            if self.rng.random() < 0.05:  # 5% of the time, draw from fat tail
                tail_shock = (
                    self.rng.pareto(regime.fat_tail_alpha)
                    * regime.volatility * np.sqrt(dt_years)
                    * self.rng.choice([-1, 1])
                )
                diffusion += tail_shock

            price_return = drift + diffusion + reversion + jump
            self._current_price *= (1 + price_return)
            self._current_price = max(self._current_price, self.tick_size)

            # Generate order book
            bp, bv, ap, av = self._generate_book(self._current_price, regime)

            self._time_us += int(dt_minutes * 60 * 1e6)

            snapshots.append(OrderBookSnapshot(
                timestamp_us=self._time_us,
                symbol=self.symbol,
                bid_prices=bp,
                bid_volumes=bv,
                ask_prices=ap,
                ask_volumes=av,
                last_trade_price=self._current_price,
                last_trade_volume=float(self.rng.exponential(10)),
                funding_rate=float(self.rng.normal(0, 0.0001)),
            ))

        return snapshots

    def generate_regime_switch(
        self,
        regimes: List[Tuple[MarketRegime, int]],
        dt_minutes: float = 1.0,
    ) -> List[OrderBookSnapshot]:
        """
        Generate snapshots with regime switching.

        Parameters
        ----------
        regimes : List of (regime, n_snapshots) tuples
            Sequence of regimes with their duration.

        Returns
        -------
        List[OrderBookSnapshot]
        """
        all_snapshots: List[OrderBookSnapshot] = []
        for regime, n in regimes:
            all_snapshots.extend(self.generate(n, regime, dt_minutes))
        return all_snapshots
