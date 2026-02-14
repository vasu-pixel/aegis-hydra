"""
aegis_hydra.market.tensor_field — Order Book → Physics Tensor Converter

Transforms raw market data (order book snapshots, trades, funding rates)
into a structured tensor that the physics engine can consume.

The MarketTensor is the "reality" that agents observe.

Dependencies: jax, polars, pydantic
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, ConfigDict, Field
import numpy as np


# ---------------------------------------------------------------------------
# Pydantic schema for type-safe market data
# ---------------------------------------------------------------------------
class OrderBookSnapshot(BaseModel):
    """Validated order book snapshot from the ingestion layer."""

    timestamp_us: int = Field(..., description="Microsecond timestamp")
    symbol: str = Field(..., description="Trading pair, e.g., 'BTC-USDT'")
    bid_prices: list[float] = Field(..., description="Bid prices, best first")
    bid_volumes: list[float] = Field(..., description="Bid volumes")
    ask_prices: list[float] = Field(..., description="Ask prices, best first")
    ask_volumes: list[float] = Field(..., description="Ask volumes")
    last_trade_price: float = Field(0.0, description="Most recent trade price")
    last_trade_volume: float = Field(0.0, description="Most recent trade volume")
    funding_rate: float = Field(0.0, description="Current funding rate (perps)")

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# MarketTensor — the physics-ready data structure
# ---------------------------------------------------------------------------
class MarketTensor:
    """
    Structured tensor representation of market state.

    Fields (all JAX arrays):
        mid_price       : float     — Current mid price
        spread          : float     — Bid-ask spread (normalized)
        volatility      : float     — Realized volatility estimate
        bid_density     : (L,)      — Volume density on bid side
        ask_density     : (L,)      — Volume density on ask side
        imbalance       : float     — Order book imbalance [-1, 1]
        flow_velocity   : (L,)      — Estimated order flow velocity field
        pressure        : (L,)      — Pressure field (price impact proxy)
        funding_rate    : float     — Funding rate for perpetual contracts
        timestamp       : float     — Normalized timestamp
    """

    __slots__ = [
        "mid_price", "spread", "volatility",
        "bid_density", "ask_density", "imbalance",
        "flow_velocity", "pressure", "funding_rate", "timestamp",
    ]

    def __init__(
        self,
        mid_price: Array,
        spread: Array,
        volatility: Array,
        bid_density: Array,
        ask_density: Array,
        imbalance: Array,
        flow_velocity: Array,
        pressure: Array,
        funding_rate: Array,
        timestamp: Array,
    ):
        self.mid_price = mid_price
        self.spread = spread
        self.volatility = volatility
        self.bid_density = bid_density
        self.ask_density = ask_density
        self.imbalance = imbalance
        self.flow_velocity = flow_velocity
        self.pressure = pressure
        self.funding_rate = funding_rate
        self.timestamp = timestamp

    def to_flat_vector(self) -> Array:
        """Flatten all fields into a single vector for agent consumption."""
        scalars = jnp.array([
            self.mid_price, self.spread, self.volatility,
            self.imbalance, self.funding_rate, self.timestamp,
        ])
        return jnp.concatenate([
            scalars, self.bid_density, self.ask_density,
            self.flow_velocity, self.pressure,
        ])

    @property
    def n_levels(self) -> int:
        return self.bid_density.shape[0]


# ---------------------------------------------------------------------------
# TensorField — the converter
# ---------------------------------------------------------------------------
class TensorField:
    """
    Converts raw OrderBookSnapshots into MarketTensor objects.

    Maintains a rolling history for volatility estimation and
    flow velocity computation.

    Parameters
    ----------
    n_levels : int
        Number of price levels to include on each side.
    vol_window : int
        Number of snapshots for realized volatility estimation.
    flow_decay : float
        Exponential decay for flow velocity EMA.
    """

    kalman_filter: Optional[Any] = None
    filter_state: Optional[Tuple[Any, Any]] = None

    def __init__(
        self,
        n_levels: int = 50,
        vol_window: int = 100,
        flow_decay: float = 0.95,
        use_filter: bool = True,
    ):
        self.n_levels = n_levels
        self.vol_window = vol_window
        self.flow_decay = flow_decay

        # Rolling state
        self._price_history: list[float] = []
        self._prev_bid_density: Optional[np.ndarray] = None
        self._prev_ask_density: Optional[np.ndarray] = None
        self._flow_ema: Optional[np.ndarray] = None
        
        if use_filter:
            try:
                from ..core.filter import UnscentedKalmanFilter
                self.kalman_filter = UnscentedKalmanFilter()
                self.filter_state = None
            except ImportError as e:
                print(f"Warning: Could not import Kalman Filter: {e}")
                self.kalman_filter = None

    def _estimate_volatility(self) -> float:
        """Realized volatility from log returns."""
        if len(self._price_history) < 2:
            return 0.01
        prices = np.array(self._price_history[-self.vol_window:])
        log_returns = np.diff(np.log(prices + 1e-30))
        return float(np.std(log_returns) + 1e-10)

    def _compute_flow_velocity(
        self,
        bid_density: np.ndarray,
        ask_density: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate order flow velocity from density changes.

        v ≈ -∂ρ/∂t (continuity equation approximation).
        """
        combined = np.concatenate([bid_density, ask_density])

        if self._prev_bid_density is None:
            flow = np.zeros_like(combined)
        else:
            prev_combined = np.concatenate([self._prev_bid_density, self._prev_ask_density])
            flow = -(combined - prev_combined)  # negative because inflow = positive

        # EMA smoothing
        if self._flow_ema is None:
            self._flow_ema = flow
        else:
            self._flow_ema = self.flow_decay * self._flow_ema + (1 - self.flow_decay) * flow

        self._prev_bid_density = bid_density.copy()
        self._prev_ask_density = ask_density.copy()

        return self._flow_ema

    def _compute_pressure(
        self,
        bid_density: np.ndarray,
        ask_density: np.ndarray,
    ) -> np.ndarray:
        """
        Pressure field P(x) ∝ ρ^γ (equation of state).

        Proxy for price impact at each level.
        """
        combined = np.concatenate([bid_density, ask_density])
        gamma = 1.4  # adiabatic index
        return np.power(combined + 1e-10, gamma)

    def process(self, snapshot: OrderBookSnapshot) -> MarketTensor:
        """
        Convert a raw order book snapshot into a MarketTensor.

        Parameters
        ----------
        snapshot : OrderBookSnapshot
            Validated market data.

        Returns
        -------
        MarketTensor
            Physics-ready tensor for agent consumption.
        """
        if snapshot.bid_volumes is None:
             print("DEBUG: bid_volumes is None")
        if snapshot.ask_volumes is None:
             print("DEBUG: ask_volumes is None")
             
        # Pad / truncate to n_levels
        try:
            bids = np.array(snapshot.bid_volumes[:self.n_levels])
            asks = np.array(snapshot.ask_volumes[:self.n_levels])
            bid_prices = np.array(snapshot.bid_prices[:self.n_levels])
            ask_prices = np.array(snapshot.ask_prices[:self.n_levels])
        except Exception as e:
            print(f"CRASH in array creation: {e}")
            raise

        if len(bids) < self.n_levels:
            bids = np.pad(bids, (0, self.n_levels - len(bids)))
        if len(asks) < self.n_levels:
            asks = np.pad(asks, (0, self.n_levels - len(asks)))

        # Core fields
        mid_price = 0.5 * (bid_prices[0] + ask_prices[0]) if len(bid_prices) > 0 else 0.0
        spread = (ask_prices[0] - bid_prices[0]) / (mid_price + 1e-10) if mid_price > 0 else 0.0

        self._price_history.append(mid_price)
        vol = self._estimate_volatility()

        # Kalman Filter Step
        if self.kalman_filter:
            measurement = jnp.array([mid_price, vol])
            
            if self.filter_state is None:
                self.filter_state = self.kalman_filter.initialize(mid_price, vol)
            
            mean, cov = self.kalman_filter.step(self.filter_state, measurement)
            self.filter_state = (mean, cov)
            
            # Use filtered state: [price, velocity, volatility]
            mid_price = float(mean[0])
            # We keep 'vol' as is for now or use mean[2] depending on trust in filter
            # Let's trust the filter for price, but maybe blend vol
            # vol = float(mean[2]) 

        # Normalize densities
        total = np.sum(bids) + np.sum(asks) + 1e-30
        bid_density = bids / total
        ask_density = asks / total

        imbalance = (np.sum(bids) - np.sum(asks)) / (total + 1e-30)

        flow = self._compute_flow_velocity(bid_density, ask_density)
        pressure = self._compute_pressure(bid_density, ask_density)

        return MarketTensor(
            mid_price=jnp.float32(mid_price),
            spread=jnp.float32(spread),
            volatility=jnp.float32(vol),
            bid_density=jnp.array(bid_density, dtype=jnp.float32),
            ask_density=jnp.array(ask_density, dtype=jnp.float32),
            imbalance=jnp.float32(imbalance),
            flow_velocity=jnp.array(flow[:2 * self.n_levels], dtype=jnp.float32),
            pressure=jnp.array(pressure[:2 * self.n_levels], dtype=jnp.float32),
            funding_rate=jnp.float32(snapshot.funding_rate),
            timestamp=jnp.float32(snapshot.timestamp_us / 1e6),
        )
