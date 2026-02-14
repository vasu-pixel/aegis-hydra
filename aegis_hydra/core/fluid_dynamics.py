"""
aegis_hydra.core.fluid_dynamics — Order Book Flow Dynamics

Navier-Stokes inspired solvers that model the order book as a
compressible fluid. Bid/ask volumes are densities, and order flow
is velocity. Viscosity captures market friction (spread, slippage).

Mathematical Foundation:
    - Continuity:     ∂ρ/∂t + ∇·(ρv) = 0
    - Momentum:       ρ(∂v/∂t + v·∇v) = -∇P + μ∇²v + f
    - Pressure:       P = P(ρ)  (equation of state)

The "fluid" is the distribution of limit orders across price levels.

Dependencies: jax, jaxlib, equinox
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


# ---------------------------------------------------------------------------
# Viscosity Estimator
# ---------------------------------------------------------------------------
class ViscosityEstimator(eqx.Module):
    """
    Estimates the effective viscosity μ of the order book fluid.

    Viscosity = market friction. High viscosity → sticky, illiquid market.
    Low viscosity → smooth, liquid market.

    Parameters
    ----------
    method : str
        Estimation method: 'spread' (from bid-ask spread),
        'volume' (from volume profile), 'combined'.
    smoothing : float
        Exponential moving average decay for temporal smoothing.
    """

    method: str = "combined"
    smoothing: float = 0.95

    def from_spread(self, bid_prices: Array, ask_prices: Array) -> Array:
        """
        Viscosity proportional to normalized spread.

        μ ∝ (ask - bid) / midprice
        """
        mid = 0.5 * (bid_prices[0] + ask_prices[0])
        spread = ask_prices[0] - bid_prices[0]
        return spread / (mid + 1e-30)

    def from_volume(self, bid_volumes: Array, ask_volumes: Array) -> Array:
        """
        Viscosity inversely proportional to total depth.

        μ ∝ 1 / total_volume
        """
        total = jnp.sum(bid_volumes) + jnp.sum(ask_volumes) + 1e-30
        return 1.0 / total

    def __call__(
        self,
        bid_prices: Array,
        ask_prices: Array,
        bid_volumes: Array,
        ask_volumes: Array,
    ) -> Array:
        """Compute viscosity using the configured method."""
        if self.method == "spread":
            return self.from_spread(bid_prices, ask_prices)
        elif self.method == "volume":
            return self.from_volume(bid_volumes, ask_volumes)
        else:  # combined
            mu_spread = self.from_spread(bid_prices, ask_prices)
            mu_volume = self.from_volume(bid_volumes, ask_volumes)
            return 0.5 * (mu_spread + mu_volume)


# ---------------------------------------------------------------------------
# Order Book Flow Solver (1-D compressible Navier-Stokes)
# ---------------------------------------------------------------------------
class OrderBookFlowSolver(eqx.Module):
    """
    1-D compressible Navier-Stokes solver for order book dynamics.

    The price axis is treated as the spatial dimension. Limit-order volume
    at each price level is the fluid density ρ(x). Order flow (market orders
    consuming liquidity) is the velocity v(x).

    Parameters
    ----------
    n_levels : int
        Number of price levels in the spatial grid.
    dx : float
        Grid spacing (price tick size, normalized).
    dt : float
        Time step for the solver.
    gamma : float
        Adiabatic index for the equation of state P = ρ^γ.
    base_viscosity : float
        Baseline viscosity if no estimator is provided.
    """

    n_levels: int = 100
    dx: float = 0.01
    dt: float = 1e-4
    gamma: float = 1.4
    base_viscosity: float = 0.01

    def equation_of_state(self, rho: Array) -> Array:
        """
        Pressure from density: P = ρ^γ.

        This encodes how "pushback" (price impact) scales with volume.
        """
        return jnp.power(jnp.maximum(rho, 0.0), self.gamma)

    def compute_flux(self, rho: Array, v: Array) -> Tuple[Array, Array]:
        """
        Compute conservative fluxes for the Euler equations.

        F_mass = ρv
        F_momentum = ρv² + P
        """
        P = self.equation_of_state(rho)
        f_mass = rho * v
        f_momentum = rho * v ** 2 + P
        return f_mass, f_momentum

    def viscous_term(self, v: Array, mu: float) -> Array:
        """
        Viscous diffusion: μ ∂²v/∂x².
        """
        return mu * (jnp.roll(v, -1) - 2 * v + jnp.roll(v, 1)) / (self.dx ** 2)

    def step(
        self,
        rho: Array,
        v: Array,
        mu: Optional[float] = None,
        source: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Single Lax-Friedrichs step of the 1-D Navier-Stokes equations.

        Parameters
        ----------
        rho : Array, shape (n_levels,)
            Density (volume) field.
        v : Array, shape (n_levels,)
            Velocity (order flow) field.
        mu : float, optional
            Viscosity. Falls back to base_viscosity.
        source : Array, optional
            External forcing term (e.g., new limit orders arriving).

        Returns
        -------
        (rho_new, v_new) : Tuple[Array, Array]
        """
        if mu is None:
            mu = self.base_viscosity
        if source is None:
            source = jnp.zeros_like(v)

        f_mass, f_mom = self.compute_flux(rho, v)

        # Lax-Friedrichs for mass
        rho_avg = 0.5 * (jnp.roll(rho, -1) + jnp.roll(rho, 1))
        flux_diff_mass = (jnp.roll(f_mass, -1) - jnp.roll(f_mass, 1)) / (2 * self.dx)
        rho_new = rho_avg - self.dt * flux_diff_mass

        # Lax-Friedrichs for momentum
        momentum = rho * v
        mom_avg = 0.5 * (jnp.roll(momentum, -1) + jnp.roll(momentum, 1))
        flux_diff_mom = (jnp.roll(f_mom, -1) - jnp.roll(f_mom, 1)) / (2 * self.dx)
        mom_new = mom_avg - self.dt * flux_diff_mom + self.dt * (
            self.viscous_term(v, mu) * rho + source
        )

        # Recover velocity from momentum
        rho_safe = jnp.maximum(rho_new, 1e-10)
        v_new = mom_new / rho_safe

        return rho_new, v_new

    def evolve(
        self,
        rho0: Array,
        v0: Array,
        n_steps: int,
        mu: Optional[float] = None,
    ) -> Tuple[Array, Array]:
        """Evolve the fluid state for n_steps."""
        def body_fn(i, state):
            rho, v = state
            return self.step(rho, v, mu=mu)
        return jax.lax.fori_loop(0, n_steps, body_fn, (rho0, v0))
