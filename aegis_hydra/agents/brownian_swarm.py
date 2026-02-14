"""
aegis_hydra.agents.brownian_swarm — Brownian Motion Agent Species (400k)

Each agent is a stochastic particle performing Langevin dynamics.
Their collective density approximates the Fokker-Planck probability
distribution of future prices.

Signal: Consensus of where the "probability mass" is flowing.

Dependencies: jax, core.stochastic
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from .base_solver import BaseSolver, AgentState
from ..core.stochastic import LangevinSolver


class BrownianSwarm(BaseSolver):
    """
    400,000 Brownian agents performing Langevin dynamics.

    Each agent independently samples a path of the SDE:
        dX = -γ(X - μ_market)dt + σ(market_vol)dW

    The swarm collectively estimates the probability distribution
    of future prices.

    Parameters
    ----------
    n_agents : int
        Number of Brownian particles (default: 400_000).
    dt : float
        Time step.
    state_dim : int
        Dimensionality (typically 1 for single-asset).
    gamma : float
        Mean-reversion rate.
    sigma_base : float
        Base noise intensity (scaled by market volatility).
    """

    n_agents: int = 400_000
    dt: float = 1e-3
    state_dim: int = 1
    gamma: float = 0.1  # Reduced from 1.0 to allow trend following
    sigma_base: float = 0.1
    coupling: float = 1.5 # Ising Coupling Constant (J)

    def initialize(self, key: jax.random.PRNGKey) -> AgentState:
        """Scatter agents around zero with small Gaussian noise."""
        return self._default_state(key)

    def step(
        self,
        state: AgentState,
        market_tensor: Array,
        key: jax.random.PRNGKey,
        magnetization: float = 0.0,
    ) -> AgentState:
        """
        Update all Brownian agents in parallel via vmap.

        dX = -gamma * (X - price) * dt + J * M * dt + sigma * dW
        """
        mid_price = market_tensor[0]
        # Index 1 is spread, Index 2 is volatility
        vol = market_tensor[2]

        sigma = self.sigma_base * (1.0 + vol)
        
        # Social Force (Ising Model Mean Field)
        # Force = J * M (where M is previous step's average signal)
        social_drift = self.coupling * magnetization

        # Vectorized Langevin step for all agents simultaneously
        # Split key so each agent gets independent noise
        key, subkey = jax.random.split(key)
        dW = jnp.sqrt(self.dt) * jax.random.normal(
            subkey, shape=(self.n_agents, self.state_dim)
        )

        drift = (-self.gamma * (state.position - mid_price) + social_drift) * self.dt
        diffusion = sigma * dW

        new_position = state.position + drift + diffusion

        # Signal: deviation from mid-price (positive = above, negative = below)
        new_signal = jnp.squeeze(new_position - mid_price, axis=-1)

        # Energy: kinetic energy of each particle
        velocity = (new_position - state.position) / self.dt
        new_energy = 0.5 * jnp.sum(velocity ** 2, axis=-1)

        return AgentState(
            position=new_position,
            momentum=velocity,
            energy=new_energy,
            signal=new_signal,
            metadata={"volatility_used": jnp.full((self.n_agents,), sigma)},
        )

    def aggregate(self, state: AgentState) -> Dict[str, Array]:
        """
        Reduce swarm into summary statistics.

        Returns
        -------
        Dict with keys:
            'mean_signal'  — average directional bias
            'variance'     — uncertainty / disagreement
            'skewness'     — asymmetry of the distribution
            'energy'       — total kinetic energy (market "temperature")
        """
        sig = state.signal
        mean = jnp.mean(sig)
        var = jnp.var(sig)
        skew = jnp.mean(((sig - mean) / (jnp.sqrt(var) + 1e-10)) ** 3)
        return {
            "mean_signal": mean,
            "variance": var,
            "skewness": skew,
            "energy": jnp.mean(state.energy),
        }
