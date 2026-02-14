"""
aegis_hydra.agents.base_solver — Abstract Base Class for Mathematical Agents

Every agent in the swarm is a "solver" — a mathematical entity that
maintains internal state and updates it each time step based on the
market tensor field.

All concrete swarms (Brownian, Entropic, Hamiltonian) inherit from BaseSolver.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


# ---------------------------------------------------------------------------
# Agent State (immutable, JAX-compatible)
# ---------------------------------------------------------------------------
class AgentState(NamedTuple):
    """
    State vector for a single agent (or a batch of agents when vectorized).

    Fields
    ------
    position : Array
        Generalized position (e.g., estimated fair price).
    momentum : Array
        Generalized momentum (e.g., order flow estimate).
    energy : Array
        Internal energy / signal strength.
    signal : Array
        Output signal fed to the Governor (buy/sell/hold strength).
    metadata : Dict[str, Array]
        Arbitrary per-agent metadata (e.g., entropy value, Lyapunov exponent).
    """
    position: Array
    momentum: Array
    energy: Array
    signal: Array
    metadata: Dict[str, Array]


# ---------------------------------------------------------------------------
# Abstract Base Solver
# ---------------------------------------------------------------------------
class BaseSolver(eqx.Module):
    """
    Abstract base class for all swarm agent species.

    Each species must implement:
        - initialize()  → create initial AgentState for N agents
        - step()        → update state by one time step given market data
        - aggregate()   → reduce N agent signals into a single summary

    Parameters
    ----------
    n_agents : int
        Number of agents in this species.
    dt : float
        Simulation time step.
    state_dim : int
        Dimensionality of position/momentum vectors.
    """

    n_agents: int
    dt: float
    state_dim: int

    @abstractmethod
    def initialize(self, key: jax.random.PRNGKey) -> AgentState:
        """
        Create initial state for all agents in this species.

        Parameters
        ----------
        key : PRNGKey
            Random key for initialization.

        Returns
        -------
        AgentState
            Batched state with arrays of shape (n_agents, state_dim).
        """
        ...

    @abstractmethod
    def step(
        self,
        state: AgentState,
        market_tensor: Array,
        key: jax.random.PRNGKey,
    ) -> AgentState:
        """
        Advance all agents by one time step.

        Parameters
        ----------
        state : AgentState
            Current batched state.
        market_tensor : Array
            Tensor field from market/tensor_field.py.
        key : PRNGKey
            Random key for stochastic updates.

        Returns
        -------
        AgentState
            Updated batched state.
        """
        ...

    @abstractmethod
    def aggregate(self, state: AgentState) -> Dict[str, Array]:
        """
        Reduce all agent signals into a summary for the Governor.

        Returns
        -------
        Dict[str, Array]
            Keys might include 'mean_signal', 'variance', 'consensus', etc.
        """
        ...

    def _default_state(self, key: jax.random.PRNGKey) -> AgentState:
        """Utility: create zero-initialized AgentState."""
        shape = (self.n_agents, self.state_dim)
        return AgentState(
            position=jax.random.normal(key, shape) * 0.01,
            momentum=jnp.zeros(shape),
            energy=jnp.ones((self.n_agents,)),
            signal=jnp.zeros((self.n_agents,)),
            metadata={},
        )
