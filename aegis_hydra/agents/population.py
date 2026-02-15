"""
aegis_hydra.agents.population — The "God Class" Orchestrator

Manages the full population of 1,000,000 agents across all three species.
Uses JAX vmap/pmap for GPU-accelerated parallel updates. No Python loops.

This is the single entry point that main.py calls to step the entire swarm.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from .base_solver import AgentState
from .brownian_swarm import BrownianSwarm
from .entropic_swarm import EntropicSwarm
from .hamiltonian_swarm import HamiltonianSwarm
from .ising_grid import IsingGrid

class Population(eqx.Module):
    """
    Orchestrator for the full 1M+ agent swarm.
    """

    brownian: BrownianSwarm
    entropic: EntropicSwarm
    hamiltonian: HamiltonianSwarm
    ising: IsingGrid
    weights: Dict[str, float]
    threshold: float

    @staticmethod
    def default(
        n_brownian: int = 400_000,
        n_entropic: int = 300_000,
        n_hamiltonian: int = 300_000,
        dt: float = 1e-3,
        coupling: float = 1.0,
        threshold: float = 0.7,
        grid_size: int = 3162,
    ) -> "Population":
        """Factory with default hyperparameters."""
        return Population(
            brownian=BrownianSwarm(n_agents=n_brownian, dt=dt),
            entropic=EntropicSwarm(n_agents=n_entropic, dt=dt),
            hamiltonian=HamiltonianSwarm(n_agents=n_hamiltonian, dt=dt),
            ising=IsingGrid(height=grid_size, width=grid_size, J=coupling),
            weights={
                "brownian": 0.1,
                "entropic": 0.1,
                "hamiltonian": 0.1,
                "ising": 0.7, # Dominant force
            },
            threshold=threshold,
        )

    def initialize(
        self, key: jax.random.PRNGKey, temperature: float = 2.27
    ) -> Dict[str, Any]:
        """
        Initialize all species.
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {
            "brownian": self.brownian.initialize(k1),
            "entropic": self.entropic.initialize(k2),
            "hamiltonian": self.hamiltonian.initialize(k3),
            "ising": self.ising.initialize(k4, temperature=temperature),
        }

    def step(
        self,
        states: Dict[str, Any],
        prev_flow_ema: jax.Array,
        price_history: jax.Array,
        key: jax.random.PRNGKey,
        coupling: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Advance all agents.
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Calculate social field (Magnetization) from previous step
        # M = mean signal of brownian swarm
        M_prev = states["ising"].magnetization
        
        # External Field for Ising
        # We use Imbalance (index 3) + Price Velocity (estimated? or from Tensor if avail)
        # market_tensor[3] is imbalance.
        # Let's use imbalance as h.
        imbalance = market_tensor[3]
        h_ext = imbalance * 1.0 # Scale factor

        return {
            "brownian": self.brownian.step(states["brownian"], market_tensor, k1, magnetization=M_prev),
            "entropic": self.entropic.step(states["entropic"], market_tensor, k2),
            "hamiltonian": self.hamiltonian.step(states["hamiltonian"], market_tensor, k3),
            "ising": self.ising.step(states["ising"], h_ext, k4, J=coupling),
        }

    def aggregate(
        self, states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produce a unified signal.
        """
        b_agg = self.brownian.aggregate(states["brownian"])
        e_agg = self.entropic.aggregate(states["entropic"])
        h_agg = self.hamiltonian.aggregate(states["hamiltonian"])
        
        # Ising Magnetization
        m_ising = states["ising"].magnetization

        # Composite signal
        composite = (
            self.weights["brownian"] * b_agg["mean_signal"]
            + self.weights["entropic"] * e_agg["entropy_gradient"]
            + self.weights["hamiltonian"] * (-h_agg["mean_lyapunov"])
            + self.weights["ising"] * m_ising
        )

        # 4. Regime Detection (Ising Model Phase Transition)
        M = m_ising
        
        # Phase Transition Logic
        is_critical = jnp.abs(M) > self.threshold
        regime = jnp.where(is_critical, 1.0, 0.0)

        return {
            "brownian": b_agg,
            "entropic": e_agg,
            "hamiltonian": h_agg,
            "composite_signal": composite,
            "chaos_fraction": h_agg["chaos_fraction"],
            "market_entropy": e_agg["mean_entropy"],
            "magnetization": M,
            "regime": regime,
        }

    @property
    def total_agents(self) -> int:
        return (
            self.brownian.n_agents
            + self.entropic.n_agents
            + self.hamiltonian.n_agents
            + (self.ising.height * self.ising.width)
        )
