"""
aegis_hydra.agents — THE SWARM ("The Particles")

1,000,000 mathematical agents organized into three species:
    - Brownian Swarm (400k)  — Price diffusion via SDEs
    - Entropic Swarm (300k)  — Information flow via entropy measures
    - Hamiltonian Swarm (300k) — Chaos detection via Lyapunov analysis

All agents are vectorized via JAX vmap — no Python for-loops.
"""

from .base_solver import BaseSolver, AgentState
from .brownian_swarm import BrownianSwarm
from .entropic_swarm import EntropicSwarm
from .hamiltonian_swarm import HamiltonianSwarm
from .population import Population

__all__ = [
    "BaseSolver",
    "AgentState",
    "BrownianSwarm",
    "EntropicSwarm",
    "HamiltonianSwarm",
    "Population",
]
