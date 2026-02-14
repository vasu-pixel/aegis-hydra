"""
aegis_hydra.agents.hamiltonian_swarm — Lyapunov / Chaos Agent Species (300k)

Each agent evolves a point in market phase space using Hamiltonian dynamics.
By measuring divergence of nearby trajectories (Lyapunov exponents),
agents detect chaos vs. stability.

Signal: Lyapunov exponent (positive = chaotic/dangerous, negative = stable).

Dependencies: jax, core.mechanics
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from .base_solver import BaseSolver, AgentState
from ..core.mechanics import HamiltonianSolver


class HamiltonianSwarm(BaseSolver):
    """
    300,000 Hamiltonian agents performing symplectic dynamics.

    Each agent tracks a trajectory in phase space (price, momentum).
    The collective divergence of trajectories signals whether the market
    is in a chaotic or stable regime.

    Parameters
    ----------
    n_agents : int
        Number of Hamiltonian particles (default: 300_000).
    dt : float
        Time step.
    state_dim : int
        Phase-space dimensionality (position only; momentum is separate).
    mass : float
        Effective market inertia.
    perturbation_scale : float
        Scale of perturbation for Lyapunov estimation.
    leapfrog_steps : int
        Number of leapfrog sub-steps per macro step.
    """

    n_agents: int = 300_000
    dt: float = 1e-3
    state_dim: int = 1
    mass: float = 1.0
    perturbation_scale: float = 1e-5
    leapfrog_steps: int = 10

    def _make_potential(self, market_tensor: Array):
        """
        Build a market-modulated potential energy function.

        The base shape is a double-well V(q) = (q² - 1)², but the
        equilibrium center and well depth are shifted by live market data:
            - market_tensor[0] (mid_price) shifts the center
            - market_tensor[2] (volatility) scales the well depth
            - market_tensor[3] (imbalance) tilts the potential asymmetrically

        Returns a closure V(q) → scalar suitable for HamiltonianSolver.
        """
        center = market_tensor[0]          # mid price
        vol = market_tensor[2]             # volatility
        imbalance = market_tensor[3]       # order book imbalance [-1, 1]
        # Well depth inversely related to volatility (high vol → shallower wells)
        depth = 1.0 / (1.0 + vol + 1e-10)

        def potential(q: Array) -> Array:
            q_centered = q - center
            # Double-well with market-driven depth + asymmetric tilt
            return jnp.sum(depth * (q_centered ** 2 - 1.0) ** 2 - imbalance * q_centered)

        return potential

    def _kinetic(self, p: Array) -> Array:
        """Standard kinetic energy T = p²/(2m)."""
        return jnp.sum(p ** 2) / (2.0 * self.mass)

    def initialize(self, key: jax.random.PRNGKey) -> AgentState:
        """Scatter agents in phase space."""
        k1, k2 = jax.random.split(key)
        shape = (self.n_agents, self.state_dim)
        return AgentState(
            position=jax.random.normal(k1, shape) * 0.5,
            momentum=jax.random.normal(k2, shape) * 0.1,
            energy=jnp.zeros(self.n_agents),
            signal=jnp.zeros(self.n_agents),
            metadata={},
        )

    def step(
        self,
        state: AgentState,
        market_tensor: Array,
        key: jax.random.PRNGKey,
    ) -> AgentState:
        """
        Evolve all Hamiltonian agents by one macro step.

        Also estimates local Lyapunov exponents by comparing
        nearby trajectories.

        Parameters
        ----------
        market_tensor : Array
            Flat vector from MarketTensor.to_flat_vector().
            Used to modulate the potential landscape in real-time.
        """
        # Dynamic potential coupled to market state
        # V(q) = (q - (price_norm))^2
        # Agents want to be near the price, but are carried by momentum
        # and trapped in local metadata-potential wells
        mid_price = market_tensor[0]
        # Normalize price for the potential (simple centering for now)
        # In a real model, this would be more complex
        center = 0.0 # mid_price is already relative in some contexts, but here we assume q is normalized
        
        # We redefine the solver's potential to be time-dependent (via closure)
        def dynamic_potential(q):
            # Standard double-well modulated by market data
            # V(q) = (q^2 - 1)^2 + coupling * (q - mid_price)
            # This is a placeholder for "market forcing"
            return jnp.sum((q ** 2 - 1.0) ** 2) + 0.1 * jnp.sum(q * mid_price)

        # Inject dynamic potential into a new solver instance or use closure
        # Since HamiltonianSolver assumes static potential in this implementation,
        # we will monkey-patch the potential function logic for this step
        # or properly: Re-instantiate solver with new potential? 
        # Better: pass potential to integrate? 
        # Current solver architecture is eqx.Module with static methods.
        # Let's use the closure method since 'integrate' calls self.step_leapfrog which calls self.potential.
        # actually, to do this correctly with Equinox, we should likely update the solver instance.
        
        solver = HamiltonianSolver(
            kinetic=self._kinetic,
            potential=dynamic_potential, # Use the dynamic one
            mass=self.mass,
            dt=self.dt,
        )

        def evolve_single(q, p, key_i):
            # Main trajectory
            q_new, p_new = solver.integrate(q, p, self.leapfrog_steps)

            # Perturbed trajectory for Lyapunov
            delta = jax.random.normal(key_i, q.shape) * self.perturbation_scale
            q_pert, p_pert = solver.integrate(q + delta, p, self.leapfrog_steps)

            # Separation
            separation = jnp.sqrt(
                jnp.sum((q_pert - q_new) ** 2 + (p_pert - p_new) ** 2)
            )
            initial_sep = jnp.sqrt(jnp.sum(delta ** 2))
            total_time = self.leapfrog_steps * self.dt

            lyapunov = jnp.log(separation / (initial_sep + 1e-30)) / (total_time + 1e-30)

            # Energy
            H = solver.hamiltonian(q_new, p_new)

            return q_new, p_new, lyapunov, H

        keys = jax.random.split(key, self.n_agents)
        q_all, p_all, lyap_all, H_all = jax.vmap(evolve_single)(
            state.position, state.momentum, keys
        )

        return AgentState(
            position=q_all,
            momentum=p_all,
            energy=H_all,
            signal=lyap_all,  # Lyapunov exponent IS the signal
            metadata={"lyapunov": lyap_all, "hamiltonian": H_all},
        )

    def aggregate(self, state: AgentState) -> Dict[str, Array]:
        """
        Reduce Hamiltonian swarm into chaos diagnostics.

        Returns
        -------
        Dict with keys:
            'mean_lyapunov'    — average Lyapunov exponent
            'chaos_fraction'   — fraction of agents in chaotic regime
            'mean_energy'      — average Hamiltonian (market "tension")
        """
        lyap = state.signal
        return {
            "mean_lyapunov": jnp.mean(lyap),
            "chaos_fraction": jnp.mean((lyap > 0).astype(jnp.float32)),
            "mean_energy": jnp.mean(state.energy),
        }
