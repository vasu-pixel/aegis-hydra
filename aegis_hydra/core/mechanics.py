"""
aegis_hydra.core.mechanics — Hamiltonian & Lagrangian Dynamics

Implements classical mechanics formalism applied to market dynamics.
The market state is treated as a point in phase space (position = price,
momentum = order flow), and we derive equations of motion via the
Action Principle.

Mathematical Foundation:
    - Hamiltonian:  H(q, p) = T(p) + V(q)
    - Hamilton's Eqs:  dq/dt = ∂H/∂p,  dp/dt = -∂H/∂q
    - Lagrangian:  L(q, q̇) = T(q̇) - V(q)
    - Action:  S[q] = ∫ L(q, q̇) dt
    - Lyapunov Exponents for chaos detection

Dependencies: jax, jaxlib, sympy (for symbolic derivation), equinox
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


# Type aliases
PotentialFn = Callable[[Array], Array]          # V(q)
KineticFn = Callable[[Array], Array]            # T(p)


# ---------------------------------------------------------------------------
# Hamiltonian Solver
# ---------------------------------------------------------------------------
class HamiltonianSolver(eqx.Module):
    """
    Symplectic integrator for Hamiltonian dynamics in market phase space.

    Phase space: q = generalized position (price), p = generalized momentum (order flow).
    H(q, p) = T(p) + V(q)

    Uses the Störmer-Verlet (leapfrog) integrator to preserve the symplectic
    structure, which is essential for long-time stability of the simulation.

    Parameters
    ----------
    kinetic : KineticFn
        Kinetic energy T(p). Default: p²/2m.
    potential : PotentialFn
        Potential energy V(q). Encodes the "market landscape".
    mass : float
        Effective mass (market inertia).
    dt : float
        Integration time step.
    """

    kinetic: KineticFn
    potential: PotentialFn
    mass: float = 1.0
    dt: float = 1e-3

    def hamiltonian(self, q: Array, p: Array) -> Array:
        """Total energy H = T + V."""
        return self.kinetic(p) + self.potential(q)

    def _grad_potential(self, q: Array) -> Array:
        """∂V/∂q via automatic differentiation."""
        return jax.grad(lambda q_: jnp.sum(self.potential(q_)))(q)

    def _grad_kinetic(self, p: Array) -> Array:
        """∂T/∂p via automatic differentiation."""
        return jax.grad(lambda p_: jnp.sum(self.kinetic(p_)))(p)

    def step_leapfrog(self, q: Array, p: Array) -> Tuple[Array, Array]:
        """
        Single Störmer-Verlet (leapfrog) step.

        Returns
        -------
        (q_new, p_new) : Tuple[Array, Array]
        """
        # Half-step momentum
        p_half = p - 0.5 * self.dt * self._grad_potential(q)
        # Full-step position
        q_new = q + self.dt * self._grad_kinetic(p_half)
        # Half-step momentum
        p_new = p_half - 0.5 * self.dt * self._grad_potential(q_new)
        return q_new, p_new

    def integrate(
        self,
        q0: Array,
        p0: Array,
        n_steps: int,
    ) -> Tuple[Array, Array]:
        """
        Integrate Hamilton's equations for n_steps.

        Returns
        -------
        (q_final, p_final) : Tuple[Array, Array]
        """
        def body_fn(i, state):
            q, p = state
            return self.step_leapfrog(q, p)

        return jax.lax.fori_loop(0, n_steps, body_fn, (q0, p0))

    def lyapunov_exponent(
        self,
        q0: Array,
        p0: Array,
        perturbation: float = 1e-6,
        n_steps: int = 1000,
    ) -> Array:
        """
        Estimate the maximal Lyapunov exponent via tangent-space evolution.

        A positive exponent signals chaotic (unpredictable) market dynamics.

        Parameters
        ----------
        q0, p0 : Array
            Initial phase-space point.
        perturbation : float
            Size of initial tangent vector.
        n_steps : int
            Integration horizon.

        Returns
        -------
        Array
            Estimated maximal Lyapunov exponent.
        """
        # Reference trajectory
        q_ref, p_ref = self.integrate(q0, p0, n_steps)

        # Perturbed trajectory
        delta = jnp.ones_like(q0) * perturbation
        q_pert, p_pert = self.integrate(q0 + delta, p0, n_steps)

        separation = jnp.sqrt(jnp.sum((q_pert - q_ref) ** 2 + (p_pert - p_ref) ** 2))
        initial_sep = jnp.sqrt(jnp.sum(delta ** 2))
        total_time = n_steps * self.dt

        return jnp.log(separation / (initial_sep + 1e-30)) / (total_time + 1e-30)


# ---------------------------------------------------------------------------
# Lagrangian Solver
# ---------------------------------------------------------------------------
class LagrangianSolver(eqx.Module):
    """
    Euler-Lagrange equation solver for market trajectories.

    L(q, q̇) = T(q̇) - V(q)
    d/dt (∂L/∂q̇) - ∂L/∂q = 0

    Parameters
    ----------
    potential : PotentialFn
        Potential energy V(q).
    mass : float
        Effective mass.
    dt : float
        Time step.
    """

    potential: PotentialFn
    mass: float = 1.0
    dt: float = 1e-3

    def lagrangian(self, q: Array, q_dot: Array) -> Array:
        """L = T - V = (m/2)|q̇|² - V(q)."""
        T = 0.5 * self.mass * jnp.sum(q_dot ** 2)
        V = self.potential(q)
        return T - V

    def euler_lagrange_acceleration(self, q: Array) -> Array:
        """
        Compute acceleration q̈ = -(1/m)∂V/∂q from the Euler-Lagrange equation.
        """
        grad_V = jax.grad(lambda q_: jnp.sum(self.potential(q_)))(q)
        return -grad_V / self.mass

    def step(self, q: Array, q_dot: Array) -> Tuple[Array, Array]:
        """Velocity-Verlet single step."""
        a = self.euler_lagrange_acceleration(q)
        q_new = q + q_dot * self.dt + 0.5 * a * self.dt ** 2
        a_new = self.euler_lagrange_acceleration(q_new)
        q_dot_new = q_dot + 0.5 * (a + a_new) * self.dt
        return q_new, q_dot_new

    def integrate(
        self,
        q0: Array,
        q_dot0: Array,
        n_steps: int,
    ) -> Tuple[Array, Array]:
        """Integrate the Euler-Lagrange equations for n_steps."""
        def body_fn(i, state):
            q, q_dot = state
            return self.step(q, q_dot)
        return jax.lax.fori_loop(0, n_steps, body_fn, (q0, q_dot0))


# ---------------------------------------------------------------------------
# Action Principle
# ---------------------------------------------------------------------------
class ActionPrinciple(eqx.Module):
    """
    Evaluates the Action functional S[q] = ∫ L dt along a trajectory.

    Used by the HJB solver to find minimum-action paths
    (optimal trading trajectories).

    Parameters
    ----------
    potential : PotentialFn
        Potential energy V(q).
    mass : float
        Effective mass.
    """

    potential: PotentialFn
    mass: float = 1.0

    def compute_action(
        self,
        trajectory: Array,
        dt: float,
    ) -> Array:
        """
        Evaluate the action integral along a discrete trajectory.

        Parameters
        ----------
        trajectory : Array, shape (T, D)
            Sequence of positions at each time step.
        dt : float
            Time step between points.

        Returns
        -------
        Array
            Total action S.
        """
        # Finite-difference velocities
        velocities = jnp.diff(trajectory, axis=0) / dt
        # Midpoint positions
        positions = 0.5 * (trajectory[:-1] + trajectory[1:])

        T = 0.5 * self.mass * jnp.sum(velocities ** 2, axis=-1)
        V = jax.vmap(self.potential)(positions)
        L = T - V
        return jnp.sum(L) * dt
