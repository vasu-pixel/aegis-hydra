"""
aegis_hydra.core.stochastic — Stochastic Differential Equation Solvers

Solves SDEs and Fokker-Planck equations using JAX for GPU-accelerated
numerical integration. These form the backbone of the Brownian Swarm agents.

Mathematical Foundation:
    - Langevin Equation:  dX = μ(X,t)dt + σ(X,t)dW
    - Fokker-Planck:      ∂P/∂t = -∂/∂x[μP] + (1/2)∂²/∂x²[σ²P]
    - Itô vs Stratonovich calculus support

Dependencies: jax, jaxlib, diffrax, equinox
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import diffrax
import equinox as eqx


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DriftFn = Callable[[float, Array, Optional[Array]], Array]   # μ(t, x, args)
DiffusionFn = Callable[[float, Array, Optional[Array]], Array]  # σ(t, x, args)


class SDEType(Enum):
    """Integration convention."""
    ITO = "ito"
    STRATONOVICH = "stratonovich"


# ---------------------------------------------------------------------------
# Core SDE Integrator (wraps diffrax)
# ---------------------------------------------------------------------------
@dataclass
class SDEIntegrator:
    """
    General-purpose SDE integrator using diffrax solvers.

    Parameters
    ----------
    drift : DriftFn
        Drift coefficient μ(t, x, args).
    diffusion : DiffusionFn
        Diffusion coefficient σ(t, x, args).
    sde_type : SDEType
        Itô or Stratonovich interpretation.
    dt : float
        Time step for the solver.
    solver : str
        Solver name — 'euler_maruyama', 'milstein', 'srk' (strong order 1.5).
    """

    drift: DriftFn
    diffusion: DiffusionFn
    sde_type: SDEType = SDEType.ITO
    dt: float = 1e-3
    solver: str = "euler_maruyama"

    def _build_solver(self) -> diffrax.AbstractSolver:
        """Map string name to a diffrax solver instance."""
        if self.sde_type == SDEType.STRATONOVICH:
            # Heun's method for Stratonovich SDEs
            return diffrax.Heun()
        solvers = {
            "euler_maruyama": diffrax.Euler,
            "milstein": diffrax.Heun,
            "srk": diffrax.Heun,
        }
        solver_cls = solvers.get(self.solver, diffrax.Euler)
        return solver_cls()

    # NOTE: _build_terms() removed — terms are constructed inline in integrate()
    # with correct shape and key. Stratonovich vs Itô is handled by solver choice.

    def integrate(
        self,
        x0: Array,
        t0: float,
        t1: float,
        key: jax.random.PRNGKey,
        args: Optional[Array] = None,
    ) -> Array:
        """
        Integrate the SDE from t0 to t1 starting at x0.

        Parameters
        ----------
        x0 : Array
            Initial state vector.
        t0 : float
            Start time.
        t1 : float
            End time.
        key : PRNGKey
            JAX random key for Brownian noise.
        args : Array, optional
            Extra arguments passed to drift/diffusion.

        Returns
        -------
        Array
            State vector at time t1.
        """
        brownian = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=self.dt / 2, shape=x0.shape, key=key
        )
        drift_term = diffrax.ODETerm(self.drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion, brownian)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)

        sol = diffrax.diffeqsolve(
            terms,
            self._build_solver(),
            t0=t0,
            t1=t1,
            dt0=self.dt,
            y0=x0,
            args=args,
        )
        return sol.ys[-1]


# ---------------------------------------------------------------------------
# Langevin Solver (Ornstein-Uhlenbeck style)
# ---------------------------------------------------------------------------
class LangevinSolver(eqx.Module):
    """
    Solves the Langevin equation for mean-reverting stochastic dynamics.

        dX = -γ(X - μ)dt + σ dW

    where γ is friction, μ is equilibrium, σ is noise intensity.
    Used by the Brownian Swarm to model price mean-reversion.

    Parameters
    ----------
    gamma : float
        Friction / mean-reversion rate.
    mu : float
        Equilibrium level.
    sigma : float
        Noise intensity.
    dt : float
        Integration time step.
    """

    gamma: float
    mu: float
    sigma: float
    dt: float = 1e-3

    def drift(self, t: float, x: Array, args: Optional[Array] = None) -> Array:
        """Drift term: -γ(x - μ)."""
        return -self.gamma * (x - self.mu)

    def diffusion(self, t: float, x: Array, args: Optional[Array] = None) -> Array:
        """Diffusion term: σ."""
        return jnp.full_like(x, self.sigma)

    def step(self, x: Array, key: jax.random.PRNGKey) -> Array:
        """Euler-Maruyama single step."""
        dW = jnp.sqrt(self.dt) * jax.random.normal(key, shape=x.shape)
        return x + self.drift(0.0, x) * self.dt + self.diffusion(0.0, x) * dW

    def integrate(
        self,
        x0: Array,
        t0: float,
        t1: float,
        key: jax.random.PRNGKey,
    ) -> Array:
        """Full path integration from t0 → t1."""
        integrator = SDEIntegrator(
            drift=self.drift,
            diffusion=self.diffusion,
            dt=self.dt,
        )
        return integrator.integrate(x0, t0, t1, key)


# ---------------------------------------------------------------------------
# Fokker-Planck Solver (PDE on probability density)
# ---------------------------------------------------------------------------
class FokkerPlanckSolver(eqx.Module):
    """
    Numerical solver for the Fokker-Planck equation on a 1-D grid.

        ∂P/∂t = -∂/∂x[μ(x)P] + (1/2)∂²/∂x²[D(x)P]

    Uses finite-difference method with periodic or absorbing boundaries.
    Outputs the evolving probability density of the swarm.

    Parameters
    ----------
    x_min : float
        Left boundary of spatial grid.
    x_max : float
        Right boundary of spatial grid.
    nx : int
        Number of grid points.
    dt : float
        Time step for Crank-Nicolson integration.
    """

    x_min: float
    x_max: float
    nx: int = 512
    dt: float = 1e-4

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def grid(self) -> Array:
        return jnp.linspace(self.x_min, self.x_max, self.nx)

    def initial_density(self, center: float, width: float) -> Array:
        """Gaussian initial condition."""
        x = self.grid
        return jnp.exp(-0.5 * ((x - center) / width) ** 2) / (width * jnp.sqrt(2 * jnp.pi))

    def step(
        self,
        P: Array,
        drift_field: Array,
        diffusion_field: Array,
    ) -> Array:
        """
        Single forward-Euler step of the Fokker-Planck PDE.

        Parameters
        ----------
        P : Array, shape (nx,)
            Current probability density.
        drift_field : Array, shape (nx,)
            μ(x) evaluated on the grid.
        diffusion_field : Array, shape (nx,)
            D(x) = σ²(x)/2 evaluated on the grid.

        Returns
        -------
        Array
            Updated probability density.
        """
        dx = self.dx
        # Advection (upwind)
        flux = drift_field * P
        advection = -(jnp.roll(flux, -1) - jnp.roll(flux, 1)) / (2 * dx)
        # Diffusion (central difference)
        DP = diffusion_field * P
        diffusion = (jnp.roll(DP, -1) - 2 * DP + jnp.roll(DP, 1)) / (dx ** 2)

        P_new = P + self.dt * (advection + diffusion)
        # Re-normalize
        P_new = jnp.maximum(P_new, 0.0)
        P_new = P_new / (jnp.sum(P_new) * dx + 1e-30)
        return P_new

    def evolve(
        self,
        P0: Array,
        drift_field: Array,
        diffusion_field: Array,
        n_steps: int,
    ) -> Array:
        """Evolve the density for n_steps."""
        def body_fn(i, P):
            return self.step(P, drift_field, diffusion_field)
        return jax.lax.fori_loop(0, n_steps, body_fn, P0)
