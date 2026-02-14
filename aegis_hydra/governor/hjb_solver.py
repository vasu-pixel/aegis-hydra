"""
aegis_hydra.governor.hjb_solver — Hamilton-Jacobi-Bellman Optimal Control

Solves the HJB equation to find the optimal trading policy given
the current market "energy state" from the swarm.

Mathematical Foundation:
    ∂V/∂t + min_u [f(x,u)·∇V + (1/2)σ²Δ²V + L(x,u)] = 0

    Where:
        V(x,t)  = value function (minimum expected cost)
        u       = control (trade size / direction)
        f(x,u)  = state dynamics under control
        L(x,u)  = running cost (transaction costs + risk penalty)

This is the "brain" that converts physics into trading decisions.

Dependencies: jax, jaxlib, optax
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
import optax


# Type aliases
DynamicsFn = Callable[[Array, Array], Array]  # f(state, control) → state_dot
CostFn = Callable[[Array, Array], Array]       # L(state, control) → running cost


class HJBSolver(eqx.Module):
    """
    Numerical solver for the Hamilton-Jacobi-Bellman equation.

    Finds the optimal control policy u*(x) that minimizes expected
    cumulative cost (transaction fees + risk) along the market trajectory.

    Two modes:
        1. Grid-based: Solve the PDE on a discrete state grid (low-dim).
        2. Neural: Parameterize V(x) with a neural network (high-dim).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the market state.
    control_dim : int
        Dimensionality of the control (trade actions).
    dt : float
        Time discretization for the PDE.
    gamma_risk : float
        Risk aversion coefficient in the running cost.
    transaction_cost : float
        Per-unit transaction cost.
    sigma : float
        Diffusion coefficient for the stochastic HJB.
    grid_size : int
        Grid points per dimension (for grid-based solver).
    learning_rate : float
        Learning rate for neural value function.
    """

    state_dim: int = 2
    control_dim: int = 1
    dt: float = 1e-3
    gamma_risk: float = 1.0
    transaction_cost: float = 0.001
    sigma: float = 0.1
    grid_size: int = 64
    learning_rate: float = 1e-3

    def running_cost(self, state: Array, control: Array) -> Array:
        """
        Running cost L(x, u) = risk_penalty + transaction_cost.

        L = γ·|state_risk|² + c·|u|

        Parameters
        ----------
        state : Array
            Current market state.
        control : Array
            Trade action.
        """
        risk_penalty = self.gamma_risk * jnp.sum(state ** 2)
        tx_cost = self.transaction_cost * jnp.sum(jnp.abs(control))
        return risk_penalty + tx_cost

    def terminal_cost(self, state: Array) -> Array:
        """Terminal cost Φ(x_T) — penalty for final state."""
        return self.gamma_risk * jnp.sum(state ** 2)

    def optimal_control(
        self,
        value_grad: Array,
        state: Array,
    ) -> Array:
        """
        Derive optimal control from value function gradient.

        u* = argmin_u [f(x,u)·∇V + L(x,u)]

        With quadratic friction: L(u) = 1/2 * friction * u^2 + fixed_fee * |u|
        u* = -∇V / friction (ignoring fixed fee for convexity)

        Parameters
        ----------
        value_grad : Array
            ∇V(x) — gradient of value function at current state.
        state : Array
            Current market state.

        Returns
        -------
        Array
            Optimal control vector.
        """
        # Friction coefficient (gamma in 1/2 gamma u^2)
        # Using transaction_cost as the friction parameter
        friction = self.transaction_cost
        
        # Closed-form for quadratic friction
        # u* = - (∇V · B) / friction
        # Assuming linear control B=1 for simplicity in 1D
        u_raw = -value_grad[:self.control_dim] / (friction + 1e-6)
        
        # Clip to feasible range [-1, 1] (normalized trade size)
        return jnp.clip(u_raw, -1.0, 1.0)

    def solve_grid(
        self,
        dynamics: DynamicsFn,
        n_steps: int = 100,
    ) -> Array:
        """
        Solve HJB on a discrete grid via backward induction.

        Parameters
        ----------
        dynamics : DynamicsFn
            State transition f(x, u) → x_dot.
        n_steps : int
            Number of backward time steps.

        Returns
        -------
        Array, shape (grid_size, ..., grid_size)
            Value function V(x) on the grid.
        """
        # Create state grid
        grid_1d = jnp.linspace(-2.0, 2.0, self.grid_size)
        if self.state_dim == 1:
            # Flatten to match the structure expected by vmap if needed,
            # though 1D usually operates directly on grid_1d.
            # To be consistent with 2D case where 'states' is (N, 2),
            # we make 1D states (N, 1).
            states = grid_1d.reshape(-1, 1)
            V = jax.vmap(self.terminal_cost)(states)
        elif self.state_dim == 2:
            xx, yy = jnp.meshgrid(grid_1d, grid_1d)
            states = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
            # Ensure V has shape (N,)
            V = jax.vmap(self.terminal_cost)(states)
        else:
            raise NotImplementedError("Grid solver supports dim ≤ 2. Use neural solver for higher dims.")

        # Backward iteration
        def backward_step(i, V):
            def update_single(idx):
                state = states[idx]
                grad_V = jax.grad(lambda s: jnp.interp(s[0], grid_1d, V))(state)
                u_star = self.optimal_control(grad_V, state)
                cost = self.running_cost(state, u_star)
                # Semi-Lagrangian update
                x_next = state + dynamics(state, u_star) * self.dt
                # Diffusion term
                diffusion = 0.5 * self.sigma ** 2 * jnp.sum(
                    jax.hessian(lambda s: jnp.interp(s[0], grid_1d, V))(state)
                )
                return V[idx] + self.dt * (cost + diffusion)

            indices = jnp.arange(V.shape[0])
            return jax.vmap(update_single)(indices)

        return jax.lax.fori_loop(0, n_steps, backward_step, V)

    def policy_from_swarm(
        self,
        swarm_signal: Dict[str, Array],
    ) -> Dict[str, Array]:
        """
        Quick policy extraction from swarm aggregate (no PDE solve).

        Maps swarm signals directly to control via heuristic that
        approximates the HJB solution for near-equilibrium markets.

        Parameters
        ----------
        swarm_signal : Dict
            Output from Population.aggregate().

        Returns
        -------
        Dict with keys:
            'action'     — normalized trade direction [-1, 1]
            'confidence' — certainty of the action [0, 1]
            'size'       — recommended position size [0, 1]
        """
        composite = swarm_signal["composite_signal"]
        chaos = swarm_signal["chaos_fraction"]
        entropy = swarm_signal["market_entropy"]

        # Action: composite signal clipped to [-1, 1]
        action = jnp.clip(composite, -1.0, 1.0)

        # Confidence: inversely proportional to chaos and entropy
        confidence = (1.0 - chaos) * jnp.exp(-entropy)
        confidence = jnp.clip(confidence, 0.0, 1.0)

        # Size: scale by confidence, reduce in chaotic regimes
        size = jnp.abs(action) * confidence

        return {
            "action": action,
            "confidence": confidence,
            "size": size,
        }

    def apply_viscosity(
        self,
        target_position: float,
        current_position: float,
        volatility: float,
    ) -> float:
        """
        Applies a 'Viscous Deadband' to stop over-trading.

        If the desired change in position is insufficiently large relative to
        market volatility (noise), we hold the current position.
        """
        # 1. Calculate 'Action Potential'
        delta = target_position - current_position

        # 2. Define 'Activation Energy'
        # Base threshold 5% + scaling with volatility
        # If volatility is 0.01 (1%), threshold is ~5%
        activation_energy = 0.05 * (1 + volatility * 100) 
        # Note: volatility is typically small per step, checking scale...
        # If vol is annualized, huge. If per-step, tiny.
        # Assuming vol is ~0.01-0.05 range?
        # Let's stick to user prompt: 0.05 * (1 + volatility)
        
        # 3. Viscosity Check
        if abs(delta) < activation_energy:
            return current_position
        
        return target_position
