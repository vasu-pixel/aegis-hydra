"""
aegis_hydra.agents.ising_grid — 2D Ising Model Kernel (10M Agents)

Implements a 3162x3162 grid of spins evolving via Metropolis-Hastings dynamics.
Uses Red-Black Checkerboard update scheme to allow parallel updates without
race conditions (neighbors of Red are all Black, and vice versa).

Physics:
    H = -J * sum(sigma_i * sigma_j) - h_ext * sigma_i
    P_flip = exp(-DeltaE / T)
"""

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from typing import NamedTuple, Tuple, Optional

class GridState(NamedTuple):
    spins: Array       # (H, W) int8 or float32 {-1, 1}
    energy: Array      # (H, W) local energy
    magnetization: float
    temperature: float

class IsingGrid(eqx.Module):
    height: int
    width: int
    J: float
    
    def __init__(self, height: int = 3162, width: int = 3162, J: float = 1.0):
        self.height = height
        self.width = width
        self.J = J

    def initialize(self, key: jax.random.PRNGKey, temperature: float = 1.0) -> GridState:
        """Create random spin grid."""
        # Random spins {-1, 1}
        # Bernouilli distribution p=0.5
        spins = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(self.height, self.width))
        return GridState(
            spins=spins,
            energy=jnp.zeros((self.height, self.width)),
            magnetization=jnp.mean(spins),
            temperature=temperature
        )

    def _compute_local_field(self, spins: Array) -> Array:
        """
        Convolution to get sum of neighbors.
        Kernel:
          [0, 1, 0]
          [1, 0, 1]
          [0, 1, 0]
        """
        # Periodic boundary conditions (Toroidal grid)
        # We can pad or use 'wrap' boundary in convolve2d if available in jax.scipy.signal
        # jax.scipy.signal.convolve2d supports 'wrap' boundary? 
        # Check docs/knowledge... JAX convolve2d supports 'fill', 'wrap', 'symm'.
        # Actually standard scipy.signal.convolve2d supports 'wrap'. JAX's might be more limited.
        # JAX scipy.signal.convolve2d: "boundary: ... 'wrap' is not supported in all backends or versions properly sometimes"
        # Safe bet: Pad manually or use 'circulant'. 
        # Actually, let's just use jnp.roll for simplicity and speed on GPU.
        
        up = jnp.roll(spins, -1, axis=0)
        down = jnp.roll(spins, 1, axis=0)
        left = jnp.roll(spins, -1, axis=1)
        right = jnp.roll(spins, 1, axis=1)
        
        neighbor_sum = up + down + left + right
        return neighbor_sum

    def step(
        self,
        state: GridState,
        external_field: float, # Scalar h_ext (e.g. from price trend)
        key: jax.random.PRNGKey,
        J: float = 1.0,        # Dynamic Coupling
    ) -> GridState:
        """
        Perform one Metropolis-Hastings step using Red-Black update.
        """
        spins = state.spins
        T = state.temperature
        
        keys = jax.random.split(key, 2)
        
        # We define a helper for the update logic to reuse for Red and Black
        def update_color(s_grid, mask, k):
            # 1. Calculate Local Field (Perception)
            neighbor_sum = self._compute_local_field(s_grid)
            total_field = J * neighbor_sum + external_field
            
            # Delta E if we flip sigma -> -sigma
            # Energy = -Field * sigma
            # Delta E = (-Field * -sigma) - (-Field * sigma) = 2 * Field * sigma
            delta_E = 2.0 * s_grid * total_field
            
            # 2. Decision (Metropolis)
            # Flip if DeltaE < 0 (Energy decreases) 
            # OR with prob exp(-DeltaE / T)
            
            # Log probability
            # We want P(accept). 
            # If delta_E < 0, transition prob is 1.
            # If delta_E > 0, transition prob is exp(-delta_E / T)
            
            # We can use uniform random number u ~ [0, 1]
            # Flip if u < exp(-delta_E / T)
            # Taking log: log(u) < -delta_E / T
            # T * log(u) < -delta_E
            # delta_E + T * log(u) < 0
            
            noise = jax.random.uniform(k, shape=s_grid.shape)
            # Avoid log(0)
            log_noise = jnp.log(noise + 1e-20)
            
            should_flip = (delta_E + T * log_noise) < 0
            
            # Apply mask (only update Red or Black pixels)
            should_flip = should_flip & mask
            
            new_spins = jnp.where(should_flip, -s_grid, s_grid)
            return new_spins

        # Checkerboard Masks
        # (0,0) is Red (sum=0 even)
        # (0,1) is Black (sum=1 odd)
        indices_y, indices_x = jnp.indices((self.height, self.width))
        phase = (indices_y + indices_x) % 2
        mask_red = (phase == 0)
        mask_black = (phase == 1)
        
        # Valid Red-Black update:
        # Update Red cells (using fixed Black neighbors)
        spins_red_updated = update_color(spins, mask_red, keys[0])
        
        # Update Black cells (using new Red neighbors)
        spins_final = update_color(spins_red_updated, mask_black, keys[1])
        
        return GridState(
            spins=spins_final,
            energy=jnp.zeros_like(spins), # Placeholder, expensive to compute full Hamiltonian every step
            magnetization=jnp.mean(spins_final),
            temperature=T
        )
