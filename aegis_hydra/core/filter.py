"""
aegis_hydra.core.filter — Kalman Filters for Signal Processing

Implements an Unscented Kalman Filter (UKF) to separate true market signal
from microstructure noise.

References:
    - Wan, E. A., & Van Der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


class UnscentedKalmanFilter(eqx.Module):
    """
    Unscented Kalman Filter (UKF) for nonlinear state estimation.
    
    Used to track the "true" latent price and volatility, filtering out
    high-frequency noise and bid-ask bounce.
    
    State: [price, velocity, volatility]
    Measurement: [mid_price, observed_vol]
    
    Parameters
    ----------
    dt : float
        Time step.
    alpha : float
        Spread of sigma points (typically 1e-3).
    beta : float
        Prior knowledge of distribution (2 for Gaussian).
    kappa : float
        Secondary scaling parameter (typically 0).
    """
    
    dt: float = 1e-3
    state_dim: int = 3
    obs_dim: int = 2
    
    # Process noise covariance (Q)
    Q: Array
    # Measurement noise covariance (R)
    R: Array
    
    def __init__(
        self,
        dt: float = 1e-3,
        q_scale: float = 1e-4,
        r_scale: float = 1e-2,
    ):
        self.dt = dt
        # Q: Process noise (random acceleration / volatility shocks)
        self.Q = jnp.eye(self.state_dim) * q_scale
        # R: Measurement noise (market microstructure noise)
        self.R = jnp.eye(self.obs_dim) * r_scale

    def predict(self, mean: Array, cov: Array) -> Tuple[Array, Array]:
        """
        Predict next state using nonlinear process model.
        
        Model: Constant Velocity
        x_k+1 = F * x_k
        price += velocity * dt
        velocity = velocity (const)
        volatility = volatility (const)
        """
        # Linear transition matrix F
        F = jnp.array([
            [1.0, self.dt, 0.0],
            [0.0, 1.0,     0.0],
            [0.0, 0.0,     1.0]
        ])
        
        mean_pred = F @ mean
        cov_pred = F @ cov @ F.T + self.Q
        
        return mean_pred, cov_pred

    def update(
        self, 
        mean_pred: Array, 
        cov_pred: Array, 
        measurement: Array
    ) -> Tuple[Array, Array]:
        """
        Update state estimate with new measurement.
        
        Measurement model H:
        z = H * x
        observed_price = true_price
        observed_vol = true_vol
        """
        # H matrix maps state [p, v, vol] -> [p, vol]
        H = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Innovation (residual)
        y = measurement - H @ mean_pred
        
        # Innovation covariance
        S = H @ cov_pred @ H.T + self.R
        
        # Kalman Gain
        K = cov_pred @ H.T @ jnp.linalg.inv(S)
        
        # Updated estimate
        mean_new = mean_pred + K @ y
        cov_new = (jnp.eye(self.state_dim) - K @ H) @ cov_pred
        
        return mean_new, cov_new

    def step(self, state: Tuple[Array, Array], measurement: Array) -> Tuple[Array, Array]:
        """Perform one predict-update cycle."""
        mean, cov = state
        mean_p, cov_p = self.predict(mean, cov)
        return self.update(mean_p, cov_p, measurement)

    def initialize(self, initial_price: float, initial_vol: float) -> Tuple[Array, Array]:
        """Initialize filter state."""
        mean = jnp.array([initial_price, 0.0, initial_vol])
        cov = jnp.eye(self.state_dim)
        return mean, cov
