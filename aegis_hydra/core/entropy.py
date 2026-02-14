"""
aegis_hydra.core.entropy — Information-Theoretic Measures

Computes Shannon and Tsallis entropy over probability distributions
and order-book microstructure. These feed the Entropic Swarm agents
with signals about market uncertainty and information flow.

Mathematical Foundation:
    - Shannon:  S = -Σ p_i ln(p_i)
    - Tsallis:  S_q = (1 - Σ p_i^q) / (q - 1)    (q → 1 recovers Shannon)
    - KL Divergence:  D_KL(P || Q) = Σ p_i ln(p_i / q_i)

Dependencies: jax, jaxlib
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


# ---------------------------------------------------------------------------
# Shannon Entropy
# ---------------------------------------------------------------------------
class ShannonEntropy(eqx.Module):
    """
    Shannon entropy calculator over discrete probability distributions.

    S(P) = -Σ p_i log(p_i)

    Parameters
    ----------
    base : str
        Logarithm base — 'e' (nats), '2' (bits), '10'.
    epsilon : float
        Small constant to avoid log(0).
    """

    base: str = "e"
    epsilon: float = 1e-12

    def _log(self, x: Array) -> Array:
        if self.base == "2":
            return jnp.log2(x)
        elif self.base == "10":
            return jnp.log10(x)
        return jnp.log(x)

    def __call__(self, p: Array) -> Array:
        """
        Compute Shannon entropy of distribution p.

        Parameters
        ----------
        p : Array, shape (..., N)
            Probability distribution (must sum to 1 along last axis).

        Returns
        -------
        Array
            Scalar entropy value(s).
        """
        p = jnp.clip(p, self.epsilon, 1.0)
        return -jnp.sum(p * self._log(p), axis=-1)

    def conditional(self, p_joint: Array) -> Array:
        """
        Conditional entropy H(X|Y) from a joint distribution.

        Parameters
        ----------
        p_joint : Array, shape (N, M)
            Joint probability table P(X, Y).

        Returns
        -------
        Array
            Conditional entropy H(X|Y).
        """
        h_joint = self(p_joint.ravel())
        p_y = jnp.sum(p_joint, axis=0)
        h_y = self(p_y)
        return h_joint - h_y


# ---------------------------------------------------------------------------
# Tsallis Entropy (Nonextensive)
# ---------------------------------------------------------------------------
class TsallisEntropy(eqx.Module):
    """
    Tsallis (non-extensive) entropy — captures fat-tail / power-law behavior.

    S_q(P) = (1 - Σ p_i^q) / (q - 1)

    When q → 1, reduces to Shannon entropy.

    Parameters
    ----------
    q : float
        Tsallis index. q > 1 → sub-additive (fat tails).
    epsilon : float
        Numerical safety floor.
    """

    q: float = 1.5
    epsilon: float = 1e-12

    def __call__(self, p: Array) -> Array:
        """
        Compute Tsallis entropy.

        Parameters
        ----------
        p : Array, shape (..., N)
            Probability distribution.

        Returns
        -------
        Array
            Tsallis entropy value(s).
        """
        p = jnp.clip(p, self.epsilon, 1.0)
        return (1.0 - jnp.sum(p ** self.q, axis=-1)) / (self.q - 1.0 + self.epsilon)


# ---------------------------------------------------------------------------
# Entropy Field (spatial entropy over order book levels)
# ---------------------------------------------------------------------------
class EntropyField(eqx.Module):
    """
    Computes a spatially-resolved entropy field over order book price levels.

    Converts bid/ask volume distributions into a 1-D entropy landscape
    that the Entropic Swarm agents use as their potential field.

    Parameters
    ----------
    n_levels : int
        Number of price levels to consider on each side.
    q : float
        Tsallis index (1.0 = Shannon).
    window : int
        Rolling window size for temporal smoothing.
    """

    n_levels: int = 50
    q: float = 1.0
    window: int = 20

    def from_order_book(
        self,
        bids: Array,
        asks: Array,
    ) -> Array:
        """
        Compute entropy field from raw bid/ask volume arrays.

        Parameters
        ----------
        bids : Array, shape (n_levels,)
            Volume at each bid price level.
        asks : Array, shape (n_levels,)
            Volume at each ask price level.

        Returns
        -------
        Array, shape (2 * n_levels,)
            Entropy density at each level (bids then asks).
        """
        volumes = jnp.concatenate([bids, asks])
        total = jnp.sum(volumes) + 1e-30
        p = volumes / total

        if self.q == 1.0:
            calculator = ShannonEntropy()
        else:
            calculator = TsallisEntropy(q=self.q)

        # Sliding-window local entropy
        def local_entropy(i):
            half_w = self.window // 2
            start = jnp.maximum(i - half_w, 0)
            end = jnp.minimum(i + half_w + 1, p.shape[0])
            local_p = jax.lax.dynamic_slice(p, (start,), (end - start,))
            local_p = local_p / (jnp.sum(local_p) + 1e-30)
            return calculator(local_p)

        indices = jnp.arange(volumes.shape[0])
        return jax.vmap(local_entropy)(indices)

    def kl_divergence(self, p: Array, q: Array) -> Array:
        """
        KL divergence D_KL(P || Q).

        Parameters
        ----------
        p, q : Array
            Probability distributions (same shape).

        Returns
        -------
        Array
            Scalar KL divergence.
        """
        eps = 1e-12
        p = jnp.clip(p, eps, 1.0)
        q_arr = jnp.clip(q, eps, 1.0)
        return jnp.sum(p * jnp.log(p / q_arr))
