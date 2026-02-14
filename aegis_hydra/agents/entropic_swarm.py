"""
aegis_hydra.agents.entropic_swarm — Information Theory Agent Species (300k)

Each agent monitors a local window of the order book and computes
entropy measures. High entropy = high uncertainty = market indecision.
Low entropy = information concentration = potential breakout.

Signal: Entropy gradient (where information is flowing).

Dependencies: jax, core.entropy
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from .base_solver import BaseSolver, AgentState
from ..core.entropy import ShannonEntropy, TsallisEntropy


class EntropicSwarm(BaseSolver):
    """
    300,000 entropy-measuring agents.

    Each agent monitors a random window of the order book and computes
    local entropy. The swarm collectively maps the information landscape.

    Parameters
    ----------
    n_agents : int
        Number of entropy agents (default: 300_000).
    dt : float
        Time step.
    state_dim : int
        Number of price levels each agent observes.
    tsallis_q : float
        Tsallis index. 1.0 = Shannon, >1.0 = fat-tail sensitive.
    entropy_threshold : float
        Below this entropy, agent signals "order" (potential breakout).
    """

    n_agents: int = 300_000
    dt: float = 1e-3
    state_dim: int = 10  # each agent watches 10 price levels
    tsallis_q: float = 1.5
    entropy_threshold: float = 0.5
    # Number of scalar fields at the start of the flat tensor vector
    # (mid_price, spread, volatility, imbalance, funding_rate, timestamp)
    _scalar_offset: int = 6

    def initialize(self, key: jax.random.PRNGKey) -> AgentState:
        """Assign each agent a random observation window into the order book region."""
        state = self._default_state(key)
        # Window offsets are indices into the ORDER BOOK portion of the flat tensor,
        # which starts at _scalar_offset. We store absolute indices so that
        # dynamic_slice operates on the correct region of the flat vector.
        # maxval = 50 means agents can watch up to level 50 of the book.
        window_offsets = (
            self._scalar_offset
            + jax.random.randint(key, shape=(self.n_agents, 1), minval=0, maxval=50)
        ).astype(jnp.float32)
        return state._replace(
            position=window_offsets * jnp.ones((1, self.state_dim)),
            metadata={"assigned_window": window_offsets},
        )

    def step(
        self,
        state: AgentState,
        market_tensor: Array,
        key: jax.random.PRNGKey,
    ) -> AgentState:
        """
        Each agent computes entropy over its assigned order-book window.

        Parameters
        ----------
        market_tensor : Array
            Must contain order book volume distribution, shape (n_levels,).
        """
        tsallis = TsallisEntropy(q=self.tsallis_q)
        shannon = ShannonEntropy()

        # Each agent samples a local window of the order book
        def compute_single_agent(agent_idx):
            # Offset by 6 because indices 0-5 are scalar fields (mid_price, spread, etc.)
            # Index 6 starts the bid_density array
            start = jnp.int32(state.metadata["assigned_window"][agent_idx, 0]) + 6
            window = jax.lax.dynamic_slice(
                market_tensor, (start,), (self.state_dim,)
            )
            # Normalize to probability
            p = window / (jnp.sum(window) + 1e-30)
            s_tsallis = tsallis(p)
            s_shannon = shannon(p)
            return s_tsallis, s_shannon

        indices = jnp.arange(self.n_agents)
        s_tsallis_all, s_shannon_all = jax.vmap(compute_single_agent)(indices)

        # Signal: negative entropy gradient → information is concentrating
        signal = -(s_tsallis_all - state.energy)  # change in entropy

        return AgentState(
            position=state.position,
            momentum=state.momentum,
            energy=s_tsallis_all,
            signal=signal,
            metadata={
                **state.metadata,
                "shannon": s_shannon_all,
                "tsallis": s_tsallis_all,
            },
        )

    def aggregate(self, state: AgentState) -> Dict[str, Array]:
        """
        Reduce entropy swarm into summary.

        Returns
        -------
        Dict with keys:
            'mean_entropy'    — average disorder level
            'entropy_gradient'— rate of information concentration
            'order_fraction'  — fraction of agents seeing "order" (low entropy)
        """
        entropy = state.energy
        signal = state.signal
        order_fraction = jnp.mean(
            (entropy < self.entropy_threshold).astype(jnp.float32)
        )
        return {
            "mean_entropy": jnp.mean(entropy),
            "entropy_gradient": jnp.mean(signal),
            "order_fraction": order_fraction,
        }
