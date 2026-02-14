"""
aegis_hydra.governor.risk_guard — The Circuit Breaker

Monitors portfolio risk in real-time and vetoes trades that would
violate risk limits. This is the final gate before execution.

Risk Measures:
    - Value at Risk (VaR) — parametric and historical
    - Conditional VaR (CVaR / Expected Shortfall)
    - Maximum drawdown tracking
    - Correlation breakdown detection
    - Position concentration limits

Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class RiskLevel(IntEnum):
    """Risk regime classification. Integer values enable correct severity comparison."""
    GREEN = 0       # Normal — full trading
    YELLOW = 1      # Elevated — reduce position sizes
    RED = 2         # Critical — no new trades, flatten if needed
    BLACK = 3       # Emergency — immediate full flatten


@dataclass
class RiskSnapshot:
    """Point-in-time risk assessment."""
    level: RiskLevel
    var_95: float                # 95% VaR as % of portfolio
    cvar_95: float               # Expected shortfall (CVaR)
    current_drawdown: float      # Current drawdown from peak
    max_drawdown: float          # Maximum drawdown observed
    correlation_stress: float    # 0-1 score (1 = correlations breaking down)
    concentration: float         # Herfindahl index of position weights
    reasons: List[str]           # Human-readable risk flags


@dataclass
class RiskGuard:
    """
    Real-time risk monitor and circuit breaker.

    Parameters
    ----------
    var_limit : float
        Maximum 95% VaR as fraction of portfolio.
    max_drawdown_limit : float
        Maximum drawdown before RED alert.
    emergency_drawdown : float
        Drawdown that triggers BLACK (emergency flatten).
    concentration_limit : float
        Max Herfindahl index (0 = perfectly diversified, 1 = all in one).
    correlation_threshold : float
        Correlation stress score that triggers YELLOW.
    lookback_window : int
        Number of returns to use for VaR calculation.
    cooldown_steps : int
        Steps to wait after RED before resuming trading.
    """

    var_limit: float = 0.05
    max_drawdown_limit: float = 0.10
    emergency_drawdown: float = 0.20
    concentration_limit: float = 0.25
    correlation_threshold: float = 0.8
    lookback_window: int = 100
    cooldown_steps: int = 50

    # Internal state
    _peak_value: float = field(default=-1.0, init=False)
    _return_history: List[float] = field(default_factory=list, init=False)
    _cooldown_remaining: int = field(default=0, init=False)

    def update_portfolio_value(self, value: float) -> None:
        """Track portfolio value for drawdown calculations."""
        if self._peak_value < 0:
            self._peak_value = value
        self._peak_value = max(self._peak_value, value)

    def add_return(self, ret: float) -> None:
        """Append a return observation to the history."""
        self._return_history.append(ret)
        if len(self._return_history) > self.lookback_window * 2:
            self._return_history = self._return_history[-self.lookback_window:]

    def compute_var(self, confidence: float = 0.95) -> float:
        """
        Parametric VaR assuming normal returns.

        VaR_α = μ - z_α · σ
        """
        if len(self._return_history) < 10:
            return 0.0
        returns = np.array(self._return_history[-self.lookback_window:])
        mu = np.mean(returns)
        sigma = np.std(returns) + 1e-10
        z = stats.norm.ppf(1 - confidence)
        return -(mu + z * sigma)

    def compute_cvar(self, confidence: float = 0.95) -> float:
        """
        Historical CVaR (Expected Shortfall).

        CVaR = E[loss | loss > VaR]
        """
        if len(self._return_history) < 10:
            return 0.0
        returns = np.array(self._return_history[-self.lookback_window:])
        var = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= var]
        return -np.mean(tail) if len(tail) > 0 else 0.0

    def compute_drawdown(self, current_value: float) -> float:
        """Current drawdown from peak."""
        return (self._peak_value - current_value) / (self._peak_value + 1e-10)

    def compute_concentration(self, weights: np.ndarray) -> float:
        """Herfindahl-Hirschman Index of position weights."""
        w = np.abs(weights)
        w = w / (np.sum(w) + 1e-10)
        return float(np.sum(w ** 2))

    def correlation_stress_score(
        self,
        return_matrix: np.ndarray,
    ) -> float:
        """
        Detect correlation breakdown / regime change.

        Compares recent correlation structure to long-run average.
        Returns a score in [0, 1] — higher = more stress.
        """
        if return_matrix.shape[0] < 20:
            return 0.0

        n = return_matrix.shape[0]
        split = n // 2
        corr_old = np.corrcoef(return_matrix[:split].T)
        corr_new = np.corrcoef(return_matrix[split:].T)

        # Frobenius norm of difference
        diff = np.linalg.norm(corr_new - corr_old, "fro")
        max_diff = np.sqrt(2 * corr_old.shape[0] ** 2)  # theoretical max
        return float(np.clip(diff / max_diff, 0.0, 1.0))

    def assess(
        self,
        current_value: float,
        weights: np.ndarray,
        return_matrix: Optional[np.ndarray] = None,
    ) -> RiskSnapshot:
        """
        Full risk assessment. Returns a RiskSnapshot with level + details.

        Parameters
        ----------
        current_value : float
            Current portfolio value.
        weights : ndarray
            Current position weights.
        return_matrix : ndarray, optional
            Matrix of recent returns per asset, shape (T, N).
        """
        self.update_portfolio_value(current_value)

        var = self.compute_var()
        cvar = self.compute_cvar()
        dd = self.compute_drawdown(current_value)
        conc = self.compute_concentration(weights)
        corr_stress = (
            self.correlation_stress_score(return_matrix)
            if return_matrix is not None
            else 0.0
        )

        reasons: List[str] = []
        level = RiskLevel.GREEN

        # Check drawdown
        if dd >= self.emergency_drawdown:
            level = RiskLevel.BLACK
            reasons.append(f"EMERGENCY: Drawdown {dd:.1%} exceeds {self.emergency_drawdown:.1%}")
        elif dd >= self.max_drawdown_limit:
            level = RiskLevel.RED
            reasons.append(f"Drawdown {dd:.1%} exceeds limit {self.max_drawdown_limit:.1%}")

        # Check VaR
        if var > self.var_limit:
            if level < RiskLevel.RED:
                level = RiskLevel.YELLOW
            reasons.append(f"VaR {var:.2%} exceeds limit {self.var_limit:.2%}")

        # Check concentration
        if conc > self.concentration_limit:
            if level < RiskLevel.YELLOW:
                level = RiskLevel.YELLOW
            reasons.append(f"Concentration {conc:.2f} exceeds {self.concentration_limit:.2f}")

        # Check correlation stress
        if corr_stress > self.correlation_threshold:
            if level < RiskLevel.YELLOW:
                level = RiskLevel.YELLOW
            reasons.append(f"Correlation stress {corr_stress:.2f}")

        # Cooldown
        if self._cooldown_remaining > 0:
            level = RiskLevel.RED
            reasons.append(f"Cooldown: {self._cooldown_remaining} steps remaining")
            self._cooldown_remaining -= 1

        if not reasons:
            reasons.append("All risk metrics within limits")

        return RiskSnapshot(
            level=level,
            var_95=var,
            cvar_95=cvar,
            current_drawdown=dd,
            max_drawdown=max(dd, getattr(self, "_max_dd_seen", 0.0)),
            correlation_stress=corr_stress,
            concentration=conc,
            reasons=reasons,
        )

    def veto(self, snapshot: RiskSnapshot) -> bool:
        """Returns True if trading should be halted."""
        if snapshot.level in (RiskLevel.RED, RiskLevel.BLACK):
            self._cooldown_remaining = self.cooldown_steps
            return True
        return False

    def scale_factor(self, snapshot: RiskSnapshot) -> float:
        """Position size scaling factor based on risk level."""
        scale_map = {
            RiskLevel.GREEN: 1.0,
            RiskLevel.YELLOW: 0.5,
            RiskLevel.RED: 0.0,
            RiskLevel.BLACK: 0.0,
        }
        return scale_map[snapshot.level]
