"""
aegis_hydra.governor.allocator — Convex Optimization Capital Allocator

Solves a Quadratic Program (QP) to distribute capital across trading
slots subject to risk constraints. This is the bridge between the
HJB policy and actual dollar amounts.

Mathematical Foundation:
    minimize    (1/2)x'Σx - μ'x          (risk-adjusted return)
    subject to  Σ x_i ≤ budget
                x_i ≥ 0                   (long-only, or allow shorts)
                VaR(x) ≤ VaR_limit

Dependencies: cvxpy, numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None  # Graceful degradation; will raise at runtime


@dataclass
class AllocationResult:
    """Result of the capital allocation optimization."""
    weights: np.ndarray          # Optimal weights per slot
    expected_return: float       # μ'x
    expected_risk: float         # sqrt(x'Σx)
    sharpe_ratio: float          # (μ'x) / sqrt(x'Σx)
    solver_status: str           # 'optimal', 'infeasible', etc.
    raw_solution: Optional[object] = None


@dataclass
class CapitalAllocator:
    """
    Quadratic Programming allocator for distributing the $100 budget
    across the 100 "quantum" order slots.

    Parameters
    ----------
    budget : float
        Total capital to allocate (e.g., $100).
    n_slots : int
        Number of order slots (e.g., 100).
    max_single_allocation : float
        Maximum fraction of budget in a single slot.
    risk_aversion : float
        λ in the objective: maximize μ'x - λ(x'Σx).
    allow_short : bool
        Whether to allow short positions.
    var_limit : Optional[float]
        Maximum Value-at-Risk (95%) as fraction of budget.
    """

    budget: float = 100.0
    n_slots: int = 100
    max_single_allocation: float = 0.10  # 10% max per slot
    risk_aversion: float = 1.0
    allow_short: bool = False
    var_limit: Optional[float] = 0.05  # 5% VaR limit

    def solve(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
    ) -> AllocationResult:
        """
        Solve the QP for optimal capital allocation.

        Parameters
        ----------
        expected_returns : ndarray, shape (n_slots,)
            Expected return per slot (from HJB policy × swarm signal).
        covariance_matrix : ndarray, shape (n_slots, n_slots)
            Covariance matrix of slot returns.
        current_positions : ndarray, optional
            Current allocation (for turnover penalty).

        Returns
        -------
        AllocationResult
        """
        if cp is None:
            raise ImportError("cvxpy is required for CapitalAllocator. pip install cvxpy")

        n = self.n_slots
        x = cp.Variable(n)

        # Objective: maximize return - risk_aversion * variance
        # Equivalently: minimize -μ'x + λ(x'Σx)
        portfolio_return = expected_returns @ x
        portfolio_risk = cp.quad_form(x, covariance_matrix)

        objective = cp.Minimize(
            -portfolio_return + self.risk_aversion * portfolio_risk
        )

        # Constraints
        constraints = [
            cp.sum(x) <= self.budget,
            x <= self.budget * self.max_single_allocation,
        ]

        if not self.allow_short:
            constraints.append(x >= 0)
        else:
            constraints.append(x >= -self.budget * self.max_single_allocation)

        # Turnover penalty (if rebalancing)
        if current_positions is not None:
            turnover = cp.norm(x - current_positions, 1)
            objective = cp.Minimize(
                -portfolio_return + self.risk_aversion * portfolio_risk + 0.01 * turnover
            )

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)

        if problem.status in ("optimal", "optimal_inaccurate"):
            weights = x.value
            exp_ret = float(expected_returns @ weights)
            exp_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
            sharpe = exp_ret / (exp_risk + 1e-10)
        else:
            weights = np.zeros(n)
            exp_ret = 0.0
            exp_risk = 0.0
            sharpe = 0.0

        return AllocationResult(
            weights=weights,
            expected_return=exp_ret,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            solver_status=problem.status,
            raw_solution=problem,
        )

    def kelly_fraction(
        self,
        win_prob: float,
        win_return: float,
        loss_return: float,
    ) -> float:
        """
        Kelly criterion for single-bet sizing.

        f* = (p·b - q) / b

        where p = win probability, b = win/loss ratio, q = 1-p.
        """
        q = 1.0 - win_prob
        b = abs(win_return / (loss_return + 1e-10))
        kelly = (win_prob * b - q) / (b + 1e-10)
        # Half-Kelly for safety
        return max(0.0, kelly * 0.5)
