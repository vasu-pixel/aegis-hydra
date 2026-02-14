"""
aegis_hydra.governor — THE CONTROL TOWER ("The Optimizer")

Takes the aggregate signal from the 1M-agent swarm and produces
optimal trading decisions subject to risk constraints.

Modules:
    hjb_solver  — Hamilton-Jacobi-Bellman optimal control
    allocator   — Convex optimization (QP) for capital allocation
    risk_guard  — Circuit breaker (VaR, correlation, drawdown limits)
"""

from .hjb_solver import HJBSolver
from .allocator import CapitalAllocator
from .risk_guard import RiskGuard

__all__ = ["HJBSolver", "CapitalAllocator", "RiskGuard"]
