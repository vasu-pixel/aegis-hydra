"""
aegis_hydra.core — THE PHYSICS ENGINE ("The Brain")

Pure mathematical solvers with zero trading logic.
Defines the "laws of physics" for the market universe.

Modules:
    stochastic       — SDE solvers, Fokker-Planck equations (JAX)
    entropy          — Shannon & Tsallis entropy calculators
    mechanics        — Hamiltonian/Lagrangian dynamics & Action principles
    fluid_dynamics   — Navier-Stokes inspired solvers for order book flow
"""

from .stochastic import FokkerPlanckSolver, LangevinSolver, SDEIntegrator
from .entropy import ShannonEntropy, TsallisEntropy, EntropyField
from .mechanics import HamiltonianSolver, LagrangianSolver, ActionPrinciple
from .fluid_dynamics import OrderBookFlowSolver, ViscosityEstimator

__all__ = [
    "FokkerPlanckSolver",
    "LangevinSolver",
    "SDEIntegrator",
    "ShannonEntropy",
    "TsallisEntropy",
    "EntropyField",
    "HamiltonianSolver",
    "LagrangianSolver",
    "ActionPrinciple",
    "OrderBookFlowSolver",
    "ViscosityEstimator",
]
