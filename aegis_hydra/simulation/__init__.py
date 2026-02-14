"""
aegis_hydra.simulation — THE LABORATORY

Backtesting and synthetic data generation for validating the physics engine
before deploying with real capital.
"""

from .backtester import Backtester, BacktestResult
from .synthetic_market import SyntheticMarket

__all__ = ["Backtester", "BacktestResult", "SyntheticMarket"]
