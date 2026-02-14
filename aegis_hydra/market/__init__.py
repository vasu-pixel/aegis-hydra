"""
aegis_hydra.market — THE REALITY INTERFACE ("The Sensors")

Connects the physics engine to the real market.

Modules:
    tensor_field  — Converts raw order book data into physics tensors
    execution     — Manages the 100 "quantum" order slots (API calls)

The Rust module `ingestion.rs` (compiled separately) handles the
high-speed WebSocket feed.
"""

from .tensor_field import TensorField, MarketTensor
from .execution import ExecutionEngine, OrderSlot, OrderSide

__all__ = [
    "TensorField",
    "MarketTensor",
    "ExecutionEngine",
    "OrderSlot",
    "OrderSide",
]
