"""
aegis_hydra.market.execution — Order Execution Engine

Manages the 100 "quantum" order slots. Each slot can hold one order
at a time. The engine translates the Governor's allocation into
actual exchange API calls via CCXT.

Dependencies: ccxt, asyncio
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    ccxt_async = None


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    EMPTY = "empty"
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OrderSlot:
    """
    A single "quantum" order slot.

    Each slot holds at most one active order at a time.
    """
    slot_id: int
    status: OrderStatus = OrderStatus.EMPTY
    side: Optional[OrderSide] = None
    symbol: str = ""
    price: float = 0.0
    size: float = 0.0
    filled_size: float = 0.0
    exchange_order_id: Optional[str] = None
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    @property
    def is_available(self) -> bool:
        return self.status in (OrderStatus.EMPTY, OrderStatus.FILLED, OrderStatus.CANCELLED)

    def reset(self) -> None:
        self.status = OrderStatus.EMPTY
        self.side = None
        self.symbol = ""
        self.price = 0.0
        self.size = 0.0
        self.filled_size = 0.0
        self.exchange_order_id = None
        self.created_at = None
        self.filled_at = None


@dataclass
class ExecutionReport:
    """Summary of an execution batch."""
    total_orders: int
    filled: int
    failed: int
    total_volume: float
    avg_fill_price: float
    slippage_bps: float  # basis points


class ExecutionEngine:
    """
    Manages the 100 order slots and communicates with the exchange.

    Parameters
    ----------
    exchange_id : str
        CCXT exchange identifier (e.g., 'binance', 'bybit').
    api_key : str
        Exchange API key.
    api_secret : str
        Exchange API secret.
    n_slots : int
        Number of order slots (default: 100).
    symbol : str
        Default trading pair.
    testnet : bool
        Use testnet/sandbox mode.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        n_slots: int = 100,
        symbol: str = "BTC/USDT",
        testnet: bool = True,
    ):
        self.exchange_id = exchange_id
        self.n_slots = n_slots
        self.symbol = symbol
        self.testnet = testnet

        # Initialize slots
        self.slots: List[OrderSlot] = [
            OrderSlot(slot_id=i) for i in range(n_slots)
        ]

        # Exchange connection (lazy init)
        self._exchange: Optional[object] = None
        self._api_key = api_key
        self._api_secret = api_secret

    async def connect(self) -> None:
        """Initialize the exchange connection."""
        if ccxt_async is None:
            raise ImportError("ccxt is required. pip install ccxt")

        exchange_class = getattr(ccxt_async, self.exchange_id)
        self._exchange = exchange_class({
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "sandbox": self.testnet,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

    async def disconnect(self) -> None:
        """Close the exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    def _find_available_slot(self) -> Optional[OrderSlot]:
        """Find the first available order slot."""
        for slot in self.slots:
            if slot.is_available:
                return slot
        return None

    async def place_order(
        self,
        side: OrderSide,
        size: float,
        price: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> Optional[OrderSlot]:
        """
        Place an order using the next available slot.

        Parameters
        ----------
        side : OrderSide
            Buy or sell.
        size : float
            Order size in base currency.
        price : float, optional
            Limit price. If None, market order.
        symbol : str, optional
            Trading pair. Defaults to self.symbol.

        Returns
        -------
        OrderSlot or None
            The slot used, or None if no slots available / order failed.
        """
        slot = self._find_available_slot()
        if slot is None:
            return None

        sym = symbol or self.symbol
        order_type = "limit" if price else "market"

        slot.status = OrderStatus.PENDING
        slot.side = side
        slot.symbol = sym
        slot.size = size
        slot.price = price or 0.0
        slot.created_at = datetime.utcnow()

        try:
            if self._exchange is None:
                raise RuntimeError("Exchange not connected. Call connect() first.")

            result = await self._exchange.create_order(
                symbol=sym,
                type=order_type,
                side=side.value,
                amount=size,
                price=price,
            )
            slot.exchange_order_id = result.get("id")
            slot.status = OrderStatus.OPEN
            return slot

        except Exception as e:
            slot.status = OrderStatus.FAILED
            slot.exchange_order_id = None
            # TODO: log error via utils/logger.py
            return slot

    async def cancel_slot(self, slot_id: int) -> bool:
        """Cancel the order in a specific slot."""
        slot = self.slots[slot_id]
        if slot.status != OrderStatus.OPEN or not slot.exchange_order_id:
            return False

        try:
            await self._exchange.cancel_order(slot.exchange_order_id, slot.symbol)
            slot.status = OrderStatus.CANCELLED
            return True
        except Exception:
            return False

    async def sync_fills(self) -> int:
        """Check exchange for filled orders and update slots. Returns fill count."""
        filled_count = 0
        for slot in self.slots:
            if slot.status == OrderStatus.OPEN and slot.exchange_order_id:
                try:
                    order = await self._exchange.fetch_order(
                        slot.exchange_order_id, slot.symbol
                    )
                    if order["status"] == "closed":
                        slot.status = OrderStatus.FILLED
                        slot.filled_size = order.get("filled", slot.size)
                        slot.filled_at = datetime.utcnow()
                        filled_count += 1
                except Exception:
                    pass
        return filled_count

    def get_report(self) -> ExecutionReport:
        """Generate summary report of all slots."""
        filled = [s for s in self.slots if s.status == OrderStatus.FILLED]
        failed = [s for s in self.slots if s.status == OrderStatus.FAILED]
        total_vol = sum(s.filled_size for s in filled)
        avg_price = (
            sum(s.price * s.filled_size for s in filled) / (total_vol + 1e-30)
            if filled else 0.0
        )
        return ExecutionReport(
            total_orders=len([s for s in self.slots if s.status != OrderStatus.EMPTY]),
            filled=len(filled),
            failed=len(failed),
            total_volume=total_vol,
            avg_fill_price=avg_price,
            slippage_bps=0.0,  # TODO: compute from expected vs actual fill
        )

    @property
    def available_slots(self) -> int:
        return sum(1 for s in self.slots if s.is_available)
