
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
import websockets
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class OrderBook:
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    bid_heap: List[float] = field(default_factory=list) # max heap (invert prices)
    ask_heap: List[float] = field(default_factory=list) # min heap
    timestamp: float = 0.0
    update_count: int = 0

    def update(self, side: str, price: float, size: float):
        import heapq
        target_dict = self.bids if side == 'bid' else self.asks
        target_heap = self.bid_heap if side == 'bid' else self.ask_heap

        if size == 0:
            target_dict.pop(price, None)
        else:
            target_dict[price] = size
            if side == 'bid':
                heapq.heappush(self.bid_heap, -price)
            else:
                heapq.heappush(self.ask_heap, price)

        self.update_count += 1
        if self.update_count % 1000 == 0:
            self._rebuild_heaps()

    def _rebuild_heaps(self):
        """Rebuild heaps from dictionaries to purge stale entries."""
        import heapq
        self.bid_heap = [-p for p in self.bids.keys()]
        self.ask_heap = [p for p in self.asks.keys()]
        heapq.heapify(self.bid_heap)
        heapq.heapify(self.ask_heap)

    @property
    def best_bid(self) -> float:
        import heapq
        while self.bid_heap:
            price = -self.bid_heap[0]
            if price in self.bids and self.bids[price] > 0:
                return price
            heapq.heappop(self.bid_heap)
        return 0.0

    @property
    def best_ask(self) -> float:
        import heapq
        while self.ask_heap:
            price = self.ask_heap[0]
            if price in self.asks and self.asks[price] > 0:
                return price
            heapq.heappop(self.ask_heap)
        return 0.0

    def get_snapshot(self, depth: int = 50) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        import heapq

        bids_out = []
        temp_bids = []
        while self.bid_heap and len(bids_out) < depth:
            p = -heapq.heappop(self.bid_heap)
            if p in self.bids and self.bids[p] > 0:
                bids_out.append((p, self.bids[p]))
            temp_bids.append(-p)
        for p in temp_bids: heapq.heappush(self.bid_heap, p)

        asks_out = []
        temp_asks = []
        while self.ask_heap and len(asks_out) < depth:
            p = heapq.heappop(self.ask_heap)
            if p in self.asks and self.asks[p] > 0:
                asks_out.append((p, self.asks[p]))
            temp_asks.append(p)
        for p in temp_asks: heapq.heappush(self.ask_heap, p)

        while len(bids_out) < depth: bids_out.append((0.0, 0.0))
        while len(asks_out) < depth: asks_out.append((0.0, 0.0))

        return bids_out, asks_out


class BinanceWebSocket:
    """
    Real-time market data via Binance US WebSocket.

    Fees: 0.1% taker (0.075% with BNB) - Legal in US!

    Uses:
    - Partial Depth Stream: btcusdt@depth20@100ms (snapshots every 100ms)
    - Trade Stream: btcusdt@trade (latest trade prices)
    """
    WS_URL = "wss://stream.binance.us:9443/stream"

    def __init__(self, product_id: str = "BTC-USDT"):
        # Convert to Binance format: BTC-USD -> btcusdt
        symbol = product_id.replace("-", "").lower()
        self.symbol = symbol
        self.product_id = product_id

        self.order_book = OrderBook()
        self.latest_ticker_price = 0.0
        self.ready = False
        self.ws = None

    async def connect(self):
        while True:
            try:
                # Subscribe to partial depth (snapshots) + trades
                streams = f"{self.symbol}@depth20@100ms/{self.symbol}@trade"
                url = f"{self.WS_URL}?streams={streams}"

                async with websockets.connect(url, max_size=None,
                                              ping_interval=20,
                                              ping_timeout=10) as ws:
                    self.ws = ws
                    logger.info(f"Connected to Binance WS ({self.product_id})")
                    logger.info(f"Streams: {streams}")

                    await self._listen()

            except Exception as e:
                logger.error(f"WS Connection Error: {e}")
                self.ready = False
                await asyncio.sleep(5)

    async def _listen(self):
        async for message in self.ws:
            try:
                data = json.loads(message)

                # Binance combined streams format: {"stream": "...", "data": {...}}
                if "stream" in data:
                    stream_name = data["stream"]
                    stream_data = data["data"]

                    if "depth" in stream_name:
                        self._handle_depth(stream_data)
                    elif "trade" in stream_name:
                        self._handle_trade(stream_data)
            except Exception as e:
                logger.error(f"Message parsing error: {e}")
                continue

    def _handle_depth(self, data):
        """
        Handle Binance partial depth updates (snapshots).

        Format:
        {
          "lastUpdateId": 160,
          "bids": [["50000.00", "1.50"], ...],  # price, quantity
          "asks": [["51000.00", "0.75"], ...]
        }
        """
        # Clear and rebuild from snapshot (partial depth sends full snapshot)
        self.order_book.bids.clear()
        self.order_book.asks.clear()
        self.order_book.bid_heap.clear()
        self.order_book.ask_heap.clear()

        # Process bids
        for price_qty in data.get("bids", []):
            price = float(price_qty[0])
            qty = float(price_qty[1])
            if qty > 0:
                self.order_book.update("bid", price, qty)

        # Process asks
        for price_qty in data.get("asks", []):
            price = float(price_qty[0])
            qty = float(price_qty[1])
            if qty > 0:
                self.order_book.update("ask", price, qty)

        self.order_book.timestamp = time.time()
        self.ready = True

    def _handle_trade(self, data):
        """
        Handle Binance trade stream.

        Format:
        {
          "e": "trade",
          "E": 1234567890,
          "s": "BTCUSDT",
          "p": "50000.00",  # price
          "q": "0.01",      # quantity
          "T": 1234567890   # trade time
        }
        """
        price = float(data.get("p", 0.0))
        if price > 0:
            self.latest_ticker_price = price

    def _handle_l2(self, data):
        """Compatibility method for hft_pipe.py which calls this."""
        # hft_pipe.py may call this with l2_data channel format
        # For Binance, we handle this in _handle_depth instead
        pass

    def _handle_ticker(self, data):
        """Compatibility method for hft_pipe.py which calls this."""
        # For Binance, we handle this in _handle_trade instead
        pass

    def get_mid_price(self) -> float:
        """Fastest path O(1) for HFT pipe."""
        if not self.ready:
            return 0.0

        # Prefer Book Mid
        if self.order_book.best_bid > 0 and self.order_book.best_ask > 0:
            return (self.order_book.best_bid + self.order_book.best_ask) / 2.0

        return self.latest_ticker_price

    def get_data(self) -> Tuple[float, List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Slow path for Dashboard / Standard Backtesters."""
        if not self.ready:
            return 0.0, [], []


class BinanceFuturesWS:
    """
    The 'Leader' Feed.
    Connects to Global Binance Futures (fstream.binance.com).
    Read-Only. No API Keys needed.
    """
    WS_URL = "wss://fstream.binance.com/ws"

    def __init__(self, symbol="btcusdt"):
        # Map Spot (BTC-USD) to Futures (BTCUSDT)
        sl = symbol.lower().replace("-", "")
        if sl == 'btcusd': self.symbol = 'btcusdt'
        elif sl == 'ethusd': self.symbol = 'ethusdt'
        else: self.symbol = sl
        
        self.price = 0.0
        self.latency = 0.0
        self.ready = False

    async def connect(self):
        # Subscribe to Aggregated Trades (Fastest public feed)
        stream = f"{self.symbol}@aggTrade"
        url = f"{self.WS_URL}/{stream}"
        
        while True:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"Connected to Binance Futures ({self.symbol})")
                    self.ready = True
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            # 'p' is price, 'E' is event timestamp (faster than T)
                            if 'p' in data:
                                self.price = float(data['p'])
                            if 'E' in data:
                                # Track latency (Futures Server -> Us)
                                self.latency = (time.time() - float(data['E'])/1000.0) * 1000.0
                        except Exception as e:
                            logger.error(f"Futures Msg Parse Error: {e}")
            except Exception as e:
                logger.error(f"Futures WS Error: {e}")
                self.ready = False
                await asyncio.sleep(5)
