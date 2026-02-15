
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
    
    def update(self, side: str, price: float, size: float):
        import heapq
        target_dict = self.bids if side == 'bid' else self.asks
        target_heap = self.bid_heap if side == 'bid' else self.ask_heap
        
        if size == 0:
            target_dict.pop(price, None)
            # Lazy removal: we don't remove from heap here.
        else:
            target_dict[price] = size
            if side == 'bid':
                heapq.heappush(self.bid_heap, -price)
            else:
                heapq.heappush(self.ask_heap, price)

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
        
        # Snapshots can be slow, but they happen in background maintenance thread or dashboard
        # For snapshot, we still need to filter the heaps
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

class CoinbaseWebSocket:
    """
    Real-time market data via Coinbase Advanced Trade WebSocket.
    """
    WS_URL = "wss://advanced-trade-ws.coinbase.com"
    
    def __init__(self, product_id: str = "BTC-USD"):
        self.product_id = product_id
        self.order_book = OrderBook()
        self.latest_ticker_price = 0.0
        self.ready = False
        self.ws = None
        
    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.WS_URL, max_size=None) as ws:
                    self.ws = ws
                    logger.info(f"Connected to Coinbase WS ({self.product_id})")
                    
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": [self.product_id],
                        "channel": "level2"
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    subscribe_ticker = {
                        "type": "subscribe",
                        "product_ids": [self.product_id],
                        "channel": "ticker"
                    }
                    await ws.send(json.dumps(subscribe_ticker))
                    
                    logger.info("Subscribed to level2 & ticker")
                    await self._listen()
                    
            except Exception as e:
                logger.error(f"WS Connection Error: {e}")
                self.ready = False
                await asyncio.sleep(5)
                
    async def _listen(self):
        async for message in self.ws:
            data = json.loads(message)
            channel = data.get("channel")
            
            if channel == "l2_data":
                self._handle_l2(data)
            elif channel == "ticker":
                self._handle_ticker(data)
                
    def _handle_l2(self, data):
        events = data.get("events", [])
        for event in events:
            if event["type"] == "snapshot":
                self.order_book.bids.clear()
                self.order_book.asks.clear()
                self.order_book.bid_heap.clear()
                self.order_book.ask_heap.clear()
                updates = event["updates"]
                for update in updates:
                    side = update["side"] # 'bid' or 'ask'
                    price = float(update["price_level"])
                    size = float(update["new_quantity"])
                    self.order_book.update(side, price, size)
                self.ready = True
                
            elif event["type"] == "update":
                updates = event["updates"]
                for update in updates:
                    side = update["side"]
                    price = float(update["price_level"])
                    size = float(update["new_quantity"])
                    self.order_book.update(side, price, size)
                    
        self.order_book.timestamp = time.time()

    def _handle_ticker(self, data):
        events = data.get("events", [])
        if events:
            for event in events:
                tickers = event.get("tickers", [])
                if tickers:
                    self.latest_ticker_price = float(tickers[0].get("price", 0.0))

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
            
        bids, asks = self.order_book.get_snapshot(depth=50)
        mid = self.get_mid_price()
        return mid, bids, asks
