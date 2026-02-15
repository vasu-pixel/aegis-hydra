
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
    best_bid: float = 0.0
    best_ask: float = 0.0
    timestamp: float = 0.0
    
    def update(self, side: str, price: float, size: float):
        target = self.bids if side == 'bid' else self.asks
        if size == 0:
            target.pop(price, None)
            if side == 'bid' and price == self.best_bid:
                self.best_bid = max(self.bids.keys()) if self.bids else 0.0
            if side == 'ask' and price == self.best_ask:
                self.best_ask = min(self.asks.keys()) if self.asks else 0.0
        else:
            target[price] = size
            if side == 'bid' and (price > self.best_bid or self.best_bid == 0):
                self.best_bid = price
            if side == 'ask' and (price < self.best_ask or self.best_ask == 0):
                self.best_ask = price

    def get_snapshot(self, depth: int = 50) -> Tuple[List[float], List[float]]:
        import heapq
        
        # Optimize: Avoid full sort (O(N log N)) using heapq (O(N log K))
        top_bids = heapq.nlargest(depth, self.bids.keys())
        sorted_bids = [(p, self.bids[p]) for p in top_bids]
        
        top_asks = heapq.nsmallest(depth, self.asks.keys())
        sorted_asks = [(p, self.asks[p]) for p in top_asks]
        
        # Periodic Cleanup - ONLY during snapshots, not during price requests
        if len(self.bids) > 10000:
            keep = heapq.nlargest(2000, self.bids.keys())
            self.bids = {k: self.bids[k] for k in keep}
            self.best_bid = max(self.bids.keys()) if self.bids else 0.0
            
        if len(self.asks) > 10000:
            keep = heapq.nsmallest(2000, self.asks.keys())
            self.asks = {k: self.asks[k] for k in keep}
            self.best_ask = min(self.asks.keys()) if self.asks else 0.0
        
        while len(sorted_bids) < depth: sorted_bids.append((0.0, 0.0))
        while len(sorted_asks) < depth: sorted_asks.append((0.0, 0.0))
        
        return sorted_bids, sorted_asks

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
                self.order_book.best_bid = 0.0
                self.order_book.best_ask = 0.0
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
