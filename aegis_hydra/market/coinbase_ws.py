
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
    timestamp: float = 0.0
    
    def update(self, side: str, price: float, size: float):
        target = self.bids if side == 'bid' else self.asks
        if size == 0:
            target.pop(price, None)
        else:
            target[price] = size

    def get_snapshot(self, depth: int = 50) -> Tuple[List[float], List[float]]:
        import heapq
        
        # Optimize: Avoid full sort (O(N log N)) using heapq (O(N log K))
        # Top bids (highest price)
        top_bids = heapq.nlargest(depth, self.bids.keys())
        sorted_bids = [(p, self.bids[p]) for p in top_bids]
        
        # Top asks (lowest price)
        top_asks = heapq.nsmallest(depth, self.asks.keys())
        sorted_asks = [(p, self.asks[p]) for p in top_asks]
        
        # Periodic Cleanup (Every 100 calls?)
        # For now, just keep growing to avoid delete overhead, or trim occasionally.
        if len(self.bids) > 5000:
            # Keep only top 1000
            keep = heapq.nlargest(1000, self.bids.keys())
            self.bids = {k: self.bids[k] for k in keep}
            
        if len(self.asks) > 5000:
            # Keep only bottom 1000
            keep = heapq.nsmallest(1000, self.asks.keys())
            self.asks = {k: self.asks[k] for k in keep}
        
        # Pad if insufficient depth
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
        self.latest_ticker: Optional[Dict] = None
        self.ready = False
        self.ws = None
        
    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.WS_URL, max_size=None) as ws:
                    self.ws = ws
                    logger.info(f"Connected to Coinbase WS ({self.product_id})")
                    
                    # Subscribe
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
                # Clear book and rebuild
                self.order_book.bids.clear()
                self.order_book.asks.clear()
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
            # Ticker update
            # Ticker structure: {'type': 'update', 'tickers': [{'type': 'ticker', 'product_id': 'BTC-USD', 'price': '96200.5', ...}]}
            for event in events:
                tickers = event.get("tickers", [])
                if tickers:
                    self.latest_ticker = tickers[0]

    def get_data(self) -> Tuple[float, List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Returns (mid_price, bids, asks)
        """
        if not self.ready:
            return 0.0, [], []
            
        bids, asks = self.order_book.get_snapshot(depth=50)
        
        # Calculate mid price from book if possible, else ticker
        if bids and asks and bids[0][0] > 0 and asks[0][0] > 0:
            mid = (bids[0][0] + asks[0][0]) / 2.0
        elif self.latest_ticker:
            mid = float(self.latest_ticker.get("price", 0.0))
        else:
            mid = 0.0
            
        return mid, bids, asks
