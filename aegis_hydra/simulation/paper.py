
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import sys
import gc

# Try CCXT
try:
    import ccxt.async_support as ccxt
except ImportError:
    print("CCXT not installed. Please pip install ccxt.")
    sys.exit(1)

import jax
import jax.numpy as jnp
import numpy as np
import functools
import equinox as eqx

# Hydra components
from ..agents.population import Population
from ..governor.hjb_solver import HJBSolver
from ..governor.risk_guard import RiskGuard
from ..market.tensor_field import process_tensor_jax

# ---------------------------------------------------------------------------
# JIT Compiled Monolithic Kernel
# ---------------------------------------------------------------------------
@eqx.filter_jit
def update_cycle_jit(
    population: Population,
    grid_state: jax.Array,
    key: jax.Array,
    # Market Data Inputs (Packed)
    market_data: jax.Array, # [mid_price, timestamp, bid_vol..., ask_vol...]
    # Tensor State Inputs
    prev_bid_vol: jax.Array,
    prev_ask_vol: jax.Array,
    prev_flow_ema: jax.Array,
    price_history: jax.Array,
):
    """
    Monolithic Update: Data -> Tensor -> Physics -> Aggregation.
    Runs entirely on GPU/Accelerator.
    Args:
        market_data: [mid_price, timestamp, bid_vol[0]..bid_vol[49], ask_vol[0]..ask_vol[49]]
    """
    n_levels = 50
    mid_price = market_data[0]
    timestamp = market_data[1]
    bid_vol = market_data[2:52]
    ask_vol = market_data[52:102]
    
    # 1. Process Market Data -> Tensor
    flat_tensor, new_flow_ema, new_price_hist, debug_flow = process_tensor_jax(
        mid_price, bid_vol, ask_vol, 
        prev_bid_vol, prev_ask_vol, 
        prev_flow_ema, price_history, timestamp
    )
    
    # 2. Step Physics
    key, step_key = jax.random.split(key)
    new_grid_state = population.step(grid_state, flat_tensor, step_key)
    
    # 3. Aggregate
    agg = population.aggregate(new_grid_state)
    
    return (
        new_grid_state, key, agg, 
        # New State
        bid_vol, ask_vol, new_flow_ema, new_price_hist
    )

class PaperTrader:
    """
    Live Paper Trading Engine (Optimized).
    """
    def __init__(
        self,
        population: Population,
        risk_guard: RiskGuard,
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0,
        exchange_id: str = "binance",
        fee_rate: float = 0.0005,
        temperature: float = 2.27,
    ):
        self.population = population
        self.risk_guard = risk_guard
        self.symbol = symbol
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.exchange_id = exchange_id
        self.temperature = temperature
        
        self.position = 0.0 
        self.history = []
        self.n_levels = 50 
        
        # Initialize JAX State Arrays
        self.prev_bid_vol = jnp.zeros(self.n_levels)
        self.prev_ask_vol = jnp.zeros(self.n_levels)
        self.prev_flow_ema = jnp.zeros(2 * self.n_levels)
        self.price_history = jnp.zeros(100) # Window size 100
        
        # Pre-allocate Host Buffer (Zero-Copy Optimization)
        # [mid, time] + [bid_vol * 50] + [ask_vol * 50]
        # 2 + 50 + 50 = 102 floats
        self.market_buffer = np.zeros(102, dtype=np.float32)
        
    async def run(self):
        print(f"=== PAPER TRADER INITIALIZED (Zero-Copy) ===")
        print(f"Exchange: {self.exchange_id.upper()}")
        print(f"Physics: T={self.temperature}")
        print("--------------------------------")
        
        # Initialize Exchange
        if not hasattr(ccxt, self.exchange_id):
            print(f"Error: Exchange '{self.exchange_id}' not found.")
            return

        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        try:
            # Initialize Population
            rng_key = jax.random.PRNGKey(int(time.time()))
            grid_state = self.population.initialize(rng_key, temperature=self.temperature)
            print("Swarm Initialized & JIT Compiling...")
            
            step = 0
            prev_price = None
            
            while True:
                loop_start = time.time()
                
                # 1. Non-Blocking Fetch
                try:
                    # Parallel fetch
                    ticker_task = exchange.fetch_ticker(self.symbol)
                    book_task = exchange.fetch_order_book(self.symbol, limit=self.n_levels)
                    
                    ticker, book = await asyncio.gather(ticker_task, book_task)
                except Exception as e:
                    print(f"Data Error: {e}")
                    await asyncio.sleep(0.1)
                    continue

                ts_sec = time.time()
                mid_price = (book['bids'][0][0] + book['asks'][0][0]) / 2.0
                
                # 2. Fill Buffer (NumPy) - Zero Allocation
                self.market_buffer[0] = mid_price
                self.market_buffer[1] = ts_sec
                
                # Fill bids/asks (upto 50)
                n_bids = min(len(book['bids']), self.n_levels)
                n_asks = min(len(book['asks']), self.n_levels)
                
                # Manual loop or slice assignment - slice is fast in numpy
                # Extract volumes from list of lists [[price, vol], ...]
                # This list comp is unavoidable unless we use a custom parser, 
                # but it's small (50 items).
                # optimization: pre-allocate temp arrays? No, CCXT returns new lists anyway.
                
                b_vols = [x[1] for x in book['bids'][:n_bids]]
                a_vols = [x[1] for x in book['asks'][:n_asks]]
                
                self.market_buffer[2:2+n_bids] = b_vols
                self.market_buffer[2+n_bids:52] = 0.0 # Pad remainder
                
                self.market_buffer[52:52+n_asks] = a_vols
                self.market_buffer[52+n_asks:102] = 0.0 # Pad remainder

                # 3. Monolithic Update (Engine Latency)
                eng_start = time.time()
                
                # Use jax.lax.device_put to move buffer to GPU?
                # Actually passing numpy array to JIT function does implicit device_put
                # But explicit might be cleaner to measure.
                # Let's just pass it.
                
                (
                    grid_state, rng_key, agg,
                    self.prev_bid_vol, self.prev_ask_vol, self.prev_flow_ema, self.price_history
                ) = update_cycle_jit(
                    self.population, grid_state, rng_key,
                    self.market_buffer, # Passing single array
                    self.prev_bid_vol, self.prev_ask_vol, self.prev_flow_ema, self.price_history
                )
                
                # We DO NOT block here. We let JAX run ahead.
                # Only if we need to trade do we check value.
                # To check trades asynchronously, we can fetch the value.
                # For this demo, to measure latency, checking IS required.
                # But let's check ONLY magnetization
                
                eng_latency = (time.time() - eng_start) * 1000 # This will be ~0ms (async dispatch)
                
                # 4. Decision & Execution (Python Scalar Logic)
                # This WILL block, revealing true compute time.
                magnetization = float(agg["magnetization"])
                
                target_pos = 0.0
                if magnetization > 0.7: target_pos = 1.0
                elif magnetization < -0.7: target_pos = -1.0
                
                if prev_price:
                    pct = (mid_price - prev_price) / prev_price
                    self.capital += self.position * self.capital * pct

                if abs(target_pos - self.position) > 0.01:
                    trade_size = target_pos - self.position
                    old_cap = self.capital
                    fee = abs(trade_size) * self.capital * self.fee_rate
                    self.capital -= fee
                    self.position = target_pos
                    
                    side = "BUY" if trade_size > 0 else "SELL"
                    print(f"\n[TRADE] {side} | Price: {mid_price:.2f} | Size: {abs(trade_size):.2f} | Cap: ${old_cap:.2f} -> ${self.capital:.2f}")
                    
                    with open("paper_trades.csv", "a") as f:
                        f.write(f"{datetime.now()},{mid_price},{side},{abs(trade_size)},{self.capital}\n")

                prev_price = mid_price
                
                # 5. Display
                net_latency = (ts_sec - loop_start) * 1000
                
                # Measure "True" latency (including block)
                true_latency = (time.time() - eng_start) * 1000
                
                status = (
                    f"Step {step:05d} | "
                    f"BTC: {mid_price:,.2f} | "
                    f"M: {magnetization:+.3f} | "
                    f"Pos: {self.position:+.1f} | "
                    f"Net: {net_latency:.0f}ms | "
                    f"Eng (Async): {eng_latency:.1f}ms | "
                    f"True: {true_latency:.1f}ms"
                )
                print(status, end="\r")
                sys.stdout.flush()
                
                # Dump State (Less frequent)
                if step % 10 == 0:
                    self.history.append({
                        "time": datetime.now().isoformat(),
                        "step": step,
                        "price": mid_price,
                        "capital": self.capital,
                        "magnetization": magnetization,
                        "position": self.position,
                        "latency": true_latency
                    })
                    
                    # Log full state for ML Training (Append Mode)
                    with open("paper_log.csv", "a") as f:
                        f.write(f"{datetime.now().isoformat()},{step},{mid_price},{self.capital},{magnetization},{self.position},{true_latency}\n")

                    # Dump State for Dashboard (More Frequent: Every 10 steps = ~2-3s)
                    if step % 10 == 0:
                        import json
                        with open("paper_state.json", "w") as f:
                            json.dump(self.history[-1000:], f)
                
                step += 1
                
                # Event-Driven: No sleep, just yield
                await asyncio.sleep(0)
                
                # Manual GC (Emergency Patch)
                if step % 1000 == 0:
                    gc.collect()

        except KeyboardInterrupt:
            print("\n\n=== PAUSED ===")
        finally:
            await exchange.close()
            print("Exchange connection closed.")
