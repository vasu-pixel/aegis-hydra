
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
    market_data: jax.Array,
    prev_bid_vol: jax.Array,
    prev_ask_vol: jax.Array,
    prev_flow_ema: jax.Array,
    price_history: jax.Array,
    coupling: float = 1.0,  # <-- NEW: Dynamic J
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
    new_grid_state = population.step(grid_state, flat_tensor, step_key, coupling=coupling)
    
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
        coupling: float = 1.0,
        viscosity_buy: float = 0.85,
        viscosity_sell: float = 0.2,
        min_hold_seconds: float = 60.0,
    ):
        self.population = population
        self.risk_guard = risk_guard
        self.symbol = symbol
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.exchange_id = exchange_id
        self.temperature = temperature
        self.coupling = coupling
        self.viscosity_buy = viscosity_buy
        self.viscosity_sell = viscosity_sell
        self.min_hold_seconds = min_hold_seconds
        
        self.position = 0.0 
        self.last_trade_time = 0.0 # For Cool-Down
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
        print(f"Physics: T={self.temperature}, J={self.coupling}")
        print(f"Strategy: Viscosity Buy>{self.viscosity_buy}, Sell<{self.viscosity_sell}")
        print(f"Cool-Down: {self.min_hold_seconds}s")
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
            last_physics_time = time.time()
            
            # Candle Aggregation Variables
            candle_start_time = time.time()
            ticks_buffer = []

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
                
                # Buffer Ticks for Candle
                ticks_buffer.append(mid_price)
                
                # 2. Check Aggregation Timer (5 Seconds)
                time_since_physics = ts_sec - last_physics_time
                if time_since_physics < 5.0:
                    await asyncio.sleep(0.1) # Fast sleep, keep fetching to fill buffer
                    continue
                
                # === PHYSICS STEP (Every 5s) ===
                last_physics_time = ts_sec
                
                # Use averaged price or last price? User said "feed it 5-second candles". 
                # Ideally we pass OHLC, but our Tensor currently takes snapshot.
                # Let's use the LAST snapshot but maybe we could smooth it?
                # For now, stick to Snapshot at t=5s to match current tensor logic.
                
                # 2. Fill Buffer (NumPy) - Zero Allocation
                self.market_buffer[0] = mid_price
                self.market_buffer[1] = ts_sec
                
                # Fill bids/asks (upto 50)
                n_bids = min(len(book['bids']), self.n_levels)
                n_asks = min(len(book['asks']), self.n_levels)
                
                b_vols = [x[1] for x in book['bids'][:n_bids]]
                a_vols = [x[1] for x in book['asks'][:n_asks]]
                
                self.market_buffer[2:2+n_bids] = b_vols
                self.market_buffer[2+n_bids:52] = 0.0 # Pad remainder
                
                self.market_buffer[52:52+n_asks] = a_vols
                self.market_buffer[52+n_asks:102] = 0.0 # Pad remainder

                # 3. Monolithic Update (Engine Latency)
                eng_start = time.time()
                
                (
                    grid_state, rng_key, agg,
                    self.prev_bid_vol, self.prev_ask_vol, self.prev_flow_ema, self.price_history
                ) = update_cycle_jit(
                    self.population, grid_state, rng_key,
                    self.market_buffer, # Passing single array
                    self.prev_bid_vol, self.prev_ask_vol, self.prev_flow_ema, self.price_history,
                    coupling=self.coupling # Pass Dynamic J
                )
                
                eng_latency = (time.time() - eng_start) * 1000 
                
                # 4. Decision & Execution (Python Scalar Logic)
                magnetization = float(agg["magnetization"])
                
                # Hysteresis (Viscosity) with Configurable Thresholds
                # Entry: Strong Signal (> viscosity_buy)
                # Exit:  Weak Signal (< viscosity_sell)
                
                target_pos = self.position # Default: Hold
                
                if abs(self.position) < 0.1: # Neutral
                    if magnetization > self.viscosity_buy: target_pos = 1.0
                    elif magnetization < -self.viscosity_buy: target_pos = -1.0
                elif self.position > 0.5: # Long
                    if magnetization < self.viscosity_sell: target_pos = 0.0
                elif self.position < -0.5: # Short
                    if magnetization > -self.viscosity_sell: target_pos = 0.0
                
                # Cool-Down Timer Check
                can_trade = (ts_sec - self.last_trade_time) > self.min_hold_seconds
                
                if prev_price:
                    pct = (mid_price - prev_price) / prev_price
                    self.capital += self.position * self.capital * pct

                if abs(target_pos - self.position) > 0.01:
                    if can_trade:
                        trade_size = target_pos - self.position
                        old_cap = self.capital
                        fee = abs(trade_size) * self.capital * self.fee_rate
                        self.capital -= fee
                        self.position = target_pos
                        self.last_trade_time = ts_sec # Reset Timer
                        
                        side = "BUY" if trade_size > 0 else "SELL"
                        print(f"\n[TRADE] {side} | Price: {mid_price:.2f} | Size: {abs(trade_size):.2f} | Cap: ${old_cap:.2f} -> ${self.capital:.2f}")
                        
                        with open("paper_trades.csv", "a") as f:
                            f.write(f"{datetime.now()},{mid_price},{side},{abs(trade_size)},{self.capital}\n")
                    else:
                        # Logic: Signal says trade, but Cool-Down prevents it.
                        # Do we log this? Maybe just ignore.
                        pass

                prev_price = mid_price
                
                # 5. Display
                net_latency = (ts_sec - loop_start) * 1000
                true_latency = (time.time() - eng_start) * 1000
                
                status = (
                    f"Step {step:05d} | "
                    f"BTC: {mid_price:,.2f} | "
                    f"M: {magnetization:+.3f} | "
                    f"Pos: {self.position:+.1f} | "
                    f"Net: {net_latency:.0f}ms | "
                    f"Wait: {(ts_sec - candle_start_time):.1f}s | "
                    f"True: {true_latency:.1f}ms"
                )
                print(status, end="\r")
                sys.stdout.flush()
                
                # Dump State
                if step % 2 == 0: # More frequent relative to steps (since steps are 5s now)
                    self.history.append({
                        "time": datetime.now().isoformat(),
                        "step": step,
                        "price": mid_price,
                        "capital": self.capital,
                        "magnetization": magnetization,
                        "position": self.position,
                        "latency": true_latency
                    })
                    
                    with open("paper_log.csv", "a") as f:
                        f.write(f"{datetime.now().isoformat()},{step},{mid_price},{self.capital},{magnetization},{self.position},{true_latency}\n")

                    import json
                    with open("paper_state.json", "w") as f:
                        json.dump(self.history[-1000:], f)
                
                step += 1
                candle_start_time = time.time() # Reset candle timer
                
                # Event-Driven: No sleep, just yield
                await asyncio.sleep(0)
                
                if step % 200 == 0:
                    gc.collect()

        except KeyboardInterrupt:
            print("\n\n=== PAUSED ===")
        finally:
            await exchange.close()
            print("Exchange connection closed.")
