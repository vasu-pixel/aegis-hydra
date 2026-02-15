
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import sys
import gc
import concurrent.futures

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

# I/O Executor for non-blocking file writes
io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="io_worker")

def write_csv_sync(filename, lines):
    """Synchronous CSV write helper for executor."""
    try:
        with open(filename, "a") as f:
            f.writelines(lines)
    except Exception:
        pass

def save_json_sync(filename, data):
    """Synchronous JSON write helper for executor."""
    try:
        import json
        with open(filename, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

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
        aggregation_seconds: float = 5.0,
        use_cpp: bool = False,
        use_hsoft: bool = False,
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
        self.aggregation_seconds = aggregation_seconds
        self.use_cpp = use_cpp
        self.use_hsoft = use_hsoft
        
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
        print(f"Exchange: COINBASE (WebSocket)")
        engine_type = "JAX (GPU/TPU)"
        if self.use_hsoft: engine_type = "C++ HSOFT (Background Thread)"
        elif self.use_cpp: engine_type = "C++ (OpenMP)"
        print(f"Engine: {engine_type}")
        print(f"Physics: T={self.temperature}, J={self.coupling}")
        print(f"Strategy: Viscosity Buy>{self.viscosity_buy}, Sell<{self.viscosity_sell}")
        print(f"Cool-Down: {self.min_hold_seconds}s")
        print(f"Aggregation: {self.aggregation_seconds}s")
        print("--------------------------------")
        
        # Initialize WebSocket
        try:
            from ..market.coinbase_ws import CoinbaseWebSocket
            ws_client = CoinbaseWebSocket(product_id="BTC-USD")
            asyncio.create_task(ws_client.connect())
            print("Coinbase WebSocket Connected (Background)")
        except ImportError:
            print("Error: Could not import CoinbaseWebSocket. Is websockets installed?")
            return

        try:
            # Initialize Physics Engine
            rng_key = None
            grid_state = None
            cpp_engine = None
            cpp_grid = None
            
            if self.use_hsoft:
                from ..agents.cpp_grid import CppEngine
                grid_size = self.population.ising.width
                print(f"Starting C++ Engine ({grid_size}x{grid_size})")
                cpp_engine = CppEngine()
                cpp_engine.start(grid_size)
                # Pass initial params
                cpp_engine.update_params(self.temperature, self.coupling, 0.0)
                
            elif self.use_cpp:
                from ..agents.cpp_grid import CppIsingGrid
                grid_size = self.population.ising.width
                print(f"Initializing C++ Grid ({grid_size}x{grid_size})")
                cpp_grid = CppIsingGrid(grid_size)
            else:
                rng_key = jax.random.PRNGKey(int(time.time()))
                grid_state = self.population.initialize(rng_key, temperature=self.temperature)
                print("Swarm Initialized & JIT Compiling...")
            
            step = 0
            prev_price = None
            last_physics_time = time.time()
            
            # Candle Aggregation Variables
            candle_start_time = time.time()
            ticks_buffer = []

            # Wait for WS Data
            print("Waiting for initial L2 snapshot...")
            while not ws_client.ready:
                await asyncio.sleep(0.5)
            print("Initial Flash Received. Starting Engine.")

            loop = asyncio.get_running_loop()

            while True:
                loop_start = time.perf_counter()  # Use perf_counter for better precision

                # 1. Non-Blocking Read from Memory (Zero Latency)
                mid_price, bids, asks = ws_client.get_data()

                if mid_price == 0:
                    await asyncio.sleep(0.1)
                    continue

                ts_sec = time.time()
                
                # Buffer Ticks for Candle
                ticks_buffer.append(mid_price)
                
                # 2. Check Aggregation Timer (Dynamic)
                time_since_physics = ts_sec - last_physics_time
                if time_since_physics < self.aggregation_seconds:
                    await asyncio.sleep(0.01) # Ultra-fast sleep
                    continue
                
                # === PHYSICS STEP ===
                last_physics_time = ts_sec
                
                # 2. Fill Buffer (NumPy) - Zero Allocation
                self.market_buffer[0] = mid_price
                self.market_buffer[1] = ts_sec
                
                # Fill bids/asks (upto 50)
                n_bids = min(len(bids), self.n_levels)
                n_asks = min(len(asks), self.n_levels)
                
                b_vols = [x[1] for x in bids[:n_bids]]
                a_vols = [x[1] for x in asks[:n_asks]]
                
                self.market_buffer[2:2+n_bids] = b_vols
                self.market_buffer[2+n_bids:52] = 0.0 # Pad remainder
                
                self.market_buffer[52:52+n_asks] = a_vols
                self.market_buffer[52+n_asks:102] = 0.0 # Pad remainder

                # 3. Monolithic Update (Engine Latency)
                eng_start = time.time()
                
                if self.use_hsoft:
                    # HSOFT LOGIC
                    # 1. Feed Market Data
                    cpp_engine.update_market(mid_price)
                    
                    # 2. Poll Result (Instant)
                    magnetization = cpp_engine.get_magnetization()
                    
                    # 3. Param Update (Optional, maybe periodically?)
                    # cpp_engine.update_params(self.temperature, self.coupling, h_ext)
                    
                    # Dummy Aggregation
                    agg = {"magnetization": magnetization}
                    
                elif self.use_cpp:
                    # C++ CORE LOGIC
                    h_ext = 0.0 
                    magnetization = cpp_grid.step(self.temperature, self.coupling, h_ext)
                    agg = {"magnetization": magnetization}
                    
                else:
                    # JAX CORE LOGIC
                    (
                        grid_state, rng_key, agg,
                        # New State
                        self.prev_bid_vol, self.prev_ask_vol, self.prev_flow_ema, self.price_history
                    ) = update_cycle_jit(
                        self.population, grid_state, rng_key,
                        self.market_buffer, # Passing single array
                        self.prev_bid_vol, self.prev_ask_vol, self.prev_flow_ema, self.price_history,
                        coupling=self.coupling # Pass Dynamic J
                    )
                    magnetization = float(agg["magnetization"])
                
                eng_latency = (time.time() - eng_start) * 1000 
                
                # 4. Decision & Execution (Python Scalar Logic)
                # magnetization variable is already set above
                
                # Hysteresis (Viscosity) with Configurable Thresholds
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

                        # Non-blocking trade log write
                        trade_line = f"{datetime.now()},{mid_price},{side},{abs(trade_size)},{self.capital}\n"
                        loop.run_in_executor(io_executor, lambda: open("paper_trades.csv", "a").write(trade_line))
                    else:
                        pass # Cooling Down

                prev_price = mid_price
                
                # 5. Display & Logging (Buffered)
                eng_latency = (time.perf_counter() - eng_start) * 1000
                loop_total = (time.perf_counter() - loop_start) * 1000

                # Update display every 20 steps (good balance: responsive but not spammy)
                if step % 20 == 0:
                    pnl = self.capital - 100.0  # PnL from $100 starting capital
                    pnl_pct = (pnl / 100.0) * 100
                    status = (
                        f"Step {step:05d} | "
                        f"BTC: {mid_price:.2f} | "
                        f"M: {magnetization:+.3f} | "
                        f"Pos: {self.position:+.1f} | "
                        f"Cap: ${self.capital:.2f} ({pnl_pct:+.2f}%) | "
                        f"Lat: {eng_latency:.1f}ms"
                    )
                    sys.stdout.write(status + "\r")
                    sys.stdout.flush()

                # Spike detection instrumentation (logs to console AND continues running)
                if loop_total > 5.0:
                    breakdown = {
                        "step": step,
                        "total_ms": f"{loop_total:.1f}",
                        "engine_ms": f"{eng_latency:.1f}",
                        "overhead_ms": f"{loop_total - eng_latency:.1f}",
                        "operations": []
                    }
                    if step % 503 == 0: breakdown["operations"].append("json_dump")
                    if hasattr(self, 'csv_buffer') and len(self.csv_buffer) >= 1000:
                        breakdown["operations"].append("csv_flush")

                    print(f"\n⚠️  SPIKE DETECTED: {breakdown}")
                
                # Capture State Every 2 Steps (High resolution)
                if step % 2 == 0:
                    # Cache timestamp (single call instead of two)
                    timestamp = datetime.now().isoformat()

                    # Add to RAM history (for JSON snapshot)
                    self.history.append({
                        "time": timestamp,
                        "step": step,
                        "price": mid_price,
                        "capital": self.capital,
                        "magnetization": magnetization,
                        "position": self.position,
                        "latency": eng_latency
                    })

                    # Buffer CSV Line (RAM Only)
                    if not hasattr(self, 'csv_buffer'): self.csv_buffer = []
                    self.csv_buffer.append(f"{timestamp},{step},{mid_price},{self.capital},{magnetization},{self.position},{eng_latency}\n")

                    # Flush Buffer to Disk (Every 1000 lines - non-blocking)
                    if len(self.csv_buffer) >= 1000:
                        lines_to_write = self.csv_buffer[:]  # Shallow copy
                        self.csv_buffer = []  # Clear buffer
                        loop.run_in_executor(io_executor, write_csv_sync, "paper_log.csv", lines_to_write)

                    # Write JSON State every 503 steps (prime number, non-blocking)
                    if step % 503 == 0:
                        snapshot = self.history[-1000:].copy()  # Shallow copy
                        loop.run_in_executor(io_executor, save_json_sync, "paper_state.json", snapshot)
                
                step += 1
                candle_start_time = time.time() # Reset candle timer

                # Event-Driven: Ultra-fast yield
                await asyncio.sleep(0) # Yield to event loop

                # REMOVED: gc.collect() - defeats gc.disable() purpose and causes 5-12ms spikes
                # if step % 1000 == 0:
                #     gc.collect()

        except KeyboardInterrupt:
            print("\n\n=== PAUSED ===")
        finally:
            print("Stopping...")

