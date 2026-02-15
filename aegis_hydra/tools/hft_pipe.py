import asyncio
import struct
import sys
import os
import time
import subprocess
import gc
from datetime import datetime

# Absolute Path to Daemon
DAEMON_PATH = os.path.join(os.path.dirname(__file__), '../cpp/aegis_daemon')

async def run_pipe(product_id="BTC-USD"):
    # 0. Zero Jitter Tuning
    gc.disable() # Stop automatic GC pauses
    try:
        os.nice(-10) # Priority boost (requires sudo/perms, fails gracefully)
    except:
        pass
        
    # 1. Connect WS
    from ..market.coinbase_ws import CoinbaseWebSocket
    ws = CoinbaseWebSocket(product_id)
    asyncio.create_task(ws.connect())
    
    print(f"Waiting for OpenBook...")
    while not ws.ready:
        await asyncio.sleep(0.5)
        
    print(f"Starting C++ Daemon: {DAEMON_PATH}")
    
    # 2. Launch Daemon
    process = subprocess.Popen(
        [DAEMON_PATH, "256"], # Grid Size 256
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # Pass stderr through
        bufsize=0 # Unbuffered
    )
    
    # 3. State Tracker
    class Tracker:
        capital = 100.0
        position = 0.0
        prev_price = 0.0
        fee_rate = 0.0005 # Same as PaperTrader

    tracker = Tracker()
    state_history = []
    data_buffer = [] # Shared buffer for background logging
    signal_buffer = []

    print("=== HIGH FREQUENCY PIPE ESTABLISHED ===")
    print("Python (WS) -> [Binary Float] -> C++ (Engine)")
    
    # 4. Background Sync Task (All I/O and GC happens here)
    async def background_maintenance():
        """Periodic background task to handle disk I/O and GC without blocking trading."""
        nonlocal data_buffer, signal_buffer
        log_file = "hft_market_data.csv"
        
        while True:
            await asyncio.sleep(0.2) # Update every 200ms
            
            # 1. Dashboard State
            if state_history:
                try:
                    with open("paper_state.json", "w") as f:
                        json.dump(state_history[-1000:], f)
                except Exception as e:
                    print(f"Dashboard Update Error: {e}")
            
            # 2. Market Data Logging (Batch)
            if data_buffer:
                with open(log_file, "a") as f:
                    f.writelines(data_buffer)
                data_buffer = []
                
            # 3. Signal Logging
            if signal_buffer:
                with open("hft_signals.csv", "a") as f:
                    f.writelines(signal_buffer)
                signal_buffer = []
            
            # 4. Incremental GC (Generation 0 only)
            gc.collect(0)

    async def read_signals(stdout):
        nonlocal signal_buffer
        while True:
            line = await loop.run_in_executor(None, stdout.readline)
            if not line: break
            decoded = line.decode().strip()
            
            if decoded.startswith("STATE "):
                # Parse STATE <step> <price> <mag> <latency>
                parts = decoded.split()
                if len(parts) >= 5:
                    step_val = int(parts[1])
                    price_val = float(parts[2])
                    mag_val = float(parts[3])
                    lat_val = float(parts[4])
                    
                    state_obj = {
                        "time": datetime.now().isoformat(),
                        "step": step_val,
                        "price": price_val,
                        "capital": tracker.capital,
                        "magnetization": mag_val,
                        "position": tracker.position,
                        "latency": lat_val
                    }
                    state_history.append(state_obj)
                    if len(state_history) > 2000:
                        state_history[:] = state_history[-1000:]
            else:
                print(f"\n[DAEMON SIGNAL] {datetime.now().strftime('%H:%M:%S.%f')} | {decoded}")
                
                # Signal Processing logic (Update position/capital)
                parts = decoded.split()
                if not parts: continue
                
                signal_type = parts[0]
                price_at_signal = float(parts[1]) if len(parts) > 1 else tracker.prev_price
                
                old_pos = tracker.position
                if signal_type == "BUY":
                    tracker.position = 1.0
                elif signal_type == "SELL":
                    tracker.position = -1.0
                elif signal_type.startswith("CLOSE"):
                    tracker.position = 0.0
                
                # Deduct fees on trade
                if tracker.position != old_pos:
                    fee = abs(tracker.position - old_pos) * tracker.capital * tracker.fee_rate
                    tracker.capital -= fee

                # Add to signal buffer for background writing
                signal_buffer.append(f"{datetime.now().isoformat()},{decoded}\n")

    loop = asyncio.get_event_loop()
    import json
    asyncio.create_task(read_signals(process.stdout))
    asyncio.create_task(background_maintenance())

    # 5. Core Low-Latency Loop (Zero I/O!)
    try:
        while True:
            # 3. Get Price (Fastest Path)
            loop_start = time.time()
            price, bids, asks = ws.get_data()
            
            if price > 0:
                # Update Capital based on price movement
                if tracker.prev_price > 0:
                    pct_change = (price - tracker.prev_price) / tracker.prev_price
                    tracker.capital += tracker.position * tracker.capital * pct_change
                tracker.prev_price = price

                # 4. Write to Pipe (4 bytes)
                try:
                    process.stdin.write(struct.pack('f', price))
                    process.stdin.flush()
                except BrokenPipeError:
                    print("Daemon Died!")
                    break
                
                feed_latency = (time.time() - loop_start) * 1000
                # Just append to memory buffer. No disk access here!
                data_buffer.append(f"{datetime.now().isoformat()},{price},{feed_latency:.3f}\n")
            
            # 1kHz Loop (No I/O)
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Final Flush (Manual)
        if data_buffer:
            with open("hft_market_data.csv", "a") as f:
                f.writelines(data_buffer)
        process.terminate()
        gc.enable()

if __name__ == "__main__":
    # Fix import path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    try:
        asyncio.run(run_pipe())
    except KeyboardInterrupt:
        pass
