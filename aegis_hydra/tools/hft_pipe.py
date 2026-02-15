
import asyncio
import struct
import sys
import os
import time
import subprocess
from datetime import datetime

# Absolute Path to Daemon
DAEMON_PATH = os.path.join(os.path.dirname(__file__), '../cpp/aegis_daemon')

async def run_pipe(product_id="BTC-USD"):
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
        [DAEMON_PATH, "1000"], # Grid Size 1000
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

    print("=== HIGH FREQUENCY PIPE ESTABLISHED ===")
    print("Python (WS) -> [Binary Float] -> C++ (Engine)")
    
    # 4. Non-blocking Signal Reader
    state_history = []
    
    async def sync_dashboard():
        """Periodic background task to update paper_state.json without blocking trading."""
        while True:
            await asyncio.sleep(0.5) # Update every 500ms
            if state_history:
                try:
                    # Write only the last 1000 points to keep it fast
                    with open("paper_state.json", "w") as f:
                        json.dump(state_history[-1000:], f)
                except Exception as e:
                    print(f"Dashboard Sync Error: {e}")

    async def read_signals(stdout):
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
                    # Cap memory usage
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

                # Append signal to a local log
                with open("hft_signals.csv", "a") as f:
                    f.write(f"{datetime.now().isoformat()},{decoded}\n")

    loop = asyncio.get_event_loop()
    import json
    asyncio.create_task(read_signals(process.stdout))
    asyncio.create_task(sync_dashboard())

    # 5. Data Storage Buffer
    data_buffer = []
    log_file = "hft_market_data.csv"
    
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
                    
                # Store data (Buffered)
                data_buffer.append(f"{datetime.now().isoformat()},{price},{feed_latency:.3f}\n")
                if len(data_buffer) >= 100:
                    with open(log_file, "a") as f:
                        f.writelines(data_buffer)
                    data_buffer = []
            
            # 1kHz Loop (Adjustable)
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Final Flush
        if data_buffer:
            with open(log_file, "a") as f:
                f.writelines(data_buffer)
        process.terminate()

if __name__ == "__main__":
    # Fix import path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    try:
        asyncio.run(run_pipe())
    except KeyboardInterrupt:
        pass
