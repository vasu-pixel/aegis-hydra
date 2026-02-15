import asyncio
import struct
import sys
import os
import time
import subprocess
import gc
import concurrent.futures
import json
from datetime import datetime

# Absolute Path to Daemon
DAEMON_PATH = os.path.join(os.path.dirname(__file__), '../cpp/aegis_daemon')

# Optimization: Global Thread Pools
# parse_executor: for JSON parsing
# io_executor: for disk writes (CSV/JSON)
parse_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def save_state_sync(data):
    try:
        with open("paper_state.json", "w") as f:
            json.dump(data, f)
    except: pass

def log_csv_sync(filename, lines):
    try:
        with open(filename, "a") as f:
            f.writelines(lines)
    except: pass

async def run_pipe(product_id="BTC-USD"):
    # 0. Zero Jitter Tuning
    gc.disable()
    try:
        os.nice(-19) # Maximum priority
    except:
        pass
        
    loop = asyncio.get_running_loop()
    
    # 1. Connect WS
    from ..market.coinbase_ws import CoinbaseWebSocket
    ws = CoinbaseWebSocket(product_id)
    
    print(f"Starting C++ Daemon: {DAEMON_PATH}")
    
    # 2. Launch Daemon
    process = subprocess.Popen(
        [DAEMON_PATH], 
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0
    )
    
    # 3. State Tracker
    class Tracker:
        capital = 100.0
        position = 0.0
        prev_price = 0.0
        fee_rate = 0.0005

    tracker = Tracker()
    state_history = []
    data_buffer = [] 
    signal_buffer = []

    print("=== HIGH FREQUENCY ZERO-JITTER PIPE ===")

    async def background_maintenance():
        nonlocal data_buffer, signal_buffer
        while True:
            await asyncio.sleep(0.2) # Reduced frequency for stability
            
            if state_history:
                # Offload slow JSON dump to thread
                loop.run_in_executor(io_executor, save_state_sync, state_history[-500:])
            
            if data_buffer:
                lines_to_write = data_buffer[:]
                data_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_market_data.csv", lines_to_write)
                
            if signal_buffer:
                sigs_to_write = signal_buffer[:]
                signal_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_signals.csv", sigs_to_write)
            
            gc.collect(0)

    # 5. Signal Reader (C++ -> Python)
    async def read_signals(stdout):
        nonlocal signal_buffer
        while True:
            line = await loop.run_in_executor(None, stdout.readline)
            if not line: break
            decoded = line.decode().strip()
            
            if decoded.startswith("STATE "):
                parts = decoded.split()
                if len(parts) >= 6:
                    state_obj = {
                        "time": datetime.now().isoformat(),
                        "step": int(parts[1]),
                        "price": float(parts[2]),
                        "capital": tracker.capital,
                        "magnetization": float(parts[3]),
                        "position": tracker.position,
                        "latency": float(parts[4]), # Phys
                        "net_latency": float(parts[5]),
                        "threshold": float(parts[6]) if len(parts) > 6 else 0.6
                    }
                    state_history.append(state_obj)
                    if len(state_history) > 1000:
                        state_history[:] = state_history[-500:]
            else:
                print(f"\n[DAEMON SIGNAL] {datetime.now().strftime('%H:%M:%S.%f')} | {decoded}")
                parts = decoded.split()
                if not parts: continue
                
                signal_type = parts[0]
                old_pos = tracker.position
                if signal_type == "BUY": tracker.position = 1.0
                elif signal_type == "SELL": tracker.position = -1.0
                elif signal_type.startswith("CLOSE"): tracker.position = 0.0
                
                if tracker.position != old_pos:
                    fee = abs(tracker.position - old_pos) * tracker.capital * tracker.fee_rate
                    tracker.capital -= fee

                signal_buffer.append(f"{datetime.now().isoformat()},{decoded}\n")

    asyncio.create_task(read_signals(process.stdout))
    asyncio.create_task(background_maintenance())

    # 6. Optimized WS Loop (Threaded Parsing + Zero-Yield)
    import websockets
    async with websockets.connect(ws.WS_URL, max_size=None) as socket:
        # Subscribe to both high-frequency book changes and trade tickers
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [product_id],
            "channel": "level2"
        }
        await socket.send(json.dumps(subscribe_msg))
        
        await socket.send(json.dumps({
            "type": "subscribe", "product_ids": [product_id], "channel": "ticker"
        }))
        print("Subscribed to level2 & ticker. Restoring high-frequency pulse...")

        try:
            while True:
                # Direct await is safer and still yields to event loop for background tasks.
                message = await socket.recv()
                start_time = time.time()
                    
                # OPTIMIZATION: Threaded JSON Parsing
                data = await loop.run_in_executor(parse_executor, json.loads, message)
                
                channel = data.get("channel")
                if channel == "l2_data":
                    ws._handle_l2(data)
                elif channel == "ticker":
                    ws._handle_ticker(data)

                # Get Price (Fast path)
                price = ws.get_mid_price()
                
                if price > 0:
                    # Network Latency calculation
                    net_latency = 0.0
                    if channel == "ticker":
                        events = data.get("events", [])
                        if events and events[0].get("tickers"):
                            server_time_str = events[0]["tickers"][0].get("time")
                            if server_time_str:
                                server_ts = datetime.fromisoformat(server_time_str.replace('Z', '')).timestamp()
                                net_latency = (time.time() - server_ts) * 1000

                    # Update Tracker PnL
                    if tracker.prev_price > 0:
                        pct_change = (price - tracker.prev_price) / tracker.prev_price
                        tracker.capital += tracker.position * tracker.capital * pct_change
                    tracker.prev_price = price

                    # Write Binary Packet (Price + Latency)
                    try:
                        process.stdin.write(struct.pack('ff', price, net_latency))
                        process.stdin.flush()
                    except BrokenPipeError:
                        break
                        
                    feed_latency = (time.time() - start_time) * 1000
                    data_buffer.append(f"{datetime.now().isoformat()},{price},{feed_latency:.3f}\n")
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            process.terminate()
            gc.enable()

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    try:
        asyncio.run(run_pipe())
    except KeyboardInterrupt:
        pass
