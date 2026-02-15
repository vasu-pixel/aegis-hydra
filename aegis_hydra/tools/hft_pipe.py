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
    # 0. Zero Jitter Tuning: Kill the Garbage Collector
    gc.collect() # Clean up once
    gc.freeze()  # Move all current objects to permanent generation 
    gc.disable() # Stop all future collections
    # Set thresholds to 0 for extra safety
    try: gc.set_threshold(0) 
    except: pass

    try:
        os.nice(-20) # Absolute Priority
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

    print("=== ZERO-WAIT DECOUPLED HFT PIPE ===")

    msg_queue = asyncio.Queue(maxsize=5000)
    packet_queue = asyncio.Queue(maxsize=5000)

    # 4. Background Tasks
    async def background_maintenance():
        nonlocal data_buffer, signal_buffer
        while True:
            await asyncio.sleep(0.5) 
            if state_history:
                loop.run_in_executor(io_executor, save_state_sync, state_history[-500:])
            if data_buffer:
                lines = data_buffer[:]
                data_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_market_data.csv", lines)
            if signal_buffer:
                sigs = signal_buffer[:]
                signal_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_signals.csv", sigs)
            gc.collect(0)

    # 4b. Non-Blocking Pipe Writer
    async def pipe_writer():
        while True:
            packet = await packet_queue.get()
            try:
                process.stdin.write(packet)
                process.stdin.flush()
            except BrokenPipeError: break
            packet_queue.task_done()

    import re
    # PRE-COMPILED REGEX for extreme speed
    # Extracts first price from Level 2 update or Ticker
    # Format: ..."price_level":"123.45"... or ..."price":"123.45"...
    price_re = re.compile(r'"price(?:_level)?":"(\d+\.?\d*)"')
    side_re = re.compile(r'"side":"(bid|ask)"')
    size_re = re.compile(r'"new_quantity":"(\d+\.?\d*)"')

    # 5. Logic Worker (Processing & Strategy)
    async def logic_worker():
        nonlocal data_buffer, signal_buffer
        while True:
            message, recv_time = await msg_queue.get()
            
            # FAST PATH: String processing (Zero JSON Allocation)
            is_l2 = '"channel":"l2_data"' in message
            is_ticker = '"channel":"ticker"' in message
            
            price = 0.0
            channel = "unknown"
            
            if is_ticker:
                # Optimized extraction for trade price
                price_match = price_re.search(message)
                if price_match:
                    price = float(price_match.group(1))
                    ws.latest_ticker_price = price
                    ws.ready = True
                    channel = "ticker"
                else:
                    data = await loop.run_in_executor(parse_executor, json.loads, message)
                    ws._handle_ticker(data)
                    price = ws.get_mid_price()
            elif is_l2:
                # Must update book to get accurate mid-price
                # But we use the now-optimized heap-based handle_l2
                data = await loop.run_in_executor(parse_executor, json.loads, message)
                ws._handle_l2(data)
                price = ws.get_mid_price()
                channel = "l2_data"
            else:
                # Other messages (Snapshots, etc.)
                data = await loop.run_in_executor(parse_executor, json.loads, message)
                channel = data.get("channel", "unknown")
                if channel == "l2_data": ws._handle_l2(data)
                elif channel == "ticker": ws._handle_ticker(data)
                price = ws.get_mid_price()

            if price > 0:
                net_latency = 0.0
                # Latency calc only on tickers (has server 'time' field)
                if is_ticker:
                    time_idx = message.find('"time":"')
                    if time_idx != -1:
                        ts_str = message[time_idx+8:time_idx+34] # Close enough to ISO
                        try:
                            server_ts = datetime.fromisoformat(ts_str.replace('Z', '')).timestamp()
                            net_latency = (time.time() - server_ts) * 1000
                        except: pass

                if tracker.prev_price > 0:
                    pct = (price - tracker.prev_price) / tracker.prev_price
                    tracker.capital += tracker.position * tracker.capital * pct
                tracker.prev_price = price

                packet_queue.put_nowait(struct.pack('ff', price, net_latency))
                
                # Zero-allocation logging (just float/timestamp)
                proc_latency = (time.time() - recv_time) * 1000
                data_buffer.append(f"{time.time()},{price},{proc_latency:.3f}\n")
            
            msg_queue.task_done()

    # 5b. Signal Reader (C++ -> Python)
    # Optimization: Re-use state object to avoid garbage collection
    shared_state = {
        "time": "", "step": 0, "price": 0.0, "capital": 100.0,
        "magnetization": 0.0, "position": 0.0, "latency": 0.0,
        "net_latency": 0.0, "threshold": 0.6
    }

    async def read_signals(stdout):
        nonlocal signal_buffer, shared_state
        while True:
            line = await loop.run_in_executor(None, stdout.readline)
            if not line: break
            decoded = line.decode().strip()
            
            if decoded.startswith("STATE "):
                parts = decoded.split()
                if len(parts) >= 6:
                    shared_state["time"] = time.time()
                    shared_state["step"] = int(parts[1])
                    shared_state["price"] = float(parts[2])
                    shared_state["capital"] = tracker.capital
                    shared_state["magnetization"] = float(parts[3])
                    shared_state["position"] = tracker.position
                    shared_state["latency"] = float(parts[4])
                    shared_state["net_latency"] = float(parts[5])
                    shared_state["threshold"] = float(parts[6]) if len(parts) > 6 else 0.6
                    
                    # Only append periodically to history to avoid memory bloat
                    if int(parts[1]) % 100 == 0:
                        state_history.append(shared_state.copy())
                        if len(state_history) > 500: state_history[:] = state_history[-200:]
            else:
                print(f"[SIGNAL] {datetime.now().strftime('%H:%M:%S.%f')} | {decoded}")
                parts = decoded.split()
                if not parts: continue
                old_pos = tracker.position
                if parts[0] == "BUY": tracker.position = 1.0
                elif parts[0] == "SELL": tracker.position = -1.0
                elif parts[0].startswith("CLOSE"): tracker.position = 0.0
                
                if tracker.position != old_pos:
                    fee = abs(tracker.position - old_pos) * tracker.capital * tracker.fee_rate
                    tracker.capital -= fee
                signal_buffer.append(f"{time.time()},{decoded}\n")

    asyncio.create_task(read_signals(process.stdout))
    asyncio.create_task(background_maintenance())
    asyncio.create_task(pipe_writer())
    asyncio.create_task(logic_worker())

    # 6. Optimized Network Listener
    import websockets
    async with websockets.connect(ws.WS_URL, max_size=None) as socket:
        await socket.send(json.dumps({"type": "subscribe", "product_ids": [product_id], "channel": "level2"}))
        await socket.send(json.dumps({"type": "subscribe", "product_ids": [product_id], "channel": "ticker"}))
        print("Network listener HOT. RESTING NO MORE.")

        try:
            while True:
                message = await socket.recv()
                msg_queue.put_nowait((message, time.time()))
        except KeyboardInterrupt: pass
        finally:
            process.terminate()
            gc.enable()

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    try:
        asyncio.run(run_pipe())
    except KeyboardInterrupt:
        pass
