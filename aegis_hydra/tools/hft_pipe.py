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

    # 5. Logic Worker (Processing & Strategy)
    async def logic_worker():
        nonlocal data_buffer, signal_buffer
        while True:
            message, recv_time = await msg_queue.get()
            
            # Offload JSON to Thread Pool
            data = await loop.run_in_executor(parse_executor, json.loads, message)
            
            channel = data.get("channel")
            if channel == "l2_data":
                ws._handle_l2(data)
            elif channel == "ticker":
                ws._handle_ticker(data)

            price = ws.get_mid_price()
            if price > 0:
                net_latency = 0.0
                if channel == "ticker":
                    events = data.get("events", [])
                    if events and events[0].get("tickers"):
                        ts_str = events[0]["tickers"][0].get("time")
                        if ts_str:
                            server_ts = datetime.fromisoformat(ts_str.replace('Z', '')).timestamp()
                            net_latency = (time.time() - server_ts) * 1000

                if tracker.prev_price > 0:
                    pct = (price - tracker.prev_price) / tracker.prev_price
                    tracker.capital += tracker.position * tracker.capital * pct
                tracker.prev_price = price

                # Push to Pipe Writer
                packet_queue.put_nowait(struct.pack('ff', price, net_latency))
                
                proc_latency = (time.time() - recv_time) * 1000
                data_buffer.append(f"{datetime.now().isoformat()},{price},{proc_latency:.3f}\n")
            
            msg_queue.task_done()

    # 5b. Signal Reader (C++ -> Python)
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
                        "step": int(parts[1]), "price": float(parts[2]),
                        "capital": tracker.capital, "magnetization": float(parts[3]),
                        "position": tracker.position, "latency": float(parts[4]),
                        "net_latency": float(parts[5]),
                        "threshold": float(parts[6]) if len(parts) > 6 else 0.6
                    }
                    state_history.append(state_obj)
                    if len(state_history) > 1000: state_history[:] = state_history[-500:]
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
                signal_buffer.append(f"{datetime.now().isoformat()},{decoded}\n")

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
