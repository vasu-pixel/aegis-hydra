
import asyncio
import struct
import sys
import os
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
    
    print("=== HIGH FREQUENCY PIPE ESTABLISHED ===")
    print("Python (WS) -> [Binary Float] -> C++ (Engine)")
    
    # 3. Non-blocking Signal Reader
    async def read_signals(stdout):
        while True:
            line = await loop.run_in_executor(None, stdout.readline)
            if not line: break
            decoded = line.decode().strip()
            print(f"\n[DAEMON SIGNAL] {datetime.now().strftime('%H:%M:%S.%f')} | {decoded}")
            # Append signal to a local log
            with open("hft_signals.csv", "a") as f:
                f.write(f"{datetime.now().isoformat()},{decoded}\n")

    loop = asyncio.get_event_loop()
    asyncio.create_task(read_signals(process.stdout))

    # 4. Data Storage Buffer
    data_buffer = []
    log_file = "hft_market_data.csv"
    
    try:
        while True:
            # 3. Get Price (Fastest Path)
            price, bids, asks = ws.get_data()
            
            if price > 0:
                # 4. Write to Pipe (4 bytes)
                try:
                    process.stdin.write(struct.pack('f', price))
                    process.stdin.flush()
                except BrokenPipeError:
                    print("Daemon Died!")
                    break
                    
                # Store data (Buffered)
                data_buffer.append(f"{datetime.now().isoformat()},{price}\n")
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
