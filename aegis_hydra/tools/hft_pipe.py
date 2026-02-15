
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
    
    try:
        while True:
            # 3. Get Price (Fastest Path)
            price, _, _ = ws.get_data()
            
            if price > 0:
                # 4. Write to Pipe (4 bytes)
                try:
                    process.stdin.write(struct.pack('f', price))
                    process.stdin.flush()
                except BrokenPipeError:
                    print("Daemon Died!")
                    break
                    
                # 5. Read Signals (Non-blocking check?)
                # For simplicity, we just PUSH data here. 
                # Reading stdout requires another thread or asyncio stream.
                # But since C++ only prints on signal, maybe we ignore reading for now?
                # Actually, let's just loop.
            
            # 1kHz Loop
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        process.terminate()

if __name__ == "__main__":
    # Fix import path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    try:
        asyncio.run(run_pipe())
    except KeyboardInterrupt:
        pass
