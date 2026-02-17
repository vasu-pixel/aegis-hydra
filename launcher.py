import subprocess
import os
import sys
import time

# --- TRI-FORCE CONFIGURATION ---
ASSETS = ["BTCUSD", "ETHUSD", "USDTUSD"]

def main():
    print("=== AEGIS HYDRA: TRI-FORCE PORTFOLIO LAUNCHER ===")
    print(f"Allocating $50 across: {', '.join(ASSETS)}")
    
    processes = []
    
    # Ensure current directory is in sys.path for the subprocesses
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    try:
        for asset in ASSETS:
            print(f"🚀 Launching Engine for {asset}...")
            # Run hft_pipe as a module to handle relative imports correctly
            proc = subprocess.Popen(
                [sys.executable, "-m", "aegis_hydra.tools.hft_pipe", asset],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env
            )
            processes.append(proc)
            time.sleep(2) # Staggered startup to avoid WS rate limits
            
        print("\n✅ All engines active. Use Ctrl+C to stop the portfolio.")
        
        # Keep launcher alive
        while True:
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    print(f"⚠️ Warning: Engine for {ASSETS[i]} died. Restarting...")
                    processes[i] = subprocess.Popen(
                        [sys.executable, "-m", "aegis_hydra.tools.hft_pipe", ASSETS[i]],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        env=env
                    )
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Portfolio Engines...")
        for proc in processes:
            proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
