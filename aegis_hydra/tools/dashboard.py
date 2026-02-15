
import json
import time
import os
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def run_dashboard():
    print("Starting Dashboard... (Watching paper_state.json)")
    # plt.ion() # Removed for headless
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    while True:
        try:
            if not os.path.exists("paper_state.json"):
                time.sleep(1)
                continue
                
            with open("paper_state.json", "r") as f:
                data = json.load(f)
            
            if not data:
                time.sleep(1)
                continue
                
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            
            # Clear axes
            ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
            
            # 1. Price & Trades
            ax1.plot(df['step'], df['price'], color='orange', label='BTC Price')
            ax1.set_ylabel("Price")
            ax1.set_title("Live Market")
            ax1.grid(True, alpha=0.3)
            
            # 2. Capital
            ax2.plot(df['step'], df['capital'], color='green', label='Equity')
            ax2.set_ylabel("Capital ($)")
            ax2.set_title(f"Equity Curve (Current: ${df['capital'].iloc[-1]:,.2f})")
            ax2.grid(True, alpha=0.3)
            
            # 3. Magnetization
            ax3.plot(df['step'], df['magnetization'], color='purple', label='Magnetization')
            ax3.set_ylabel("M (-1 to 1)")
            ax3.set_title("Ising Regime")
            ax3.axhline(0.7, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(-0.7, color='red', linestyle='--', alpha=0.5)
            ax3.set_ylim(-1.1, 1.1)
            ax3.grid(True, alpha=0.3)

            # 4. Latency
            if 'latency' in df.columns:
                ax4.plot(df['step'], df['latency'], color='blue', label='Latency (ms)')
                ax4.set_ylabel("Latency (ms)")
                ax4.set_title(f"System Latency (Current: {df['latency'].iloc[-1]:.1f}ms)")
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("live_dashboard.png")
            # plt.pause(0.1) # Don't need to show window if headless, saving png is enough for VS Code
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Dashboard Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_dashboard()
