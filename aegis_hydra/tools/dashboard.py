
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
    print("=== CRITICAL FLOW SNIPER DASHBOARD ===")
    print("Displaying: MLOFI, Hawkes Criticality (n), Volatility, Dynamic Threshold")
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
            
            # 3. MLOFI vs Threshold (Order Flow Imbalance)
            if 'mlofi' in df.columns:
                ax3.plot(df['step'], df['mlofi'], color='blue', linewidth=2, label='MLOFI')
                if 'threshold' in df.columns:
                    ax3.plot(df['step'], df['threshold'], color='red', linestyle='--', alpha=0.7, label='BUY Threshold')
                    ax3.plot(df['step'], -df['threshold'], color='green', linestyle='--', alpha=0.7, label='SELL Threshold')
                ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax3.set_ylabel("MLOFI")
                ax3.set_title("Multi-Level Order Flow Imbalance (Triggers when MLOFI crosses threshold)")
                ax3.legend(loc='upper right', fontsize=8)
                ax3.grid(True, alpha=0.3)

            # 4. Hawkes Criticality + Volatility
            ax4_twin = ax4.twinx()  # Create twin axis for volatility

            if 'criticality' in df.columns:
                line1 = ax4.plot(df['step'], df['criticality'], color='purple', linewidth=2, label='Hawkes n')
                ax4.axhline(0.6, color='red', linestyle='--', alpha=0.5, label='Critical Threshold')
                ax4.set_ylabel("Hawkes Criticality (n)", color='purple')
                ax4.set_ylim(0, 1.0)
                ax4.tick_params(axis='y', labelcolor='purple')

            if 'volatility' in df.columns:
                line2 = ax4_twin.plot(df['step'], df['volatility'], color='orange', linewidth=2, label='Volatility σ')
                ax4_twin.set_ylabel("Volatility (σ)", color='orange')
                ax4_twin.tick_params(axis='y', labelcolor='orange')

            # Combine legends
            if 'criticality' in df.columns and 'volatility' in df.columns:
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper right', fontsize=8)

            ax4.set_title("Market Regime Indicators (Trade only when n > 0.6)")
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("live_dashboard.png")
            # plt.pause(0.1) # Don't need to show window if headless, saving png is enough for VS Code
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Dashboard Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_dashboard()
