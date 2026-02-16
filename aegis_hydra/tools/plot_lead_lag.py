
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import time

# Styling
plt.style.use('dark_background')

# File paths
ANALYSIS_FILE = "hft_analysis.csv"
SIGNALS_FILE = "hft_signals.csv"

# Config
WINDOW_SIZE = 500  # Number of points to show

def read_data():
    try:
        if not os.path.exists(ANALYSIS_FILE):
             return None
             
        # Read Analysis Data (cols: time, price, z_score, criticality, mlofi)
        # It has no header, need to assign names
        df = pd.read_csv(ANALYSIS_FILE, names=['time', 'price', 'z_score', 'n', 'mlofi'])
        
        # Filter to last N points
        if len(df) > WINDOW_SIZE:
            df = df.iloc[-WINDOW_SIZE:]
            
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def read_signals():
    try:
        if not os.path.exists(SIGNALS_FILE):
             return None
        # Format: time, decoded_msg, price, pnl
        df = pd.read_csv(SIGNALS_FILE, names=['time', 'msg', 'price', 'pnl'])
        return df
    except:
        return None

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Aegis Hydra: Lead-Lag Monitor', fontsize=16, color='white')

def animate(i):
    df = read_data()
    if df is None or df.empty:
        print("Waiting for data...")
        return

    # Convert timestamps to relative seconds
    t0 = df['time'].iloc[0]
    t = df['time'] - t0
    
    # 1. Price Plot
    ax1.clear()
    ax1.plot(t, df['price'], color='cyan', label='Spot Price', linewidth=1)
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')
    
    # Overlay Signals
    sigs = read_signals()
    if sigs is not None and not sigs.empty:
        # Filter signals within the current time window
        mask = (sigs['time'] >= df['time'].min()) & (sigs['time'] <= df['time'].max())
        recent_sigs = sigs[mask]
        
        for idx, row in recent_sigs.iterrows():
            rel_t = row['time'] - t0
            if "BUY" in row['msg']:
                ax1.plot(rel_t, row['price'], '^', color='lime', markersize=10)
            elif "SELL" in row['msg']:
                ax1.plot(rel_t, row['price'], 'v', color='red', markersize=10)
            elif "CLOSE" in row['msg']:
                ax1.plot(rel_t, row['price'], 'o', color='yellow', markersize=8)

    # 2. Z-Score Plot (Lead-Lag)
    ax2.clear()
    ax2.plot(t, df['z_score'], color='magenta', label='Z-Score (Fut - Spot)', linewidth=1)
    ax2.axhline(3.0, color='lime', linestyle='--', alpha=0.5)
    ax2.axhline(-3.0, color='lime', linestyle='--', alpha=0.5)
    ax2.fill_between(t, df['z_score'], 3.0, where=(df['z_score'] > 3.0), color='lime', alpha=0.3)
    ax2.fill_between(t, df['z_score'], -3.0, where=(df['z_score'] < -3.0), color='salmon', alpha=0.3)
    ax2.set_ylabel('Z-Score')
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper left')

    # 3. Criticality Plot (Hawkes)
    ax3.clear()
    ax3.plot(t, df['n'], color='orange', label='Criticality (n)', linewidth=1)
    ax3.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='Entry (0.8)')
    ax3.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='Exit (0.2)')
    ax3.fill_between(t, df['n'], 0.8, where=(df['n'] > 0.8), color='red', alpha=0.3)
    ax3.set_ylabel('Branching Ratio (n)')
    ax3.set_ylim(0, 1.2)
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper left')

    plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=1000)

if __name__ == "__main__":
    print(f"Monitoring {ANALYSIS_FILE}...")
    plt.show()
