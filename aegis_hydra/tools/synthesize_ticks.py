import argparse
import pandas as pd
import numpy as np
import datetime

def brownian_bridge(start, end, high, low, n=60):
    """
    Generate a Brownian Bridge path between start and end with constraints.
    Simplified version: meaningful random walk scaled to hit target.
    """
    dt = 1.0
    sigma = (high - low) / np.sqrt(n) * 2.0 # Volatility estimate
    
    # Standard Brownian Motion
    W = np.cumsum(np.random.normal(0, 1, n))
    
    # Brownian Bridge formula: B(t) = W(t) - (t/T) * W(T)
    # pinned at 0 and 0
    bridge = W - (np.arange(n) / (n - 1)) * W[-1]
    
    # Scale to match price constraints (approx)
    # We add this to a linear interpolation from start to end
    linear = np.linspace(start, end, n)
    
    path = linear + bridge * sigma * 0.5
    
    # Simple clamp to high/low (not perfect but valid for testing)
    path = np.clip(path, low, high)
    
    # Force start/end exact
    path[0] = start
    path[-1] = end
    
    return path

def synthesize_ticks(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    all_ticks = []
    
    total = len(df)
    
    for i, row in df.iterrows():
        print(f"Processing candle {i}/{total}...", end="\r")
        
        start_ts = row["timestamp_us"]
        open_p = row["open"]
        high_p = row["high"]
        low_p = row["low"]
        close_p = row["close"]
        vol = row["volume"]
        
        # 60 seconds
        n_ticks = 60
        prices = brownian_bridge(open_p, close_p, high_p, low_p, n=n_ticks)
        
        # Linear volume dist
        vols = np.full(n_ticks, vol / n_ticks)
        
        # Timestamps
        # 1m = 60s = 60,000,000 us
        step_us = 1_000_000
        timestamps = [start_ts + j * step_us for j in range(n_ticks)]
        
        for j in range(n_ticks):
            all_ticks.append({
                "timestamp_us": timestamps[j],
                "symbol": row["symbol"],
                "open": prices[j],
                "high": prices[j], # High/Low same as price for tick
                "low": prices[j],
                "close": prices[j],
                "volume": vols[j],
                "funding_rate": row["funding_rate"]
            })
            
    print(f"\nGenerated {len(all_ticks)} ticks.")
    
    tick_df = pd.DataFrame(all_ticks)
    tick_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    synthesize_ticks(args.input, args.output)
