import csv
import random
import math
import os

def generate_asset_data(filename, days, start_price, vol_base, drift_base, regime_type):
    """
    regime_type: 
    - 'trending' (BTC): High drift, high vol shocks.
    - 'noisy' (ETH): High vol, low drift, frequent oscillations.
    - 'coiling' (ETH Trap): Low vol coiling, then massive crash.
    - 'pegged' (USDT): Mean reverting to 1.0, tiny vol, rare de-peg shock.
    """
    SECONDS = days * 24 * 3600
    start_ts = 1704067200 # Jan 1 2024
    
    price = start_price
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'price', 'regime'])
        
        for t in range(0, SECONDS, 10): # 10 second steps for speed
            # Determine regime specific shock
            if regime_type == 'trending':
                # BTC style: Moderate trend
                mu = (drift_base * 0.1 / SECONDS) * 10 # 2% drift over 7 days
                vol = (vol_base * 0.5 / math.sqrt(SECONDS)) * math.sqrt(10)
                regime = "Excited"
            elif regime_type == 'noisy':
                mu = 0
                vol = (vol_base * 1.5 / math.sqrt(SECONDS)) * math.sqrt(10)
                regime = "Excited"
            elif regime_type == 'coiling':
                # Siren's Call: First 3 days low vol, then crash
                if t < SECONDS * 0.4:
                    mu = 0
                    vol = (0.05 / math.sqrt(SECONDS)) * math.sqrt(10) # Tiny vol
                    regime = "Ground"
                else:
                    # Realistic Trap: -10% drop
                    mu = (-0.1 / (0.6 * SECONDS)) * 10 
                    vol = (0.5 / math.sqrt(SECONDS)) * math.sqrt(10)
                    regime = "Excited"
            else:
                mu = (1.0 - price) * 0.1 
                vol = (0.0001 / math.sqrt(SECONDS)) * math.sqrt(10)
                regime = "Ground"
                
            shock = random.gauss(mu, vol)
            price *= (1.0 + shock)
            writer.writerow([start_ts + t, f"{price:.6f}", regime])

if __name__ == "__main__":
    print("Generating Tri-Force Synthetic Data...")
    # BTC: $90k, High Vol (80%), Positive Drift
    generate_asset_data("backtest_BTCUSD.csv", 7, 90000.0, 0.8, 0.2, 'trending')
    print("Done: backtest_BTCUSD.csv")
    
    # ETH: $3k, Siren's Call Trap
    generate_asset_data("backtest_ETHUSD.csv", 7, 3000.0, 1.0, 0.0, 'coiling')
    print("Done: backtest_ETHUSD.csv")
    
    # USDT: $1.00, Pegged (0.5% vol), Mean Reverting
    generate_asset_data("backtest_USDTUSD.csv", 7, 1.0, 0.005, 0.0, 'pegged')
    print("Done: backtest_USDTUSD.csv")
