import csv
import random
import math
import os

def generate_chaos_data(filename, asset_type):
    """
    asset_type: 'BTC', 'ETH', 'USDT'
    Simulates 30 days of sequential extreme regimes.
    """
    days = 30
    SECONDS = days * 24 * 3600
    start_ts = 1704067200 
    
    price = 90000.0 if asset_type == 'BTC' else (3000.0 if asset_type == 'ETH' else 1.0)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'price', 'regime'])
        
        for t in range(0, SECONDS, 10):
            day = t / (24 * 3600)
            
            mu = 0
            vol = 0
            regime = "Ground"
            
            # --- CHAOS TIMELINE ---
            
            # Phase 1: Lead Rally (Days 0-5)
            if day < 5:
                if asset_type == 'BTC': mu, vol = 0.05/SECONDS, 0.4/math.sqrt(SECONDS)
                elif asset_type == 'ETH': mu, vol = 0.08/SECONDS, 0.6/math.sqrt(SECONDS)
                else: mu, vol = (1.0 - price)*0.1, 0.0001/math.sqrt(SECONDS)
                regime = "Excited"
                
            # Phase 2: Siren's Call (Days 5-7)
            elif day < 7:
                if asset_type == 'BTC': mu, vol = 0, 0.1/math.sqrt(SECONDS) # Consolidation
                elif asset_type == 'ETH':
                    if day < 6.5: mu, vol = 0, 0.02/math.sqrt(SECONDS) # Coiling Trap
                    else: mu, vol = -0.2/SECONDS*100, 1.5/math.sqrt(SECONDS) # Flash Crash 20%
                else: mu, vol = (1.0 - price)*0.1, 0.0001/math.sqrt(SECONDS)
                regime = "Ground" if (asset_type == 'ETH' and day < 6.5) else "Excited"
                
            # Phase 3: Hyper-Vol (Days 7-12)
            elif day < 12:
                mu = math.sin(t / 3600) * 0.001 # Large oscillations
                vol = 1.2/math.sqrt(SECONDS)
                regime = "Excited"
                
            # Phase 4: Peg Break (Days 12-14)
            elif day < 14:
                if asset_type == 'USDT':
                    mu, vol = (0.95 - price)*0.01, 0.01/math.sqrt(SECONDS) # De-peg to 0.95
                else:
                    mu, vol = 0, 0.8/math.sqrt(SECONDS)
                regime = "Excited"
                
            # Phase 5: Recovery (Days 14-30)
            else:
                mu, vol = 0.1/SECONDS, 0.4/math.sqrt(SECONDS)
                regime = "Excited"

            shock = random.gauss(mu * 10, vol * math.sqrt(10))
            price *= (1.0 + shock)
            writer.writerow([start_ts + t, f"{price:.6f}", regime])

if __name__ == "__main__":
    print("🚀 GENERATING CHAOS STRESS TEST DATA (30 DAYS)...")
    generate_chaos_data("chaos_BTCUSD.csv", 'BTC')
    print("Done: chaos_BTCUSD.csv")
    generate_chaos_data("chaos_ETHUSD.csv", 'ETH')
    print("Done: chaos_ETHUSD.csv")
    generate_chaos_data("chaos_USDTUSD.csv", 'USDT')
    print("Done: chaos_USDTUSD.csv")
