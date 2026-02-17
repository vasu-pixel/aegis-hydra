
import argparse
import csv
import random
import math

def generate_stress_test(output_file, days=30):
    print(f"Generating {days} days of Synthetic Quantum Stress Test...")
    
    # Constants
    SECONDS = days * 24 * 3600
    START_PRICE = 50000.0
    VOL_GROUND = 0.0001
    VOL_EXCITED = 0.0010
    CRASH_MAGNITUDE = -0.05 
    
    price = START_PRICE
    start_ts = 1700000000.0
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'price', 'regime'])
        
        for t in range(0, SECONDS, 1): # 1 second steps
            hour = (t / 3600.0) % 24.0 # Daily Cycle
            day = t / (24.0 * 3600.0)
            
            # Scenario:
            # Days 0-5: Normal
            # Day 6: CRASH
            # Day 7: Recovery
            
            regime = "GROUND"
            vol = VOL_GROUND
            mu = 0.0
            
            if day > 6.0 and day < 6.1:
                 # The Crash
                 regime = "TUNNELING"
                 vol = VOL_EXCITED * 5.0
                 mu = -0.0002 # Drift down
            elif 14.0 <= hour < 16.0:
                # Daily high vol session
                regime = "EXCITED"
                vol = VOL_EXCITED
            else:
                regime = "GROUND"
                vol = VOL_GROUND
            
            # Random Walk
            shock = random.gauss(mu, vol)
            price *= (1.0 + shock)
            
            if t % 3600 == 0:
                 print(f"Day {day:.1f} Hour {hour:.1f}: ${price:.2f} [{regime}]", end='\r')
            
            writer.writerow([start_ts + t, f"{price:.2f}", regime])
            
    print(f"\nSaved {SECONDS} ticks to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_stress_test.csv")
    parser.add_argument("--days", type=int, default=7) 
    args = parser.parse_args()
    generate_stress_test(args.output, days=args.days)
