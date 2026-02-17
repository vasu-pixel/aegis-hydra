import csv
import math
import statistics
import sys
import numpy as np

# A Quantum Market Analyzer
# "Energy" = Jarque-Bera Test Statistic (Deviation from Gaussian)
# Ground State (n=0): Energy ~ 0 (Gaussian) -> Harvest
# Excited State (n>0): Energy >> 0 (Fat Tailed/Skewed) -> Trend

FILE_PATH = '../hft_market_data.csv'
WINDOW_SIZE = 50

def calculate_moments(window):
    n = len(window)
    if n < 3: return 0, 0, 0
    
    mean = statistics.mean(window)
    std_dev = statistics.stdev(window)
    if std_dev == 0: return 0, 0, 0
    
    # Skewness (3rd Moment)
    skew_sum = sum(((x - mean) / std_dev) ** 3 for x in window)
    skew = skew_sum / n
    
    # Kurtosis (4th Moment) - Pearson (Normal = 3.0)
    kurt_sum = sum(((x - mean) / std_dev) ** 4 for x in window)
    kurt = kurt_sum / n
    
    return mean, skew, kurt

def calculate_energy(skew, kurt, n):
    # Jarque-Bera Statistic
    # JB = (n/6) * (S^2 + (1/4)*(K-3)^2)
    # Energy = JB / Normalization (to map to roughly 0.0 - 1.0 range for simple thresholding)
    
    excess_kurt = kurt - 3.0
    jb = (n / 6.0) * (skew**2 + 0.25 * (excess_kurt**2))
    
    # Log-Energy implies Order of Magnitude deviation
    # We clamp it and scale it
    energy = math.log1p(jb) / 10.0 # Heuristic scaling
    return energy

def analyze_market_quantum_state(path):
    print(f"🔮 QUANTUM ANALYZER: Scanning {path}...")
    
    prices = []
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2: continue
                try:
                    prices.append(float(row[1])) # Price
                except: continue
    except FileNotFoundError:
        print("File not found.")
        return

    print(f"Loaded {len(prices)} ticks.")
    print("Timestamp | Price  | Skew   | Kurt   | Energy (n) | State")
    print("-" * 70)
    
    energies = []
    
    # Analysis Loop
    for i in range(WINDOW_SIZE, len(prices), 100): # Skip steps for speed
        window = prices[i-WINDOW_SIZE:i]
        
        # We calculate moments of RETURNS, not Prices 
        # (Prices are random walk, Returns should be Gaussian in Ground State)
        returns = [math.log(window[j]/window[j-1]) for j in range(1, len(window))]
        
        if len(returns) < 10: continue
        
        mean, skew, kurt = calculate_moments(returns)
        energy = calculate_energy(skew, kurt, len(returns))
        energies.append(energy)
        
        state = "GROUND (Harvest)" if energy < 0.1 else "EXCITED (Danger!)"
        if energy > 0.4: state = "PLASMA (Crash)"
        
        # Print sample lines
        if i % 5000 == 0:
            print(f"{i:9d} | {window[-1]:.2f} | {skew:6.2f} | {kurt:6.2f} | {energy:6.3f}     | {state}")

    print("-" * 70)
    avg_energy = statistics.mean(energies)
    max_energy = max(energies)
    print(f"Avg Quantum Energy: {avg_energy:.4f}")
    print(f"Max Quantum Energy: {max_energy:.4f}")
    print("=" * 70)
    
    print("\nRECOMMENDED THRESHOLDS:")
    print(f"Harvest Limit (n=0): < {avg_energy:.3f}")
    print(f"Trend Start   (n=1): > {avg_energy + statistics.stdev(energies):.3f}")

if __name__ == "__main__":
    analyze_market_quantum_state(FILE_PATH)
