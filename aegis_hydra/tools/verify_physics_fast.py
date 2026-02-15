import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np

# Import IsingGrid
from aegis_hydra.agents.ising_grid import IsingGrid

def load_data(path: str):
    """Load and preprocess data for scan."""
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Calculate simple imbalance signal from OHLCV
    # Signal = (Close - Open) / Price * Scale
    # Volatility = (High - Low) / Price
    
    close = jnp.array(df["close"].values, dtype=jnp.float32)
    open_p = jnp.array(df["open"].values, dtype=jnp.float32)
    high = jnp.array(df["high"].values, dtype=jnp.float32)
    low = jnp.array(df["low"].values, dtype=jnp.float32)
    
    # Vectorized signal calculation
    delta = close - open_p
    price = close + 1e-10
    
    # External Field h = delta / price * sensitivity
    # We want a 1% move to be a strong signal (h ~ 0.1 to 0.5)
    h_ext = (delta / price) * 100.0 
    
    return h_ext, close

def run_fast_scan(data_path, grid_size=100, temp=2.27, j_coupling=1.0):
    # 1. Setup
    h_ext_seq, prices = load_data(data_path)
    n_steps = len(h_ext_seq)
    print(f"Data loaded: {n_steps} steps.")
    
    # 2. Initialize Grid
    print(f"Initializing Ising Grid ({grid_size}x{grid_size})...")
    grid = IsingGrid(height=grid_size, width=grid_size, J=j_coupling)
    key = jax.random.PRNGKey(42)
    init_state = grid.initialize(key, temperature=temp)
    
    # 3. Define Scan Function
    # scan(carry, x) -> (carry, y)
    
    def step_fn(carry, h):
        state, key = carry
        key, subkey = jax.random.split(key)
        new_state = grid.step(state, h, subkey)
        m = new_state.magnetization
        return (new_state, key), m

    # 4. JIT Compile & Run Scan
    # We must JIT the scan itself to avoid Python dispatch overhead for 600k loading
    print("Compiling & Running JAX Scan (JIT Mode)...")
    start_t = time.time()
    
    @eqx.filter_jit
    def run_scan_jit(init_s, k, h_seq):
        final_c, m_seq = jax.lax.scan(step_fn, (init_s, k), h_seq)
        return m_seq

    magnetization_seq = run_scan_jit(init_state, key, h_ext_seq)
    
    # Block until ready
    magnetization_seq.block_until_ready()
    end_t = time.time()
    
    duration = end_t - start_t
    mps = n_steps / duration
    print(f"Done! Processed {n_steps} steps in {duration:.2f}s ({mps:.0f} steps/sec)")
    
    return magnetization_seq, prices

def plot_results(m_seq, prices, output_file="physics_verification.png"):
    print("Plotting results...")
    
    # Move to CPU
    m = np.array(m_seq)
    p = np.array(prices)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Price
    ax1.plot(p, color="orange", label="BTC Price")
    ax1.set_title("Market Data (1s Ticks)")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    
    # Magnetization
    ax2.plot(m, color="purple", label="Magnetization", linewidth=0.5)
    ax2.axhline(0.7, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(-0.7, color="red", linestyle="--", alpha=0.5)
    ax2.set_title("Ising Domain Response (Phase Transitions)")
    ax2.set_ylabel("Magnetization")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--grid-size", type=int, default=100)
    args = parser.parse_args()
    
    m, p = run_fast_scan(args.data, args.grid_size)
    plot_results(m, p)
