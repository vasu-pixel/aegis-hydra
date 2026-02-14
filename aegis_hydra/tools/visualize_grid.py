"""
aegis_hydra.tools.visualize_grid — Phase 1 Visualizer

Runs the Ising Grid simulation on historical data and producing a heatmap animation.
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import pandas as pd
from aegis_hydra.agents.ising_grid import IsingGrid

def load_data(path: str, noise: bool = False):
    # Dummy loader or real CSV
    if noise:
        # White Noise (Random Walk or just random fluctuations?)
        # "Feed the system White Noise (random data)."
        # If we feed it random PRICES, h = diff(prices) is also random.
        # Let's generate random price changes directly (h).
        # Actually load_data returns prices.
        # Random Walk:
        steps = 200
        np.random.seed(42)
        returns = np.random.normal(0, 1, steps)
        prices = 100 + np.cumsum(returns)
        return prices
    elif not path:
        # Synthetic sine wave
        t = np.linspace(0, 4*np.pi, 200)
        prices = 100 + 10 * np.sin(t)
        return prices
    else:
        # Simple CSV loader without pandas
        try:
            # Assuming header row, and 'close' is maybe 4th column? 
            # Too brittle. Let's just try basic numpy load
            data = np.genfromtxt(path, delimiter=',', names=True)
            if 'close' in data.dtype.names:
                return data['close']
            else:
                return np.genfromtxt(path, delimiter=',')[:, 4] # Guess close is 4
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return np.zeros(200)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="", help="Path to BTC CSV")
    parser.add_argument("--temp", type=float, default=2.27, help="Temperature (Critical is ~2.269)")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--noise", action="store_true", help="Generate White Noise input")
    parser.add_argument("--output", type=str, default="grid_viz.gif", help="Output filename")
    args = parser.parse_args()

    print("Initializing Grid (10M Agents)...")
    grid_model = IsingGrid()
    key = jax.random.PRNGKey(42)
    state = grid_model.initialize(key, temperature=args.temp)
    
    # Jit the step function
    step_fn = jax.jit(grid_model.step)
    
    # Data
    prices = load_data(args.data, noise=args.noise)
    # Normalize prices to external field h
    # h ~ (price - mean) / std * scale
    # Simple momemtum: h = price_change
    price_changes = np.diff(prices, prepend=prices[0])
    # Scale h to be visible but not overwhelming J=1
    # h of 0.1 is significant
    h_ext = np.clip(price_changes * 0.1, -0.5, 0.5)
    
    print("Simulating...")
    
    frames = []
    magnetizations = []
    
    # Run loop
    # Storing 3162x3162 frames in RAM is heavy (10MB per frame * 200 = 2GB). Feasible.
    
    curr_state = state
    for i in range(min(args.steps, len(h_ext))):
        key, subkey = jax.random.split(key)
        h = h_ext[i]
        curr_state = step_fn(curr_state, h, subkey)
        
        # Store for animation (downsample for viz speed?)
        # 3k x 3k is too big for MPL imshow animation in real time
        # Downsample to 512x512
        img = jax.image.resize(curr_state.spins, (512, 512), method='nearest')
        
        # Convert to numpy
        frames.append(np.array(img))
        magnetizations.append(float(curr_state.magnetization))
        
        if i % 10 == 0:
            print(f"Step {i}/{args.steps}, M={curr_state.magnetization:.3f}, h={h:.3f}")

    print("Rendering Animation...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Heatmap
    im = ax1.imshow(frames[0], cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_title("Ising Grid Domain Structure")
    ax1.axis('off')
    
    # Graph
    line, = ax2.plot([], [], lw=2, color='purple')
    ax2.set_xlim(0, len(frames))
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Magnetization")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.7, color='r', ls='--', alpha=0.5)
    ax2.axhline(-0.7, color='r', ls='--', alpha=0.5)
    
    def update(frame_idx):
        im.set_data(frames[frame_idx])
        line.set_data(range(frame_idx+1), magnetizations[:frame_idx+1])
        return im, line
    
    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    # Use Pillow writer for GIF
    ani.save(args.output, writer='pillow', fps=15)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
