import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import matplotlib.pyplot as plt
import numpy as np
# CORRECT IMPORT:
from aegis_hydra.agents.ising_grid import IsingGrid

def run_calibration(target_j: float, temp: float, n_steps: int = 500):
    print(f"Calibrating Ising Grid (J={target_j}, T={temp})...")
    
    # Initialize 10M Agent Grid (or smaller for calib speed)
    # Using 1000x1000 = 1M agents for fast calibration
    grid = IsingGrid(height=1000, width=1000, J=target_j) 
    key = jax.random.PRNGKey(42)
    state = grid.initialize(key, temperature=temp)
    
    magnetization_curve = []
    
    # Jit the step function
    # Note: IsingGrid.step takes (state, external_field, key)
    step_fn = eqx.filter_jit(grid.step)
    
    current_state = state
    
    # No external field (h=0) to test spontaneous magnetization
    h = 0.0
    
    keys = jax.random.split(key, n_steps)
    
    for i in range(n_steps):
        # We need a new key for each step
        current_state = step_fn(current_state, h, keys[i])
        magnetization_curve.append(float(current_state.magnetization))
        if i % 50 == 0:
            print(f"Step {i}: M={magnetization_curve[-1]:.4f}")
        
    return magnetization_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=2.27)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    
    m_curve = run_calibration(args.j, args.temp, args.steps)
    
    plt.figure(figsize=(10, 6))
    plt.plot(m_curve)
    plt.title(f"Ising Grid Magnetization (J={args.j}, T={args.temp})")
    plt.ylim(-1.1, 1.1) # Correct bounds for Ising
    plt.grid(True, alpha=0.3)
    plt.xlabel("Step")
    plt.ylabel("Magnetization")
    output_file = f"calibration_ising_T{args.temp}.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    main()
