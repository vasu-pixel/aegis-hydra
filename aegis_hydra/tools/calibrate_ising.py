import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import matplotlib.pyplot as plt
import numpy as np
from aegis_hydra.agents.brownian_swarm import BrownianSwarm, AgentState
from aegis_hydra.market.tensor_field import MarketTensor

def run_calibration(target_j: float, n_steps: int = 500):
    print(f"Calibrating Ising Model with J = {target_j}...")
    
    # Initialize swarm
    # We use a smaller swarm for calibration speed
    swarm = BrownianSwarm(n_agents=10000, coupling=target_j, dt=0.01)
    key = jax.random.PRNGKey(0)
    state = swarm.initialize(key)
    
    # Dummy market tensor (neutral)
    # [mid, spread, vol, imbalance, ...]
    # We set imbalance to 0 to test purely social phase transition
    tensor = jnp.zeros((50,))
    tensor = tensor.at[0].set(100.0) # mid
    tensor = tensor.at[2].set(0.01)  # vol
    
    magnetization_curve = []
    
    step_fn = eqx.filter_jit(swarm.step)
    
    current_state = state
    
    # Initial push to break symmetry?
    # No, we want to see if spontaneous ordering happens or if it amplifies noise
    
    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        
        # Calculate M
        m = jnp.mean(current_state.signal)
        magnetization_curve.append(float(m))
        
        # Step
        current_state = step_fn(current_state, tensor, subkey, magnetization=m)
        
    return magnetization_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_j", type=float, default=1.5, help="Coupling constant to test")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
    args = parser.parse_args()
    
    m_curve = run_calibration(args.target_j, args.steps)
    
    m_final = np.mean(mouse_curve if (mouse_curve := m_curve[-50:]) else 0.0)
    m_abs = abs(m_final)
    
    print(f"Final Magnetization (last 50 steps mean): {m_final:.4f}")
    if m_abs > 0.7:
        print("SUCCESS: Phase Transition Detected (Fermi-Dirac / Ising Criticality reached)")
    else:
        print("FAIL: Swarm remains in disordered (Paramagnetic) state.")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(m_curve, label=f"Magnetization (J={args.target_j})")
    plt.axhline(0.7, color='r', linestyle='--')
    plt.axhline(-0.7, color='r', linestyle='--')
    plt.title(f"Ising Model Calibration (J={args.target_j})")
    plt.xlabel("Step")
    plt.ylabel("Magnetization (M)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"calibration_j_{args.target_j}.png")
    print(f"Plot saved to calibration_j_{args.target_j}.png")

if __name__ == "__main__":
    main()
