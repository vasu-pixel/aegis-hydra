
import time
import argparse
import sys
import os

# Ensure import path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from aegis_hydra.agents.cpp_grid import CppEngine

def benchmark(size=3162, steps=1000):
    print(f"=== C++ ENGINE BENCHMARK ({size}x{size}) ===")
    
    try:
        engine = CppEngine()
    except FileNotFoundError:
        print("Error: Engine library not found. Run 'make' in aegis_hydra/cpp/")
        return

    # Start Engine
    print("Starting Engine...")
    engine.start(size)
    
    print("Engine Running (Background Thread). Measuring poll rate...")
    
    start_time = time.time()
    last_step = 0
    poll_count = 0
    
    while time.time() - start_time < 5.0:
        mag = engine.get_magnetization()
        steps_done = engine.get_steps()
        
        if steps_done > last_step:
            # print(f"Step {steps_done}: M={mag:.4f}")
            last_step = steps_done
            
        poll_count += 1
        # time.sleep(0.001) # Poll at 1kHz
        
    duration = time.time() - start_time
    total_steps = engine.get_steps()
    
    rate = total_steps / duration
    latency = 1000.0 / rate if rate > 0 else float('inf')
    
    print(f"\n--- RESULTS ---")
    print(f"Duration: {duration:.2f}s")
    print(f"Steps: {total_steps}")
    print(f"Speed: {rate:.2f} steps/sec")
    print(f"Latency per Step: {latency:.2f} ms")
    
    engine.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=3162)
    args = parser.parse_args()
    
    benchmark(size=args.size)
