#!/usr/bin/env python3
"""
Real-time spike diagnostics for HFT pipe.
Analyzes hft_market_data.csv and paper_log.csv to identify spike patterns.
"""

import sys
import time
from collections import defaultdict

def monitor_spikes(filename, interval=2.0):
    """Monitor file for new entries and report spikes."""
    last_position = 0
    spikes_by_step_mod = defaultdict(int)
    total_samples = 0
    spike_count = 0
    
    print(f"📊 Monitoring {filename} for latency spikes...")
    print(f"   Threshold: >5ms")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        while True:
            try:
                with open(filename, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                
                for line in new_lines:
                    parts = line.strip().split(',')
                    if len(parts) < 3:
                        continue
                    
                    try:
                        # hft_market_data.csv format: timestamp,price,latency
                        # paper_log.csv format: timestamp,step,price,capital,mag,pos,latency
                        if len(parts) >= 7:  # paper_log
                            step = int(parts[1])
                            latency = float(parts[6])
                        else:  # hft_market_data
                            step = total_samples
                            latency = float(parts[2])
                        
                        total_samples += 1
                        
                        if latency > 5.0:
                            spike_count += 1
                            spikes_by_step_mod[step % 1000] += 1
                            
                            # Identify potential causes
                            causes = []
                            if step % 503 == 0: causes.append("json_dump")
                            if step % 1000 == 0: causes.append("csv_flush")
                            if step % 100 == 0: causes.append("display")
                            
                            print(f"⚠️  SPIKE at step {step}: {latency:.2f}ms {causes if causes else ''}")
                    
                    except (ValueError, IndexError):
                        continue
                
            except FileNotFoundError:
                print(f"⏳ Waiting for {filename}...")
            
            time.sleep(interval)
            
            # Periodic summary
            if total_samples > 0 and total_samples % 100 == 0:
                spike_rate = (spike_count / total_samples) * 100
                print(f"\n📈 Summary: {total_samples} samples, {spike_count} spikes ({spike_rate:.1f}%)")
                
                if spikes_by_step_mod:
                    print("   Most common spike positions (step % 1000):")
                    sorted_mods = sorted(spikes_by_step_mod.items(), key=lambda x: x[1], reverse=True)
                    for mod, count in sorted_mods[:5]:
                        print(f"      step % 1000 == {mod}: {count} spikes")
                print()
    
    except KeyboardInterrupt:
        print(f"\n\n📊 Final Summary:")
        print(f"   Total samples: {total_samples}")
        print(f"   Spikes (>5ms): {spike_count} ({(spike_count/total_samples)*100:.1f}%)")
        print(f"\n   Spike distribution by step % 1000:")
        sorted_mods = sorted(spikes_by_step_mod.items(), key=lambda x: x[1], reverse=True)
        for mod, count in sorted_mods[:10]:
            print(f"      step {mod}: {count} spikes")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "hft_market_data.csv"
    monitor_spikes(filename)
