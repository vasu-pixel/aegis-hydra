#!/usr/bin/env python3
"""
Real-time latency visualization for HFT system.
Reads hft_latency_breakdown.csv and displays live statistics.
"""

import sys
import time
from collections import deque
import statistics

def visualize_latency(filename="hft_latency_breakdown.csv", window_size=100):
    """
    Monitor and visualize latency breakdown in real-time.

    CSV format: timestamp,network,parse,physics,signal_read,total
    """

    # Rolling windows for each metric
    windows = {
        'network': deque(maxlen=window_size),
        'parse': deque(maxlen=window_size),
        'physics': deque(maxlen=window_size),
        'signal_read': deque(maxlen=window_size),
        'total': deque(maxlen=window_size)
    }

    last_position = 0
    sample_count = 0

    print("=" * 80)
    print("📊 REAL-TIME LATENCY VISUALIZATION")
    print("=" * 80)
    print(f"Window size: {window_size} samples")
    print(f"Reading from: {filename}")
    print("\nMetrics:")
    print("  Network:     Exchange → Your Server (network transmission)")
    print("  Parse:       WebSocket parsing + data extraction")
    print("  Physics:     C++ Ising model computation")
    print("  Signal Read: C++ → Python signal transmission")
    print("  Total:       End-to-end processing latency")
    print("\n" + "=" * 80)
    print()

    try:
        while True:
            try:
                with open(filename, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                for line in new_lines:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue

                    try:
                        timestamp = float(parts[0])
                        network = float(parts[1])
                        parse = float(parts[2])
                        physics = float(parts[3])
                        signal_read = float(parts[4])
                        total = float(parts[5])

                        # Add to windows
                        windows['network'].append(network)
                        windows['parse'].append(parse)
                        windows['physics'].append(physics)
                        windows['signal_read'].append(signal_read)
                        windows['total'].append(total)

                        sample_count += 1

                        # Update display every 10 samples
                        if sample_count % 10 == 0:
                            display_stats(windows, sample_count)

                    except (ValueError, IndexError):
                        continue

            except FileNotFoundError:
                print(f"⏳ Waiting for {filename}...")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("📊 FINAL STATISTICS")
        print("=" * 80)
        display_stats(windows, sample_count, final=True)
        print("\nVisualization stopped.")

def display_stats(windows, sample_count, final=False):
    """Display current statistics."""

    if not windows['total']:
        return

    # Clear screen for live update (unless final)
    if not final:
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        print("=" * 80)
        print("📊 REAL-TIME LATENCY STATISTICS")
        print("=" * 80)
        print(f"Samples processed: {sample_count}")
        print()

    # Calculate statistics for each metric
    metrics = {}
    for name, data in windows.items():
        if data:
            metrics[name] = {
                'min': min(data),
                'avg': statistics.mean(data),
                'p50': statistics.median(data),
                'p95': sorted(data)[int(len(data) * 0.95)] if len(data) > 20 else max(data),
                'p99': sorted(data)[int(len(data) * 0.99)] if len(data) > 100 else max(data),
                'max': max(data),
            }

    # Display table
    print(f"{'Metric':<15} {'Min':>8} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print("-" * 80)

    for name in ['network', 'parse', 'physics', 'signal_read', 'total']:
        if name in metrics:
            m = metrics[name]
            # Color code based on values
            color = ""
            reset = ""

            if name == 'total':
                if m['avg'] > 5.0:
                    color = "\033[91m"  # Red
                    reset = "\033[0m"
                elif m['avg'] > 2.0:
                    color = "\033[93m"  # Yellow
                    reset = "\033[0m"
                else:
                    color = "\033[92m"  # Green
                    reset = "\033[0m"

            print(f"{color}{name.capitalize():<15} {m['min']:7.2f}ms {m['avg']:7.2f}ms "
                  f"{m['p50']:7.2f}ms {m['p95']:7.2f}ms {m['p99']:7.2f}ms {m['max']:7.2f}ms{reset}")

    print()

    # Component breakdown (percentage of total)
    if 'total' in metrics and metrics['total']['avg'] > 0:
        print("Component Breakdown (% of total latency):")
        print("-" * 80)
        total_avg = metrics['total']['avg']

        for name in ['network', 'parse', 'physics', 'signal_read']:
            if name in metrics:
                pct = (metrics[name]['avg'] / total_avg) * 100
                bar_length = int(pct / 2)  # Scale to 50 chars max
                bar = "█" * bar_length
                print(f"  {name.capitalize():<15} {pct:5.1f}% {bar}")
        print()

    # Latency targets
    if 'total' in metrics:
        total_avg = metrics['total']['avg']
        total_p99 = metrics['total']['p99']

        print("Performance Targets:")
        print("-" * 80)
        print(f"  Average < 2ms:     {'✅ PASS' if total_avg < 2.0 else '❌ FAIL'} ({total_avg:.2f}ms)")
        print(f"  P99 < 5ms:         {'✅ PASS' if total_p99 < 5.0 else '❌ FAIL'} ({total_p99:.2f}ms)")

        if 'physics' in metrics:
            phys_p99 = metrics['physics']['p99']
            print(f"  Physics P99 < 1ms: {'✅ PASS' if phys_p99 < 1.0 else '⚠️  CHECK'} ({phys_p99:.2f}ms)")

    if not final:
        print("\nPress Ctrl+C to stop and see final summary...")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "hft_latency_breakdown.csv"
    visualize_latency(filename)
