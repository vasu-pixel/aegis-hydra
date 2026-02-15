#!/usr/bin/env python3
"""
Validation script for latency fixes.
Analyzes paper_log.csv to verify spike reduction.
"""

import sys
from pathlib import Path

def analyze_latency_log(filename="paper_log.csv", min_samples=100):
    """Analyze latency log and report statistics."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ {filename} not found. Run paper trader first.")
        return False

    latencies = []
    spike_details = []

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 7:
            try:
                step = int(parts[1])
                lat = float(parts[6])
                latencies.append(lat)

                if lat > 5.0:  # Spike threshold
                    spike_details.append((step, lat))
            except (ValueError, IndexError):
                continue

    if len(latencies) < min_samples:
        print(f"⚠️  Only {len(latencies)} samples. Need at least {min_samples} for valid analysis.")
        print("   Run the system for a few more minutes.")
        return False

    # Calculate statistics
    latencies.sort()
    n = len(latencies)

    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_lat = sum(latencies) / n
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]

    # Count spikes
    spikes_5ms = sum(1 for x in latencies if x > 5.0)
    spikes_10ms = sum(1 for x in latencies if x > 10.0)
    spikes_18ms = sum(1 for x in latencies if x > 18.0)

    # Report
    print("=" * 60)
    print("📊 LATENCY ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n📈 Sample Statistics:")
    print(f"   Total samples: {n:,}")
    print(f"   Time span: ~{n * 2} steps (~{n * 2 / 120:.1f} minutes at 2 steps/sec)")

    print(f"\n⚡ Latency Distribution:")
    print(f"   Min:     {min_lat:6.2f} ms")
    print(f"   Average: {avg_lat:6.2f} ms")
    print(f"   P50:     {p50:6.2f} ms")
    print(f"   P95:     {p95:6.2f} ms")
    print(f"   P99:     {p99:6.2f} ms")
    print(f"   Max:     {max_lat:6.2f} ms")

    print(f"\n🎯 Target Validation:")
    baseline_ok = avg_lat < 0.5
    p99_ok = p99 < 5.0
    max_ok = max_lat < 10.0

    print(f"   Average < 0.5ms:  {'✅ PASS' if baseline_ok else '❌ FAIL'} ({avg_lat:.2f}ms)")
    print(f"   P99 < 5ms:        {'✅ PASS' if p99_ok else '❌ FAIL'} ({p99:.2f}ms)")
    print(f"   Max < 10ms:       {'✅ PASS' if max_ok else '❌ FAIL'} ({max_lat:.2f}ms)")

    print(f"\n⚠️  Spike Analysis:")
    print(f"   >5ms:   {spikes_5ms:5d} ({100*spikes_5ms/n:5.2f}%) {'✅ OK' if spikes_5ms/n < 0.05 else '⚠️  HIGH'}")
    print(f"   >10ms:  {spikes_10ms:5d} ({100*spikes_10ms/n:5.2f}%) {'✅ OK' if spikes_10ms/n < 0.01 else '❌ FAIL'}")
    print(f"   >18ms:  {spikes_18ms:5d} ({100*spikes_18ms/n:5.2f}%) {'✅ OK' if spikes_18ms == 0 else '❌ FAIL'}")

    if spike_details:
        print(f"\n🔍 Top 10 Worst Spikes:")
        spike_details.sort(key=lambda x: x[1], reverse=True)
        for step, lat in spike_details[:10]:
            mod_503 = "json_dump" if step % 503 == 0 else ""
            mod_1000 = "csv_flush" if step % 1000 == 0 else ""
            operations = ", ".join([op for op in [mod_503, mod_1000] if op])
            print(f"   Step {step:6d}: {lat:6.2f}ms {f'[{operations}]' if operations else ''}")

    # Overall verdict
    print(f"\n{'=' * 60}")
    all_ok = baseline_ok and p99_ok and max_ok and spikes_18ms == 0
    if all_ok:
        print("✅ ALL TESTS PASSED - Latency spikes eliminated!")
    elif p99_ok and spikes_18ms == 0:
        print("⚠️  PARTIAL SUCCESS - 18ms spikes gone but some outliers remain")
    else:
        print("❌ ISSUES DETECTED - Further optimization needed")
    print("=" * 60)

    return all_ok

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "paper_log.csv"
    success = analyze_latency_log(filename)
    sys.exit(0 if success else 1)
