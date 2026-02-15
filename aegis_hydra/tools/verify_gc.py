#!/usr/bin/env python3
"""
Verify that garbage collection is properly disabled.
Run this while HFT pipeline is running in another terminal.
"""

import gc
import sys

print("=== GARBAGE COLLECTION STATUS ===\n")

# Check if GC is enabled
enabled = gc.isenabled()
print(f"GC Enabled: {enabled}")

if enabled:
    print("⚠️  WARNING: GC is ENABLED (should be disabled for HFT!)")
else:
    print("✅ GC is disabled (good!)")

# Check thresholds
thresholds = gc.get_threshold()
print(f"\nGC Thresholds: {thresholds}")
if thresholds[0] == 0:
    print("✅ Thresholds set to 0 (no collections)")
else:
    print(f"⚠️  Thresholds not zero: {thresholds}")

# Check counts
counts = gc.get_count()
print(f"\nGC Counts: {counts}")

# Check stats (if available)
try:
    stats = gc.get_stats()
    print(f"\nGC Stats:")
    for i, stat in enumerate(stats):
        print(f"  Generation {i}: {stat}")
except AttributeError:
    print("\nGC stats not available (Python < 3.4)")

# Check frozen objects
try:
    frozen = gc.get_freeze_count()
    print(f"\nFrozen objects: {frozen}")
except AttributeError:
    print("\nget_freeze_count() not available")

# Recommendations
print("\n=== RECOMMENDATIONS ===")
if enabled or thresholds[0] != 0:
    print("❌ GC not properly disabled!")
    print("   Add to your code:")
    print("   gc.disable()")
    print("   gc.freeze()")
    print("   gc.set_threshold(0)")
else:
    print("✅ GC configuration looks good!")

print("\n=== LIVE MONITORING ===")
print("To monitor GC during runtime:")
print("  python3 -c 'import gc; print(gc.isenabled(), gc.get_count())'")
