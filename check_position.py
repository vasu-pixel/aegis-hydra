#!/usr/bin/env python3
"""
Check current trading position and state.
Run this while HFT pipeline is running.
"""

import json
import os

# Check if paper_state.json exists
if not os.path.exists("paper_state.json"):
    print("❌ paper_state.json not found!")
    print("   The HFT pipeline might not be running or hasn't saved state yet.")
    exit(1)

# Read latest state
with open("paper_state.json", "r") as f:
    data = json.load(f)

if not data:
    print("❌ paper_state.json is empty!")
    exit(1)

# Get latest entry
latest = data[-1]

print("=" * 60)
print("🔍 CURRENT TRADING STATE")
print("=" * 60)

print(f"\n📊 Account:")
print(f"   Capital:      ${latest.get('capital', 0):.2f}")
print(f"   Position:     {latest.get('position', 0):+.1f}")
print(f"   Position Size: ", end="")
if latest.get('position', 0) > 0:
    print("LONG 📈")
elif latest.get('position', 0) < 0:
    print("SHORT 📉")
else:
    print("FLAT (no position)")

print(f"\n💰 Market:")
print(f"   Current Price: ${latest.get('price', 0):,.2f}")
print(f"   Magnetization: {latest.get('magnetization', 0):+.6f}")
print(f"   Threshold:     {latest.get('threshold', 0.6):.6f}")

print(f"\n⚡ Performance:")
print(f"   Latency:       {latest.get('latency', 0):.3f}ms")

# Calculate P&L from start
initial_capital = 100.0
pnl = latest.get('capital', 100) - initial_capital
pnl_pct = (pnl / initial_capital) * 100

print(f"\n📈 P&L:")
print(f"   Total P&L:     ${pnl:+.2f} ({pnl_pct:+.2f}%)")

# Explain why capital might not be moving
print(f"\n💡 Capital Movement:")
if latest.get('position', 0) == 0:
    print("   ⚠️  Position is FLAT - capital won't change until you enter a position")
    print("   ℹ️  System is waiting for magnetization signal to enter")
else:
    print(f"   ✅ Position is ACTIVE ({latest.get('position', 0):+.1f})")
    print("   ℹ️  Capital updates with every price tick")

# Show data age
print(f"\n🕐 Data:")
print(f"   Timestamp:     {latest.get('time', 'unknown')}")
print(f"   Total samples: {len(data)}")

print("=" * 60)
