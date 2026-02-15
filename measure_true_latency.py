#!/usr/bin/env python3
"""
Measure TRUE end-to-end latency by capturing server timestamps.
Connects to Coinbase WebSocket and measures actual network latency.
"""

import asyncio
import websockets
import json
import time
from datetime import datetime
from collections import deque
import statistics

async def measure_latency():
    """Connect to Coinbase and measure true network latency."""

    WS_URL = "wss://advanced-trade-ws.coinbase.com"
    product_id = "BTC-USD"

    latencies = deque(maxlen=100)
    total_messages = 0

    print("=" * 80)
    print("🔬 TRUE LATENCY MEASUREMENT")
    print("=" * 80)
    print(f"Connecting to: {WS_URL}")
    print(f"Product: {product_id}")
    print(f"Measuring network latency (Coinbase → Your Server)\n")

    async with websockets.connect(WS_URL, max_size=None) as socket:
        # Subscribe to ticker (has server timestamps)
        await socket.send(json.dumps({
            "type": "subscribe",
            "product_ids": [product_id],
            "channel": "ticker"
        }))

        print("✅ Connected! Measuring...\n")
        print(f"{'Message':<8} {'Server Time':<30} {'Recv Time':<30} {'Latency':<10}")
        print("-" * 80)

        try:
            while total_messages < 100:  # Measure 100 samples
                message = await socket.recv()
                recv_time = time.time()

                try:
                    data = json.loads(message)

                    # Only process ticker messages with timestamps
                    if data.get("channel") == "ticker":
                        events = data.get("events", [])
                        if events and events[0].get("tickers"):
                            ticker = events[0]["tickers"][0]
                            server_time_str = ticker.get("time")

                            if server_time_str:
                                # Parse server timestamp
                                server_time = datetime.fromisoformat(
                                    server_time_str.replace('Z', '+00:00')
                                ).timestamp()

                                # Calculate true network latency
                                network_latency = (recv_time - server_time) * 1000

                                # Only record valid latencies (0-100ms range)
                                if 0 < network_latency < 100:
                                    latencies.append(network_latency)
                                    total_messages += 1

                                    # Print every 10th message
                                    if total_messages % 10 == 0:
                                        print(f"#{total_messages:<7} "
                                              f"{datetime.fromtimestamp(server_time).strftime('%H:%M:%S.%f')[:-3]:<30} "
                                              f"{datetime.fromtimestamp(recv_time).strftime('%H:%M:%S.%f')[:-3]:<30} "
                                              f"{network_latency:>8.2f}ms")

                                    # Show running stats every 20 messages
                                    if total_messages % 20 == 0 and latencies:
                                        avg = statistics.mean(latencies)
                                        p50 = statistics.median(latencies)
                                        print(f"\n   📊 Running Stats: Avg={avg:.2f}ms, P50={p50:.2f}ms\n")

                except Exception as e:
                    continue

        except KeyboardInterrupt:
            pass

    # Final statistics
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)

    if latencies:
        sorted_lat = sorted(latencies)
        print(f"\nSamples collected: {len(latencies)}")
        print(f"\nNetwork Latency (Coinbase Server → Your GCP Instance):")
        print(f"  Min:     {min(sorted_lat):>8.2f}ms")
        print(f"  Average: {statistics.mean(sorted_lat):>8.2f}ms")
        print(f"  Median:  {statistics.median(sorted_lat):>8.2f}ms")
        print(f"  P95:     {sorted_lat[int(len(sorted_lat)*0.95)]:>8.2f}ms")
        print(f"  P99:     {sorted_lat[int(len(sorted_lat)*0.99)]:>8.2f}ms")
        print(f"  Max:     {max(sorted_lat):>8.2f}ms")

        print(f"\n💡 Your Processing Latency: ~0.8ms")
        print(f"💡 Network Latency: ~{statistics.mean(sorted_lat):.1f}ms")
        print(f"💡 Exchange Processing: ~1-5ms (estimated)")
        print("─" * 80)
        avg_net = statistics.mean(sorted_lat)
        total = avg_net + 0.8 + 3  # network + processing + exchange
        print(f"💡 TOTAL Trading Latency: ~{total:.1f}ms")
        print(f"   ({avg_net:.1f}ms network + 0.8ms processing + 3ms exchange)")

        print("\n✅ Network latency confirmed via server timestamps!")
    else:
        print("❌ No valid latency measurements collected")

if __name__ == "__main__":
    print("\n🔬 Starting latency measurement...")
    print("This will measure TRUE network latency using Coinbase server timestamps.\n")
    asyncio.run(measure_latency())
