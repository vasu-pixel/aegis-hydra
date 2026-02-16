import asyncio
import struct
import sys
import os
import time
import subprocess
import gc
import concurrent.futures
import json
import random
from datetime import datetime

# Absolute Path to Daemon
DAEMON_PATH = os.path.join(os.path.dirname(__file__), '../cpp/critical_flow_daemon')

# Optimization: Global Thread Pools
# parse_executor: for JSON parsing
# io_executor: for disk writes (CSV/JSON)
parse_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def save_state_sync(data):
    try:
        with open("paper_state.json", "w") as f:
            json.dump(data, f)
    except: pass

def log_csv_sync(filename, lines):
    try:
        with open(filename, "a") as f:
            f.writelines(lines)
    except: pass

async def run_pipe(product_id="BTCUSD"):
    # 0. Zero Jitter Tuning: Kill the Garbage Collector
    gc.collect() # Clean up once
    gc.freeze()  # Move all current objects to permanent generation 
    gc.disable() # Stop all future collections
    # Set thresholds to 0 for extra safety
    try: gc.set_threshold(0) 
    except: pass

    try:
        os.nice(-20) # Absolute Priority
    except:
        pass
        
    loop = asyncio.get_running_loop()
    
    # 1. Connect WS
    from ..market.binance_ws import BinanceWebSocket, BinanceFuturesWS
    ws = BinanceWebSocket(product_id)
    fut_ws = BinanceFuturesWS(product_id.replace("-", "").lower()) # Global Futures
    
    print(f"Starting C++ Daemon: {DAEMON_PATH}")
    
    # 2. Launch Daemon
    process = subprocess.Popen(
        [DAEMON_PATH], 
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0
    )

    
    # 3. State Tracker
    class Tracker:
        capital = 100.0
        position = 0.0
        prev_price = 0.0
        entry_price = 0.0      # Track entry price for P&L calculation
        entry_time = 0.0       # Track entry time for stats
        fee_rate = 0.000       # Fees Set to 0.0% (VIP / Paper Mode)

        # HFT MODE: ZERO RESTRICTIONS
        min_hold_seconds = 0.0  # NO MINIMUM HOLD - Trade at physics speed!
        stop_loss_pct = -2.0    # Wide stop loss (only for disasters)

        # Stats tracking
        total_trades = 0
        winning_trades = 0
        losing_trades = 0

    tracker = Tracker()
    state_history = []
    data_buffer = [] 
    signal_buffer = []
    analysis_buffer = [] # NEW: High-res analysis log

    print("=== ZERO-WAIT DECOUPLED HFT PIPE ===")

    msg_queue = asyncio.Queue(maxsize=5000)
    packet_queue = asyncio.Queue(maxsize=5000)

    # Trade counting for Hawkes estimator
    trade_count_per_tick = 0

    # Latency tracking
    latency_buffer = []
    latency_stats = {
        'network': [], 'parse': [], 'physics': [],
        'signal_read': [], 'total': []
    }

    # 4. Background Tasks
    maintenance_counter = 0
    async def background_maintenance():
        nonlocal data_buffer, signal_buffer, latency_buffer, analysis_buffer, latency_stats, maintenance_counter
        while True:
            await asyncio.sleep(0.5)
            maintenance_counter += 1

            if state_history:
                # Convert state history to CSV lines
                # Format: time, price, z_score, criticality, mlofi, latency
                analysis_lines = []
                for s in state_history:
                    # Only log if not already logged (need a cursor? or just dump last chunk?)
                    # Simple approach: Plotter reads last N lines. We append new lines.
                    # But state_history is a rolling buffer. 
                    # Let's just log the *latest* state to a file if it changed?
                    pass 
                
                # Better: Log to CSV immediately when parsing STATE
                # logic is in read_signals.
                pass

            if data_buffer:
                lines = data_buffer[:]
                data_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_market_data.csv", lines)
            if signal_buffer:
                sigs = signal_buffer[:]
                signal_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_signals.csv", sigs)
            if analysis_buffer:
                lines = analysis_buffer[:]
                analysis_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_analysis.csv", lines)
            if latency_buffer:
                lat_lines = latency_buffer[:]
                latency_buffer = []
                loop.run_in_executor(io_executor, log_csv_sync, "hft_latency_breakdown.csv", lat_lines)

            # Print HFT stats every 20 cycles (10 seconds)
            if maintenance_counter % 20 == 0:
                pnl = tracker.capital - 100.0
                pnl_pct = (pnl / 100.0) * 100
                pos_str = "LONG 📈" if tracker.position > 0 else ("SHORT 📉" if tracker.position < 0 else "FLAT")

                # Calculate win rate
                total = tracker.winning_trades + tracker.losing_trades
                win_rate = (tracker.winning_trades / total * 100) if total > 0 else 0

                print(f"\n⚡ HFT STATS | Cap: ${tracker.capital:.2f} | P&L: {pnl:+.2f} ({pnl_pct:+.2f}%) | "
                      f"Trades: {tracker.total_trades} | W/L: {tracker.winning_trades}/{tracker.losing_trades} ({win_rate:.0f}%) | Pos: {pos_str}\n")

            # Print latency stats every 500ms
            if latency_stats['total']:
                import statistics
                print(f"\n📊 Latency Stats (last 500ms):")
                print(f"   Network:  {statistics.mean(latency_stats['network']):.2f}ms avg (Spot)")
                print(f"   FutLat:   {fut_ws.latency:.2f}ms lag (Futures)")
                print(f"   Parse:    {statistics.mean(latency_stats['parse']):.2f}ms avg")
                print(f"   Physics:  {statistics.mean(latency_stats['physics']):.2f}ms avg")
                print(f"   SigRead:  {statistics.mean(latency_stats['signal_read']):.2f}ms avg")
                print(f"   TOTAL:    {statistics.mean(latency_stats['total']):.2f}ms avg (Spot E2E)")
                
                # Clear stats for next interval
                for key in latency_stats:
                    latency_stats[key] = []

            # REMOVED: gc.collect(0) - causes 3-8ms spikes, defeats gc.disable()
            # gc.collect(0)

    # 4b. Non-Blocking Pipe Writer
    async def pipe_writer():
        while True:
            packet = await packet_queue.get()
            try:
                process.stdin.write(packet)
                process.stdin.flush()
            except BrokenPipeError:
                print("\n⚠️  C++ daemon pipe broken!")
                break
            packet_queue.task_done()

    import re
    # PRE-COMPILED REGEX for extreme speed
    # Binance format: "p":"50000.00" (trade price), "E":1234567890 (timestamp ms)
    price_re = re.compile(r'"p":"(\d+\.?\d*)"')
    timestamp_re = re.compile(r'"E":(\d+)')

    # 5. Logic Worker (Processing & Strategy)
    async def logic_worker():
        nonlocal data_buffer, signal_buffer, latency_buffer, trade_count_per_tick
        while True:
            message, recv_time, server_time = await msg_queue.get()
            parse_start = time.perf_counter()

            # FAST PATH: String processing (websocket returns strings, not bytes)
            # Binance uses "stream":"symbol@trade" and "stream":"symbol@depth"
            is_trade = '@trade' in message
            is_depth = '@depth' in message

            price = 0.0
            channel = "unknown"
            net_latency = 0.0

            if is_trade:
                # Count this trade for Hawkes estimator
                trade_count_per_tick += 1

                # Debug: Log trade counting every 10 trades
                if trade_count_per_tick % 10 == 0:
                    print(f"[TRADES] Counted {trade_count_per_tick} trades in current window")

                # Optimized extraction for trade price (message is already a string)
                price_match = price_re.search(message)
                if price_match:
                    price = float(price_match.group(1))
                    ws.latest_ticker_price = price
                    ws.ready = True
                    channel = "trade"

                    # Extract timestamp for network latency (fast path)
                    # Binance format: "E":1234567890123 (milliseconds)
                    ts_match = timestamp_re.search(message)
                    if ts_match:
                        try:
                            server_ts_ms = int(ts_match.group(1))
                            server_ts = server_ts_ms / 1000.0  # Convert to seconds
                            net_latency = (time.time() - server_ts) * 1000
                        except:
                            pass
                # Skip fallback to JSON parsing - if regex fails, just skip this message
            elif is_depth:
                # Must parse full message to update book
                data = await loop.run_in_executor(parse_executor, json.loads, message)
                # Binance combined stream format: {"stream":"...","data":{...}}
                if "data" in data:
                    ws._handle_depth(data["data"])
                price = ws.get_mid_price()
                channel = "depth"
            else:
                # Other messages
                data = await loop.run_in_executor(parse_executor, json.loads, message)
                if "stream" in data and "data" in data:
                    stream_name = data["stream"]
                    stream_data = data["data"]
                    if "trade" in stream_name:
                        ws._handle_trade(stream_data)
                        channel = "trade"
                    elif "depth" in stream_name:
                        ws._handle_depth(stream_data)
                        channel = "depth"
                price = ws.get_mid_price()

            if price > 0:
                if tracker.prev_price > 0:
                    pct = (price - tracker.prev_price) / tracker.prev_price
                    tracker.capital += tracker.position * tracker.capital * pct
                tracker.prev_price = price

                # Track parse time
                parse_time = (time.perf_counter() - parse_start) * 1000

                # Send to C++ daemon with order book data (5 levels)
                pipe_send_time = time.perf_counter()

                # Extract order book snapshot (5 levels of bid/ask)
                bids, asks = ws.order_book.get_snapshot(depth=5)

                # Build arrays for binary packet
                bid_prices = [b[0] for b in bids[:5]]
                bid_sizes = [b[1] for b in bids[:5]]
                ask_prices = [a[0] for a in asks[:5]]
                ask_sizes = [a[1] for a in asks[:5]]

                # DEBUG: Verify flows
                # if random.random() < 0.01:
                #    print(f"DEBUG: Spot={price:.2f} Fut={fut_ws.price:.2f} Z={shared_state.get('z_score',0):.2f}")

                # Pack: fffd (mid, fut, net_lat, recv_time) + I (trade_count) + 20f (5 levels x 4 arrays)
                if fut_ws.price == 0.0 and random.random() < 0.001:
                     print(f"⚠️  WARNING: Futures Price is 0.0! (Feed not ready?)")
                
                # Use '=' for standard alignment (no padding) to match C++ packed struct
                packet = struct.pack('=fffdI20f',
                    price, fut_ws.price, net_latency, recv_time, trade_count_per_tick,
                    *bid_prices, *bid_sizes, *ask_prices, *ask_sizes)

                packet_queue.put_nowait(packet)

                # Debug: Log when sending trade counts
                if trade_count_per_tick > 0:
                    print(f"[PACKET] Sending {trade_count_per_tick} trades to C++ daemon")

                # Reset trade count for next tick
                trade_count_per_tick = 0

                # Store timing info for signal correlation
                ws.last_packet_time = recv_time
                ws.last_server_time = server_time
                ws.last_parse_time = parse_time
                ws.last_net_latency = net_latency

                # Zero-allocation logging (just float/timestamp)
                proc_latency = (time.perf_counter() - parse_start) * 1000
                data_buffer.append(f"{time.time()},{price},{proc_latency:.3f}\n")

            msg_queue.task_done()

    # 5b. Signal Reader (C++ -> Python)
    # Optimization: Re-use state object to avoid garbage collection
    shared_state = {
        "time": "", "step": 0, "price": 0.0, "capital": 100.0,
        "mlofi": 0.0, "criticality": 0.0, "volatility": 0.0,
        "position": 0.0, "latency": 0.0,
        "net_latency": 0.0, "threshold": 0.0,
        "parse_latency": 0.0, "read_latency": 0.0, "total_latency": 0.0
    }

    async def read_signals(stdout):
        nonlocal signal_buffer, shared_state, latency_buffer, latency_stats
        while True:
            signal_recv_time = time.perf_counter()
            line = await loop.run_in_executor(None, stdout.readline)
            if not line: break
            decoded = line.decode().strip()

            if decoded.startswith("STATE "):
                parts = decoded.split()
                if len(parts) >= 9:  # STATE step price mlofi cpp_lat net_lat criticality volatility threshold
                    shared_state["time"] = time.time()
                    shared_state["step"] = int(parts[1])
                    shared_state["price"] = float(parts[2])
                    shared_state["capital"] = tracker.capital
                    shared_state["mlofi"] = float(parts[3])  # Multi-Level Order Flow Imbalance
                    shared_state["position"] = tracker.position

                    # Parse latency components from C++ daemon
                    physics_latency = float(parts[4])  # C++ computation time
                    net_latency_reported = float(parts[5])  # Network latency from ticker
                    shared_state["latency"] = physics_latency
                    shared_state["net_latency"] = net_latency_reported

                    # New Critical Flow metrics
                    shared_state["criticality"] = float(parts[6])  # Hawkes branching ratio (n)
                    shared_state["volatility"] = float(parts[7])   # Price volatility (σ)
                    shared_state["threshold"] = float(parts[8])    # Dynamic threshold
                    
                    # Z-Score (Lead-Lag) - added in recent C++ update
                    if len(parts) > 9:
                        shared_state["z_score"] = float(parts[9])
                    else:
                        shared_state["z_score"] = 0.0

                    # Log to Analysis CSV (for Plotter)
                    # Format: time, price, z_score, criticality, mlofi
                    analysis_buffer.append(
                        f"{shared_state['time']},{shared_state['price']},{shared_state['z_score']},"
                        f"{shared_state['criticality']},{shared_state['mlofi']}\n"
                    )
                    
                    # Z-Score (Lead-Lag) - added in recent C++ update
                    if len(parts) > 9:
                        shared_state["z_score"] = float(parts[9])
                    else:
                        shared_state["z_score"] = 0.0

                    # Calculate full latency breakdown
                    if hasattr(ws, 'last_packet_time'):
                        # Total latency from when we received data to now
                        total_latency = (signal_recv_time - ws.last_packet_time) * 1000

                        # Detailed breakdown
                        net_lat = ws.last_net_latency if hasattr(ws, 'last_net_latency') else 0.0
                        parse_lat = ws.last_parse_time if hasattr(ws, 'last_parse_time') else 0.0
                        phys_lat = physics_latency

                        # Signal read latency = total - parse - physics
                        # (Network is separate, already measured)
                        read_lat = max(0.0, total_latency - parse_lat - phys_lat)

                        # Store in shared state for dashboard
                        shared_state["parse_latency"] = parse_lat
                        shared_state["read_latency"] = read_lat
                        shared_state["total_latency"] = total_latency

                        # Store for stats (only valid positive values)
                        if net_lat >= 0:
                            latency_stats['network'].append(net_lat)
                        if parse_lat >= 0:
                            latency_stats['parse'].append(parse_lat)
                        if phys_lat >= 0:
                            latency_stats['physics'].append(phys_lat)
                        if read_lat >= 0:
                            latency_stats['signal_read'].append(read_lat)
                        if total_latency >= 0:
                            latency_stats['total'].append(total_latency)

                        # Log detailed breakdown (only if valid)
                        if total_latency >= 0 and phys_lat >= 0:
                            latency_buffer.append(
                                f"{time.time()},{net_lat:.3f},{parse_lat:.3f},"
                                f"{phys_lat:.3f},{read_lat:.3f},{total_latency:.3f}\n"
                            )

                            # Print every 10 steps with color-coded warnings
                            if int(parts[1]) % 10 == 0:
                                color = ""
                                reset = ""

                                # Color based on physics latency (most important)
                                if phys_lat > 5.0:
                                    color = "\033[91m"  # Red
                                    reset = "\033[0m"
                                elif phys_lat > 1.0:
                                    color = "\033[93m"  # Yellow
                                    reset = "\033[0m"
                                else:
                                    color = "\033[92m"  # Green
                                    reset = "\033[0m"

                                print(f"{color}[LATENCY] Net: {net_lat:5.2f}ms | Parse: {parse_lat:4.2f}ms | "
                                      f"Phys: {phys_lat:4.2f}ms | Read: {read_lat:4.2f}ms | "
                                      f"TOTAL: {total_latency:5.2f}ms{reset}")

                    # Append to history every 10 steps for responsive dashboard
                    if int(parts[1]) % 10 == 0:
                        state_history.append(shared_state.copy())
                        if len(state_history) > 500: state_history[:] = state_history[-200:]
            else:
                print(f"[SIGNAL] {datetime.now().strftime('%H:%M:%S.%f')} | {decoded}")
                parts = decoded.split()
                if not parts: continue

                # Extract current price from signal (last part is usually price)
                current_price = float(parts[-1]) if len(parts) > 1 else tracker.prev_price

                old_pos = tracker.position
                pnl_pct = 0.0

                # Entry signals: VACUUM EXECUTION (Dynamic Size & Limit)
                if parts[0] == "BUY":
                    # Futures > Spot. We want to BUY Spot up to (Futures - Margin).
                    # 1. Calc Vacuum Limit Price
                    # Default: Use Spot Price if Futures unavailable? No, Z-score implies Futures exist.
                    fut_price = fut_ws.price if fut_ws.ready else current_price * 1.001
                    limit_price = fut_price * (1 - 0.0003) # 0.03% Vacuum Margin below Futures

                    # 2. Calc Max Size (50% of Capital)
                    usd_available = tracker.capital * 0.50
                    if limit_price <= 0: return # Prevent ZeroDivisionError
                    qty = usd_available / limit_price
                    
                    # 3. Execution (Simulated IOC)
                    # In real life: await sniper.snipe_stale_order('buy', qty, limit_price)
                    # In Paper: We fill at current_price because it is < limit_price (Arb exists)
                    
                    tracker.position = qty / (tracker.capital / current_price) # Normalized to 1.0ish
                    tracker.entry_price = current_price # We still fill at Best Ask (Start of sweep)
                    tracker.entry_time = time.time()
                    tracker.vacuum_limit = limit_price # Store for logging
                    
                    print(f"  🌪️ VACUUM LONG: Sweeping {qty:.4f} BTC up to ${limit_price:.2f} (Fut: {fut_price:.2f})")
                    print(f"     Filled @ Best Ask: ${current_price:.2f}")

                elif parts[0] == "SELL":
                    # Futures < Spot. We want to SELL Spot down to (Futures + Margin).
                    fut_price = fut_ws.price if fut_ws.ready else current_price * 0.999
                    limit_price = fut_price * (1 + 0.0003) # 0.03% Vacuum Margin above Futures

                    # 2. Calc Max Size (100% of Position or 50% Short Capability)
                    # Paper Trade: Assume we short 1.0 unit equivalent
                    qty = (tracker.capital * 0.50) / current_price
                    
                    tracker.position = - (qty / (tracker.capital / current_price))
                    tracker.entry_price = current_price
                    tracker.entry_time = time.time()
                    tracker.vacuum_limit = limit_price

                    print(f"  🌪️ VACUUM SHORT: Dumping {qty:.4f} BTC down to ${limit_price:.2f} (Fut: {fut_price:.2f})")
                    print(f"     Filled @ Best Bid: ${current_price:.2f}")

                # Exit signals: CLOSE_LONG or CLOSE_SHORT - IMMEDIATE EXECUTION
                elif parts[0].startswith("CLOSE"):
                    if tracker.entry_price > 0 and old_pos != 0:
                        # Calculate P&L %
                        if old_pos > 0:  # Closing long
                            pnl_pct = ((current_price - tracker.entry_price) / tracker.entry_price) * 100
                        else:  # Closing short
                            pnl_pct = ((tracker.entry_price - current_price) / tracker.entry_price) * 100

                        hold_time = time.time() - tracker.entry_time if tracker.entry_time > 0 else 0

                        # Track win/loss
                        if pnl_pct > 0:
                            tracker.winning_trades += 1
                        else:
                            tracker.losing_trades += 1

                        # Wide stop loss (only for disasters)
                        if pnl_pct <= tracker.stop_loss_pct:
                            print(f"  🛑 STOP: {pnl_pct:+.2f}% | {hold_time:.1f}s")
                        else:
                            print(f"  ⚡ EXIT: {pnl_pct:+.2f}% | {hold_time:.2f}s")

                    tracker.position = 0.0

                # Execute ALL position changes immediately (HFT mode)
                if tracker.position != old_pos:
                    tracker.total_trades += 1
                    fee = abs(tracker.position - old_pos) * tracker.capital * tracker.fee_rate
                    tracker.capital -= fee

                    # Calculate net P&L after fees
                    if old_pos != 0 and pnl_pct != 0:
                        gross_pnl = tracker.capital * (pnl_pct / 100)
                        net_pnl = gross_pnl - fee
                        print(f"  💰 Net: {net_pnl:+.4f} | Fee: {fee:.4f} | Cap: ${tracker.capital:.2f}")

                    signal_buffer.append(f"{time.time()},{decoded},{current_price:.2f},{pnl_pct:.4f}\n")

    asyncio.create_task(fut_ws.connect()) # Start Futures Feed
    asyncio.create_task(read_signals(process.stdout))
    asyncio.create_task(background_maintenance())
    asyncio.create_task(pipe_writer())
    asyncio.create_task(logic_worker())

    # 6. Optimized Network Listener
    import websockets
    # Build Binance stream URL: symbol@depth + symbol@trade
    symbol = product_id.replace("-", "").lower()
    streams = f"{symbol}@depth20@100ms/{symbol}@trade"
    binance_url = f"{ws.WS_URL}?streams={streams}"

    async with websockets.connect(binance_url, max_size=None,
                                   ping_interval=20, ping_timeout=10) as socket:
        # Binance doesn't require subscription messages - streams are in URL
        print("⚡ ULTRA-HFT MODE ACTIVATED ⚡")
        print(f"   Exchange: Binance.US")
        print(f"   Symbol: {product_id}")
        print(f"   Fees: 0.1% taker")
        print(f"   Speed: ZERO RESTRICTIONS - Trade at physics speed!")
        print(f"   Min Hold: {tracker.min_hold_seconds:.1f}s (INSTANT)")
        print(f"   Stop Loss: {tracker.stop_loss_pct:.1f}% (Wide)\n")
        print("📊 Latency Display Format:")
        print("   Net    = Network latency (exchange to you)")
        print("   Parse  = Python message parsing time")
        print("   Phys   = C++ physics computation time")
        print("   Read   = Signal read from C++ to Python")
        print("   TOTAL  = End-to-end processing time\n")

        try:
            while True:
                recv_time = time.time()
                message = await socket.recv()

                # Try to extract server timestamp for network latency
                # Binance format: "E":1234567890123 (milliseconds)
                server_time = 0.0
                if '"E":' in message:
                    ts_match = timestamp_re.search(message)
                    if ts_match:
                        try:
                            server_ts_ms = int(ts_match.group(1))
                            server_time = server_ts_ms / 1000.0
                        except:
                            pass

                msg_queue.put_nowait((message, recv_time, server_time))
        except KeyboardInterrupt: pass
        finally:
            process.terminate()
            gc.enable()

            # Final stats summary
            print("\n\n📊 SESSION LATENCY SUMMARY:")
            if latency_buffer:
                print("   Detailed breakdown saved to: hft_latency_breakdown.csv")
            print("   Exiting...")

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    try:
        asyncio.run(run_pipe())
    except KeyboardInterrupt:
        pass
