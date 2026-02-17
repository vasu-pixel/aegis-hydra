import argparse
import sys
import csv
import math
import statistics
import numpy as np

# --- CONFIGURATION ---
DEFAULT_FILE_PATH = '../hft_market_data.csv' 
INITIAL_CAPITAL = 100.0
TRADE_SIZE_USD = 50.0  
MAKER_FEE = 0.000      
SPREAD_BPS = 0.0005

class StrategyConfig:
    def __init__(self, name, j, alpha, cooldown_mode, use_spread_filter, use_quantum_filter, is_leader=True):
        self.name = name
        self.J = j
        self.alpha = alpha
        self.cooldown_mode = cooldown_mode # 'fixed', 'adaptive', 'quantum_viscous'
        self.use_spread_filter = use_spread_filter
        self.use_quantum_filter = use_quantum_filter
        self.is_leader = is_leader

def load_data(path):
    print(f"Loading {path}...")
    data = []
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers: return []
            
            # Detect Columns
            try:
                # Case A: HFT Data provided (timestamp, price)
                if 'price' in headers and 'timestamp' in headers:
                    t_idx = headers.index('timestamp')
                    p_idx = headers.index('price')
                    
                    for row in reader:
                        if len(row) < 2: continue
                        t = float(row[t_idx])
                        p = float(row[p_idx])
                        data.append({'timestamp': t, 'price': p})
                        
                # Case B: Standard OHLCV / Fetch Data (timestamp_us, close)
                elif 'close' in headers and ('timestamp_us' in headers or 'timestamp' in headers):
                    t_idx = headers.index('timestamp_us') if 'timestamp_us' in headers else headers.index('timestamp')
                    p_idx = headers.index('close')
                    
                    for row in reader:
                        if len(row) < 2: continue
                        t = float(row[t_idx])
                        # If us, convert to seconds for consistency? 
                        # HFT data was likely seconds. Let's keep usage consistent in strategy.
                        # If t > 10 billion, it's probably us or ms.
                        if t > 1e11: t /= 1_000_000.0 # Convert us to s
                        elif t > 1e10: t /= 1000.0 # Convert ms to s
                        
                        p = float(row[p_idx])
                        data.append({'timestamp': t, 'price': p})
                
                # Case C: No Headers / Raw (Assuming t, p)
                else:
                    # Reset file
                    f.seek(0)
                    for row in reader:
                        if len(row) < 2: continue
                        try:
                            # Try first two columns
                            t = float(row[0])
                            p = float(row[1])
                            data.append({'timestamp': t, 'price': p})
                        except: continue

            except ValueError as e:
                print(f"⚠️ CSV Parsing Error: {e}")
                return []
                
    except FileNotFoundError:
        print(f"❌ Error: {path} not found.")
        return []
        
    print(f"Loaded {len(data)} rows.")
    return data

def calculate_energy(prices):
    if len(prices) < 20: return 0.5
    
    # Log returns
    returns = [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices))]
    
    n = len(returns)
    mean = statistics.mean(returns)
    std_dev = statistics.stdev(returns)
    if std_dev == 0: return 0.0
    
    skew_sum = sum(((x - mean) / std_dev) ** 3 for x in returns)
    skew = skew_sum / n
    
    kurt_sum = sum(((x - mean) / std_dev) ** 4 for x in returns)
    kurt = kurt_sum / n
    
    excess_kurt = kurt - 3.0
    jb = (n / 6.0) * (skew**2 + 0.25 * (excess_kurt**2))
    
    return math.log1p(jb) / 10.0

def calculate_hurst_proxy(prices):
    # Variance Ratio Test
    if len(prices) < 20: return 0.5
    
    # Log Returns
    r1 = [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices))]
    
    if len(prices) > 5:
        r5 = [math.log(prices[i]/prices[i-5]) for i in range(5, len(prices))]
    else:
        return 0.5
        
    var1 = statistics.variance(r1) if len(r1) > 1 else 0.0
    var5 = statistics.variance(r5) if len(r5) > 1 else 0.0
    
    if var1 < 1e-12: return 0.5
    
    vr = var5 / (5.0 * var1)
    
    # Map VR to H roughly
    return 0.5 + 0.5 * math.log(vr) if vr > 0 else 0.5

def run_strategy(data, config, initial_capital, trade_size, lead_data=None):
    print(f"Running {config.name} (Capital: ${initial_capital}, Lead-Lag: {'ON' if lead_data else 'OFF'})...")
    cash = initial_capital
    inventory = 0.0
    M = 0.0
    last_action_time = 0.0
    
    # Pre-calculate Lead Hurst if data provided
    lead_hurst_map = {}
    if lead_data:
        print("Pre-calculating Lead Hurst signals (BTC)...")
        lead_prices = [d['price'] for d in lead_data]
        h_vals = []
        for idx in range(20, len(lead_data)):
            h = calculate_hurst_proxy(lead_prices[max(0, idx-100):idx+1])
            lead_hurst_map[lead_data[idx]['timestamp']] = h
            h_vals.append(h)
        print(f"Lead Hurst: Avg={statistics.mean(h_vals):.2f}, Max={max(h_vals):.2f}")
    
    trades = [] 
    closed_trades = [] 
    long_inventory = [] 
    short_inventory = []
    
    prices = [d['price'] for d in data]
    window_size = 50
    energy_window = 100 
    
    equity_curve = [] 
    
    hurst_vals = []
    energy_vals = []
    
    for i in range(1, len(data)):
        curr = data[i]
        price = curr['price']
        timestamp = curr['timestamp']
        
        # 1. Volatility
        start_idx = max(0, i - window_size)
        window = prices[start_idx:i+1]
        
        if len(window) > 1:
            std_dev = statistics.stdev(window)
            mean_p = statistics.mean(window)
            if mean_p == 0:
                vol_bps = 0.0
            else:
                vol_bps = (std_dev / mean_p) * 10000.0
        else:
            vol_bps = 5.0 
            
        # 2. Quantum Energy & Hurst
        energy = 0.5
        hurst = 0.5
        
        if config.use_quantum_filter:
            start_e = max(0, i - energy_window)
            e_window = prices[start_e:i+1]
            energy = calculate_energy(e_window)
            hurst = calculate_hurst_proxy(e_window)
            
            # Simple Smoothing
            if energy_vals: energy = 0.9*energy_vals[-1] + 0.1*energy
            if hurst_vals: hurst = 0.9*hurst_vals[-1] + 0.1*hurst
            
            energy_vals.append(energy)
            hurst_vals.append(hurst)

        # 3. Physics
        prev_price = data[i-1]['price']
        momentum = (price - prev_price) * 1000.0
        imbalance_proxy = max(-1.0, min(1.0, momentum))
        
        h = imbalance_proxy * 5.0
        T = vol_bps * 0.5 + 0.1 
        M = math.tanh((config.J * M + h) / T)
        
        # 4. Regime Logic (Quantum Viscosity)
        width = 0.5 # Default Viscous Width
        cooldown = 0.5
        
        is_ground = False
        is_excited = False
        is_crash = False
        
        if config.cooldown_mode == 'quantum_viscous':
            # Thresholds
            ENERGY_CRASH = 0.8
            ENERGY_EXCITED = 0.4
            HURST_TRENDING = 0.55
            
            if energy > ENERGY_CRASH: is_crash = True
            elif energy > ENERGY_EXCITED or hurst > HURST_TRENDING: is_excited = True
            else: is_ground = True
            
            # THE ETH PATCH: Lead-Lag Shield. If Lead is trending, Stand Down.
            if lead_data:
                l_h = lead_hurst_map.get(timestamp, 0.5)
                if l_h > 0.55: # BTC is trending
                    equity_curve.append((timestamp, cash + inventory*price))
                    continue # STAND DOWN

            if is_crash: 
                equity_curve.append((timestamp, cash + inventory*price))
                continue # STOP TRADING
            
            if is_excited:
                # Defense Mode
                width = 1.0 # DOUBLE SPREAD
                cooldown = 1.0 # Slow Down
            else:
                # Ground Mode (Viscous)
                width = 0.5 if config.is_leader else 1.0 # ETH Patch: 2x Width (10bps)
                cooldown = 0.2 # Fast
        
        elif config.cooldown_mode == 'fixed':
            cooldown = 0.5
            
        if timestamp - last_action_time < cooldown:
            equity_curve.append((timestamp, cash + inventory*price))
            continue 

        # 5. Filters
        sim_spread_bps = max(1.5, vol_bps * 0.5) 
        sim_spread_abs = price * (sim_spread_bps / 10000.0)
        
        if config.use_spread_filter:
            if sim_spread_bps < 1.0 or sim_spread_bps > 50.0:
                equity_curve.append((timestamp, cash + inventory*price))
                continue

        # 6. Pricing
        fair_value = price
        risk_aversion = 10.0
        inv_skew = inventory * risk_aversion * (vol_bps/100.0)
        book_skew = imbalance_proxy * (sim_spread_abs * 0.4)
        
        fair_value += book_skew - inv_skew
        if abs(M) > 0.5: fair_value += M * config.alpha

        # Quotes
        my_bid = fair_value - (sim_spread_abs * width)
        my_ask = fair_value + (sim_spread_abs * width)
        
        # 7. Execution
        if i + 1 >= len(data): break
        next_price = data[i+1]['price']
        
        can_buy = abs(inventory) < 0.002
        can_sell = inventory > 0 or abs(inventory) < 0.002
        
        filled_qty = 0.0
        filled_price = 0.0
        side = ''
        
        # Physical Crash Protection (Always On)
        phy_crash = (vol_bps > 3.0 and M < -0.6)
        phy_fomo = (vol_bps > 3.0 and M > 0.6)
        
        if not phy_crash and can_buy and next_price <= my_bid:
            filled_qty = trade_size / price
            filled_price = price
            side = 'BUY'
        elif not phy_fomo and can_sell and next_price >= my_ask:
            filled_qty = trade_size / price
            filled_price = price
            side = 'SELL'
            
        if side:
            last_action_time = timestamp
            if side == 'BUY':
                cash -= filled_qty * filled_price
                inventory += filled_qty
                qty_rem = filled_qty
                while qty_rem > 0 and short_inventory:
                    entry = short_inventory.pop(0)
                    qty_closed = min(qty_rem, entry['qty'])
                    pnl = (entry['price'] - filled_price) * qty_closed
                    closed_trades.append(pnl)
                    qty_rem -= qty_closed
                    entry['qty'] -= qty_closed
                    if entry['qty'] > 0: short_inventory.insert(0, entry)
                if qty_rem > 0: long_inventory.append({'price': filled_price, 'qty': qty_rem})
            elif side == 'SELL':
                cash += filled_qty * filled_price
                inventory -= filled_qty
                qty_rem = filled_qty
                while qty_rem > 0 and long_inventory:
                    entry = long_inventory.pop(0)
                    qty_closed = min(qty_rem, entry['qty'])
                    pnl = (filled_price - entry['price']) * qty_closed
                    closed_trades.append(pnl)
                    qty_rem -= qty_closed
                    entry['qty'] -= qty_closed
                    if entry['qty'] > 0: long_inventory.insert(0, entry)
                if qty_rem > 0: short_inventory.append({'price': filled_price, 'qty': qty_rem})
            trades.append(side)

        equity_curve.append((timestamp, cash + (inventory * price)))

    # OUTPUT STATS
    final_equity = cash + (inventory * prices[-1])
    net_pnl = final_equity - initial_capital
    
    wins = [p for p in closed_trades if p > 0]
    losses = [p for p in closed_trades if p <= 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0
    
    print(f"\n--- {config.name} REPORT ---")
    print(f"Net PnL:       ${net_pnl:.2f}")
    print(f"Trades:        {len(trades)}")
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # TIMEFRAME DISTRIBUTION ANALYSIS
    analyze_timeframes(equity_curve)

def analyze_timeframes(equity_curve):
    if not equity_curve: return
    
    start_time = equity_curve[0][0]
    final_time = equity_curve[-1][0]
    duration = final_time - start_time
    
    buckets = {
        '1-Minute': 60,
        '15-Minute': 900,
        '1-Hour': 3600
    }
    
    print("\n--- Return Distribution by Timeframe ---")
    
    for name, seconds in buckets.items():
        if duration < seconds: continue
        
        returns = []
        next_snapshot = start_time + seconds
        snapshot_val = equity_curve[0][1]
        
        temp_returns = []
        for t, eq in equity_curve:
            if t >= next_snapshot:
                ret = (eq - snapshot_val)
                temp_returns.append(ret)
                snapshot_val = eq
                next_snapshot += seconds
        
        if not temp_returns: continue
        
        avg = statistics.mean(temp_returns)
        std = statistics.stdev(temp_returns) if len(temp_returns) > 1 else 0.0
        n = len(temp_returns)
        
        skew = 0.0
        if std > 0: skew = sum(((x - avg)/std)**3 for x in temp_returns) / n
        
        wins = len([x for x in temp_returns if x > 0])
        bucket_wr = (wins / n) * 100
        
        print(f"[{name}] (N={n})")
        print(f"  Mean PnL:   ${avg:.4f}")
        print(f"  Std Dev:    ${std:.4f}")
        print(f"  Skewness:   {skew:.2f}")
        print(f"  Win %:      {bucket_wr:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_FILE_PATH, help="Path to input CSV data")
    parser.add_argument("--lead", default=None, help="Path to lead asset CSV (e.g. BTC)")
    parser.add_argument("--capital", type=float, default=100.0, help="Starting capital")
    parser.add_argument("--trade_size", type=float, default=50.0, help="Trade size in USD")
    args = parser.parse_args()
    
    data = load_data(args.input)
    lead_data = load_data(args.lead) if args.lead else None
    
    if not data:
        print("No data loaded. Exiting.")
        sys.exit(1)
    
    # Determine if this run should act as a Leader or Follower
    is_leader = (lead_data is None)
    
    # 1. Viscous Maker (Benchmark)
    cfg_viscous = StrategyConfig("Strategy A (Viscous)", j=1.5, alpha=20.0, cooldown_mode='fixed', use_spread_filter=False, use_quantum_filter=False, is_leader=is_leader)
    run_strategy(data, cfg_viscous, args.capital, args.trade_size, lead_data)
    
    # 2. Quantum Viscosity (Defense Mode)
    cfg_quantum = StrategyConfig("Strategy D (Quantum Viscosity)", j=1.2, alpha=20.0, cooldown_mode='quantum_viscous', use_spread_filter=True, use_quantum_filter=True, is_leader=is_leader)
    run_strategy(data, cfg_quantum, args.capital, args.trade_size, lead_data)
