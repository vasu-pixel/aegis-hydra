import argparse
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import sys
import os

# Constants
COINAPI_URL = "https://rest.coinapi.io/v1/ohlcv/{}/history"
LIMIT = 10000  # 10k data points per call (100 credits)

def fetch_real_data(api_key: str, output_file: str):
    """
    Downloads 7 days of 1-second tick data from CoinAPI.
    Cost: ~6,000 credits (~$16.00).
    """
    symbol_id = "BINANCE_SPOT_BTC_USDT"
    period_id = "1SEC"
    
    # Start 7 days ago
    # CoinAPI requires ISO 8601, e.g. 2016-01-01T00:00:00
    start_dt = datetime.utcnow() - timedelta(days=7)
    start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    headers = {'X-CoinAPI-Key': api_key}
    url = COINAPI_URL.format(symbol_id)
    
    all_data = []
    total_credits = 0
    current_start = start_time
    
    print(f"--- STARTING DOWNLOAD: 7 Days of Real 1s Data ---")
    print(f"Target: {output_file}")
    print(f"Est. Cost: 6,000 Credits")
    
    try:
        while True:
            params = {
                'period_id': period_id,
                'time_start': current_start,
                'limit': LIMIT
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            # Rate Limit Handling
            if response.status_code == 429:
                print("Rate limit hit. Sleeping 60s...")
                time.sleep(60)
                continue
            elif response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break
            
            data = response.json()
            if not data:
                print("Download complete (no more data).")
                break
                
            # Track Usage
            total_credits += (LIMIT / 100)
            
            # Process Batch
            for row in data:
                # CoinAPI returns ISO time, we need microseconds for Hydra
                # CoinAPI format example: 2023-10-01T00:00:00.0000000Z
                # We need to handle the Z
                dt = datetime.fromisoformat(row['time_period_start'].replace('Z', '+00:00'))
                ts_us = int(dt.timestamp() * 1_000_000)
                
                all_data.append({
                    "timestamp_us": ts_us,
                    "symbol": "BTC/USDT",
                    "open": row['price_open'],
                    "high": row['price_high'],
                    "low": row['price_low'],
                    "close": row['price_close'],
                    "volume": row['volume_traded'],
                    "funding_rate": 0.0 # Spot data has no funding
                })
            
            print(f"Fetched {len(all_data)} ticks | Credits Used: ~{int(total_credits)}", end="\r")
            
            # Advance Time
            last_entry = data[-1]
            current_start = last_entry['time_period_end']
            
            # Safety Brake: Stop after 7 days (approx 600k points) to save wallet
            # 7 days * 24 h * 60 m * 60 s = 604,800 points
            if len(all_data) >= 610000:
                print("\nReached 7-day limit. Stopping to preserve credits.")
                break
                
            time.sleep(0.5) # Be nice to the API

    except KeyboardInterrupt:
        print("\nUser stopped download. Saving partial data...")
    except Exception as e:
        print(f"\nCritical Error: {e}")

    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nSUCCESS. Saved {len(df)} rows to {output_file}")
    else:
        print("\nNo data fetched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Your CoinAPI Key")
    parser.add_argument("--output", default="aegis_hydra/data/btc_1s_real.csv")
    args = parser.parse_args()
    
    fetch_real_data(args.key, args.output)
