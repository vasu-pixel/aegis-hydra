import argparse
import sys
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import ccxt.async_support as ccxt  # asynchronous ccxt
import pandas as pd

async def fetch_data(symbol: str, days: int, interval: str, output_path: str):
    """Fetch OHLCV data from Binance asynchronously."""
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to Coinbase to fetch {days} days of {symbol} ({interval})...")
    exchange = ccxt.coinbase()
    
    try:
        # Calculate start time
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        
        all_ohlcv = []
        limit = 1000
        
        while True:
            # Fetch batch
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1ms to get next batch
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # If we reached current time, stop
            if since > int(now.timestamp() * 1000):
                break
                
            print(f"Fetched {len(all_ohlcv)} candles...", end="\r")
        
        print(f"\nTotal fetched: {len(all_ohlcv)} candles.")
        
        if not all_ohlcv:
            print("No data fetched.")
            return

        # Create DataFrame
        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Process for Aegis Hydra format
        # Convert timestamp to microseconds
        df["timestamp_us"] = df["timestamp"] * 1000
        df["symbol"] = symbol
        df["funding_rate"] = 0.0  # Placeholder since public OHLCV doesn't have funding
        
        # Reorder and select columns
        cols = ["timestamp_us", "symbol", "open", "high", "low", "close", "volume", "funding_rate"]
        final_df = df[cols]
        
        final_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await exchange.close()

def main():
    parser = argparse.ArgumentParser(description="Fetch historical data for Aegis Hydra")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch")
    parser.add_argument("--interval", type=str, default="1m", help="Candle interval (1m, 5m, 1h)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(fetch_data(args.symbol, args.days, args.interval, args.output))
            
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

if __name__ == "__main__":
    main()
