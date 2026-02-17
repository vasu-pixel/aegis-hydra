
import ccxt
import sys

def check_symbols():
    try:
        exchange = ccxt.binanceus()
        markets = exchange.load_markets()
        
        print("Available BTC Pairs on Binance.US:")
        btc_pairs = [m for m in markets if 'BTC' in m]
        for p in sorted(btc_pairs):
            print(f" - {p}")
            
        print("\nChecking specifically for BTC/USDT:")
        if 'BTC/USDT' in markets:
            print("✅ BTC/USDT EXISTS")
        else:
            print("❌ BTC/USDT NOT FOUND")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_symbols()
