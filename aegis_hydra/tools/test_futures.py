
import asyncio
import websockets
import json

async def test_futures():
    url = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
    print(f"Connecting to {url}...")
    try:
        async with websockets.connect(url) as ws:
            print("Connected!")
            async for msg in ws:
                data = json.loads(msg)
                print(f"Received: {data}")
                break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_futures())
