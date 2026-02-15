import json
import os

try:
    path = os.path.expanduser("~/aegis-hydra/paper_state.json")
    with open(path) as f:
        d = json.load(f)

    print(f"Log Start: {d[0]['time']}")
    print(f"Log End:   {d[-1]['time']}")

    # Find trades (position change)
    trades = []
    for i in range(1, len(d)):
        if d[i]['position'] != d[i-1]['position']:
            trades.append(d[i])

    if trades:
        print(f"First Trade: {trades[0]['time']}")
        print(f"Last Trade:  {trades[-1]['time']}")
        print(f"Total Trades: {len(trades)}")
    else:
        print("No trades found.")

except Exception as e:
    print(f"Error: {e}")
