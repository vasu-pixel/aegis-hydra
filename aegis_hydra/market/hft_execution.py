"""
aegis_hydra.market.hft_execution — The Sniper Rifle
Zero-overhead execution for MFT/HFT.

⚠️  NOT YET ACTIVE - FOR FUTURE LIVE TRADING
Currently paper trading for strategy validation.
"""
import ccxt.async_support as ccxt
import asyncio

class SniperEngine:
    """
    Direct CCXT execution engine for live trading.

    Features:
    - Zero-overhead order placement
    - No intermediate "slots" or reports
    - Fire-and-forget execution
    - Binance.US optimized

    Usage:
        sniper = SniperEngine(api_key, api_secret, symbol="BTC/USD")
        await sniper.warm_up()
        await sniper.snipe_order('buy', 0.001)
    """

    def __init__(self, api_key, api_secret, symbol="BTC/USD"):
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret

        # Direct Binance.US connection (Low Overhead)
        self.exchange = ccxt.binanceus({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': False,  # WE CONTROL THE SPEED
            'options': {'adjustForTimeDifference': True}
        })

        self.active = False

    async def warm_up(self):
        """
        Load markets once so we don't wait on first trade.
        Pre-warms the connection and caches market data.
        """
        await self.exchange.load_markets()
        self.active = True
        print("⚡ Sniper Scope Loaded (Binance.US)")
        print(f"   Symbol: {self.symbol}")
        print(f"   Ready for live execution")

    async def snipe_order(self, side: str, quantity: float):
        """
        FIRE & FORGET order placement.

        Args:
            side: 'buy' or 'sell'
            quantity: Amount in BTC (e.g., 0.001 = $70 at $70k/BTC)

        Returns:
            order dict or None on failure

        Notes:
            - Uses MARKET orders for speed (taker fee: 0.1%)
            - For maker rebates, switch to 'limit' with postOnly=True
            - No retry logic - fire once, move on
        """
        if not self.active:
            print("❌ Engine not warmed up - call warm_up() first")
            return None

        try:
            # SPEED MODE: Market order (1-5ms execution)
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=quantity
            )

            print(f"✅ ORDER FILLED: {side.upper()} {quantity} {self.symbol} @ market")
            return order

        except ccxt.InsufficientFunds as e:
            print(f"❌ INSUFFICIENT FUNDS: {e}")
            return None
        except ccxt.InvalidOrder as e:
            print(f"❌ INVALID ORDER: {e}")
            return None
        except ccxt.NetworkError as e:
            print(f"❌ NETWORK ERROR: {e}")
            return None
        except Exception as e:
            print(f"❌ MISFIRE: {e}")
            return None

    async def snipe_stale_order(self, side, qty, limit_price):
        """
        Attempts to pick off a specific price level (IOC).
        """
        if not self.active: return None
        try:
            return await self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=side,
                amount=qty,
                price=limit_price,
                params={'timeInForce': 'IOC'} # Fill exactly at this price or cancel
            )
        except Exception as e:
            print(f"❌ SNIPE FAILED: {e}")
            return None

    async def snipe_limit(self, side: str, quantity: float, price: float):
        """
        Limit order version (maker rebate potential).

        Args:
            side: 'buy' or 'sell'
            quantity: Amount in BTC
            price: Limit price

        Returns:
            order dict or None

        Note:
            - Use postOnly=True to guarantee maker fee (0% or negative)
            - May not fill immediately (sits on book)
        """
        if not self.active:
            print("❌ Engine not warmed up")
            return None

        try:
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=price,
                params={'postOnly': True}  # Ensure maker fee
            )

            print(f"✅ LIMIT ORDER: {side.upper()} {quantity} @ ${price:.2f}")
            return order

        except Exception as e:
            print(f"❌ LIMIT ORDER FAILED: {e}")
            return None

    async def close_all(self):
        """
        Panic button - close all open positions.

        Note:
            - Fetches current positions
            - Market sells everything
            - Use for emergency exit only
        """
        try:
            # Fetch balance
            balance = await self.exchange.fetch_balance()

            # Get BTC balance
            btc_balance = balance.get('BTC', {}).get('free', 0)

            if btc_balance > 0.0001:  # Min order size
                print(f"🚨 PANIC CLOSE: Dumping {btc_balance} BTC")
                await self.snipe_order('sell', btc_balance)
            else:
                print("ℹ️  No position to close")

        except Exception as e:
            print(f"❌ PANIC CLOSE FAILED: {e}")

    async def get_balance(self):
        """Get current USD and BTC balance."""
        try:
            balance = await self.exchange.fetch_balance()
            return {
                'USD': balance.get('USD', {}).get('free', 0),
                'BTC': balance.get('BTC', {}).get('free', 0)
            }
        except Exception as e:
            print(f"❌ Balance fetch failed: {e}")
            return {'USD': 0, 'BTC': 0}

    async def close(self):
        """Cleanup - close exchange connection."""
        await self.exchange.close()
        self.active = False
        print("🔌 Sniper Engine shutdown")


# ============================================================================
# INTEGRATION TEMPLATE (For future hft_pipe.py integration)
# ============================================================================

"""
When ready to go live, integrate with hft_pipe.py:

# At top of hft_pipe.py:
from ..market.hft_execution import SniperEngine

# In run_pipe(), after creating Tracker:
sniper = None
if LIVE_TRADING_ENABLED:
    sniper = SniperEngine(API_KEY, API_SECRET, product_id)
    await sniper.warm_up()

# In signal processing (replace paper tracking):
if parts[0] == "BUY" and sniper:
    # Live execution
    await sniper.snipe_order('buy', POSITION_SIZE)
elif parts[0] == "SELL" and sniper:
    await sniper.snipe_order('sell', POSITION_SIZE)
elif parts[0].startswith("CLOSE") and sniper:
    await sniper.close_all()
"""
