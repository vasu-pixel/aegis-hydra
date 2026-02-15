"""
aegis_hydra.simulation.backtester — Historical Replay Engine

Replays historical order book data through the full physics pipeline
(Tensor Field → Population → Governor → Execution) to measure
performance without risking real capital.

Dependencies: polars, numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

try:
    import polars as pl
except ImportError:
    pl = None

from ..market.tensor_field import TensorField, OrderBookSnapshot, MarketTensor
from ..agents.population import Population
from ..governor.hjb_solver import HJBSolver
from ..governor.allocator import CapitalAllocator
from ..governor.risk_guard import RiskGuard, RiskLevel


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    pnl_curve: List[float]
    risk_events: List[Dict]
    signals: List[Dict]

    @property
    def annualized_return(self) -> float:
        """Assuming ~252 trading days, ~1440 minutes per day."""
        n_steps = len(self.pnl_curve)
        if n_steps < 2:
            return 0.0
        total_minutes = n_steps  # assuming 1-minute steps
        years = total_minutes / (252 * 1440)
        return (1 + self.total_return) ** (1 / max(years, 1e-6)) - 1

    def summary(self) -> Dict:
        return {
            "total_return": f"{self.total_return:.2%}",
            "annualized_return": f"{self.annualized_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "total_trades": self.total_trades,
        }


class Backtester:
    """
    Replays historical data through the full Aegis Hydra pipeline.

    Parameters
    ----------
    population : Population
        The agent swarm (can be scaled down for speed).
    hjb : HJBSolver
        The optimal control solver.
    allocator : CapitalAllocator
        Capital allocation optimizer.
    risk_guard : RiskGuard
        Risk management circuit breaker.
    initial_capital : float
        Starting capital for the simulation.
    """

    def __init__(
        self,
        population: Optional[Population] = None,
        hjb: Optional[HJBSolver] = None,
        allocator: Optional[CapitalAllocator] = None,
        risk_guard: Optional[RiskGuard] = None,
        initial_capital: float = 100.0,
        fee_rate: float = 0.0005,
        temperature: float = 2.27,
    ):
        self.population = population or Population.default(
            n_brownian=1000, n_entropic=1000, n_hamiltonian=1000
        )
        self.hjb = hjb or HJBSolver()
        self.allocator = allocator or CapitalAllocator(budget=initial_capital)
        self.risk_guard = risk_guard or RiskGuard()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.temperature = temperature
        self.tensor_field = TensorField()

    def run(
        self,
        snapshots: List[OrderBookSnapshot],
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Execute a full backtest over a sequence of order book snapshots.

        Parameters
        ----------
        snapshots : List[OrderBookSnapshot]
            Historical data to replay.
        verbose : bool
            Print progress.

        Returns
        -------
        BacktestResult
        """
        import jax
        import jax.numpy as jnp

        key = jax.random.PRNGKey(42)
        states = self.population.initialize(key, temperature=self.temperature)
        
        # JIT compile the step function for performance
        @eqx.filter_jit
        def step_fn(states, tensor_flat, key):
            return self.population.step(states, tensor_flat, key)

        capital = self.initial_capital
        pnl_curve = [capital]
        signals_log: List[Dict] = []
        risk_events: List[Dict] = []
        trades = 0
        wins = 0
        prev_price = None

        print("Compiling physics engine (this may take a minute)...")
        # Warmup / Trigger JIT
        if not snapshots:
             print("Error: Snapshots list is empty or None")
             return BacktestResult(...) 
        print(f"Starting loop with {len(snapshots)} snapshots")
        
        magnetization_curve = []
        price_curve = []
        current_pos = 0.0
        
        for i, snapshot in enumerate(snapshots):
            key, step_key = jax.random.split(key)

            if verbose and i % 100 == 0:
                print(f"Step {i}/{len(snapshots)}...")

            # 1. Convert to tensor
            tensor = self.tensor_field.process(snapshot)
            flat = tensor.to_flat_vector()

            # 2. Step the swarm
            states = step_fn(states, flat, step_key)

            # 3. Aggregate signals
            agg = self.population.aggregate(states)
            signals_log.append({k: float(v) for k, v in agg.items() if isinstance(v, (float, jnp.ndarray))})
            
            # Regime Detection
            magnetization = float(agg.get("magnetization", 0.0))
            magnetization_curve.append(magnetization)

            # 4. Get policy from HJB
            policy = self.hjb.policy_from_swarm(agg)
            
            # 5. Viscosity Deadband (Social Physics Upgrade)
            raw_action = float(policy["action"])
            target_size = float(policy["size"])
            # Current effective position (normalized -1 to 1)
            # We track absolute position, but here we need relative target
            # For simplicity in this engine, action * size IS the target relative position
            target_pos = raw_action * target_size
            
            # Get current volatility from tensor
            vol = float(tensor.volatility)
            
            # Apply Hysteresis
            # We need to know our current exposure. 
            # In this simple backtester, we don't hold state perfectly in 'self.execution' 
            # properly enough for this without loop variable.
            # We have 'current_position' variable in the loop? 
            # Wait, looking at previous code, we calculate 'pnl' but don't explicitly track 'current_position' 
            # persisting across steps in a variable named 'current_position'.
            # We have 'policy' from previous step? No.
            # We need to track 'current_position' in the loop.
            
            # Apply Viscosity
            final_pos = self.hjb.apply_viscosity(target_pos, current_pos, vol)
            
            # 6. Risk check 
            risk = self.risk_guard.assess(
                capital,
                np.array([abs(final_pos)]),
            )
            if risk.level != RiskLevel.GREEN:
                risk_events.append({"step": i, "level": risk.level.name, "reasons": risk.reasons})
                # Scale down intended position if risk is elevated
                scale = self.risk_guard.scale_factor(risk)
                final_pos = final_pos * scale
            
            if self.risk_guard.veto(risk):
                 final_pos = 0.0 

            # Update loop state
            previous_pos = current_pos
            current_pos = final_pos
            
            # 7. Simulated execution / PnL
            # PnL comes from holding 'previous_pos' over the price change 'mid - prev_price'
            mid = float(tensor.mid_price)
            price_curve.append(mid)
            
            if prev_price is not None:
                price_change = (mid - prev_price) / (prev_price + 1e-10)
                # PnL = exposure * capital * %return
                exposure = previous_pos # We held this entering the step
                pnl = exposure * capital * price_change
                capital += pnl
                
                if final_pos != previous_pos:
                     trades += 1
                     # Fee
                     trade_size = abs(final_pos - previous_pos)
                     fee = trade_size * capital * self.fee_rate
                     capital -= fee

                if pnl > 0:
                    wins += 1
                
                self.risk_guard.add_return(pnl / max(capital, 1e-10))

            prev_price = mid
            pnl_curve.append(capital)

        # Compute summary stats
        pnl_arr = np.array(pnl_curve)
        
        self.plot_results(pnl_curve, price_curve, magnetization_curve, snapshots)
        
        if len(pnl_arr) < 2:
            return BacktestResult(
                total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                win_rate=0.0, total_trades=0, pnl_curve=pnl_curve,
                risk_events=risk_events, signals=signals_log,
            )
        # Percentage returns (not absolute)
        returns = np.diff(pnl_arr) / np.maximum(np.abs(pnl_arr[:-1]), 1e-6)
        total_return = (capital - self.initial_capital) / self.initial_capital
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252 * 1440)
        peak = np.maximum.accumulate(pnl_arr)
        drawdowns = (peak - pnl_arr) / np.maximum(peak, 1e-10)
        max_dd = float(np.max(drawdowns))
        win_rate = wins / max(trades, 1)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=trades,
            pnl_curve=pnl_curve,
            risk_events=risk_events,
            signals=signals_log,
        )

    @staticmethod
    def load_from_csv(path: str) -> List[OrderBookSnapshot]:
        """
        Load historical snapshots from a CSV file.

        Expected columns: timestamp_us, symbol, bid_price_0..N, bid_vol_0..N,
        ask_price_0..N, ask_vol_0..N, last_trade_price, last_trade_volume, funding_rate.

        This is a stub — implement parsing based on your data format.
        """
        if pl is None:
            raise ImportError("polars is required for CSV loading")

        df = pl.read_csv(path)
        
        # Ensure required columns exist
        required = {"timestamp_us", "symbol"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")

        snapshots = []
        rows = df.to_dicts()
        
        for row in rows:
            # Extract bids/asks dynamically
            # Assuming columns like bid_price_0, bid_vol_0, etc.
            # OR simple flat structure if user provides it differently.
            # For now, let's support a format compatible with what fetch_data.py will produce.
            # We'll assume the CSV has JSON-dumped arrays for bids/asks OR specific columns.
            # SIMPLER: Let's assume the CSV just has OHLCV and we simulate a dummy order book
            # if explicit order book columns aren't present.
            
            # Check if we have order book columns
            has_ob = "bid_price_0" in row
            
            if has_ob:
                # Parse explicit 50 levels (or however many found)
                n_levels = 50
                bid_prices = [row.get(f"bid_price_{i}", 0.0) for i in range(n_levels)]
                bid_volumes = [row.get(f"bid_vol_{i}", 0.0) for i in range(n_levels)]
                ask_prices = [row.get(f"ask_price_{i}", 0.0) for i in range(n_levels)]
                ask_volumes = [row.get(f"ask_vol_{i}", 0.0) for i in range(n_levels)]
            else:
                # Synthesize order book from OHLCV (Close price)
                price = row.get("close", row.get("price", 100.0))
                open_price = row.get("open", price)
                vol = row.get("volume", 1.0)
                
                # Directional bias from candle body
                delta = price - open_price
                # Sigmoid-like skew: if delta > 0, bid_bias > 0.5
                # Scale delta relative to price (e.g. 1% move -> strong bias)
                rel_delta = delta / (price + 1e-10)
                bias = 0.5 + np.clip(rel_delta * 100.0, -0.4, 0.4) 
                
                bid_vol_total = vol * bias
                ask_vol_total = vol * (1.0 - bias)

                # Create a symmetric spread but asymmetric depth
                spread = price * 0.0001
                bid_prices = [price - spread * (i + 1) for i in range(50)]
                ask_prices = [price + spread * (i + 1) for i in range(50)]
                bid_volumes = [bid_vol_total / 50] * 50
                ask_volumes = [ask_vol_total / 50] * 50

            # Last trade
            last_price = row.get("close", row.get("last_trade_price", 0.0))
            last_vol = row.get("volume", row.get("last_trade_volume", 0.0))
            funding = row.get("funding_rate", 0.0)

            snapshots.append(OrderBookSnapshot(
                timestamp_us=int(row["timestamp_us"]),
                symbol=row["symbol"],
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
                last_trade_price=float(last_price),
                last_trade_volume=float(last_vol),
                funding_rate=float(funding),
            ))
        return snapshots
    def plot_results(self, pnl, prices, magnetization, snapshots):
        """Generate performance visualization."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            times = range(len(pnl))
            # Align lengths if needed
            min_len = min(len(times), len(prices), len(magnetization))
            times = times[:min_len]
            pnl = pnl[:min_len]
            prices = prices[:min_len]
            magnetization = magnetization[:min_len]

            # 1. PnL
            ax1.plot(times, pnl, label="Portfolio Value", color="green")
            ax1.set_ylabel("USD")
            ax1.set_title(f"Equity Curve (Fees: {self.fee_rate:.2%})")
            ax1.grid(True, alpha=0.3)
            
            # 2. BTC Price
            ax2.plot(times, prices, label="BTC Price", color="orange")
            ax2.set_ylabel("Price")
            ax2.set_title("Market Price")
            ax2.grid(True, alpha=0.3)
            
            # 3. Regime (Magnetization)
            ax3.plot(times, magnetization, label="Magnetization (M)", color="purple", linewidth=0.5)
            ax3.axhline(0.7, color="red", linestyle="--", alpha=0.5, label="Critical (+)")
            ax3.axhline(-0.7, color="red", linestyle="--", alpha=0.5, label="Critical (-)")
            ax3.fill_between(times, 0.7, 1.0, color="red", alpha=0.1)
            ax3.fill_between(times, -1.0, -0.7, color="red", alpha=0.1)
            ax3.set_ylabel("Magnetization (-1 to 1)")
            ax3.set_title("Regime Detection (Ising Model)")
            ax3.legend(loc="upper right")
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("backtest_results.png")
            print("Visualization saved to backtest_results.png")
            plt.close()
        except ImportError:
            print("matplotlib not installed, skipping visualization")
        except Exception as e:
            print(f"Visualization failed: {e}")
        return snapshots
