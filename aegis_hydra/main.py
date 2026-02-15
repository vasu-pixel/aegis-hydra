"""
aegis_hydra.main — The Launchpad

Entry point for the Aegis Hydra Econophysics Engine.

Modes:
    live      — Connect to exchange, run the full pipeline in real-time
    backtest  — Replay historical data through the engine
    simulate  — Generate synthetic data and test the math
    demo      — Small-scale demo with console output

Usage:
    python -m aegis_hydra.main --mode demo
    python -m aegis_hydra.main --mode backtest --data ./data/btc_2024.csv
    python -m aegis_hydra.main --mode live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from .utils.logger import get_logger


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file, expanding ${ENV_VAR} references."""
    config_path = Path(__file__).parent / path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        raw = f.read()
    # Expand shell-style ${VAR} and $VAR references to actual env values
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)


def run_demo(config: dict) -> None:
    """
    Small-scale demo: generate synthetic data, run 100 steps,
    print the swarm signals and a mini backtest result.
    """
    import jax
    import jax.numpy as jnp

    from .agents.population import Population
    from .governor.hjb_solver import HJBSolver
    from .governor.risk_guard import RiskGuard
    from .simulation.synthetic_market import SyntheticMarket, REGIME_CALM, REGIME_VOLATILE
    from .simulation.backtester import Backtester

    logger = get_logger("demo")
    logger.info("=== AEGIS HYDRA — Demo Mode ===")

    # Create small population for demo
    sim_cfg = config.get("simulation", {})
    phys_cfg = config.get("physics", {})
    pop = Population.default(
        n_brownian=sim_cfg.get("backtest_brownian", 1000),
        n_entropic=sim_cfg.get("backtest_entropic", 1000),
        n_hamiltonian=sim_cfg.get("backtest_hamiltonian", 1000),
        dt=phys_cfg.get("dt", 1e-3),
    )
    logger.info(f"Population initialized: {pop.total_agents:,} agents")

    # Generate synthetic data
    market = SyntheticMarket(initial_price=50000.0, seed=sim_cfg.get("seed", 42))
    snapshots = market.generate_regime_switch([
        (REGIME_CALM, 500),
        (REGIME_VOLATILE, 200),
        (REGIME_CALM, 300),
    ])
    logger.info(f"Generated {len(snapshots)} synthetic snapshots")

    # Run backtest
    backtester = Backtester(
        population=pop,
        initial_capital=sim_cfg.get("initial_capital", 100.0),
    )
    logger.info("Running backtest...")
    result = backtester.run(snapshots, verbose=True)

    logger.info("=== Backtest Results ===")
    for k, v in result.summary().items():
        logger.info(f"  {k}: {v}")

    logger.info(f"  Risk events: {len(result.risk_events)}")
    logger.info("=== Demo Complete ===")


async def run_live(config: dict) -> None:
    """
    Live trading mode. Connects to exchange and runs the full pipeline.
    """
    import jax

    from .agents.population import Population
    from .governor.hjb_solver import HJBSolver
    from .governor.allocator import CapitalAllocator
    from .governor.risk_guard import RiskGuard, RiskLevel
    from .market.tensor_field import TensorField, OrderBookSnapshot
    from .market.execution import ExecutionEngine, OrderSide

    log_cfg = config.get("logging", {})
    logger = get_logger("live", level=log_cfg.get("level", "INFO"))
    logger.info("=== AEGIS HYDRA — Live Mode ===")
    logger.warning("THIS IS LIVE TRADING. Proceed with caution.")

    # Initialize components with safe config access
    phys = config.get("physics", {})
    gov = config.get("governor", {})
    mkt = config.get("market", {})
    hjb_cfg = gov.get("hjb", {})
    alloc_cfg = gov.get("allocator", {})
    risk_cfg = gov.get("risk_guard", {})
    tf_cfg = mkt.get("tensor_field", {})

    pop = Population.default(
        n_brownian=phys.get("n_brownian", 400_000),
        n_entropic=phys.get("n_entropic", 300_000),
        n_hamiltonian=phys.get("n_hamiltonian", 300_000),
        dt=phys.get("dt", 1e-3),
    )
    hjb = HJBSolver(
        gamma_risk=hjb_cfg.get("gamma_risk", 1.0),
        transaction_cost=hjb_cfg.get("transaction_cost", 0.001),
    )
    allocator = CapitalAllocator(
        budget=alloc_cfg.get("budget", 100.0),
        n_slots=alloc_cfg.get("n_slots", 100),
    )
    risk_guard = RiskGuard(
        var_limit=risk_cfg.get("var_limit", 0.05),
        max_drawdown_limit=risk_cfg.get("max_drawdown", 0.10),
    )
    tensor_field = TensorField(
        n_levels=mkt.get("n_levels", 50),
        vol_window=tf_cfg.get("vol_window", 100),
    )
    execution = ExecutionEngine(
        exchange_id=mkt.get("exchange", "binance"),
        symbol=mkt.get("symbol", "BTC/USDT"),
        testnet=mkt.get("testnet", True),
    )

    # Connect to exchange
    await execution.connect()
    logger.info(f"Connected to {mkt.get('exchange', 'binance')} ({'testnet' if mkt.get('testnet', True) else 'LIVE'})")

    # Initialize swarm
    key = jax.random.PRNGKey(0)
    states = pop.initialize(key)
    logger.info(f"Swarm initialized: {pop.total_agents:,} agents")

    # Main loop
    logger.info("Entering main loop... (Ctrl+C to stop)")
    try:
        step = 0
        while True:
            key, step_key = jax.random.split(key)

            # TODO: Replace with real WebSocket data from ingestion.rs
            # For now, this is a placeholder that would be fed by the Rust ingestion module
            logger.debug(f"Step {step}: Awaiting market data...")

            # The actual loop would:
            # 1. Receive snapshot from ingestion
            # 2. tensor = tensor_field.process(snapshot)
            # 3. states = pop.step(states, tensor.to_flat_vector(), step_key)
            # 4. agg = pop.aggregate(states)
            # 5. policy = hjb.policy_from_swarm(agg)
            # 6. risk = risk_guard.assess(...)
            # 7. if not risk_guard.veto(risk): execution.place_order(...)

            await asyncio.sleep(1)  # Placeholder tick rate
            step += 1

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await execution.disconnect()
        logger.info("Disconnected. Goodbye.")


async def run_paper(config: dict, args: argparse.Namespace) -> None:
    """
    Paper Trading Mode.
    Connects to exchange (read-only) and mocks execution.
    """
    from .simulation.paper import PaperTrader
    from .agents.population import Population
    from .governor.risk_guard import RiskGuard

    # Configuration
    sim_cfg = config.get("simulation", {})
    
    # Initialize Population (Grid Size from args)
    print(f"Initializing Ising Grid ({args.grid_size}x{args.grid_size})...")
    pop = Population.default(
        coupling=args.coupling, 
        threshold=args.threshold, 
        grid_size=args.grid_size
    )
    
    # Risk Guard
    rg = RiskGuard(concentration_limit=1.0)
    
    # Initialize Paper Trader
    trader = PaperTrader(
        population=pop,
        risk_guard=rg,
        symbol="BTC/USD", # Coinbase uses BTC/USD usually backtester/fetch_data logic
        initial_capital=sim_cfg.get("initial_capital", 10000.0),
        exchange_id=args.exchange,
        temperature=args.temp,
        coupling=args.coupling,
        viscosity_buy=args.viscosity_buy,
        viscosity_sell=args.viscosity_sell,
        min_hold_seconds=args.min_hold_seconds,
        aggregation_seconds=args.aggregation,
        use_cpp=args.use_cpp
    )
    
    await trader.run()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aegis Hydra — Econophysics Trading Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "backtest", "live", "simulate", "paper"],
        default="demo",
        help="Execution mode",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to historical data CSV (backtest mode)",
    )
    parser.add_argument("--temp", type=float, default=2.27, help="Ising Temperature")
    parser.add_argument("--coupling", type=float, default=1.0, help="Ising Coupling J")
    parser.add_argument("--threshold", type=float, default=0.7, help="Magnetization Threshold")
    parser.add_argument("--grid-size", type=int, default=3162, help="Grid dim (size x size)")
    parser.add_argument("--exchange", type=str, default="coinbase", help="Exchange ID (ccxt)")
    parser.add_argument("--viscosity_buy", type=float, default=0.85, help="Buy Threshold (>0.85)")
    parser.add_argument("--viscosity_sell", type=float, default=0.2, help="Sell Threshold (<0.2)")
    parser.add_argument("--min_hold_seconds", type=float, default=60.0, help="Minimum Hold Time")
    parser.add_argument("--aggregation", type=float, default=5.0, help="Candle Aggregation Time (s)")
    parser.add_argument("--use-cpp", action="store_true", help="Use C++ Core for Physics")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "demo":
        run_demo(config)
    elif args.mode == "live":
        asyncio.run(run_live(config))
    elif args.mode == "backtest":
        if args.data is None:
            print("Error: --data required for backtest mode")
            sys.exit(1)
            
        from .simulation.backtester import Backtester
        from .agents.population import Population
        
        print(f"Loading backtest data from {args.data}...")
        try:
            snapshots = Backtester.load_from_csv(args.data)
            print(f"Loaded {len(snapshots)} snapshots.")
            
            # Initialize backtester with default population (10,000 agents)
            # FAST_DEV_RUN: Set to True to use small population for debugging
            FAST_DEV_RUN = False
            
            if FAST_DEV_RUN:
                 print("WARNING: Running in FAST_DEV_RUN mode (tiny population)")
                 pop = Population.default(n_brownian=10, n_entropic=10, n_hamiltonian=10, coupling=args.coupling, threshold=args.threshold, grid_size=args.grid_size)
            else:
                 print(f"Initializing full population (+ Ising Grid {args.grid_size}x{args.grid_size} T={args.temp} J={args.coupling})...")
                 pop = Population.default(coupling=args.coupling, threshold=args.threshold, grid_size=args.grid_size)

            # Initialize RiskGuard with relaxed concentration for single-asset backtest
            from .governor.risk_guard import RiskGuard
            rg = RiskGuard(concentration_limit=1.0)

            backtester = Backtester(
                population=pop,
                risk_guard=rg,
                initial_capital=config["simulation"]["initial_capital"],
                fee_rate=0.0005,  # 0.05% per trade
                temperature=args.temp # Pass temp to backtester to pass to pop.initialize? 
                # Backtester calls pop.initialize internally?? 
                # Backtester.__init__ calls pop.initialize IF passed? No.
                # Backtester.run loop manages state.
                # Let's check Backtester.run
            )
            
            print("Running backtest...")
            result = backtester.run(snapshots, verbose=True)
            
            print("=== Backtest Results ===")
            for k, v in result.summary().items():
                print(f"  {k}: {v}")
                
            print(f"  Risk events: {len(result.risk_events)}")
            
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.mode == "simulate":
        run_demo(config)  # Simulate = demo with synthetic data
    elif args.mode == "paper":
        # Paper Trading Mode
        asyncio.run(run_paper(config, args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
