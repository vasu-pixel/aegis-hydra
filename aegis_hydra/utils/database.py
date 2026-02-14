"""
aegis_hydra.utils.database — Database Connectors

Redis: Fast in-memory store for the swarm "pheromone" state.
TimescaleDB: Time-series database for historical data and backtest results.

Dependencies: redis, psycopg2 (for TimescaleDB)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import redis
except ImportError:
    redis = None

import json


@dataclass
class RedisConnector:
    """
    Redis connector for the swarm pheromone state database.

    Stores fast-changing agent state that needs to be shared
    across processes (e.g., when using Ray for distribution).

    Parameters
    ----------
    host : str
        Redis host.
    port : int
        Redis port.
    db : int
        Redis database index.
    prefix : str
        Key prefix for all Aegis Hydra keys.
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    prefix: str = "aegis:"

    def __post_init__(self):
        self._client: Optional[object] = None

    def connect(self) -> None:
        if redis is None:
            raise ImportError("redis is required. pip install redis")
        self._client = redis.Redis(
            host=self.host, port=self.port, db=self.db, decode_responses=True
        )

    def _key(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def _require_connection(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise RuntimeError("RedisConnector not connected. Call connect() first.")

    def set_state(self, key: str, state: Dict[str, Any], ttl: int = 60) -> None:
        """Store agent state with TTL (seconds)."""
        self._require_connection()
        self._client.setex(self._key(key), ttl, json.dumps(state, default=str))

    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent state."""
        self._require_connection()
        val = self._client.get(self._key(key))
        return json.loads(val) if val else None

    def publish_signal(self, channel: str, signal: Dict) -> None:
        """Publish a signal to a Redis pub/sub channel."""
        self._require_connection()
        self._client.publish(self._key(channel), json.dumps(signal, default=str))

    def get_latest_signals(self, key: str, n: int = 10) -> List[Dict]:
        """Get last N signals from a Redis list."""
        self._require_connection()
        items = self._client.lrange(self._key(key), 0, n - 1)
        return [json.loads(item) for item in items]

    def push_signal(self, key: str, signal: Dict, max_len: int = 1000) -> None:
        """Push a signal to a bounded list."""
        self._require_connection()
        pipe = self._client.pipeline()
        pipe.lpush(self._key(key), json.dumps(signal, default=str))
        pipe.ltrim(self._key(key), 0, max_len - 1)
        pipe.execute()

    def close(self) -> None:
        if self._client:
            self._client.close()


@dataclass
class TimeScaleConnector:
    """
    TimescaleDB connector for time-series storage.

    Stores order book snapshots, backtest results, and agent
    history for long-term analysis.

    Parameters
    ----------
    host : str
        Database host.
    port : int
        Database port.
    dbname : str
        Database name.
    user : str
        Database user.
    password : str
        Database password.
    """

    host: str = "localhost"
    port: int = 5432
    dbname: str = "aegis_hydra"
    user: str = "aegis"
    password: str = ""

    def __post_init__(self):
        self._conn: Optional[object] = None

    def connect(self) -> None:
        """Establish database connection."""
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
            )
        except ImportError:
            raise ImportError("psycopg2 is required for TimescaleDB. pip install psycopg2-binary")

    def create_tables(self) -> None:
        """Create the required hypertables (run once)."""
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        ddl = """
        CREATE TABLE IF NOT EXISTS order_book_snapshots (
            time        TIMESTAMPTZ NOT NULL,
            symbol      TEXT NOT NULL,
            mid_price   DOUBLE PRECISION,
            spread      DOUBLE PRECISION,
            imbalance   DOUBLE PRECISION,
            bid_depth   DOUBLE PRECISION,
            ask_depth   DOUBLE PRECISION
        );
        SELECT create_hypertable('order_book_snapshots', 'time', if_not_exists => TRUE);

        CREATE TABLE IF NOT EXISTS agent_signals (
            time              TIMESTAMPTZ NOT NULL,
            composite_signal  DOUBLE PRECISION,
            brownian_mean     DOUBLE PRECISION,
            entropy_gradient  DOUBLE PRECISION,
            chaos_fraction    DOUBLE PRECISION
        );
        SELECT create_hypertable('agent_signals', 'time', if_not_exists => TRUE);

        CREATE TABLE IF NOT EXISTS trades (
            time        TIMESTAMPTZ NOT NULL,
            side        TEXT,
            size        DOUBLE PRECISION,
            price       DOUBLE PRECISION,
            pnl         DOUBLE PRECISION,
            risk_level  TEXT
        );
        SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);
        """
        with self._conn.cursor() as cur:
            cur.execute(ddl)
        self._conn.commit()

    def insert_snapshot(self, data: Dict) -> None:
        """Insert a single order book snapshot."""
        if self._conn is None:
            return
        sql = """
        INSERT INTO order_book_snapshots (time, symbol, mid_price, spread, imbalance, bid_depth, ask_depth)
        VALUES (NOW(), %(symbol)s, %(mid_price)s, %(spread)s, %(imbalance)s, %(bid_depth)s, %(ask_depth)s)
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, data)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
