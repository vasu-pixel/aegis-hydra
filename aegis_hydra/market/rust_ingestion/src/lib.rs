//! aegis_ingestion — High-Speed WebSocket Ingestion Module
//!
//! Compiled as a Python extension (via PyO3 + maturin).
//! Listens to Binance/Solana WebSockets, parses JSON order book
//! updates, and pushes structured data into shared memory for
//! the Python physics engine.
//!
//! Build: `cd market/rust_ingestion && maturin develop --release`

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------------------

/// Raw order book update from exchange WebSocket.
#[derive(Debug, Deserialize, Clone)]
struct OrderBookUpdate {
    /// Microsecond timestamp
    #[serde(rename = "T", default)]
    timestamp: u64,

    /// Symbol (e.g., "BTCUSDT")
    #[serde(rename = "s", default)]
    symbol: String,

    /// Bid price-quantity pairs [[price, qty], ...]
    #[serde(rename = "b", default)]
    bids: Vec<[String; 2]>,

    /// Ask price-quantity pairs [[price, qty], ...]
    #[serde(rename = "a", default)]
    asks: Vec<[String; 2]>,
}

/// Parsed, typed order book snapshot ready for Python consumption.
#[derive(Debug, Clone)]
struct ParsedSnapshot {
    timestamp_us: u64,
    symbol: String,
    bid_prices: Vec<f64>,
    bid_volumes: Vec<f64>,
    ask_prices: Vec<f64>,
    ask_volumes: Vec<f64>,
}

impl ParsedSnapshot {
    fn from_update(update: &OrderBookUpdate) -> Self {
        let parse_level = |pair: &[String; 2]| -> (f64, f64) {
            let price = pair[0].parse::<f64>().unwrap_or(0.0);
            let volume = pair[1].parse::<f64>().unwrap_or(0.0);
            (price, volume)
        };

        let (bid_prices, bid_volumes): (Vec<f64>, Vec<f64>) =
            update.bids.iter().map(parse_level).unzip();
        let (ask_prices, ask_volumes): (Vec<f64>, Vec<f64>) =
            update.asks.iter().map(parse_level).unzip();

        ParsedSnapshot {
            timestamp_us: update.timestamp * 1000, // ms → μs
            symbol: update.symbol.clone(),
            bid_prices,
            bid_volumes,
            ask_prices,
            ask_volumes,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared State (lock-free ring buffer would be ideal; Mutex for scaffold)
// ---------------------------------------------------------------------------

/// Thread-safe snapshot buffer shared between WebSocket thread and Python.
struct SnapshotBuffer {
    latest: Mutex<Option<ParsedSnapshot>>,
    count: Mutex<u64>,
}

impl SnapshotBuffer {
    fn new() -> Self {
        SnapshotBuffer {
            latest: Mutex::new(None),
            count: Mutex::new(0),
        }
    }

    fn push(&self, snapshot: ParsedSnapshot) {
        let mut latest = self.latest.lock().unwrap();
        *latest = Some(snapshot);
        let mut count = self.count.lock().unwrap();
        *count += 1;
    }

    fn take(&self) -> Option<ParsedSnapshot> {
        let mut latest = self.latest.lock().unwrap();
        latest.take()
    }

    fn message_count(&self) -> u64 {
        *self.count.lock().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Python-Exposed Module
// ---------------------------------------------------------------------------

/// The main ingestion engine, exposed to Python.
#[pyclass]
struct IngestionEngine {
    buffer: Arc<SnapshotBuffer>,
    url: String,
    is_running: Arc<Mutex<bool>>,
}

#[pymethods]
impl IngestionEngine {
    /// Create a new ingestion engine.
    ///
    /// # Arguments
    /// * `ws_url` - WebSocket URL (e.g., "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms")
    #[new]
    fn new(ws_url: String) -> Self {
        IngestionEngine {
            buffer: Arc::new(SnapshotBuffer::new()),
            url: ws_url,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the WebSocket listener in a background thread.
    ///
    /// This spawns a Tokio runtime and connects to the exchange.
    /// Call `get_latest()` from Python to retrieve the most recent snapshot.
    fn start(&self) -> PyResult<()> {
        let buffer = Arc::clone(&self.buffer);
        let url = self.url.clone();
        let is_running = Arc::clone(&self.is_running);

        {
            let mut running = is_running.lock().unwrap();
            if *running {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Ingestion already running",
                ));
            }
            *running = true;
        }

        // Spawn background thread with Tokio runtime
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                // TODO: Implement actual WebSocket connection
                // For scaffold, this is a placeholder loop
                //
                // Production code would:
                // 1. Connect to `url` via tokio_tungstenite
                // 2. Parse each message as OrderBookUpdate
                // 3. Convert to ParsedSnapshot
                // 4. Push to buffer
                //
                // Example:
                // let (ws_stream, _) = connect_async(&url).await.unwrap();
                // let (_, mut read) = ws_stream.split();
                // while let Some(msg) = read.next().await {
                //     let text = msg.unwrap().into_text().unwrap();
                //     let update: OrderBookUpdate = serde_json::from_str(&text).unwrap();
                //     buffer.push(ParsedSnapshot::from_update(&update));
                // }

                tracing::info!("Ingestion engine started (scaffold mode)");
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    if !*is_running.lock().unwrap() {
                        break;
                    }
                }
            });
        });

        Ok(())
    }

    /// Stop the WebSocket listener.
    fn stop(&self) -> PyResult<()> {
        let mut running = self.is_running.lock().unwrap();
        *running = false;
        Ok(())
    }

    /// Get the latest order book snapshot as a Python dict.
    ///
    /// Returns None if no data is available yet.
    fn get_latest(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.buffer.take() {
            Some(snap) => {
                let dict = PyDict::new(py);
                dict.set_item("timestamp_us", snap.timestamp_us)?;
                dict.set_item("symbol", &snap.symbol)?;
                dict.set_item("bid_prices", &snap.bid_prices)?;
                dict.set_item("bid_volumes", &snap.bid_volumes)?;
                dict.set_item("ask_prices", &snap.ask_prices)?;
                dict.set_item("ask_volumes", &snap.ask_volumes)?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    /// Get the total number of messages received.
    fn message_count(&self) -> u64 {
        self.buffer.message_count()
    }

    /// Check if the engine is currently running.
    fn is_running(&self) -> bool {
        *self.is_running.lock().unwrap()
    }
}

/// Python module definition.
#[pymodule]
fn aegis_ingestion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IngestionEngine>()?;
    Ok(())
}
