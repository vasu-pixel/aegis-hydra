#ifndef CRITICAL_FLOW_STRATEGY_H
#define CRITICAL_FLOW_STRATEGY_H

#include "order_book.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>

// --- THE UNIFIED FIELD STRATEGY (HJB + ISING) ---
// Type: Market Maker (Provides Liquidity)
// Goal: Capture Spread (0.05%) + Avoid Fees (0.0%)
class CriticalFlowStrategy {
public:
  enum class Signal { HOLD, BUY, SELL };

private:
  // --- CONFIGURATION ---
  // Maker Fees are 0.0% (Tier 0) or negative (Rebates).
  // We effectively have a negative cost basis.
  static constexpr float MAKER_FEE = 0.0f;

  // HJB Parameters (Inventory Control)
  static constexpr float RISK_AVERSION =
      5.0f; // Gamma (Aggressiveness to dump inventory)
  static constexpr float MAX_INVENTORY =
      0.01f; // Max BTC to hold before panic dumping

  // Ising Parameters (Trend Following)
  static constexpr float ISING_ALPHA = 10.0f; // Trend influence on Fair Value
  static constexpr float COUPLING_J = 1.0f;   // Ising Interaction Strength
  static constexpr float CRITICAL_TEMP = 2.269f; // Curie Temperature

  // Market Physics
  static constexpr double COOLDOWN_SEC = 0.1; // Fast requoting

  // State
  std::deque<float> price_history;
  double last_action_time = 0.0;
  float inventory_btc = 0.0f;   // Current Position (q)
  float M_prev = 0.0f;          // Previous Magnetization
  float entry_price = 0.0f;     // For tracking
  float last_fair_value = 0.0f; // For logging

public:
  // 1. Ingest Prices & Track Volatility
  void update_price(float mid_price, double timestamp) {
    if (price_history.size() > 100)
      price_history.pop_front();
    price_history.push_back(mid_price);
  }

  // Calculate Volatility (Sigma) for HJB Risk Term
  float calculate_volatility() const {
    if (price_history.size() < 10)
      return 5.0f; // Default high vol

    float sum = 0.0f, sq_sum = 0.0f;
    for (float p : price_history)
      sum += p;
    float mean = sum / price_history.size();

    for (float p : price_history)
      sq_sum += (p - mean) * (p - mean);
    return std::sqrt(sq_sum / price_history.size());
  }

  // 2. Solve Ising Magnetization (Mean Field Approximation)
  // 2. Solve Ising Magnetization (Mean Field Approximation)
  float solve_ising_magnetization(float imbalance, float spread_vol) {
    // Map Market to Physics Fields
    float h = imbalance * 5.0f;         // External Field = Order Flow
    float T = spread_vol * 0.1f + 0.1f; // Temperature = Volatility

    // Mean Field Iteration: M = tanh((J*M + h)/T)
    float m = M_prev;
    for (int i = 0; i < 3; ++i) { // Fast convergence
      m = std::tanh((COUPLING_J * m + h) / T);
    }
    M_prev = m;
    return m;
  }

  // 3. Generate Maker Signals (HJB Logic)
  Signal generate_signal(const OrderBook &book, double current_time,
                         float futures_price) {
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;

    // --- STEP A: CALCULATE "FAIR VALUE" (The HJB Price) ---
    // This is the core "Proprietary" alpha.

    // 1. Inventory Risk (q * gamma * sigma^2)
    float vol = calculate_volatility();
    float inventory_skew = inventory_btc * RISK_AVERSION * vol;

    // 2. Trend Bias (Ising Magnetization)
    float bid_qty = book.bid_sizes[0];
    float ask_qty = book.ask_sizes[0];
    float imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-5f);
    float M = solve_ising_magnetization(imbalance, vol);
    float trend_skew = M * ISING_ALPHA;

    // 3. The "Unified Field" Price
    // If we are Long (q>0), Price drops -> We are more eager to Sell, less
    // eager to Buy. If M > 0 (Trend Up), Price rises -> We chase the trend.
    float fair_value = mid_price + trend_skew - inventory_skew;
    last_fair_value = fair_value; // Store for logging

    // --- STEP B: GENERATE QUOTES ---

    // Target Spread (Minimum to cover risk + profit)
    // 5bps (0.05%) spread is our target
    float half_spread = mid_price * 0.00025f;

    float my_bid = fair_value - half_spread;
    float my_ask = fair_value + half_spread;

    // --- STEP C: EXECUTION LOGIC (Maker) ---
    // We signal BUY if the market Best Bid is below our Fair Bid (We want to
    // post there) We signal SELL if the market Best Ask is above our Fair Ask

    // Safety: Don't spam signals
    if (current_time - last_action_time < COOLDOWN_SEC)
      return Signal::HOLD;

    // Logic: If our calculated "Safe Buy Price" (my_bid) is higher than the
    // current best bid, it means the market is "cheap" relative to our alpha.
    // We post a limit buy. NOTE: In a real Maker bot, we would return the
    // PRICE. Here we return SIGNAL to trigger the Python logic.

    // Dynamic Adjustment:
    // If we have NO inventory, we want to acquire (BUY).
    // If we have inventory, we want to dispose (SELL).

    if (std::abs(inventory_btc) < MAX_INVENTORY) {
      // We are building position
      if (my_bid > book.bid_prices[0]) {
        // Our alpha says price is going up, join the bid
        return Signal::BUY;
      }
    }

    if (inventory_btc > 0) {
      // We are Long, looking to Sell
      if (my_ask < book.ask_prices[0]) {
        // Our alpha says price is dropping (or we have risk), join the ask
        return Signal::SELL;
      }
    }

    return Signal::HOLD;
  }

  // --- STATE TRACKING ---
  // Python must call this when fills happen to update 'q' (Inventory)
  void update_inventory(float qty_change) { inventory_btc += qty_change; }

  // Standard helpers
  void open_position(Signal s, float p, double t, bool b) {
    last_action_time = t;
  }
  void close_position(double t) { last_action_time = t; }
  void record_trade() {}

  struct Metrics {
    float M;
    float q;
    float fair;
    float vol;
    size_t n;
    float z;
  };
  Metrics get_metrics(const OrderBook &) const {
    return {M_prev, inventory_btc, last_fair_value, 0.0f, 0, 0.0f};
  }
};

#endif
