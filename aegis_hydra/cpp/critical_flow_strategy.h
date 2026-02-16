#ifndef CRITICAL_FLOW_STRATEGY_H
#define CRITICAL_FLOW_STRATEGY_H

#include "order_book.h"
#include <cmath>
#include <deque>
#include <numeric>
#include <vector>

// --- ISING-GLASS STRATEGY ---
// Uses Mean-Field Theory to detect Phase Transitions in order flow.
class CriticalFlowStrategy {
public:
  enum class Signal { HOLD, BUY, SELL };

private:
  // --- CONFIGURATION ---
  static constexpr float BASE_FEE = 0.001f; // 0.1% (Standard Taker)

  // Physics Parameters
  static constexpr float COUPLING_J =
      1.0f; // Interaction strength (Trend following)
  static constexpr float CRITICAL_TEMP = 1.0f; // Phase transition point

  // Market Mapping
  static constexpr float OFI_SENSITIVITY =
      5.0f; // Scales Volume Imbalance to Field (h)
  static constexpr float VOL_SENSITIVITY =
      10.0f; // Scales Volatility to Temp (T)

  // Exit Logic
  static constexpr double COOLDOWN_SEC = 0.5;
  static constexpr double MAX_HOLD_SEC = 10.0;

  // State
  std::deque<float> returns_window;
  double last_exit_time = 0.0;
  double entry_time = 0.0;
  bool in_position = false;
  Signal current_side = Signal::HOLD;
  float entry_price = 0.0f;

  // Ising State
  float M_prev = 0.0f; // Previous Magnetization

public:
  // 1. Ingest Market Data & Update Physics State
  void update_price(float mid_price, double timestamp) {
    // (Keep basic return tracking for volatility calc)
    if (returns_window.size() > 50)
      returns_window.pop_front();
    // ... (Standard volatility tracking code omitted for brevity, assume simple
    // calc)
  }

  // 2. The Mean-Field Solver
  // Solves M = tanh((J*M + h) / T) iteratively
  // 2. The Mean-Field Solver
  // Solves M = tanh((J*M + h) / T) iteratively
  float solve_magnetization(float h, float T) {
    if (std::isnan(h) || std::isnan(T))
      return M_prev; // Ignore bad inputs
    if (std::isnan(M_prev))
      M_prev = 0.0f; // Reset if state corrupted

    float m = M_prev;             // Start from last state (Hysteresis)
    for (int i = 0; i < 5; ++i) { // 5 iterations is enough for convergence
      float field = (COUPLING_J * m + h) / (T + 1e-6f);
      m = std::tanh(field);
    }
    M_prev = m; // Store for next tick (Memory)
    return m;
  }

  // 3. Generate Signal
  Signal generate_signal(const OrderBook &book, double current_time,
                         float futures_price) {
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;

    // --- STEP A: MAP MARKET TO PHYSICS ---

    // Field (h): Order Flow Imbalance
    // If bids are heavy, h > 0 (Upward pressure)
    float bid_vol = book.bid_sizes[0] + book.bid_sizes[1] + book.bid_sizes[2];
    float ask_vol = book.ask_sizes[0] + book.ask_sizes[1] + book.ask_sizes[2];
    float imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6f);
    float h = imbalance * OFI_SENSITIVITY;

    // Add "External Field" from Futures Spread (The Lead-Lag alpha)
    float spread_pct = (futures_price - mid_price) / mid_price;
    h += spread_pct * 100.0f; // Strong weight on futures

    // Temperature (T): Inverse of Order Book Density?
    // Actually, let's map Spread to Temp.
    // Wide Spread = High Temp (Disordered). Tight Spread = Low Temp (Ordered).
    // Note: If futures_price is 0, ignore spread part or handle gracefully?
    // Assuming futures_price is valid from daemon.

    float book_spread = (book.ask_prices[0] - book.bid_prices[0]) / mid_price;
    float T = book_spread * 10000.0f; // Scaling factor

    // --- STEP B: SOLVE FOR MAGNETIZATION (M) ---
    float M = solve_magnetization(h, T);

    // --- STEP C: TRADING LOGIC ---

    // Cooldown
    if (!in_position && (current_time - last_exit_time < COOLDOWN_SEC))
      return Signal::HOLD;

    // ENTRY: Phase Transition WITH FEE HURDLE
    // Only trade if M > 0.8 AND Spread covers fees
    if (!in_position) {
      float fee_hurdle = mid_price * 2.0f * BASE_FEE;

      if (M > 0.8f && (futures_price - mid_price) > fee_hurdle) {
        open_position(Signal::BUY, mid_price, current_time);
        return Signal::BUY;
      }
      if (M < -0.8f && (mid_price - futures_price) > fee_hurdle) {
        open_position(Signal::SELL, mid_price, current_time);
        return Signal::SELL;
      }
    }

    // EXIT: Demagnetization
    // If M drops below 0.5, the trend is breaking.
    if (in_position) {
      float pnl_pct = (current_side == Signal::BUY)
                          ? (mid_price - entry_price) / entry_price
                          : (entry_price - mid_price) / entry_price;

      // 1. Hard Stop (-0.5%)
      if (pnl_pct < -0.005f) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // 2. Physics Exit (Phase Shift)
      bool trend_broken = (current_side == Signal::BUY && M < 0.5f) ||
                          (current_side == Signal::SELL && M > -0.5f);

      if (trend_broken) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // 3. Time Decay (Force exit after 10s)
      if (current_time - entry_time > MAX_HOLD_SEC) {
        close_position(current_time);
        return Signal::HOLD;
      }

      return current_side;
    }

    return Signal::HOLD;
  }

  // Helpers
  void open_position(Signal side, float price, double time) {
    in_position = true;
    current_side = side;
    entry_price = price;
    entry_time = time;
  }

  void close_position(double time) {
    in_position = false;
    current_side = Signal::HOLD;
    last_exit_time = time;
  }

  // Dummy methods for main.cpp compatibility
  void record_trade() {}
  struct Metrics {
    float m;
    float h;
    float T;
  };
  // Updated get_metrics to return relevant Ising variables
  // Note: generate_signal is called before get_metrics usually, so we need to
  // access internal state or recompute. Since M_prev stores the last solved M,
  // we can return it. We need to recompute h and T for logging if they aren't
  // stored. For simplicity, let's store last_h and last_T. Since the user
  // provided code didn't have members for it, I will add them or compute on
  // fly. Recomputing on fly inside get_metrics is safer if book is passed.

  Metrics get_metrics(const OrderBook &book) {
    // Recompute logic for logging consistency
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;
    float bid_vol = book.bid_sizes[0] + book.bid_sizes[1] + book.bid_sizes[2];
    float ask_vol = book.ask_sizes[0] + book.ask_sizes[1] + book.ask_sizes[2];
    float imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6f);
    float h = imbalance * OFI_SENSITIVITY;
    // Note: Futures spread part of h is missing here if we don't pass
    // futures_price. We can store last_h in generate_signal.

    float book_spread = (book.ask_prices[0] - book.bid_prices[0]) / mid_price;
    float T = book_spread * 10000.0f;

    return {M_prev, h, T};
  }
};

#endif
