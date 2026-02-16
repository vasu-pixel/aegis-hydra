#ifndef CRITICAL_FLOW_STRATEGY_H
#define CRITICAL_FLOW_STRATEGY_H

#include "hawkes_estimator.h"
#include "mlofi_calculator.h"
#include "order_book.h"
#include <cmath>
#include <deque>
#include <numeric>

class CriticalFlowStrategy {
public:
  enum class Signal { HOLD, BUY, SELL };

private:
  HawkesEstimator hawkes;
  MLOFICalculator mlofi_calc;
  std::deque<float> price_changes;
  static constexpr int VOL_WINDOW = 100;

  // --- CONFIGURATION ---
  // ✅ FIXED: Fee is 0.1% (Standard Taker)
  // This prevents the bot from taking trades that lose money on fees.
  static constexpr float BASE_FEE = 0.001f;

  static constexpr float LATENCY_MS = 8.0f;
  static constexpr float VOL_MULTIPLIER = 3.0f;
  static constexpr float URGENCY_DISCOUNT = 0.0005f;

  static constexpr float ENTRY_CRITICALITY = 0.8f;
  static constexpr float SURRENDER_CRITICALITY = 0.2f;

  // ✅ FIXED: Grace Period Removed (0.0s)
  // We want to exit INSTANTLY if the trade goes bad or fills.
  static constexpr double GRACE_PERIOD_SEC = 0.0;
  static constexpr double COOLDOWN_SEC = 0.5;

  float prev_mid_price = 0.0f;
  uint64_t trade_count_per_tick = 0;

  double last_exit_time = 0.0;
  double entry_time = 0.0;
  bool in_position = false;
  bool is_arb_trade = false;
  float entry_price = 0.0f;
  Signal current_side = Signal::HOLD;

  float spread_mean = 0.0f;
  float spread_var = 0.0f;
  float last_z_score = 0.0f;
  static constexpr float SPREAD_ALPHA = 0.001f;

public:
  void update_price(float mid_price, double timestamp) {
    if (prev_mid_price > 0.0f) {
      float price_change = (mid_price - prev_mid_price) / prev_mid_price;
      if (std::abs(price_change) < 0.1f) {
        price_changes.push_back(price_change);
        if (price_changes.size() > VOL_WINDOW)
          price_changes.pop_front();
      }
    }
    prev_mid_price = mid_price;
    hawkes.update(trade_count_per_tick, timestamp);
    trade_count_per_tick = 0;
  }

  void record_trade() { trade_count_per_tick++; }

  float calculate_volatility() const {
    if (price_changes.size() < 20)
      return 0.0001f;
    float sum =
        std::accumulate(price_changes.begin(), price_changes.end(), 0.0f);
    float mean = sum / price_changes.size();
    float sq_sum = 0.0f;
    for (float change : price_changes) {
      float diff = change - mean;
      sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / price_changes.size());
  }

  float calculate_threshold() const {
    float volatility = calculate_volatility();
    float criticality = hawkes.calculate_criticality();
    return (2.0f * BASE_FEE) +
           (VOL_MULTIPLIER * volatility * std::sqrt(LATENCY_MS / 1000.0f)) -
           (URGENCY_DISCOUNT * criticality);
  }

  Signal generate_signal(const OrderBook &book, double current_time,
                         float futures_price) {
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;
    float spread = futures_price - mid_price;

    if (futures_price > 0.0f) {
      spread_mean = (1 - SPREAD_ALPHA) * spread_mean + SPREAD_ALPHA * spread;
      spread_var = (1 - SPREAD_ALPHA) * spread_var +
                   SPREAD_ALPHA * std::pow(spread - spread_mean, 2);
    }
    float sigma = std::sqrt(spread_var);
    float z_score = (sigma > 1e-6f) ? (spread - spread_mean) / sigma : 0.0f;
    const_cast<CriticalFlowStrategy *>(this)->last_z_score = z_score;

    float mlofi = mlofi_calc.calculate_normalized_mlofi(book);
    float criticality = hawkes.calculate_criticality();
    float threshold = calculate_threshold();

    if (!in_position && (current_time - last_exit_time < COOLDOWN_SEC))
      return Signal::HOLD;

    // --- ENTRY LOGIC ---
    if (!in_position) {
      // Enter only if spread covers fees (0.2% round trip)
      if (z_score > 3.0f && spread > (mid_price * 2.0f * BASE_FEE)) {
        open_position(Signal::BUY, mid_price, current_time, true);
        return Signal::BUY;
      }
      if (z_score < -3.0f && spread < -(mid_price * 2.0f * BASE_FEE)) {
        open_position(Signal::SELL, mid_price, current_time, true);
        return Signal::SELL;
      }
    }

    // --- POSITION MANAGEMENT ---
    if (in_position) {
      float pnl_pct = (current_side == Signal::BUY)
                          ? (mid_price - entry_price) / entry_price
                          : (entry_price - mid_price) / entry_price;

      // 1. Hard Stop Loss (-0.5%)
      if (pnl_pct < -0.005f) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // 2. Take Profit (Fee Hurdle + 0.05% Pure Profit)
      // If we made money, BANK IT. Don't wait for Z-score.
      if (pnl_pct > (2.0f * BASE_FEE + 0.0005f)) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // 3. ARBITRAGE EXIT (Dynamic Decay)
      if (is_arb_trade) {
        double hold_duration = current_time - entry_time;
        float exit_threshold = 0.5f; // Default: Wait for perfect reversion

        // If held > 2 seconds, accept "good enough" (Z < 1.5)
        if (hold_duration > 2.0)
          exit_threshold = 1.5f;

        // If held > 5 seconds, GET OUT (Z < 2.5)
        if (hold_duration > 5.0)
          exit_threshold = 2.5f;

        bool converged =
            (current_side == Signal::BUY && z_score < exit_threshold) ||
            (current_side == Signal::SELL && z_score > -exit_threshold);

        if (converged) {
          close_position(current_time);
          return Signal::HOLD;
        }
        return current_side;
      }

      // 4. FLOW EXIT (Surrender)
      if (criticality < SURRENDER_CRITICALITY) {
        close_position(current_time);
        return Signal::HOLD;
      }

      return current_side;
    }

    // --- FLOW ENTRY (Secondary) ---
    if (criticality < ENTRY_CRITICALITY)
      return Signal::HOLD;
    if (mlofi > threshold) {
      open_position(Signal::BUY, mid_price, current_time, false);
      return Signal::BUY;
    } else if (mlofi < -threshold) {
      open_position(Signal::SELL, mid_price, current_time, false);
      return Signal::SELL;
    }

    return Signal::HOLD;
  }

  void open_position(Signal side, float price, double time, bool is_arb) {
    in_position = true;
    current_side = side;
    entry_price = price;
    entry_time = time;
    is_arb_trade = is_arb;
  }

  void close_position(double time) {
    in_position = false;
    current_side = Signal::HOLD;
    last_exit_time = time;
  }

  // (Metrics struct kept same as before...)
  struct Metrics {
    float mlofi;
    float criticality;
    float volatility;
    float threshold;
    size_t hawkes_samples;
    float z_score;
  };

  Metrics get_metrics(const OrderBook &book) const {
    return {mlofi_calc.calculate_normalized_mlofi(book),
            hawkes.calculate_criticality(),
            calculate_volatility(),
            calculate_threshold(),
            hawkes.sample_count(),
            last_z_score};
  }
};
#endif
