#ifndef CRITICAL_FLOW_STRATEGY_H
#define CRITICAL_FLOW_STRATEGY_H

#include "hawkes_estimator.h"
#include "mlofi_calculator.h"
#include "order_book.h"
#include <cmath>
#include <deque>
#include <numeric>

// Critical-Flow Sniper Strategy
// Combines OFI + Hawkes Criticality + Dynamic Thresholds
class CriticalFlowStrategy {
public:
  enum class Signal { HOLD, BUY, SELL };

private:
  HawkesEstimator hawkes;
  MLOFICalculator mlofi_calc;

  // Volatility tracking (for dynamic threshold)
  std::deque<float> price_changes;
  static constexpr int VOL_WINDOW = 100;

  // Strategy parameters
  static constexpr float BASE_FEE = 0.001f;          // 0.1% taker fee
  static constexpr float LATENCY_MS = 8.0f;          // 8ms total latency
  static constexpr float VOL_MULTIPLIER = 3.0f;      // Volatility buffer
  static constexpr float URGENCY_DISCOUNT = 0.0005f; // Criticality discount

  // Dynamic Surrender & Churn Reduction
  static constexpr float ENTRY_CRITICALITY =
      0.7f; // Only enter on strong bursts
  static constexpr float SURRENDER_CRITICALITY = 0.2f; // Exit if signal decays
  static constexpr float MAINTENANCE_CRITICALITY = 0.3f; // Hold zone

  static constexpr double MIN_PROFIT_PCT = 0.0005; // 0.05% net profit target
  static constexpr double COOLDOWN_SEC = 2.0;      // 2s cooldown after exit

  float prev_mid_price = 0.0f;
  uint64_t trade_count_per_tick = 0;

  // State Tracking
  double last_exit_time = 0.0;
  bool in_position = false;
  float entry_price = 0.0f;
  Signal current_side = Signal::HOLD;

public:
  enum class Signal { HOLD, BUY, SELL };

  // Update with new price tick
  void update_price(float mid_price, double timestamp) {
    if (prev_mid_price > 0.0f) {
      float price_change = (mid_price - prev_mid_price) / prev_mid_price;

      // Sanity check: reject extreme price changes (> 10%)
      // This filters corrupted data
      if (std::abs(price_change) < 0.1f) {
        price_changes.push_back(price_change);

        if (price_changes.size() > VOL_WINDOW) {
          price_changes.pop_front();
        }
      }
    }
    prev_mid_price = mid_price;

    // Update Hawkes with trade count
    hawkes.update(trade_count_per_tick, timestamp);
    trade_count_per_tick = 0; // Reset for next tick
  }

  // Increment trade count (call on each trade event)
  void record_trade() { trade_count_per_tick++; }

  // Calculate current volatility (standard deviation of returns)
  float calculate_volatility() const {
    if (price_changes.size() < 20) {
      return 0.0001f; // Default minimum
    }

    // Mean
    float sum =
        std::accumulate(price_changes.begin(), price_changes.end(), 0.0f);
    float mean = sum / price_changes.size();

    // Variance
    float sq_sum = 0.0f;
    for (float change : price_changes) {
      float diff = change - mean;
      sq_sum += diff * diff;
    }
    float variance = sq_sum / price_changes.size();

    // Safety: prevent NaN/inf
    if (variance < 0.0f || !std::isfinite(variance)) {
      return 0.0001f;
    }

    float vol = std::sqrt(variance);

    // Clamp to reasonable range (0.01% to 10% volatility)
    if (!std::isfinite(vol) || vol < 0.0001f)
      return 0.0001f;
    if (vol > 0.1f)
      return 0.1f;

    return vol;
  }

  // Calculate dynamic threshold
  // H = 2*Fee + σ*sqrt(Δ) - γ*n
  float calculate_threshold() const {
    float volatility = calculate_volatility();
    float criticality = hawkes.calculate_criticality();

    // Base cost (2x fee for round-trip)
    float base_cost = 2.0f * BASE_FEE;

    // Delay risk (volatility * sqrt(latency in seconds))
    float latency_sec = LATENCY_MS / 1000.0f;
    float delay_risk = VOL_MULTIPLIER * volatility * std::sqrt(latency_sec);

    // Urgency discount (lower threshold when market is critical)
    float urgency = URGENCY_DISCOUNT * criticality;

    float threshold = base_cost + delay_risk - urgency;

    // Ensure positive threshold
    if (threshold < BASE_FEE)
      threshold = BASE_FEE;

    return threshold;
  }

  // Generate trading signal with Dynamic Surrender
  Signal generate_signal(const OrderBook &book, double current_time) {
    // Calculate parameters
    float mlofi = mlofi_calc.calculate_normalized_mlofi(book);
    float criticality = hawkes.calculate_criticality();
    float threshold = calculate_threshold();
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;

    // 1. Cooldown Check
    if (!in_position && (current_time - last_exit_time < COOLDOWN_SEC)) {
      return Signal::HOLD;
    }

    // 2. Position Management (Surrender Logic)
    if (in_position) {
      // Calculate unrealized P&L %
      float pnl_pct = 0.0f;
      if (current_side == Signal::BUY) {
        pnl_pct = (mid_price - entry_price) / entry_price;
      } else if (current_side == Signal::SELL) {
        pnl_pct = (entry_price - mid_price) / entry_price;
      }

      // CRITICAL: Dynamic Surrender
      // Exit if signal decays (n < 0.2) OR MLOFI reverses
      bool signal_decayed = criticality < SURRENDER_CRITICALITY;
      bool signal_reversed = (current_side == Signal::BUY && mlofi < -0.1f) ||
                             (current_side == Signal::SELL && mlofi > 0.1f);

      // Hard Stop Loss (safety net)
      if (pnl_pct < -0.005f) { // -0.5% stop
        close_position(current_time);
        return Signal::HOLD;
      }

      // Take Profit (if target met)
      if (pnl_pct > MIN_PROFIT_PCT) {
        // Trailing logic could go here, for now take profit
        close_position(current_time);
        return Signal::HOLD;
      }

      // Surrender: Close even at small loss if signal is dead
      if (signal_decayed || signal_reversed) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // Maintenance: Hold if still critical (n > 0.3)
      return current_side; // START HOLDING
    }

    // 3. Entry Logic (Sniping)
    // Only enter if Market is SUPER CRITICAL (n > 0.7)
    if (criticality < ENTRY_CRITICALITY) {
      return Signal::HOLD;
    }

    if (mlofi > threshold) {
      open_position(Signal::BUY, mid_price);
      return Signal::BUY;
    } else if (mlofi < -threshold) {
      open_position(Signal::SELL, mid_price);
      return Signal::SELL;
    }

    return Signal::HOLD;
  }

  void open_position(Signal side, float price) {
    in_position = true;
    current_side = side;
    entry_price = price;
  }

  void close_position(double time) {
    in_position = false;
    current_side = Signal::HOLD;
    last_exit_time = time;
  }

  // Get current metrics (for monitoring)
  struct Metrics {
    float mlofi;
    float criticality;
    float volatility;
    float threshold;
    size_t hawkes_samples;
  };

  Metrics get_metrics(const OrderBook &book) const {
    return {mlofi_calc.calculate_normalized_mlofi(book),
            hawkes.calculate_criticality(), calculate_volatility(),
            calculate_threshold(), hawkes.sample_count()};
  }
};

#endif // CRITICAL_FLOW_STRATEGY_H
