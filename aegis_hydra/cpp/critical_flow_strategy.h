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
      5.0f; // 5.0 = Disabled (Pure Arb Mode)
  static constexpr float SURRENDER_CRITICALITY = 0.2f; // Exit if signal decays
  static constexpr float MAINTENANCE_CRITICALITY = 0.3f; // Hold zone

  static constexpr double MIN_PROFIT_PCT = 0.0005; // 0.05% net profit target
  static constexpr double COOLDOWN_SEC = 2.0;      // 2s cooldown after exit
  static constexpr double GRACE_PERIOD_SEC =
      5.0; // 5s immunity after entry (Increased)

  float prev_mid_price = 0.0f;
  uint64_t trade_count_per_tick = 0;

  // State Tracking
  double last_exit_time = 0.0;
  double entry_time = 0.0;
  bool in_position = false;
  bool is_arb_trade = false; // NEW: Distinguish Arb vs Flow trades
  float entry_price = 0.0f;
  Signal current_side = Signal::HOLD;

  // Lead-Lag Arbitrage State
  float spread_mean = 0.0f;
  float spread_var = 0.0f;
  float last_z_score = 0.0f;                    // For display
  static constexpr float SPREAD_ALPHA = 0.001f; // EMA decay

public:
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

  // Generate trading signal with Dynamic Surrender + Lead-Lag Arb
  Signal generate_signal(const OrderBook &book, double current_time,
                         float futures_price) {
    // Calculate parameters
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;
    float spread = futures_price - mid_price;

    // 0. Update Spread Stats (Lead-Lag)
    if (futures_price > 0.0f) {
      spread_mean = (1 - SPREAD_ALPHA) * spread_mean + SPREAD_ALPHA * spread;
      spread_var = (1 - SPREAD_ALPHA) * spread_var +
                   SPREAD_ALPHA * std::pow(spread - spread_mean, 2);
    }

    float sigma = std::sqrt(spread_var);
    float z_score = (sigma > 1e-6f) ? (spread - spread_mean) / sigma : 0.0f;
    const_cast<CriticalFlowStrategy *>(this)->last_z_score =
        z_score; // Update display metric

    float mlofi = mlofi_calc.calculate_normalized_mlofi(book);
    float criticality = hawkes.calculate_criticality();
    float threshold = calculate_threshold();

    // 1. Cooldown Check
    if (!in_position && (current_time - last_exit_time < COOLDOWN_SEC)) {
      return Signal::HOLD;
    }

    // 3. Lead-Lag Arbitrage Entry (Priority 1)
    if (!in_position) {
      if (z_score > 3.0f && spread > (mid_price * 2.0f * BASE_FEE)) {
        open_position(Signal::BUY, mid_price, current_time, true); // True = Arb
        return Signal::BUY;
      }
      if (z_score < -3.0f && spread < -(mid_price * 2.0f * BASE_FEE)) {
        open_position(Signal::SELL, mid_price, current_time,
                      true); // True = Arb
        return Signal::SELL;
      }
    }

    // 2. Position Management
    if (in_position) {
      // Calculate unrealized P&L %
      float pnl_pct = 0.0f;
      if (current_side == Signal::BUY) {
        pnl_pct = (mid_price - entry_price) / entry_price;
      } else if (current_side == Signal::SELL) {
        pnl_pct = (entry_price - mid_price) / entry_price;
      }

      // Hard Stop Loss (safety net) - ALWAYS ACTIVE
      if (pnl_pct < -0.005f) { // -0.5% stop
        close_position(current_time);
        return Signal::HOLD;
      }

      // Grace Period: Ignore noise/surrender for first 5s
      if (current_time - entry_time < GRACE_PERIOD_SEC) {
        return current_side;
      }

      // Take Profit (Gross PnL must cover fees + tail risk)
      float fee_hurdle = 2.0f * BASE_FEE;
      if (pnl_pct > (fee_hurdle + MIN_PROFIT_PCT)) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // ARBITRAGE EXIT LOGIC
      if (is_arb_trade) {
        // Exit if Z-Score reverts (Mean Reversion)
        // Threshold: 0.5 (close enough to mean)
        bool converged = (current_side == Signal::BUY && z_score < 0.5f) ||
                         (current_side == Signal::SELL && z_score > -0.5f);

        if (converged) {
          close_position(current_time);
          return Signal::HOLD;
        }
        // IGNORE MLOFI/HAWKES Surrender for Arb trades
        return current_side;
      }

      // FLOW EXIT LOGIC (Surrender)
      bool signal_decayed = criticality < SURRENDER_CRITICALITY;
      bool signal_reversed = (current_side == Signal::BUY && mlofi < -0.1f) ||
                             (current_side == Signal::SELL && mlofi > 0.1f);

      if (signal_decayed || signal_reversed) {
        close_position(current_time);
        return Signal::HOLD;
      }

      // Maintenance: Hold if still critical (n > 0.3)
      return current_side; // START HOLDING
    }

    // 3. Entry Logic (Sniping - Flow Based)
    // Only enter if Market is SUPER CRITICAL (n > 0.7)
    if (criticality < ENTRY_CRITICALITY) {
      return Signal::HOLD;
    }

    if (mlofi > threshold) {
      open_position(Signal::BUY, mid_price, current_time,
                    false); // False = Flow
      return Signal::BUY;
    } else if (mlofi < -threshold) {
      open_position(Signal::SELL, mid_price, current_time,
                    false); // False = Flow
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

  // Get current metrics (for monitoring)
  struct Metrics {
    float mlofi;
    float criticality;
    float volatility;
    float threshold;
    size_t hawkes_samples;
    float z_score; // NEW: Lead-Lag Z-Score
  };

  Metrics get_metrics(const OrderBook &book) const {
    // Re-calculate Z-Score for display (slightly wasteful but safe)
    float sigma = std::sqrt(spread_var);
    float z =
        (sigma > 1e-6f)
            ? (spread_mean / sigma)
            : 0.0f; // Approx (using mean as current spread proxy for display)
    // Actually, we want the *current* Z-score.
    // Let's store the last Z-score as a member variable for display.
    return {mlofi_calc.calculate_normalized_mlofi(book),
            hawkes.calculate_criticality(),
            calculate_volatility(),
            calculate_threshold(),
            hawkes.sample_count(),
            last_z_score};
  }
};

#endif // CRITICAL_FLOW_STRATEGY_H
