#ifndef CRITICAL_FLOW_STRATEGY_H
#define CRITICAL_FLOW_STRATEGY_H

#include "order_book.h"
#include "hawkes_estimator.h"
#include "mlofi_calculator.h"
#include <deque>
#include <cmath>
#include <numeric>

// Critical-Flow Sniper Strategy
// Combines OFI + Hawkes Criticality + Dynamic Thresholds
class CriticalFlowStrategy {
private:
    HawkesEstimator hawkes;
    MLOFICalculator mlofi_calc;

    // Volatility tracking (for dynamic threshold)
    std::deque<float> price_changes;
    static constexpr int VOL_WINDOW = 100;

    // Strategy parameters
    static constexpr float BASE_FEE = 0.001f;         // 0.1% taker fee
    static constexpr float LATENCY_MS = 8.0f;         // 8ms total latency
    static constexpr float VOL_MULTIPLIER = 3.0f;     // Volatility buffer
    static constexpr float URGENCY_DISCOUNT = 0.0005f; // Criticality discount

    static constexpr float MIN_HAWKES = 0.6f;  // Minimum n to trade
    static constexpr float IDEAL_HAWKES = 0.7f; // Ideal n for trading

    float prev_mid_price = 0.0f;
    uint64_t trade_count_per_tick = 0;

public:
    enum class Signal {
        HOLD,
        BUY,
        SELL
    };

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
        trade_count_per_tick = 0;  // Reset for next tick
    }

    // Increment trade count (call on each trade event)
    void record_trade() {
        trade_count_per_tick++;
    }

    // Calculate current volatility (standard deviation of returns)
    float calculate_volatility() const {
        if (price_changes.size() < 20) {
            return 0.0001f;  // Default minimum
        }

        // Mean
        float sum = std::accumulate(price_changes.begin(),
                                    price_changes.end(), 0.0f);
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
        if (!std::isfinite(vol) || vol < 0.0001f) return 0.0001f;
        if (vol > 0.1f) return 0.1f;

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
        if (threshold < BASE_FEE) threshold = BASE_FEE;

        return threshold;
    }

    // Generate trading signal
    Signal generate_signal(const OrderBook& book) {
        // Calculate MLOFI
        float mlofi = mlofi_calc.calculate_normalized_mlofi(book);

        // Get Hawkes criticality
        float criticality = hawkes.calculate_criticality();

        // Don't trade if market is not critical
        if (criticality < MIN_HAWKES) {
            return Signal::HOLD;
        }

        // Calculate dynamic threshold
        float threshold = calculate_threshold();

        // Generate signal
        if (mlofi > threshold) {
            return Signal::BUY;  // Strong buy pressure
        } else if (mlofi < -threshold) {
            return Signal::SELL; // Strong sell pressure
        }

        return Signal::HOLD;
    }

    // Get current metrics (for monitoring)
    struct Metrics {
        float mlofi;
        float criticality;
        float volatility;
        float threshold;
        size_t hawkes_samples;
    };

    Metrics get_metrics(const OrderBook& book) const {
        return {
            mlofi_calc.calculate_normalized_mlofi(book),
            hawkes.calculate_criticality(),
            calculate_volatility(),
            calculate_threshold(),
            hawkes.sample_count()
        };
    }
};

#endif // CRITICAL_FLOW_STRATEGY_H
