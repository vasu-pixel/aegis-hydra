#ifndef HAWKES_ESTIMATOR_H
#define HAWKES_ESTIMATOR_H

#include <deque>
#include <cmath>
#include <numeric>

// Real-time Hawkes Process Branching Ratio Estimator
// Based on Filimonov & Sornette (2012)
// Uses variance/mean method for computational efficiency
class HawkesEstimator {
private:
    static constexpr int WINDOW_SIZE = 600;  // 1 minute at ~100ms updates
    static constexpr float MIN_CRITICALITY = 0.0f;
    static constexpr float MAX_CRITICALITY = 0.95f;

    std::deque<uint64_t> trade_counts;  // Number of trades per tick
    std::deque<double> timestamps;      // Timestamps for windowing

    double window_duration = 60.0;  // 60 seconds

public:
    // Add new trade count observation
    void update(uint64_t count, double timestamp) {
        trade_counts.push_back(count);
        timestamps.push_back(timestamp);

        // Remove old observations outside window
        while (!timestamps.empty() &&
               (timestamp - timestamps.front()) > window_duration) {
            trade_counts.pop_front();
            timestamps.pop_front();
        }
    }

    // Calculate branching ratio n = 1 - sqrt(E[N] / Var[N])
    // Returns: 0.0 (noise) to 0.95 (critical)
    float calculate_criticality() const {
        if (trade_counts.size() < 30) {
            return 0.0f;  // Not enough data
        }

        // Calculate mean
        double sum = std::accumulate(trade_counts.begin(),
                                     trade_counts.end(), 0.0);
        double mean = sum / trade_counts.size();

        if (mean < 0.01) {
            return 0.0f;  // No activity
        }

        // Calculate variance
        double sq_sum = 0.0;
        for (uint64_t count : trade_counts) {
            double diff = count - mean;
            sq_sum += diff * diff;
        }
        double variance = sq_sum / trade_counts.size();

        if (variance < 0.01) {
            return 0.0f;  // No variance = noise
        }

        // Branching ratio: n = 1 - sqrt(mean / variance)
        float n = 1.0f - std::sqrt(mean / variance);

        // Clamp to valid range
        if (n < MIN_CRITICALITY) return MIN_CRITICALITY;
        if (n > MAX_CRITICALITY) return MAX_CRITICALITY;

        return n;
    }

    // Check if market is in critical regime (n > threshold)
    inline bool is_critical(float threshold = 0.7f) const {
        return calculate_criticality() > threshold;
    }

    // Get current sample count
    inline size_t sample_count() const {
        return trade_counts.size();
    }
};

#endif // HAWKES_ESTIMATOR_H
