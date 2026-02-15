#ifndef HAWKES_ESTIMATOR_H
#define HAWKES_ESTIMATOR_H

#include <cmath>
#include <deque>
#include <iostream>
#include <numeric>

// Real-time Hawkes Process Branching Ratio Estimator
// Based on Filimonov & Sornette (2012)
// Uses variance/mean method for computational efficiency
class HawkesEstimator {
private:
  static constexpr int WINDOW_SIZE = 100; // 10 seconds at ~100ms updates
  static constexpr float MIN_CRITICALITY = 0.0f;
  static constexpr float MAX_CRITICALITY = 0.95f;

  std::deque<double> trade_counts; // float for EWMA smoothing
  std::deque<double> timestamps;

  // Bucketing for Variance Calculation
  double current_bucket_sum = 0.0;
  double current_bucket_start = 0.0;
  double bucket_size = 1.0; // 1 second buckets (maximize variance)

  double window_duration = 10.0; // 10 seconds (HFT scale)
  double ewma_alpha = 0.2;       // Smoothing factor

public:
  // Add new trade count observation with Time Bucketing
  void update(uint64_t raw_count, double timestamp) {
    if (current_bucket_start == 0.0) {
      current_bucket_start = timestamp;
    }

    // Add to current bucket
    current_bucket_sum += raw_count;

    // If bucket is full (or time jumped), push to deque
    if (timestamp - current_bucket_start >= bucket_size) {

      // Push the accumulated bucket count (keep it raw/integer-like for
      // variance) We can apply slight smoothing if needed, but let's try raw
      // first to get Var > Mean
      trade_counts.push_back(current_bucket_sum);
      timestamps.push_back(current_bucket_start);

      // Reset bucket
      current_bucket_sum = 0.0;
      current_bucket_start = timestamp;
    }

    // Remove old observations outside window
    while (!timestamps.empty() &&
           (timestamp - timestamps.front()) > window_duration) {
      trade_counts.pop_front();
      timestamps.pop_front();
    }
  }

  // Ratio-Based Estimator: n = 1 - (Mean / Variance)
  // Works better for slightly under-dispersed data
  float calculate_criticality() const {
    if (trade_counts.size() < 5) {
      return 0.0f; // Not enough data
    }

    // Calculate Mean
    double sum = std::accumulate(trade_counts.begin(), trade_counts.end(), 0.0);
    double mean = sum / trade_counts.size();

    if (mean < 0.001) {
      return 0.0f; // No activity
    }

    // Calculate Variance
    double sq_sum = 0.0;
    for (double count : trade_counts) {
      double diff = count - mean;
      sq_sum += diff * diff;
    }
    double variance = sq_sum / trade_counts.size();

    // DEBUG: Print internal stats occasionally
    static int debug_counter = 0;
    if (debug_counter++ % 10 == 0) { // Print more often (every 10s)
      std::cerr << "[HAWKES DEBUG] Mean: " << mean << " | Var: " << variance
                << " | Count: " << trade_counts.size() << " | Window: "
                << (timestamps.empty()
                        ? 0.0
                        : (timestamps.back() - timestamps.front()))
                << "s"
                << " | Raw Counts: [";
      for (size_t i = 0; i < trade_counts.size(); ++i) {
        std::cerr << trade_counts[i]
                  << (i == trade_counts.size() - 1 ? "" : ", ");
      }
      std::cerr << "]" << std::endl;
    }

    if (variance < 0.0001) {
      return 0.0f; // No variance
    }

    // Ratio-Based Estimator
    // If Var > Mean, it's clustered (n > 0).
    // n = 1 - (Mean / Variance)
    float n = 0.0f;
    if (variance > mean) {
      n = 1.0f - (float)(mean / variance);
    }

    // Clamp to valid range
    if (n < MIN_CRITICALITY)
      return MIN_CRITICALITY;
    if (n > MAX_CRITICALITY)
      return MAX_CRITICALITY;

    return n;
  }

  // Check if market is in critical regime
  inline bool is_critical(float threshold = 0.5f) const {
    return calculate_criticality() > threshold;
  }

  // Get current sample count
  inline size_t sample_count() const { return trade_counts.size(); }
};

#endif // HAWKES_ESTIMATOR_H
