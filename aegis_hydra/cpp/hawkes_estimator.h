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

  double window_duration = 10.0; // 10 seconds (HFT scale)
  double ewma_alpha = 0.2;       // Smoothing factor

public:
  // Add new trade count observation with EWMA smoothing
  void update(uint64_t raw_count, double timestamp) {
    double smoothed_count = (double)raw_count;

    if (!trade_counts.empty()) {
      double prev = trade_counts.back();
      smoothed_count = (ewma_alpha * raw_count) + ((1.0 - ewma_alpha) * prev);
    }

    trade_counts.push_back(smoothed_count);
    timestamps.push_back(timestamp);

    // Remove old observations outside window
    while (!timestamps.empty() &&
           (timestamp - timestamps.front()) > window_duration) {
      trade_counts.pop_front();
      timestamps.pop_front();
    }
  }

  // Moment-Based Estimator: n = (Var - Mean) / Var
  // Measures clustering (excess variance) relative to random Poisson (Var=Mean)
  float calculate_criticality() const {
    if (trade_counts.size() < 10) {
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
    if (debug_counter++ % 50 == 0) {
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

    // Moment-Based Estimator
    // If Var > Mean, it's clustered (n > 0).
    // If Var <= Mean, it's random or regular (n = 0).
    float n = 0.0f;
    if (variance > mean) {
      n = (float)((variance - mean) / variance);
    }

    // HFT adjustment: Amplify weak signals slightly
    n = std::pow(n, 0.8f);

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
