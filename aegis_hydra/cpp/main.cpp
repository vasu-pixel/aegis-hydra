#include "ising_engine.cpp"
#include "risk_guard.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Calculation for Standard Deviation (Volatility)
double calculate_volatility(const std::vector<float> &history) {
  if (history.empty())
    return 0.0;
  double sum = std::accumulate(history.begin(), history.end(), 0.0);
  double mean = sum / history.size();
  double sq_sum =
      std::inner_product(history.begin(), history.end(), history.begin(), 0.0);
  double variance = (sq_sum / history.size()) - (mean * mean);
  return std::sqrt(std::max(0.0, variance));
}

// Aegis Daemon: Zero-Jitter HFT Edition with Dynamic Thresholds
int main(int argc, char *argv[]) {
  // Phase 20/21: LOCKED FOR HFT
  const int size = 256;
  const float T = 2.27f; // Critical Point
  const float J = 1.0f;  // Reactive
  const uint32_t seed = 42;

  RiskGuard risk_guard;

  // Circular Buffer for Magnetization (Last 1000 ticks)
  const int history_size = 1000;
  std::vector<float> mag_history;
  mag_history.reserve(history_size);
  int history_idx = 0;

  std::cerr << "=== AEGIS HFT DAEMON STARTING (" << size << "x" << size
            << ") ===" << std::endl;
  std::cerr << "=== CRITICALITY: T=" << T << " J=" << J << " ===" << std::endl;

  // Start Engine (Background Thread)
  Engine_start(size, size, seed);
  Engine_update_params(T, J, 0.0f);

  // Buffer for input
  struct {
    float price;
    float latency;
  } packet;

  float last_mag = 0.0f;
  long total_steps = 0;

  // Read binary packet (Price + Latency)
  while (std::cin.read(reinterpret_cast<char *>(&packet), sizeof(packet))) {
    float price_in = packet.price;
    float lat_in = packet.latency;

    // 1. Update Engine
    Engine_update_market(price_in);

    // 2. Poll Result
    float mag = Engine_get_magnetization();
    long steps = Engine_get_steps();
    float phys_latency = Engine_get_latency();

    // Update History
    if (mag_history.size() < history_size) {
      mag_history.push_back(mag);
    } else {
      mag_history[history_idx] = mag;
      history_idx = (history_idx + 1) % history_size;
    }

    // Calculate Dynamic Threshold
    double vol = calculate_volatility(mag_history);
    double buy_thresh = 0.60 + (0.5 * vol);
    double sell_thresh = -0.60 - (0.5 * vol);
    double exit_thresh = 0.40;

    // Heartbeat every 100 prices
    if (total_steps % 100 == 0) {
      std::cerr << "\r[DAEMON] Price: " << price_in << " | M: " << mag
                << " | Vol: " << vol << " | Thresh: " << buy_thresh
                << " | Phys: " << phys_latency << "ms " << std::flush;

      std::cout << "STATE " << total_steps << " " << price_in << " " << mag
                << " " << phys_latency << " " << lat_in << " " << buy_thresh
                << std::endl;
    }
    total_steps++;

    // 3. HFT Logic with RiskGuard & Dynamic Thresholds
    if (mag > (float)buy_thresh && last_mag <= (float)buy_thresh) {
      if (risk_guard.can_trade(lat_in)) {
        std::cout << "BUY " << price_in << std::endl;
      }
    } else if (mag < (float)sell_thresh && last_mag >= (float)sell_thresh) {
      if (risk_guard.can_trade(lat_in)) {
        std::cout << "SELL " << price_in << std::endl;
      }
    } else if (mag < (float)exit_thresh && last_mag >= (float)exit_thresh &&
               last_mag > 0) {
      std::cout << "CLOSE_LONG " << price_in << std::endl;
    } else if (mag > -(float)exit_thresh && last_mag <= -(float)exit_thresh &&
               last_mag < 0) {
      std::cout << "CLOSE_SHORT " << price_in << std::endl;
    }

    last_mag = mag;
  }

  Engine_stop();
  return 0;
}
