#ifndef RISK_GUARD_H
#define RISK_GUARD_H

#include <atomic>
#include <chrono>
#include <iostream>

class RiskGuard {
private:
  const int MAX_ORDERS_PER_SEC = 10;
  const double MAX_LATENCY_MS = 5.0;

  std::atomic<int> orders_this_second{0};
  std::atomic<long long> last_second_timestamp{0};
  std::atomic<bool> is_circuit_broken{false};

public:
  bool can_trade(double current_network_latency) {
    if (is_circuit_broken.load())
      return false;

    // 1. Latency Watchdog
    if (current_network_latency > MAX_LATENCY_MS) {
      std::cerr << "\n[CRITICAL] Latency Spike: " << current_network_latency
                << "ms. HALTING." << std::endl;
      is_circuit_broken.store(true);
      return false;
    }

    // 2. Machine Gun Brake
    long long current_time =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    if (current_time > last_second_timestamp.load()) {
      orders_this_second.store(0);
      last_second_timestamp.store(current_time);
    }

    if (orders_this_second.load() >= MAX_ORDERS_PER_SEC) {
      std::cerr << "\n[EMERGENCY] Machine Gun Logic Detected. KILLING PROCESS."
                << std::endl;
      exit(1);
    }

    orders_this_second++;
    return true;
  }

  void reset() { is_circuit_broken.store(false); }
};

#endif
