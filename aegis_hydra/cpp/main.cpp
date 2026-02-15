
#include "ising_engine.cpp"
#include <iostream>
#include <string>
#include <vector>

// Aegis Daemon: Standalone Binary
// Reads Market Data from STDIN (Price)
// Outputs Trading Signals to STDOUT (Size)
int main(int argc, char *argv[]) {
  int size = 1000;
  float T = 2.27f;
  float J = 1.0f;
  uint32_t seed = 42;

  // Parse quick args if needed (skipped for speed)
  if (argc > 1)
    size = std::stoi(argv[1]);

  std::cerr << "=== AEGIS DAEMON STARTING (" << size << "x" << size
            << ") ===" << std::endl;

  // Start Engine (Background Thread)
  Engine_start(size, size, seed);
  Engine_update_params(T, J, 0.0f);

  // Main Loop (Consume STDIN)
  // Format: 4 byte float (Price)
  float price_in;
  float last_mag = 0.0f;
  long total_steps = 0;

  while (std::cin.read(reinterpret_cast<char *>(&price_in), sizeof(float))) {
    // 1. Update Engine (Atomic)
    Engine_update_market(price_in);

    // 2. Poll Result (Atomic)
    float mag = Engine_get_magnetization();
    long steps = Engine_get_steps();

    // Heartbeat every 100 prices
    if (total_steps % 100 == 0) {
      std::cerr << "\r[DAEMON] Price: " << price_in << " | M: " << mag
                << " | Steps: " << steps << std::flush;
    }
    total_steps++;

    // 3. Logic (Simple Viscosity)
    // If mag crosses threshold, emit signal to stdout
    // Threshold: 0.85
    if (mag > 0.85f && last_mag <= 0.85f) {
      std::cout << "BUY " << price_in << std::endl;
    } else if (mag < -0.2f && last_mag >= -0.2f) {
      std::cout << "SELL " << price_in << std::endl;
    } else if (mag > -0.2f && last_mag <= -0.2f) {
      std::cout << "CLOSE_SHORT " << price_in << std::endl;
    } else if (mag < 0.85f && last_mag >= 0.85f) {
      std::cout << "CLOSE_LONG " << price_in << std::endl;
    }

    last_mag = mag;
  }

  Engine_stop();
  return 0;
}
