
#include "ising_kernel.cpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

// Global Engine State
class IsingEngine {
public:
  IsingModel *model = nullptr;
  std::atomic<bool> running{false};
  std::thread worker;

  // Shared State (Python <-> C++)
  std::atomic<float> current_price{0.0f};
  std::atomic<float> current_magnetization{0.0f};
  std::atomic<float> T{2.27f};
  std::atomic<float> J{1.0f};
  std::atomic<float> h{0.0f};

  // Metrics
  std::atomic<long> steps{0};

  void start(int height, int width, uint32_t seed) {
    if (running)
      return;

    model = new IsingModel(height, width, seed);
    running = true;

    worker = std::thread([this]() { this->run_loop(); });
  }

  void stop() {
    if (!running)
      return;
    running = false;
    if (worker.joinable()) {
      worker.join();
    }
    delete model;
    model = nullptr;
  }

  void run_loop() {
    while (running) {
      // Physics Step (1 MC Sweep)
      // Using atomic relaxed for speed, Python updates T/J occasionally
      float t_val = T.load(std::memory_order_relaxed);
      float j_val = J.load(std::memory_order_relaxed);
      float h_val = h.load(std::memory_order_relaxed);

      // TODO: Calculate h_val based on Price Dynamics?
      // Currently Python sets 'h', or we can implement logic here.
      // For Phase 11 MVP: Python sets 'h'.

      model->step(t_val, j_val, h_val);

      // Update Magnetization (Heavy calculation!)
      // Doing this every step might be too slow for 10M agents.
      // Maybe every 10 steps?
      // User wants max speed. Let's do every step for now and measure.

      float m = model->magnetization();
      current_magnetization.store(m, std::memory_order_relaxed);

      steps++;

      // Aggressive Yield? No yield -> 100% Core Usage.
      // std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }
};

// Singleton Instance
IsingEngine engine;

extern "C" {
void Engine_start(int h, int w, uint32_t seed) { engine.start(h, w, seed); }

void Engine_stop() { engine.stop(); }

// Python feeds Market Data here
void Engine_update_market(float price) {
  engine.current_price.store(price, std::memory_order_relaxed);
}

// Python updates Physics Parameters
void Engine_update_params(float t, float j, float h_ext) {
  engine.T.store(t, std::memory_order_relaxed);
  engine.J.store(j, std::memory_order_relaxed);
  engine.h.store(h_ext, std::memory_order_relaxed);
}

// Python polls Magnetization (instant read)
float Engine_get_magnetization() {
  return engine.current_magnetization.load(std::memory_order_relaxed);
}

long Engine_get_steps() { return engine.steps.load(std::memory_order_relaxed); }
}
