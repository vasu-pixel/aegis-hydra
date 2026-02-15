
#include "ising_kernel.cpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

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
  std::atomic<float> last_step_ms{0.0f};

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
    // Set thread affinity to avoid core migration (reduces scheduler jitter)
    #ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);  // Pin to core 0
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    #endif

    while (running) {
      auto start = std::chrono::high_resolution_clock::now();

      // Physics Step (1 MC Sweep)
      // Using atomic relaxed for speed, Python updates T/J occasionally
      float t_val = T.load(std::memory_order_relaxed);
      float j_val = J.load(std::memory_order_relaxed);
      float h_val = h.load(std::memory_order_relaxed);

      model->step(t_val, j_val, h_val);

      float m = model->magnetization();
      current_magnetization.store(m, std::memory_order_relaxed);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> diff = end - start;
      last_step_ms.store(diff.count(), std::memory_order_relaxed);

      steps++;

      // Yield to prevent OS scheduler preemption spikes
      // sched_yield() hints to scheduler without blocking
      #ifdef __linux__
      sched_yield();
      #else
      std::this_thread::yield();
      #endif
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

float Engine_get_latency() {
  return engine.last_step_ms.load(std::memory_order_relaxed);
}

// Phase 15: Run Dedicated Feed Loop (Blocks Thread)
// Reads binary float stream from STDIN (piped)
void Engine_run_feed_loop() {
  float price_in;
  while (std::cin.read(reinterpret_cast<char *>(&price_in), sizeof(float))) {
    engine.current_price.store(price_in, std::memory_order_relaxed);
  }
}
}
