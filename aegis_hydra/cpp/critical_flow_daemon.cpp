#include "critical_flow_strategy.h"
#include "order_book.h"
#include <iostream>
#include <chrono>
#include <cstring>

// Binary packet from Python
// Use packed attribute to match Python struct packing exactly
struct __attribute__((packed)) InputPacket {
    float mid_price;
    float net_latency;
    double recv_time;
    uint32_t trade_count;  // Number of trades since last tick

    // Order book levels (5 levels of bid/ask)
    float bid_prices[5];
    float bid_sizes[5];
    float ask_prices[5];
    float ask_sizes[5];
};

int main() {
    std::cerr << "=== CRITICAL FLOW SNIPER DAEMON ===" << std::endl;
    std::cerr << "=== Strategy: OFI + Hawkes Criticality ===" << std::endl;

    CriticalFlowStrategy strategy;
    OrderBook book;

    InputPacket packet;
    long total_ticks = 0;

    CriticalFlowStrategy::Signal last_signal = CriticalFlowStrategy::Signal::HOLD;

    // Read binary packets from stdin
    while (std::cin.read(reinterpret_cast<char*>(&packet), sizeof(packet))) {
        auto start = std::chrono::high_resolution_clock::now();

        // Validate price (BTC should be between 1,000 and 200,000)
        if (packet.mid_price < 1000.0f || packet.mid_price > 200000.0f) {
            std::cerr << "\n⚠️  Bad price: " << packet.mid_price << " - skipping tick" << std::endl;
            continue;  // Skip this tick
        }

        // Update order book
        book.update_snapshot(packet.bid_prices, packet.bid_sizes,
                            packet.ask_prices, packet.ask_sizes, 5);

        // Record trades for Hawkes estimator
        if (packet.trade_count > 0) {
            std::cerr << "[C++] Received " << packet.trade_count << " trades" << std::endl;
        }
        for (uint32_t i = 0; i < packet.trade_count; ++i) {
            strategy.record_trade();
        }

        // Update strategy with new price
        strategy.update_price(packet.mid_price, packet.recv_time);

        // Generate signal
        auto signal = strategy.generate_signal(book);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> latency = end - start;

        // Print state every 10 ticks
        if (total_ticks % 10 == 0) {
            auto metrics = strategy.get_metrics(book);

            std::cerr << "\r[DAEMON] Price: " << packet.mid_price
                      << " | MLOFI: " << metrics.mlofi
                      << " | n: " << metrics.criticality
                      << " | Samples: " << metrics.hawkes_samples  // Show sample count
                      << " | σ: " << metrics.volatility
                      << " | Thresh: " << metrics.threshold
                      << " | Latency: " << latency.count() << "ms "
                      << std::flush;

            // Output state for Python (including hawkes_samples for debugging)
            std::cout << "STATE " << total_ticks
                      << " " << packet.mid_price
                      << " " << metrics.mlofi
                      << " " << latency.count()
                      << " " << packet.net_latency
                      << " " << metrics.criticality
                      << " " << metrics.volatility
                      << " " << metrics.threshold
                      << " " << metrics.hawkes_samples  // Add sample count to output
                      << std::endl;
        }

        // Emit signals on transitions
        if (signal != last_signal && signal != CriticalFlowStrategy::Signal::HOLD) {
            if (signal == CriticalFlowStrategy::Signal::BUY) {
                std::cout << "BUY " << packet.mid_price << std::endl;
            } else if (signal == CriticalFlowStrategy::Signal::SELL) {
                std::cout << "SELL " << packet.mid_price << std::endl;
            }
        }
        // Emit close signals when reverting to HOLD from active position
        else if (last_signal != CriticalFlowStrategy::Signal::HOLD &&
                 signal == CriticalFlowStrategy::Signal::HOLD) {
            if (last_signal == CriticalFlowStrategy::Signal::BUY) {
                std::cout << "CLOSE_LONG " << packet.mid_price << std::endl;
            } else if (last_signal == CriticalFlowStrategy::Signal::SELL) {
                std::cout << "CLOSE_SHORT " << packet.mid_price << std::endl;
            }
        }

        last_signal = signal;
        total_ticks++;
    }

    std::cerr << "\n=== DAEMON SHUTDOWN ===" << std::endl;
    return 0;
}
