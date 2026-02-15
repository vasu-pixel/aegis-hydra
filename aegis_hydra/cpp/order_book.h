#ifndef ORDER_BOOK_H
#define ORDER_BOOK_H

#include <array>
#include <cstdint>

// Multi-Level Order Book (5 levels of depth)
// Optimized for L2 cache (256 bytes total)
struct OrderBook {
    static constexpr int MAX_LEVELS = 5;

    // Price levels (bid descending, ask ascending)
    std::array<float, MAX_LEVELS> bid_prices{};
    std::array<float, MAX_LEVELS> ask_prices{};

    // Quantities at each level
    std::array<float, MAX_LEVELS> bid_sizes{};
    std::array<float, MAX_LEVELS> ask_sizes{};

    // Previous state (for OFI calculation)
    std::array<float, MAX_LEVELS> prev_bid_prices{};
    std::array<float, MAX_LEVELS> prev_ask_prices{};
    std::array<float, MAX_LEVELS> prev_bid_sizes{};
    std::array<float, MAX_LEVELS> prev_ask_sizes{};

    uint64_t update_count = 0;

    // Update book from snapshot
    void update_snapshot(const float* bid_p, const float* bid_s,
                        const float* ask_p, const float* ask_s, int levels);

    // Calculate mid price
    inline float mid_price() const {
        return (bid_prices[0] + ask_prices[0]) * 0.5f;
    }

    // Calculate spread
    inline float spread() const {
        return ask_prices[0] - bid_prices[0];
    }
};

#endif // ORDER_BOOK_H
