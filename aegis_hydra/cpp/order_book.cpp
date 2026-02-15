#include "order_book.h"
#include <cstring>
#include <algorithm>

void OrderBook::update_snapshot(const float* bid_p, const float* bid_s,
                                const float* ask_p, const float* ask_s,
                                int levels) {
    // Save previous state for OFI calculation
    std::memcpy(prev_bid_prices.data(), bid_prices.data(),
                MAX_LEVELS * sizeof(float));
    std::memcpy(prev_ask_prices.data(), ask_prices.data(),
                MAX_LEVELS * sizeof(float));
    std::memcpy(prev_bid_sizes.data(), bid_sizes.data(),
                MAX_LEVELS * sizeof(float));
    std::memcpy(prev_ask_sizes.data(), ask_sizes.data(),
                MAX_LEVELS * sizeof(float));

    // Update with new snapshot (up to MAX_LEVELS)
    int copy_levels = std::min(levels, MAX_LEVELS);

    std::memcpy(bid_prices.data(), bid_p, copy_levels * sizeof(float));
    std::memcpy(ask_prices.data(), ask_p, copy_levels * sizeof(float));
    std::memcpy(bid_sizes.data(), bid_s, copy_levels * sizeof(float));
    std::memcpy(ask_sizes.data(), ask_s, copy_levels * sizeof(float));

    // Fill remaining levels with zeros if snapshot has fewer levels
    if (copy_levels < MAX_LEVELS) {
        std::fill(bid_prices.begin() + copy_levels, bid_prices.end(), 0.0f);
        std::fill(ask_prices.begin() + copy_levels, ask_prices.end(), 0.0f);
        std::fill(bid_sizes.begin() + copy_levels, bid_sizes.end(), 0.0f);
        std::fill(ask_sizes.begin() + copy_levels, ask_sizes.end(), 0.0f);
    }

    update_count++;
}
