#ifndef MLOFI_CALCULATOR_H
#define MLOFI_CALCULATOR_H

#include "order_book.h"
#include <cmath>
#include <array>

// Multi-Level Order Flow Imbalance Calculator
// Based on Cont, Kukanov, Stoikov (2014) & Xu et al. (2019)
class MLOFICalculator {
private:
    static constexpr int MAX_LEVELS = OrderBook::MAX_LEVELS;
    static constexpr float LAMBDA = 1.5f;  // Decay factor for level weighting

    // Level weights: 1/k^λ (Level 1 strongest)
    std::array<float, MAX_LEVELS> level_weights;

public:
    MLOFICalculator() {
        // Pre-compute level weights
        for (int k = 0; k < MAX_LEVELS; ++k) {
            level_weights[k] = 1.0f / std::pow(k + 1.0f, LAMBDA);
        }
    }

    // Calculate OFI at single level k
    // e_k(t) = I(P_k^B > P_{k-1}^B)q_k^B - I(P_k^B < P_{k-1}^B)q_{k-1}^B
    //        - I(P_k^A < P_{k-1}^A)q_k^A + I(P_k^A > P_{k-1}^A)q_{k-1}^A
    float calculate_level_ofi(const OrderBook& book, int level) const {
        if (level < 0 || level >= MAX_LEVELS) return 0.0f;

        float ofi = 0.0f;

        // Bid side contribution
        if (book.bid_prices[level] > book.prev_bid_prices[level]) {
            ofi += book.bid_sizes[level];
        } else if (book.bid_prices[level] < book.prev_bid_prices[level]) {
            ofi -= book.prev_bid_sizes[level];
        }

        // Ask side contribution
        if (book.ask_prices[level] < book.prev_ask_prices[level]) {
            ofi -= book.ask_sizes[level];
        } else if (book.ask_prices[level] > book.prev_ask_prices[level]) {
            ofi += book.prev_ask_sizes[level];
        }

        return ofi;
    }

    // Calculate weighted Multi-Level OFI
    // MLOFI = Σ(OFI_k / k^λ) for k=1..5
    float calculate_mlofi(const OrderBook& book) const {
        float mlofi = 0.0f;

        for (int k = 0; k < MAX_LEVELS; ++k) {
            float level_ofi = calculate_level_ofi(book, k);
            mlofi += level_ofi * level_weights[k];
        }

        return mlofi;
    }

    // Normalize MLOFI by total book depth (for comparability)
    float calculate_normalized_mlofi(const OrderBook& book) const {
        float mlofi = calculate_mlofi(book);

        // Total depth
        float total_depth = 0.0f;
        for (int k = 0; k < MAX_LEVELS; ++k) {
            total_depth += book.bid_sizes[k] + book.ask_sizes[k];
        }

        if (total_depth < 0.001f) return 0.0f;

        return mlofi / total_depth;
    }

    // Simple imbalance (Level 1 only, for comparison)
    float calculate_simple_imbalance(const OrderBook& book) const {
        float bid_qty = book.bid_sizes[0];
        float ask_qty = book.ask_sizes[0];
        float total = bid_qty + ask_qty;

        if (total < 0.001f) return 0.0f;

        return (bid_qty - ask_qty) / total;
    }
};

#endif // MLOFI_CALCULATOR_H
