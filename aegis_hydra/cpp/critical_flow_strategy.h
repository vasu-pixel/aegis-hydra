#ifndef CRITICAL_FLOW_STRATEGY_H
#define CRITICAL_FLOW_STRATEGY_H

#include "order_book.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <vector>

// --- QUANTUM VISCOSITY STRATEGY (Final Form) ---
// Integrates:
// 1. Quantum Energy (Jarque-Bera): Detects Fat Tails (Crash Risk).
// 2. Multiscale Hurst (Variance Ratio): Detects Trend vs Mean Reversion.
// 3. Ising Model: Detects Directional Pressure.
//
// Action Matrix:
// - Ground State (H < 0.5, E ~ 0): Viscous Making (Safe).
// - Excited State (H > 0.5, E > 0): Defense Mode (Wide Spreads).
// - Tunneling (E >> 0): Stop / Retreat.

class CriticalFlowStrategy {
public:
  enum class Signal { HOLD, BUY, SELL };

private:
  static constexpr float MAKER_FEE = 0.0f;

  // Risk & Inventory
  static constexpr float RISK_AVERSION = 10.0f;
  static constexpr float MAX_INVENTORY = 0.002f;

  // Physics
  static constexpr float ISING_ALPHA = 2.0f; // ~2bps fair value shift
  static constexpr float COUPLING_J = 1.2f;
  static constexpr float CRITICAL_M = 0.6f;

  // SPREAD FILTERS
  static constexpr float MIN_SPREAD_BPS = 0.0001f; // 1bp (tight BTC markets)
  static constexpr float MAX_SPREAD_BPS =
      0.0080f; // 80bps (Allow wider captures)

  // THRESHOLDS
  static constexpr float ENERGY_CRASH_LIMIT = 0.80f; // Stop Trading
  static constexpr float ENERGY_EXCITED = 0.40f;     // Widen Spreads
  static constexpr float HURST_TRENDING = 0.55f; // Widen Spreads (Proxy > 1.1)

  // State
  std::deque<float> price_history;
  double last_action_time = 0.0;
  float inventory_btc = 0.0f;
  float M_prev = 0.0f;
  float last_fair_value = 0.0f;
  float last_energy = 0.0f;
  float last_hurst = 0.5f;
  bool is_leader = true;

  // USDT-USD Spread EMA (Z-Score Normalization)
  float spread_ema = 0.0f;
  bool spread_initialized = false;
  static constexpr float SPREAD_EMA_ALPHA = 0.005f; // Slow EMA (~200 ticks)

public:
  void update_price(float mid_price, double timestamp) {
    if (price_history.size() > 60)
      price_history.pop_front();
    price_history.push_back(mid_price);
  }

  // Calculate "Market Energy" using Jarque-Bera on Returns
  float calculate_market_energy() const {
    if (price_history.size() < 20)
      return 0.5f;

    std::vector<float> returns;
    returns.reserve(price_history.size());
    for (size_t i = 1; i < price_history.size(); ++i) {
      float r = std::log(price_history[i] / price_history[i - 1]);
      returns.push_back(r);
    }

    float sum = 0.0f;
    for (float r : returns)
      sum += r;
    float mean = sum / returns.size();

    float sq_sum = 0.0f;
    float cube_sum = 0.0f;
    float quart_sum = 0.0f;

    for (float r : returns) {
      float d = r - mean;
      sq_sum += d * d;
      cube_sum += d * d * d;
      quart_sum += d * d * d * d;
    }

    float n = (float)returns.size();
    float variance = sq_sum / n;
    float std_dev = std::sqrt(variance);

    if (variance < 1e-12f)
      return 0.0f;

    float skewness = (cube_sum / n) / (std_dev * std_dev * std_dev);
    float kurtosis = (quart_sum / n) / (variance * variance);

    float excess_kurt = kurtosis - 3.0f;
    float jb = (n / 6.0f) *
               (skewness * skewness + 0.25f * (excess_kurt * excess_kurt));

    return std::log1p(jb) / 10.0f;
  }

  // Calculate Proxy for Hurst Exponent (Variance Ratio)
  // VR = Var(r_k) / (k * Var(r_1))
  // If VR < 1 -> Mean Reverting (H < 0.5)
  // If VR > 1 -> Trending (H > 0.5)
  // We allow k=5 for short-term memory
  float calculate_hurst_proxy() const {
    if (price_history.size() < 20)
      return 0.5f;

    std::vector<float> r1; // 1-step returns
    std::vector<float> r5; // 5-step returns

    for (size_t i = 1; i < price_history.size(); ++i) {
      r1.push_back(std::log(price_history[i] / price_history[i - 1]));
    }

    if (price_history.size() > 5) {
      for (size_t i = 5; i < price_history.size(); ++i) {
        r5.push_back(std::log(price_history[i] / price_history[i - 5]));
      }
    } else {
      return 0.5f;
    }

    // Calc Variance 1
    float sum1 = 0.0f, sq_sum1 = 0.0f;
    for (float r : r1) {
      sum1 += r;
      sq_sum1 += r * r;
    }
    float var1 =
        (sq_sum1 / r1.size()) - (sum1 / r1.size()) * (sum1 / r1.size());

    // Calc Variance 5
    float sum5 = 0.0f, sq_sum5 = 0.0f;
    for (float r : r5) {
      sum5 += r;
      sq_sum5 += r * r;
    }
    float var5 =
        (sq_sum5 / r5.size()) - (sum5 / r5.size()) * (sum5 / r5.size());

    if (var1 < 1e-12f)
      return 0.5f;

    float vr = var5 / (5.0f * var1);

    // Map VR to H roughly: H = 0.5 + 0.5 * log(VR)
    // VR=1 -> H=0.5. VR=0.5 -> H<0.5. VR=2 -> H>0.5
    return 0.5f + 0.5f * std::log(vr);
  }

  float calculate_volatility_bps() const {
    if (price_history.size() < 10)
      return 10.0f;
    float sum = 0.0f;
    for (float p : price_history)
      sum += p;
    float mean = sum / price_history.size();

    float sq_sum = 0.0f;
    for (float p : price_history)
      sq_sum += (p - mean) * (p - mean);
    float std_dev = std::sqrt(sq_sum / price_history.size());

    return (std_dev / mean) * 10000.0f;
  }

  float solve_ising(float imbalance, float vol_bps) {
    float h = imbalance * 1.0f;      // Weak coupling to prevent saturation
    float T = vol_bps * 0.5f + 1.5f; // Base > J=1.2 ensures paramagnetic phase

    float m = M_prev;
    for (int i = 0; i < 3; ++i)
      m = std::tanh((COUPLING_J * m + h) / T);

    M_prev = m;
    return m;
  }

  Signal generate_signal(const OrderBook &book, double current_time,
                         float futures_price, float lead_hurst) {
    float mid_price = (book.bid_prices[0] + book.ask_prices[0]) / 2.0f;
    float vol_bps = calculate_volatility_bps();

    // --- 1. QUANTUM & HURST METRICS ---
    float energy = calculate_market_energy();
    float hurst = calculate_hurst_proxy();

    last_energy = energy;
    last_hurst = hurst;

    // --- 2. REGIME DETECTION ---
    bool is_ground_state = (energy < 0.3f && hurst < 0.52f);
    bool is_excited_state = (energy > ENERGY_EXCITED || hurst > HURST_TRENDING);

    // LEAD-LAG SHIELD: If BTC (Lead) is trending, ETH (Lag) must go defensive.
    if (!is_leader && lead_hurst > 0.55f) {
      is_ground_state = false;
      is_excited_state = true;
    }

    bool is_tunneling_crash = (energy > ENERGY_CRASH_LIMIT); // CRASH DETECTED

    // --- 3. DYNAMIC ACTIONS ---

    // 1. LEAD-LAG EJECT (Follower Protection)
    // If we are ETH (Follower) and BTC (Leader) is trending hard, get flat.
    if (!is_leader && lead_hurst > 0.55f) {
      if (inventory_btc > 0)
        return Signal::SELL; // Panic Sell
      if (inventory_btc < 0)
        return Signal::BUY; // Panic Buy (Cover)
      return Signal::HOLD;
    }

    // 2. CRASH EJECT (Physics Protection)
    // If Energy is critical, get flat immediately.
    if (is_tunneling_crash) {
      if (inventory_btc > 0)
        return Signal::SELL;
      if (inventory_btc < 0)
        return Signal::BUY;
      return Signal::HOLD;
    }

    // LEVEL 2: EXCITED / TRENDING --> DEFENSE MODE
    // Widen spreads, slow down.
    float width_factor = 0.5f; // Standard Viscous
    double cooldown = 0.5;

    if (is_excited_state) {
      width_factor = 1.0f; // DOUBLE SPREAD (Defense)
      cooldown = 1.0;
    }
    // LEVEL 1: GROUND STATE --> VISCOUS HARVEST
    else if (is_ground_state) {
      width_factor = 1.2f; // Keep spread wide — "Fishing Lines" further out
      cooldown = 0.5;      // Slow down firing
    }

    if (current_time - last_action_time < cooldown)
      return Signal::HOLD;

    // --- 4. SPREAD & PRICING (always compute M/Fair for metrics) ---
    float spread_abs = book.ask_prices[0] - book.bid_prices[0];
    float spread_bps = spread_abs / mid_price;

    float bid_qty = book.bid_sizes[0] + book.bid_sizes[1];
    float ask_qty = book.ask_sizes[0] + book.ask_sizes[1];
    float imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-5f);

    float M = solve_ising(imbalance, vol_bps);

    // HJB Inventory Skew
    float inventory_skew = inventory_btc * RISK_AVERSION * (vol_bps / 100.0f);
    float book_skew = imbalance * (spread_abs * 0.4f);

    // USDT-USD Local Lead Alpha (Z-Score Normalized):
    float usdt_lead_skew = 0.0f;
    if (futures_price > 0.0f && mid_price > 0.0f) {
      float raw_spread = futures_price - mid_price;

      if (!spread_initialized) {
        spread_ema = raw_spread;
        spread_initialized = true;
      } else {
        spread_ema = SPREAD_EMA_ALPHA * raw_spread +
                     (1.0f - SPREAD_EMA_ALPHA) * spread_ema;
      }

      float raw_signal = raw_spread - spread_ema;
      float max_skew = spread_abs * 1.5f;
      raw_signal = std::max(-max_skew, std::min(max_skew, raw_signal));
      usdt_lead_skew = raw_signal * 0.3f;
    }

    float fair_value = mid_price + book_skew - inventory_skew + usdt_lead_skew;

    if (std::abs(M) > 0.5f) {
      fair_value += M * ISING_ALPHA * (mid_price * 0.0001f);
    }

    last_fair_value = fair_value;

    // Spread filter: gate execution only (M/Fair already computed above)
    if (spread_bps < MIN_SPREAD_BPS || spread_bps > MAX_SPREAD_BPS)
      return Signal::HOLD;

    // --- 6. TARGET QUOTES ---
    float premium_buffer = mid_price * 0.0002f; // Fixed 2bp premium buffer
    float my_bid = fair_value - (spread_abs * width_factor) - premium_buffer;
    float my_ask = fair_value + (spread_abs * width_factor) + premium_buffer;

    // --- 7. EXECUTION LOGIC ---

    // CRASH PROTECTION (Absolute Check)
    if (vol_bps > 3.0f && M < -CRITICAL_M) {
      if (my_ask < book.ask_prices[0]) {
        last_action_time = current_time;
        return Signal::SELL;
      }
      return Signal::HOLD;
    }
    if (vol_bps > 3.0f && M > CRITICAL_M) {
      if (my_bid > book.bid_prices[0]) {
        last_action_time = current_time;
        return Signal::BUY;
      }
      return Signal::HOLD;
    }

    // Standard Making
    bool can_buy = std::abs(inventory_btc) < MAX_INVENTORY;
    bool can_sell =
        inventory_btc > 0 || std::abs(inventory_btc) < MAX_INVENTORY;

    if (inventory_btc < 0 && can_buy && my_bid >= book.bid_prices[0]) {
      last_action_time = current_time;
      return Signal::BUY;
    }
    if (inventory_btc > 0 && can_sell && my_ask <= book.ask_prices[0]) {
      last_action_time = current_time;
      return Signal::SELL;
    }

    if (can_buy && my_bid >= book.bid_prices[0]) {
      last_action_time = current_time;
      return Signal::BUY;
    }
    if (can_sell && my_ask <= book.ask_prices[0]) {
      last_action_time = current_time;
      return Signal::SELL;
    }

    return Signal::HOLD;
  }

  void update_inventory(float qty) { inventory_btc += qty; }
  void set_leader(bool leader) { is_leader = leader; }
  void record_trade() {}

  struct Metrics {
    float M;
    float q;
    float fair;
    float vol;
    size_t n;
    float z;
  };
  Metrics get_metrics(const OrderBook &) const {
    return {M_prev, inventory_btc, last_fair_value, last_energy, 0, last_hurst};
  }
};

#endif
