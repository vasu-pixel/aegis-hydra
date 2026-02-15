# Critical Flow Sniper Strategy

**Replaces:** Ising Model (archived)
**Strategy:** Multi-Level Order Flow Imbalance + Hawkes Criticality
**Latency:** Optimized for 8ms total latency

---

## 🎯 **Core Concept**

Instead of predicting price movements, we **detect order flow imbalance** before price moves.

```
OLD: Price moves → React → LATE (lose)
NEW: Order flow builds → Detect → EARLY (win)
```

---

## 📐 **Components**

### **1. Order Book Manager** (`order_book.h/cpp`)
- Tracks 5 levels of bid/ask depth
- Stores previous state for OFI calculation
- Cache-optimized (256 bytes total)

### **2. Hawkes Estimator** (`hawkes_estimator.h`)
- Calculates branching ratio **n** in real-time
- Uses variance/mean method (computational efficiency)
- Filters noise: Only trade when n > 0.6 (market is critical)

**Formula:**
```
n = 1 - sqrt(E[trades] / Var[trades])
n → 0: Random noise (don't trade)
n → 1: Self-reinforcing (trade!)
```

### **3. MLOFI Calculator** (`mlofi_calculator.h`)
- Multi-Level Order Flow Imbalance across 5 depth levels
- Weighted by 1/k^λ (Level 1 strongest, Level 5 weakest)
- Detects "gravity" pulling price before it moves

**Formula:**
```
MLOFI = Σ(OFI_k / k^λ) for k=1..5
OFI_k = (bid flow change) - (ask flow change) at level k
```

### **4. Critical Flow Strategy** (`critical_flow_strategy.h`)
- Combines OFI + Hawkes + Dynamic Thresholds
- Adjusts threshold based on volatility and urgency
- Only generates signals during critical regimes

**Dynamic Threshold:**
```
H = 2×Fee + 3×σ×sqrt(Δ) - 0.0005×n
    ↑ cost   ↑ delay risk  ↑ urgency discount
```

---

## 🏗️ **Architecture**

```
Python (WebSocket) → Binary Pipe → C++ Daemon
                                    ↓
                            Order Book Update
                                    ↓
                            MLOFI Calculation
                                    ↓
                            Hawkes n Estimation
                                    ↓
                            Signal Generation
                                    ↓
                    BUY/SELL/CLOSE → Python → Execution
```

---

## 🔧 **Build**

```bash
cd cpp
make -f Makefile_new clean
make -f Makefile_new
```

**Output:** `critical_flow_daemon`

---

## 📊 **Performance Targets**

| Metric | Target | Why |
|--------|--------|-----|
| **Win Rate** | >55% | Need >52% to beat 0.2% fees |
| **Trades/Hour** | 5-20 | Down from 50+ (90% filtered) |
| **Avg Latency** | <2ms | OFI + Hawkes + Threshold calc |
| **P99 Latency** | <5ms | Must stay sub-10ms |

---

## 📈 **Expected Behavior**

### **Before (Ising Model):**
```
Trades: 500 in 10 hours
Win Rate: 45%
Result: -70% capital loss
Reason: Random signals, no edge
```

### **After (Critical Flow):**
```
Trades: ~50 in 10 hours (90% filtered)
Win Rate: 55-60% (only critical regimes)
Result: +5-10% capital gain
Reason: Order flow has predictive power
```

---

## 🔬 **Academic Basis**

1. **OFI Theory:** Cont, Kukanov, Stoikov (2014)
   *The Price Impact of Order Book Events*

2. **Hawkes Processes:** Filimonov & Sornette (2012)
   *Quantifying reflexivity in financial markets*

3. **Criticality Detection:** Hardiman et al. (2013)
   *Critical reflexivity in financial markets*

4. **Impulse Control:** arXiv:2501.03296 (2026)
   *Optimal stochastic impulse control with delay*

---

## 🎯 **Key Advantage**

**At 8ms latency:**
- ❌ Can't arbitrage against sub-ms HFTs
- ✅ **CAN detect order flow before price moves**
- ✅ **CAN filter noise with Hawkes criticality**
- ✅ **CAN capture edge with dynamic thresholds**

**This strategy is designed FOR your latency profile, not against it!**

---

## 📝 **Next Steps**

1. ✅ Build daemon: `make -f Makefile_new`
2. ⏳ Update Python pipe to send order book data
3. ⏳ Test on paper trading
4. ⏳ Validate win rate > 55%
5. ⏳ Deploy to live trading (if profitable)

---

**Status:** ✨ **Ready to Build**

Run `make -f Makefile_new` to compile the new strategy!
