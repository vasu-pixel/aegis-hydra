"""
aegis_hydra.governor.adaptive — Meta-Learning Logic (Phase 9 Concept)

"It reacts, but it does not remember."

This module defines the 'Adaptive Governor' which bridges the gap between
Pure Physics (Reactive) and AI (Reinforcement Learning).

Logic:
    The Physics Engine (Ising Grid) is the "Body" — it reacts to immediate stimuli.
    The Adaptive Governor is the "Subconscious" — it remembers pain (Loss) and 
    adjusts the body's sensitivity (Coupling J / Temperature T) to avoid repeating mistakes.

Usage:
    governor = AdaptiveGovernor()
    governor.update_learning(trade_pnl)
    J, T = governor.get_physics_params()
"""

class AdaptiveGovernor:
    def __init__(self):
        self.J_coupling = 1.0  # Starting peer pressure
        self.temperature = 2.27 # Critical point
        self.recent_pnl = []
        self.history_len = 5

    def update_learning(self, trade_result: float):
        """
        Learns from the last trade.
        """
        self.recent_pnl.append(trade_result)
        if len(self.recent_pnl) > self.history_len:
            self.recent_pnl.pop(0)
        
        # 1. If we are losing money (Negative PnL)
        # "Pain Avoidance"
        if sum(self.recent_pnl) < 0:
            print("[LEARNING] Swarm is too aggressive (Loss Detected). Increasing caution (J).")
            # Increase Coupling (Make agents harder to convince)
            # Higher J = More "Groupthink" = Less flipping on noise
            self.J_coupling *= 1.05 
            
        # 2. If we are missing trades (Flat PnL in moving market)
        # "Boredom / Opportunity Seeking" (Logic to be refined)
        # elif sum(self.recent_pnl) == 0 and market_volatility > high:
        #     print("Swarm is too sleepy. Waking them up.")
        #     # Decrease Coupling (Make agents more reactive)
        #     self.J_coupling *= 0.95

    def get_physics_params(self):
        return self.J_coupling, self.temperature
