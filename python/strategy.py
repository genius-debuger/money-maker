"""
Avellaneda-Stoikov Market Making Strategy extended with PPO
Manages inventory and quotes based on LiT predictions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class StrategyState:
    """Current strategy state"""
    inventory: float  # q(t)
    mid_price: float  # s(t)
    volatility: float  # σ
    time_to_horizon: float  # T - t
    lit_prediction: float  # P_up - P_down from LiT
    confidence: float  # Confidence in prediction


@dataclass
class OrderQuote:
    """Order quote from strategy"""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float


class PPOPolicy(nn.Module):
    """
    PPO Policy Network
    Outputs dynamic risk aversion parameter γ based on market state
    """
    
    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        Returns: (B, 1) - risk aversion parameter γ (scaled to reasonable range)
        """
        gamma_raw = self.network(state)
        # Scale to [0.01, 10.0] for practical risk aversion
        gamma = 0.01 + gamma_raw * 9.99
        return gamma
    
    def get_gamma(self, state: StrategyState) -> float:
        """Get gamma for a single state"""
        state_tensor = torch.tensor([
            state.inventory,
            state.mid_price,
            state.volatility,
            state.time_to_horizon,
            state.lit_prediction,
            state.confidence,
        ], dtype=torch.float32).unsqueeze(0)
        
        with torch.inference_mode():
            gamma = self.forward(state_tensor).item()
        
        return gamma


class AvellanedaStoikovStrategy:
    """
    Avellaneda-Stoikov Market Making Strategy
    Extended with PPO for dynamic risk aversion
    """
    
    def __init__(
        self,
        initial_gamma: float = 0.1,
        alpha: float = 0.1,  # Inventory penalty in reward
        use_ppo: bool = True,
    ):
        self.gamma_base = initial_gamma
        self.alpha = alpha
        self.use_ppo = use_ppo
        
        if use_ppo:
            self.ppo_policy = PPOPolicy()
            # In production, this would load trained weights
            # self.ppo_policy.load_state_dict(torch.load('ppo_policy.pt'))
        else:
            self.ppo_policy = None
    
    def compute_reservation_price(
        self,
        state: StrategyState,
        gamma: Optional[float] = None,
    ) -> float:
        """
        Compute reservation price: r(t) = s(t) - q(t) * γ * σ² * (T - t)
        
        Args:
            state: Current strategy state
            gamma: Risk aversion parameter (if None, uses base or PPO)
        
        Returns:
            Reservation price
        """
        if gamma is None:
            if self.use_ppo and self.ppo_policy is not None:
                gamma = self.ppo_policy.get_gamma(state)
            else:
                gamma = self.gamma_base
        
        # Adjust gamma based on LiT prediction
        # If LiT predicts up with high confidence, reduce risk aversion for buys
        if state.lit_prediction > 0.5 and state.confidence > 0.7:
            gamma *= 0.7  # More aggressive on predicted moves
        
        reservation_price = (
            state.mid_price
            - state.inventory * gamma * (state.volatility ** 2) * state.time_to_horizon
        )
        
        return reservation_price
    
    def compute_spread(
        self,
        state: StrategyState,
        gamma: Optional[float] = None,
        kappa: float = 1.5,  # Spread adjustment factor
    ) -> float:
        """
        Compute optimal spread
        
        Args:
            state: Current strategy state
            gamma: Risk aversion parameter
            kappa: Spread adjustment factor
        
        Returns:
            Half-spread (distance from mid to quote)
        """
        if gamma is None:
            if self.use_ppo and self.ppo_policy is not None:
                gamma = self.ppo_policy.get_gamma(state)
            else:
                gamma = self.gamma_base
        
        # Base spread from Avellaneda-Stoikov
        half_spread = (
            gamma * (state.volatility ** 2) * state.time_to_horizon
            + (2 / gamma) * math.log(1 + gamma / kappa)
        )
        
        # Tighten spread if LiT predicts movement (capture momentum)
        if abs(state.lit_prediction) > 0.6 and state.confidence > 0.7:
            half_spread *= 0.8
        
        # Widen spread if inventory is skewed
        inventory_penalty = 1.0 + 0.5 * abs(state.inventory) / 10.0
        half_spread *= inventory_penalty
        
        return half_spread
    
    def compute_quotes(
        self,
        state: StrategyState,
        min_spread: float = 0.0001,  # Minimum 0.01% spread
        max_size: float = 1.0,
    ) -> OrderQuote:
        """
        Compute bid/ask quotes
        
        Args:
            state: Current strategy state
            min_spread: Minimum spread to maintain
            max_size: Maximum order size
        
        Returns:
            OrderQuote with bid/ask prices and sizes
        """
        gamma = None
        if self.use_ppo and self.ppo_policy is not None:
            gamma = self.ppo_policy.get_gamma(state)
        
        reservation_price = self.compute_reservation_price(state, gamma)
        half_spread = max(self.compute_spread(state, gamma), min_spread)
        
        # Compute bid and ask prices
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread
        
        # Size adjustment based on inventory
        # Reduce size on the side we're already long/short
        base_size = max_size
        
        if state.inventory > 0:
            # Long inventory: reduce bid size, increase ask size
            bid_size = base_size * (1.0 - 0.3 * min(state.inventory / 10.0, 1.0))
            ask_size = base_size * (1.0 + 0.2 * min(state.inventory / 10.0, 1.0))
        elif state.inventory < 0:
            # Short inventory: increase bid size, reduce ask size
            bid_size = base_size * (1.0 + 0.2 * min(abs(state.inventory) / 10.0, 1.0))
            ask_size = base_size * (1.0 - 0.3 * min(abs(state.inventory) / 10.0, 1.0))
        else:
            bid_size = base_size
            ask_size = base_size
        
        # Adjust sizes based on LiT prediction
        if state.lit_prediction > 0.3:  # Predict up
            bid_size *= 1.2  # Increase buy interest
            ask_size *= 0.9
        elif state.lit_prediction < -0.3:  # Predict down
            bid_size *= 0.9
            ask_size *= 1.2  # Increase sell interest
        
        return OrderQuote(
            bid_price=max(bid_price, state.mid_price * 0.999),  # Safety bounds
            ask_price=min(ask_price, state.mid_price * 1.001),
            bid_size=max(0.01, min(bid_size, max_size)),
            ask_size=max(0.01, min(ask_size, max_size)),
        )
    
    def compute_reward(
        self,
        realized_pnl: float,
        rebates: float,
        inventory: float,
        trades_executed: int = 1,
    ) -> float:
        """
        Reward function for PPO training
        R = PnL_realized + Rebates - α * (Inventory)²
        
        Args:
            realized_pnl: Realized profit/loss
            rebates: Maker rebates collected
            inventory: Current inventory position
            trades_executed: Number of trades executed
        
        Returns:
            Reward value
        """
        reward = realized_pnl + rebates - self.alpha * (inventory ** 2)
        
        # Penalize holding inventory (force delta neutrality)
        if abs(inventory) > 5.0:
            reward -= 10.0 * (abs(inventory) - 5.0) ** 2
        
        return reward / max(trades_executed, 1)  # Normalize by trades


class InventoryManager:
    """Manages inventory and enforces delta neutrality"""
    
    def __init__(self, target_inventory: float = 0.0, max_inventory: float = 10.0):
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.current_inventory = 0.0
    
    def update_inventory(self, delta: float):
        """Update inventory after trade"""
        self.current_inventory += delta
    
    def get_inventory_skew(self) -> float:
        """Get inventory skew as percentage of max"""
        return (self.current_inventory / self.max_inventory) * 100.0 if self.max_inventory > 0 else 0.0
    
    def should_force_close(self) -> bool:
        """Check if inventory is too skewed"""
        return abs(self.current_inventory) > self.max_inventory * 0.9
    
    def reset(self):
        """Reset inventory to target"""
        self.current_inventory = self.target_inventory


if __name__ == "__main__":
    # Test strategy
    strategy = AvellanedaStoikovStrategy(use_ppo=False)
    
    state = StrategyState(
        inventory=2.5,
        mid_price=50000.0,
        volatility=0.02,  # 2% volatility
        time_to_horizon=1.0,  # 1 hour
        lit_prediction=0.7,  # Predict up
        confidence=0.85,
    )
    
    quotes = strategy.compute_quotes(state)
    print(f"Mid Price: ${state.mid_price:.2f}")
    print(f"Bid: ${quotes.bid_price:.2f} (size: {quotes.bid_size:.2f})")
    print(f"Ask: ${quotes.ask_price:.2f} (size: {quotes.ask_size:.2f})")
    print(f"Spread: ${quotes.ask_price - quotes.bid_price:.2f}")

