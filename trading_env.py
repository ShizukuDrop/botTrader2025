# trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100, commission_rate=0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        
        # Action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + technical indicators + account info
        num_features = len(df.columns) + 3  # +3 for balance, holdings, position_value
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        self.net_worth_history = []
        self.trade_history = []  # [(step, price, action)]
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_fees = 0
        self.net_worth_history = [self.initial_balance]
        self.trade_history = []  # Reset trade history
        return self._get_observation()
        
    def get_trades(self):
        """Return trade history"""
        return self.trade_history
    
    def _get_observation(self):
        current_data = self.df.iloc[self.current_step].values
        position_value = self.holdings * current_data[3]  # close price
        
        # Concatenate market data with account info
        obs = np.concatenate([
            current_data,
            [self.balance, self.holdings, position_value]
        ])
        return obs
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute trading action
        done = False
        reward = 0
        
        if action == 1:  # Buy
            max_buyable = self.balance / (current_price * (1 + self.commission_rate))
            if max_buyable > 0:
                self.holdings += max_buyable
                fee = max_buyable * current_price * self.commission_rate
                self.balance -= (max_buyable * current_price + fee)
                self.total_fees += fee
                self.trade_history.append(('buy', current_price, max_buyable))
                
        elif action == 2:  # Sell
            if self.holdings > 0:
                sale_value = self.holdings * current_price
                fee = sale_value * self.commission_rate
                self.balance += (sale_value - fee)
                self.total_fees += fee
                self.trade_history.append(('sell', current_price, self.holdings))
                self.holdings = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.df) - 1:
            done = True
            # Force sell at the end
            if self.holdings > 0:
                final_price = self.df.iloc[-1]['close']
                sale_value = self.holdings * final_price
                fee = sale_value * self.commission_rate
                self.balance += (sale_value - fee)
                self.total_fees += fee
                self.holdings = 0
        
        # Calculate reward
        net_worth = self.balance + (self.holdings * current_price)
        reward = (net_worth - self.initial_balance) - self.total_fees
        
        # Add risk penalty based on holdings
        position_value = self.holdings * current_price
        risk_penalty = 0.001 * position_value  # 0.1% risk penalty on position size
        reward -= risk_penalty
        
        # Track net worth
        self.net_worth_history.append(net_worth)
        
        if action in [1, 2]:  # Buy or Sell
            self.trade_history.append((self.current_step, current_price, action))
        
        return self._get_observation(), reward, done, {}
    
    def render_net_worth(self):
        plt.figure(figsize=(15, 7))
        plt.plot(self.net_worth_history, label='Net Worth', color='blue')
        plt.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
        
        # Plot trade points
        for step, price, action in self.trade_history:
            if action == 1:  # Buy
                plt.scatter(step, self.net_worth_history[step], color='green', marker='^', s=100)
            elif action == 2:  # Sell
                plt.scatter(step, self.net_worth_history[step], color='red', marker='v', s=100)
        
        plt.title('Trading Bot Net Worth Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Net Worth ($)')
        plt.legend()
        plt.grid(True)
        plt.show()