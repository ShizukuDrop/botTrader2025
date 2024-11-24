# evaluate.py
import pandas as pd
import torch
import matplotlib.pyplot as plt
from trading_env import TradingEnv
from train import DQN

# Load processed data and create environment
df = pd.read_csv('DOGEUSDT_test.csv')
env = TradingEnv(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(checkpoint_path):
    # Get input/output dimensions from environment
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # Initialize model
    model = DQN(input_dim, output_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def evaluate_model(model, num_episodes=1):
    total_rewards = []
    trade_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from trained policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).max(1)[1].item()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        trade_history = env.trade_history
        
        print(f"Episode {episode} Reward: {episode_reward:.2f}")
        env.render_net_worth()
        
        # Calculate and display metrics
        total_trades = len(trade_history)
        if total_trades > 0:
            profitable_trades = sum(1 for trade in trade_history if trade[2] > 0)
            win_rate = profitable_trades / total_trades * 100
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Trades: {total_trades}")
            print(f"Final Portfolio Value: ${env.balance:.2f}")
            print(f"Return: {((env.balance - env.initial_balance) / env.initial_balance * 100):.2f}%")

if __name__ == "__main__":
    # Load and evaluate model
    model = load_trained_model('trading_model_checkpoint.pth')
    evaluate_model(model)