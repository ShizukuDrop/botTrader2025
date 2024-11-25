import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from trading_env import TradingEnv
import psutil
from torch.utils.data import DataLoader, TensorDataset

# Read processed data
df = pd.read_csv('DOGEUSDT_train.csv')

# Create environment
env = TradingEnv(df)

# Check device availability for M2 Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(device)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(device)
        return self.network(x)

# Calculate optimal batch size based on available memory
def get_optimal_batch_size(min_size=32, max_size=512):
    # Get available memory in GB
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    # Scale batch size with available memory
    optimal_size = int(min(max(min_size, available_memory * 32), max_size))
    # Round to nearest power of 2 for better performance
    return 2 ** int(np.log2(optimal_size))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        # Convert to numpy arrays with float32 for M2 optimization
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Convert to numpy arrays first
        states = np.array([s[0] for s in batch], dtype=np.float32)
        actions = np.array([s[1] for s in batch], dtype=np.int64)
        rewards = np.array([s[2] for s in batch], dtype=np.float32)
        next_states = np.array([s[3] for s in batch], dtype=np.float32)
        dones = np.array([s[4] for s in batch], dtype=np.float32)
        
        # Convert to tensors
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(rewards).to(device),
            torch.from_numpy(next_states).to(device),
            torch.from_numpy(dones).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# Helper function for tensor conversion
def to_tensor(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)

def train(env, episodes=1000, batch_size=64, gamma=0.99, 
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
          learning_rate=0.001, target_update=10):
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer()
    epsilon = epsilon_start
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() > epsilon:
                with torch.no_grad():
                    # Convert to numpy array first, then to tensor
                    state_array = np.array([state], dtype=np.float32)
                    state_tensor = torch.from_numpy(state_array).to(device)
                    action = policy_net(state_tensor).max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # Training step
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = batch
                
                # Use helper function for all tensor conversions
                states = to_tensor(states, dtype=torch.float32)
                actions = to_tensor(actions, dtype=torch.long)
                rewards = to_tensor(rewards, dtype=torch.float32)
                next_states = to_tensor(next_states, dtype=torch.float32)
                dones = to_tensor(dones, dtype=torch.float32)
                
                # Compute Q values
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q = target_net(next_states).max(1)[0].detach()
                target_q = rewards + gamma * next_q * (1 - dones)
                
                # Update policy network
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Log progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")
            env.render_net_worth()
            
            # Save checkpoint
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'reward': episode_reward
            }, 'trading_model_checkpoint.pth')

if __name__ == "__main__":
    batch_size = get_optimal_batch_size()
    print(f"Using optimal batch size: {batch_size}")
    train(env)