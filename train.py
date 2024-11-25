import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from trading_env import TradingEnv
import psutil
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.multiprocessing as mp
import os

# Set number of threads
num_workers = os.cpu_count()
torch.set_num_threads(num_workers)

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

class ReplayBuffer(Dataset):  # Inherit from Dataset
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def __getitem__(self, idx):
        return self.buffer[idx]
        
    def __len__(self):
        return len(self.buffer)

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

# Helper function for tensor conversion
def to_tensor(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)

def load_checkpoint(model, optimizer, epsilon_start, checkpoint_path='trading_model_checkpoint.pth'):
    """Load checkpoint if exists, otherwise return initial values"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        return start_episode, epsilon
    print("No checkpoint found, starting from scratch")
    return 0, epsilon_start

# Modified training function
def train(env, checkpoint_path='trading_model_checkpoint.pth', episodes=1000, batch_size=256, gamma=0.99,
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
          learning_rate=0.001, target_update=10):
    
    # Enable cuDNN benchmark for optimal performance
    if device.type == 'cuda' or device.type == 'mps':
        torch.backends.cudnn.benchmark = True
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Load checkpoint if exists
    start_episode, epsilon = load_checkpoint(
        policy_net, 
        optimizer, 
        epsilon_start,  # Pass epsilon_start here
        checkpoint_path
    )
    
    # Update target network with loaded weights
    target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Resuming training from episode {start_episode}")
    
    # Use multiple workers for data loading
    memory = ReplayBuffer()
    dataloader = DataLoader(
        memory,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Enable anomaly detection during training
    torch.autograd.set_detect_anomaly(True)
    
    for episode in range(start_episode, episodes):
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
        if episode % 1 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")
            # env.render_net_worth()
            
            # Save checkpoint
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'reward': episode_reward
            }, 'trading_model_checkpoint.pth')

if __name__ == "__main__":
    # batch_size = get_optimal_batch_size()
    batch_size = 1024*32
    print(f"Using optimal batch size: {batch_size}")
    train(env)