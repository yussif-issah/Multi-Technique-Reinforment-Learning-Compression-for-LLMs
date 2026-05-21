import random
import numpy as np
from collections import deque
import torch
class ReplayBuffer:
    def __init__(self, max_capacity = 10000):
        self.max_capacity = max_capacity
        self.buffer = deque(maxlen=max_capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
        torch.tensor(np.array(states),      dtype=torch.float32),   # (32, 4)
        torch.tensor(np.array(actions),      dtype=torch.long),       # (32,)
        torch.tensor(np.array(rewards),      dtype=torch.float32),    # (32,)
        torch.tensor(np.array(next_states), dtype=torch.float32),    # (32, 4)
        torch.tensor(np.array(dones),   dtype=torch.float32),    # (32,)
    )
    def __len__(self):
        return len(self.buffer)
    