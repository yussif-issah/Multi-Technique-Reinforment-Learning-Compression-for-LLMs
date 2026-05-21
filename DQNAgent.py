from Qnetwork import Qnetwork
from ReplayBuffer import ReplayBuffer
import torch
import random
import numpy as np
import torch.nn as nn

class DQNAgent:
    def __init__(
            self,
            obs_shape: int,
            num_actions: int,
            lr = 1e-3,
            gamma = 0.99,
            epsilon_start = 1.0,
            epsilon_end = 0.05,
            epsilon_decay = 0.995,
            batch_size = 32,
            min_experiences = 1000,
            target_update_freq = 1000

    ):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_experiences = min_experiences
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Networks
        self.q_network = Qnetwork(obs_shape, num_actions)
        self.target_network = Qnetwork(obs_shape, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer()
    #agents acts in the environment and stores the experience in the replay buffer
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax().item()
    #store  the experience in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    #agent learns from the experience stored in the replay buffer
    # DQNAgent.py — fixed learn()
    def learn(self):
        if len(self.buffer) < self.min_experiences:
            return

        self.steps += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # current Q values — already tensors from buffer
        predictions = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions    = self.q_network(next_states).argmax(1, keepdim=True)
            target_q_next   = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            targets         = rewards + self.gamma * target_q_next * (1 - dones)

        loss = nn.MSELoss()(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # soft update instead of hard update
        tau = 0.005
        for tp, op in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

        return loss.item()

        

