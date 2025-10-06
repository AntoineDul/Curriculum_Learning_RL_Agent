import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int = 100000, gamma: float = 0.99, N: int = 1):
        self.capacity = capacity
        self.gamma = gamma
        self.N = N
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=N)
    
    def push(self, obs, action, reward, next_obs, done):
        # Add new transition to n-step buffer
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        # Wait until buffer is full enough for N-step transition
        if len(self.n_step_buffer) < self.N:
            return

        # Compute N-step return from the buffer
        total_return = 0
        done_flag = False
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            total_return += (self.gamma ** idx) * r
            if d:
                done_flag = True
                break

        obs0, action0, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_obs_N, _ = self.n_step_buffer[-1]

        # Store the N-step experience
        self.buffer.append((obs0, action0, total_return, next_obs_N, done_flag))

        # If episode ends, clear buffer (reset for next episode)
        if done:
            self.n_step_buffer.clear()
    
    def sample(self, batch_size: int = 32):
        return random.sample(self.buffer, batch_size)

class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(observation_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,action_size)

        init.uniform_(self.fc1.weight, -0.001,0.001)
        init.uniform_(self.fc2.weight, -0.001,0.001)

        self.model = nn.Sequential(
            self.fc1, 
            nn.Tanh(),
            self.fc2,
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
class DQN:
    def __init__(self,observation_size : int, action_size : int, hidden_size: int = 64, epsilon : float = 1, gamma: float = 0.999, N_steps: int = 1, learning_rate : float = 0.001, train_freq: int = 1, replay_capacity: int = 1000000):
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.N_steps = N_steps
        self.replay_capacity = replay_capacity
        self.train_count = 0
        self.train_freq = train_freq

        self.replay_buffer = ReplayBuffer(capacity=replay_capacity,gamma=gamma, N=N_steps)
        self.Q_network = QNetwork(observation_size,action_size,hidden_size)
        self.target_Q_network = QNetwork(observation_size,action_size,hidden_size)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=learning_rate)
    
    def sample_action(self,state : np.ndarray, training_mode: bool = True, action_mask: np.ndarray = None):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.Q_network(state).cpu().numpy()[0]

        if action_mask is not None:
            q_values = np.where(action_mask, q_values, -np.inf)
        
        #print("State:")
        #print(state)
        #print("Q Values:")
        #print(q_values)

        if training_mode and np.random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.flatnonzero(action_mask)
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(0, self.action_size)
        else:
            max_value = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_value)
            return np.random.choice(best_actions)
    
    def store_transition(self,state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size: int = 32):
        self.train_count += 1

        if self.train_count % self.train_freq != 0:
            return

        if len(self.replay_buffer.buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.Q_network(states).gather(1, actions)

        with torch.no_grad():
            all_q_values = self.Q_network(states)  # shape [batch, num_actions]
            max_q_per_state = all_q_values.max(dim=1)[0]  # shape [batch]
            avg_q = max_q_per_state.mean().item()  # scalar

            next_actions = self.Q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_Q_network(next_states).gather(1, next_actions)
            targets = rewards + (1 - dones) * (self.gamma ** self.N_steps) * next_q_values

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()

        return loss.item(), avg_q
    
    def update_target_network(self, tau=0.005):
         for target_param, local_param in zip(self.target_Q_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def update_epsilon(self, decay_rate: float = 0.992, min_epsilon: float = 0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    #save agent
    def save_agent(self, save_path):
        torch.save({
            'model_state_dict': self.Q_network.state_dict(),
        }, save_path)
        print(f"Agent saved at {save_path}")

    #load agent
    def load_agent(self, load_path):
        checkpoint = torch.load(load_path)
        self.Q_network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Agent loaded from {load_path}")