import torch
import numpy as np


class LAP:
    def __init__(self, state_dim, action_dim, device, capacity=1e6, normalize_action=True, max_action=1, prioritized=True):
        # Set the device
        self.device = device

        # Set the replay buffer capacity
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0

        # Set the action normalization factor
        self.do_normalize_action = normalize_action
        self.normalize_action = max_action if normalize_action else 1
        self.max_action = max_action

        # Set the prioritized flag
        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(self.capacity, device=device)
            self.max_priority = 1

        # Initialize the replay buffer
        self.state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action / self.normalize_action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        if self.prioritized:
            self.priority[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.prioritized:
            csum = torch.cumsum(self.priority[:self.size], 0)
            val = torch.rand(size=(batch_size,), device=self.device) * csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.state_buffer[self.ind]).to(self.device)
        actions = torch.FloatTensor(self.action_buffer[self.ind]).to(self.device)
        rewards = torch.FloatTensor(self.reward_buffer[self.ind]).to(self.device)
        next_states = torch.FloatTensor(self.next_state_buffer[self.ind]).to(self.device)
        dones = torch.FloatTensor(self.done_buffer[self.ind]).to(self.device)

        return states, actions, rewards, next_states, dones

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[:self.size].max())
