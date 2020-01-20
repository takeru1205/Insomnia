import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, *input_shape))
        self.new_state_memory = torch.zeros((self.mem_size, *input_shape))
        self.action_memory = torch.zeros((self.mem_size, n_actions))
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = torch.from_numpy(state)
        self.new_state_memory[index] = torch.from_numpy(state_)
        self.action_memory[index] = torch.from_numpy(action)
        self.reward_memory[index] = torch.from_numpy(np.array([reward]).astype(np.float))
        self.terminal_memory[index] = torch.from_numpy(np.array([1 - done]).astype(np.uint8))
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # return states, actions, rewards, states_, terminal
        return states, states_, actions, rewards, terminal
