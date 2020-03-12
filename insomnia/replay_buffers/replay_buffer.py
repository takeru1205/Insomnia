import numpy as np
import torch


class ReplayBuffer:
    """Experience Replay.

    Experience Replay is just sampling randomly from buffer.

    Attributes:
        max_size (int): maximum size of replay buffer.
        mem_control (int): to point the head of buffer.
        state_memory (torch.Tensor): the buffer of state.
        new_state_memory (torch.Tensor): the buffer of next state.
        action_memory (torch.Tensor): the buffer of action.
        reward_memory (torch.Tensor): the buffer of reward.
        terminal_memory(torch.Tensor): the buffer for done or not.
    """
    def __init__(self, state_dim, act_dim, cuda, max_size=10000):
        """Initial of ReplayBuffer

        Args:
            state_dim (list): the dimension of state
            act_dim (int): the dimension of action
            max_size (int): maximum size of replay buffer
        """
        self.max_size = max_size
        self.cuda = cuda
        self.mem_control = 0
        self.state_memory = torch.zeros((self.max_size, *state_dim), dtype=torch.float)
        self.new_state_memory = torch.zeros((self.max_size, *state_dim), dtype=torch.float)
        self.action_memory = torch.zeros((self.max_size, act_dim), dtype=torch.uint8)
        self.reward_memory = torch.zeros(self.max_size, dtype=torch.float)
        self.terminal_memory = torch.zeros(self.max_size, dtype=torch.uint8)

    def store_transition(self, state, action, reward, state_, done):
        """To store state transition

        Args:
            state (np.ndarray): state by observe environment
            action (np.ndarray): action
            reward (np.ndarray): reward by agent decided action
            state_ (np.ndarray): next state by observe environment
            done (np.ndarray): episode is finished or not
        """
        index = self.mem_control % self.max_size
        self.state_memory[index] = torch.from_numpy(state)
        self.new_state_memory[index] = torch.from_numpy(state_)
        self.action_memory[index] = torch.tensor(action)
        self.reward_memory[index] = torch.from_numpy(np.array([reward]).astype(np.float))
        self.terminal_memory[index] = torch.from_numpy(np.array([1 - done]).astype(np.uint8))
        self.mem_control += 1

    def sample_buffer(self, batch_size):
        """Sampling data randomly from buffer

        Args:
            batch_size (int): number of sampling data

        Returns: :obj:`torch.Tensor` sampling data of the size of batch_size

        """
        mem_size = min(self.mem_control, self.max_size)

        batch = np.random.choice(mem_size, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        if self.cuda:
            return states.to('cuda'), states_.to('cuda'), actions.to('cuda'), rewards.to('cuda'), terminal

        return states, states_, actions, rewards, terminal

    def __len__(self):
        """

        Returns: :obj:`int` length of buffer

        """
        return self.mem_control
