from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from ..networks import simple_net
from ..replay_buffers import replay_buffer


class DQN:
    def __init__(self, env, net=None, buffer=None, gamma=0.95, criterion=None, optimizer=None, cuda=False):
        self.env = env
        self.policy_net = simple_net.SimpleNet(env.observation_space.shape[0],
                                               env.action_space.n) if net is None else net
        self.target_net = deepcopy(self.policy_net)
        self.buffer = replay_buffer.ReplayBuffer([env.observation_space.shape[0]], 1, cuda) if buffer is None else buffer
        self.gamma = gamma
        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001) if optimizer is None else optimizer
        self.cuda = cuda
        if self.cuda:
            self.policy_net.to('cuda')
            self.target_net.to('cuda')

    def decide_action(self, state, episode):
        epsilon = 0.3 - 0.0001 * episode

        if epsilon <= random.random():
            with torch.no_grad():
                if self.cuda:
                    return self.policy_net(state.to('cuda', torch.float)).max(0)[1].item()

                return self.policy_net(state.to(torch.float)).max(0)[1].item()
        else:
            return self.env.action_space.sample()

    def target_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def model_update(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return

        states, states_, actions, rewards, terminals = self.buffer.sample_buffer(batch_size)
        state_action_values = self.policy_net(states).gather(1, actions.to(torch.long))

        with torch.no_grad():
            next_action_values = self.target_net(states).max(1)[0].detach()
            expected_state_action_values = self.gamma * next_action_values + rewards

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, state, action, reward, state_, done):
        self.buffer.store_transition(state, action, reward, state_, done)

    def print_network(self):
        print(self.policy_net)





