from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from ..networks import simple_net
from ..replay_buffers import replay_buffer


class DQN:
    """Deep Q Network.

    Implementation of https://www.nature.com/articles/nature14236.

    Attributes:
        env (gym.env): the environment which agents act fot reinforcement learning.
        policy_net (torch.nn.Module): policy network approximate state value function.
        target_net (torch.nn.Module): temporally goal for policy network. This makes learning stable.
        buffer (insomnia.replay_buffer): replay buffer. it contains how to sampling data.
        gamma (int): discount factor.
        criterion (torch.nn): loss function.
        optimizer (torch.optim): optimization function.
        cuda (bool): whether use GPUs for learning or not.
    """
    def __init__(self, env, net=None, buffer=None, gamma=0.95, criterion=None, optimizer=None, cuda=False):
        """Initial of DQN.

        Args:
            env (gym.env): object of gym environment made by gym.make(ENV NAME).
            net (torch.nn.Module): the policy network of neural network. default network is 3 fully connected network.
            buffer (insomnia.replay_buffer): the replay buffer. default buffer is randomly sampling data.
            gamma (float): set between 0 to 1. default value is 0.95 based on original paper.
            criterion (torch.nn): what loss function use. default function is MeanSquaredError. original is HuberLoss.
            optimizer (torch.optim): what optimization function use.
            cuda (bool): True makes using GPUs for learning.
        """
        self.env = env
        self.policy_net = simple_net.SimpleNet(env.observation_space.shape[0],
                                               env.action_space.n) if net is None else net
        self.target_net = deepcopy(self.policy_net)
        self.buffer = replay_buffer.ReplayBuffer(
            [env.observation_space.shape[0]], 1, cuda) if buffer is None else buffer
        self.gamma = gamma
        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001) if optimizer is None else optimizer
        self.cuda = cuda
        if self.cuda:
            self.policy_net.to('cuda')
            self.target_net.to('cuda')

    def decide_action(self, state, episode):
        """To decide action with ε-greedy

        Args:
            state (torch.Tensor): state of observed from environment.
            episode (int): the number of training episode.

        Returns (np.ndarray): action

        """
        epsilon = 0.3 - 0.0002 * episode

        if epsilon <= random.random():
            with torch.no_grad():
                if self.cuda:
                    return self.policy_net(state.to('cuda', torch.float)).max(0)[1].item()

                return self.policy_net(state.to(torch.float)).max(0)[1].item()
        else:
            return self.env.action_space.sample()

    def target_update(self):
        """Update the target network

          run this method regularly and makes learning stable.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def model_update(self, batch_size=32):
        """update policy network weights

        calculate loss from outputted state action value from policy_net and calculated state action value from
        target_net. the output of target_net is calculated with gamma * output + rewards.

        Args:
            batch_size (int): mini-batch size.

        Returns: None

        """
        if len(self.buffer) < batch_size:
            return

        states, states_, actions, rewards, terminals = self.buffer.sample_buffer(batch_size)
        state_action_values = self.policy_net(states).gather(1, actions.to(torch.long))

        with torch.no_grad():
            next_state_values = self.target_net(states_).max(1)[0].detach()
            expected_state_action_values = self.gamma * next_state_values + rewards

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, state, action, reward, state_, done):
        """To store experience to replay buffer

        Args:
            state (np.ndarray): state of observed from environment.
            action (int): chosen action when this state.
            reward (int): reward from this state transition.
            state_ (np.ndarray): next state from this state transition.
            done (bool): whether this state transition makes episode finished or not.
        """
        self.buffer.store_transition(state, action, reward, state_, done)

    def save_model(self, path):
        """To save the weights of policy_net

        To save weights to path.

        Args:
            path: directory and file name.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """To load the weights for policy_net.

        To load weights from path.

        Args:
            path: directory and file name.
        """
        self.policy_net.load_state_dict(torch.load(path))
