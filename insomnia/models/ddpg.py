import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from insomnia.explores import GaussianActionNoise
from insomnia.replay_buffers import ReplayBuffer

from PIL import Image


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        """
        :param beta: learning rate
        :param input_dims: input dimension for model
        :param fc1_dims: finput dimension for first fully connected layer
        :param n_actions: number of actions therefore output of model
        :param name: use name when save this model
        :param chkpt_dir: to save directory
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        self.fc_input = 3136

        self.conv1 = nn.Conv2d(3, 32, kernel_size=12, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(self.fc_input, 1280)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn4 = nn.LayerNorm(1280)

        self.fc2 = nn.Linear(1280, self.fc1_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn5 = nn.LayerNorm(self.fc1_dims)

        self.action_value = nn.Linear(self.n_actions, fc1_dims)

        f3 = 0.003
        self.q = nn.Linear(self.fc1_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=1e-2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.conv1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.conv2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.conv3(state_value)
        state_value = self.bn3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc1(state_value.view(-1, self.fc_input))
        state_value = self.bn4(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn5(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file), strict=False)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        """
        :param alpha: learnign rate
        :param input_dims: input dimension for model
        :param fc1_dims: input dimension for first fully connected layer
        :param n_actions: number of actions therefore output of model
        :param name: the name of when save checkpoint file
        :param chkpt_dir: the directory when save checkpoint file
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        self.fc_input = 3136

        self.conv1 = nn.Conv2d(3, 32, kernel_size=12, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(self.fc_input, 1280)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn4 = nn.LayerNorm(1280)

        self.fc2 = nn.Linear(1280, self.fc1_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn5 = nn.LayerNorm(self.fc1_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc1_dims, self.n_actions)
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.to(self.device)

    def forward(self, state):
        x = self.conv1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc1(x.view(-1, self.fc_input))
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file), strict=False)


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2,
                 max_size=100000, layer1_size=300, batch_size=64):
        """

        :param alpha: learning rate for actor network
        :param beta:  learning rate for cirtic network
        :param input_dims: input dimension for model
        :param tau:  describe later
        :param gamma: discount coefficient
        :param n_actions: number of actions therefore output og model
        :param max_size: the maximum size of replay buffer
        :param layer1_size: input dimension for first fully connected layer
        :param batch_size: batch size to input model
        """
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, n_actions=n_actions, name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, n_actions=n_actions, name='TargetActor')

        self.critic = CriticNetwork(beta, input_dims, layer1_size, n_actions=n_actions, name='Critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, n_actions=n_actions, name='TargetCritic')

        # self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.noise = GaussianActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)  # how often update target network

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def choose_action(self, observation):
        self.actor.eval()
        observation = self.normalize_frame(observation)  # normalize
        mu = self.actor.forward(observation).to(self.actor.device, dtype=torch.float)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, current_step, writer, episode):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.uint8).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        new_state = self.normalize_frame(new_state)
        state = self.normalize_frame(state)
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        critic_loss = F.mse_loss(target, critic_value)
        writer.add_scalar("Loss/Critic", critic_loss, episode)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        writer.add_scalar("Loss/Actor", actor_loss, episode)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        if current_step % 100 == 0:
            self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict, strict=False)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict, strict=False)

    def test_action(self, observation):
        self.actor.eval()
        observation = self.normalize_frame(observation)  # normalize
        mu = self.actor.forward(observation)
        return mu.cpu().detach().numpy()

    def normalize_frame(self, frames):
        return frames.to(torch.float) / 255

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.target_actor.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.target_actor.load_checkpoint()
