from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque
import random


class Net(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, state_dim, act_dim, max_size=10000):
        self.max_size = max_size
        self.mem_control = 0
        self.state_memory = torch.zeros((self.max_size, *state_dim), dtype=torch.double)
        self.new_state_memory = torch.zeros((self.max_size, *state_dim), dtype=torch.double)
        self.action_memory = torch.zeros((self.max_size, act_dim))
        self.reward_memory = torch.zeros(self.max_size, dtype=torch.float)
        self.terminal_memory = torch.zeros(self.max_size, dtype=torch.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_control % self.max_size
        self.state_memory[index] = torch.from_numpy(state)
        self.new_state_memory[index] = torch.from_numpy(state_)
        self.action_memory[index] = torch.tensor(action)
        self.reward_memory[index] = torch.from_numpy(np.array([reward]).astype(np.float))
        self.terminal_memory[index] = torch.from_numpy(np.array([1 - done]).astype(np.uint8))
        self.mem_control += 1

    def sample_buffer(self, batch_size):
        mem_size = min(self.mem_control, self.max_size)

        batch = np.random.choice(mem_size, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, states_, actions, rewards, terminal

    def __len__(self):
        return self.mem_control


class Trainer:
    def __init__(self, env, model, buffer, max_epoch=3000, max_timestep=200, rendering=False):
        self.env = env
        self.policy_net = model
        self.policy_net = self.policy_net.to(torch.double)
        self.target_net = deepcopy(self.policy_net)
        self.buffer = buffer
        self.max_epoch = max_epoch
        self.max_timestep = max_timestep
        self.rendering = rendering
        self.gamma = 0.95
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

    def train(self):
        self.target_update()
        recent_10_steps = deque(maxlen=10)
        total_step = 0
        for epoch in range(self.max_epoch):
            state = self.env.reset()

            for timestep in range(self.max_timestep):
                action = self.decide_action(torch.tensor(state), epoch)
                next_state, reward, done, _ = self.env.step(action)

                total_step += 1

                if self.rendering:
                    self.env.render()

                terminal = 0

                if done:
                    terminal = 1
                    if timestep < 195:
                        reward = -1

                    recent_10_steps.append(timestep+1)

                self.buffer.store_transition(state, action, reward, next_state, terminal)
                state = next_state
                self.model_update()

                if done:
                    average_recent_10_rewards = sum(list(recent_10_steps)) / 10
                    if epoch % 10 == 0:
                        print('epoch:{}, timestep:{}, mean of recent 10 time steps:{}'.format(
                            epoch, timestep, average_recent_10_rewards))
                    break

            if epoch % 10 == 0:
                self.target_update()
    
    def model_update(self, batch_size=32):
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
    
    def target_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decide_action(self, state, epoch):

        epsilon = 0.3 - 0.0001 * epoch

        if epsilon <= random.random():
            with torch.no_grad():
                return self.policy_net(state).max(0)[1].item()
        else:
            return self.env.action_space.sample()


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make('CartPole-v0')
    model = Net(env.observation_space.shape[0], env.action_space.n)
    buffer = ReplayBuffer([env.observation_space.shape[0]], 1)
    trainer = Trainer(env, model, buffer)
    trainer.train()


if __name__ == '__main__':
    main()

