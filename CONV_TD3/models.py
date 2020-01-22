import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[2])))
        linear_input_size = convw * convh * 32
        self.out = nn.Linear(linear_input_size, action_dim)

    def forward(self, obs):
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.out(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[2])))
        linear_input_size = convw * convh * 32
        self.out1 = nn.Linear(linear_input_size + action_dim, action_dim)

        self.out2 = nn.Linear(linear_input_size + action_dim, action_dim)

    def forward(self, obs, action):
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)

        x1 = self.out1(x)

        x2 = self.out2(x)

        return x1, x2

    def q1(self, obs, action):
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)

        x1 = self.out1(x)

        return x1
