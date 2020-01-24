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
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[2])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[1])))
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

        self.conv4 = nn.Conv2d(state_dim[0], 16, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[2])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[1])))
        linear_input_size = convw * convh * 32
        self.out1 = nn.Linear(linear_input_size + action_dim, 1)

        self.out2 = nn.Linear(linear_input_size + action_dim, 1)

    def forward(self, obs, action):
        x1 = F.relu(self.bn1(self.conv1(obs)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))

        x1 = x1.view(x1.size(0), -1)
        x1 = torch.cat([x1, action], dim=1)

        x1 = self.out1(x1)

        x2 = F.relu(self.bn4(self.conv4(obs)))
        x2 = F.relu(self.bn5(self.conv5(x2)))
        x2 = F.relu(self.bn6(self.conv6(x2)))

        x2 = x2.view(x2.size(0), -1)
        x2 = torch.cat([x2, action], dim=1)

        x2 = self.out2(x2)

        return x1, x2

    def q1(self, obs, action):
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)

        x1 = self.out1(x)

        return x1
