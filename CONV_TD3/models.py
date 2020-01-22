import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_channel, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=4, stride=2)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        #self.bn3 = nn.BatchNorm2d(64)

        self.fc_input = 10816
        self.fc1 = nn.Linear(self.fc_input, 800)
        #self.bn4 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 400)
        #self.bn5 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, action_dim)

    def forward(self, obs):
        #x = F.relu(self.bn1(self.conv1(obs)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.bn4(self.fc1(x)))
        #x = F.relu(self.bn5(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_channel, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=4, stride=2)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        #self.bn3 = nn.BatchNorm2d(64)

        # convの後に全結合層を挟んでからconcatするか、挟まずにconcatするか

        self.fc_input = 10816
        self.fc1 = nn.Linear(self.fc_input + action_dim, 800)
        #self.bn4 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 400)
        #self.bn5 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, action_dim)

        self.fc4 = nn.Linear(self.fc_input+action_dim, 800)
        #self.bn6 = nn.BatchNorm1d(800)
        self.fc5 = nn.Linear(800, 400)
        #self.bn7 = nn.BatchNorm1d(400)
        self.fc6 = nn.Linear(400, action_dim)

    def forward(self, obs, action):
        #x = F.relu(self.bn1(self.conv1(obs)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.fc_input)
        x = torch.cat([x, action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        #x1 = F.relu(self.bn4(self.fc1(x)))
        #x1 = F.relu(self.bn5(self.fc2(x1)))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        #x2 = F.relu(self.bn6(self.fc4(x)))
        #x2 = F.relu(self.bn7(self.fc5(x2)))
        x2 = self.fc6(x2)

        return x1, x2

    def q1(self, obs, action):
        #x = F.relu(self.bn1(self.conv1(obs)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.fc_input)
        x = torch.cat([x, action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        #x1 = F.relu(self.bn4(self.fc1(x)))
        #x1 = F.relu(self.bn5(self.fc2(x1)))
        x1 = self.fc3(x1)

        return x1
