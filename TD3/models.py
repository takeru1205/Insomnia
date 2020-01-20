import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3+1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, act_limit=2):
        super().__init__()

        self.actor = Actor()
        self.critic1 = Critic()
        self.critic2 = Critic()

        self.act_limit = act_limit

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.actor_target = Actor()
        self.critic1_target = Critic()
        self.critic2_target = Critic()

        self.copy_params()

    def copy_params(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def get_action(self, obs):
        pi = self.act_limit * self.actor(obs)
        pi += self.act_limit * torch.rand_like(pi)
        pi.clamp(max=self.act_limit, min=-self.act_limit)
        return pi.squeeze()

    def get_target_action(self, obs, noise_scale=0.2, clip_param=2):
        pi = self.act_limit * self.actor_target(obs)
        eps = noise_scale * torch.randn_like(pi)
        eps.clamp(max=clip_param, min=-clip_param)
        pi += eps
        pi.clamp(max=self.act_limit, min=-self.act_limit)
        return pi.detach()

    def update_target(self, tau):
        # compute theta_taget = tau * target_p + (1 - tau) * policy_p
        for actor_p, actor_target_p in zip(self.actor.parameters(), self.actor_target.parameters()):
            actor_target_p.data = tau * actor_target_p.data + (1-tau) * actor_p.data

        for critic_p, critic_target_p in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            critic_target_p.data = tau * critic_target_p.data + (1-tau) * critic_p.data

        for critic_p, critic_target_p in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            critic_target_p.data = tau * critic_target_p.data + (1-tau) * critic_p.data

    def compute_target(self, obs, pi, gamma, rewards, done):
        # compute r + gamma * (1 - d) * Q(s', mu_targ(s'))
        q1 = self.critic1_target(obs, pi.reshape(-1, 1))
        q2 = self.critic2_target(obs, pi.reshape(-1, 1))
        q = torch.min(q1, q2)
        return (rewards + gamma * (1-done) * q.squeeze()).detach()

    def q_function(self, obs, detach=True, action=None):
        # compute Q(s, a) or Q(s, mu(s))
        if action is None:
            pi = self.act_limit * self.actor(obs)
        else:
            pi = action
        if detach:
            pi = pi.detach()
        return self.critic1(obs, pi.reshape(-1, 1)).squeeze(),\
               self.critic2(obs, pi.reshape(-1, 1)).squeeze()







