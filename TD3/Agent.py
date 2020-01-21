import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from models import Actor, Critic


class TD3:
    def __init__(self, env, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.env = env
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        action = self.actor(state.to(self.device)).data.cpu().numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1

        states, states_, actions, rewards, terminal = replay_buffer.sample_buffer(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions.to(self.device))
                     * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(states_.to(self.device))
                           + noise).clamp(-self.max_action, self.max_action)

            # compute the target Q value
            target_q1, target_q2 = self.critic_target(states_.to(self.device), next_action.to(self.device))
            target_q = torch.min(target_q1, target_q2)
            # target_q = rewards + terminal * self.gamma + target_q.cpu()
            # target_q = rewards + (terminal.reshape(256, 1) * self.gamma * target_q).detach()
            target_q = rewards + terminal * self.gamma * target_q[:,0].cpu()

        # Get current Q value
        current_q1, current_q2 = self.critic(states.to(self.device), actions.to(self.device))

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1[:,0], target_q.to(self.device)) + F.mse_loss(current_q2[:,0], target_q.to(self.device))

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compote actor loss
            actor_loss = -self.critic.q1(states.to(self.device), self.actor(states.to(self.device))).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))



