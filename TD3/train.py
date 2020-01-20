import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from models import Actor, Critic, ActorCritic
from replay_buffer import ReplayBuffer


def td3(steps_per_epoch=5000, epochs=100, gamma=0.99, polyak=0.995, pi_lr=1e-3,
        q_lr=1e-3, batch_size=100, start_steps=10000, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, max_ep_len=1000, policy_delay=2):
    replay_buffer = ReplayBuffer(10000, [3], 1)

    env = gym.make('Pendulum-v0')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor_critic = ActorCritic()

    critic1_optimizer = optim.Adam(actor_critic.critic1.parameters(), q_lr)
    critic2_optimizer = optim.Adam(actor_critic.critic2.parameters(), q_lr)
    actor_optimizer = optim.Adam(actor_critic.actor.parameters(), pi_lr)

    obs, ret, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    for t in range(total_steps):
        # if t > 50000:
        if t > 0:
            env.render()
        if t > start_steps:
            obs_tens = torch.from_numpy(obs).float().reshape(1, -1)
            act = actor_critic.get_action(obs_tens, act_noise).detach().numpy().reshape(-1)
        else:
            act = env.action_space.sample()

        obs2, ret, done, _ = env.step(act)

        ep_ret += ret
        ep_len += 1

        done = False if ep_len==max_ep_len else done

        replay_buffer.store_transition(obs, act, ret, obs2, done)

        obs = obs2

        if done or (ep_len == max_ep_len):
            for j in range(ep_len):
                obs1_tens, obs2_tens, acts_tens, rews_tens, done_tens = replay_buffer.sample_buffer(batch_size)
                # compute Q(s, a)
                q1, q2 = actor_critic.q_function(obs1_tens, action=acts_tens)
                # compute r + gamma * (1 - d) * Q(s', mu_targ(s'))
                pi_targ = actor_critic.get_target_action(obs2_tens, target_noise, noise_clip)
                q_targ = actor_critic.compute_target(obs2_tens, pi_targ, gamma, rews_tens, done_tens)
                # compute (Q(s,a) - y(r, s', d))^2
                q_loss = (q1-q_targ).pow(2).mean() + (q2 - q_targ).pow(2).mean()

                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                q_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.step()

                if j % policy_delay == 0:
                    # compute Q(s, mu(s))
                    actor_loss, _ = actor_critic.q_function(obs1_tens, detach=False)
                    actor_loss = -actor_loss.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # compute tau * tar_p + (1 - tau) * main_p
                    actor_critic.update_target(polyak)

            obs, ret, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0


if __name__ == '__main__':
    td3()


