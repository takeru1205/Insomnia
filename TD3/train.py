import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from models import Actor, Critic, ActorCritic
from replay_buffer import ReplayBuffer
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def td3(steps_per_epoch=5000, epochs=1000, gamma=0.99, polyak=0.995, pi_lr=1e-3,
        q_lr=1e-3, batch_size=100, start_steps=50000, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, max_ep_len=1000, policy_delay=2):
    replay_buffer = ReplayBuffer(int(1e6), [3], 1)

    env = gym.make('Pendulum-v0')

    writer = SummaryWriter(log_dir="logs/numeric")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor_critic = ActorCritic()

    critic1_optimizer = optim.Adam(actor_critic.critic1.parameters(), q_lr)
    critic2_optimizer = optim.Adam(actor_critic.critic2.parameters(), q_lr)
    actor_optimizer = optim.Adam(actor_critic.actor.parameters(), pi_lr)

    obs, ret, done, ep_ret, ep_len, cum_reward = env.reset(), 0, False, 0, 0, []
    total_steps = steps_per_epoch * epochs

    epoch = 0

    for t in range(total_steps):
        # if t > 50000:
        if t > 0:
            env.render()
        if t > start_steps:
            obs_tens = torch.from_numpy(obs).float().reshape(1, -1)
            act = actor_critic.get_action(obs_tens.cuda(), act_noise).detach().cpu().numpy().reshape(-1)
        else:
            act = env.action_space.sample()

        obs2, ret, done, _ = env.step(act)
        cum_reward.append(ret)

        ep_ret += ret
        ep_len += 1

        done = False if ep_len==max_ep_len else done

        replay_buffer.store_transition(obs, act, ret, obs2, done)

        obs = obs2

        if done or (ep_len == max_ep_len):
            epoch += 1
            for j in tqdm(range(ep_len)):
                obs1_tens, obs2_tens, acts_tens, rews_tens, done_tens = replay_buffer.sample_buffer(batch_size)
                # compute Q(s, a)
                q1, q2 = actor_critic.q_function(obs1_tens.cuda(), action=acts_tens.cuda())
                # compute r + gamma * (1 - d) * Q(s', mu_targ(s'))
                pi_targ = actor_critic.get_target_action(obs2_tens.cuda(), target_noise, noise_clip)
                q_targ = actor_critic.compute_target(obs2_tens.cuda(), pi_targ.cuda(), gamma, rews_tens, done_tens)
                # compute (Q(s,a) - y(r, s', d))^2
                q_loss = (q1.cpu()-q_targ.cpu()).pow(2).mean() + (q2.cpu() - q_targ.cpu()).pow(2).mean()

                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                q_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.step()

                if j % policy_delay == 0:
                    # compute Q(s, mu(s))
                    actor_loss, _ = actor_critic.q_function(obs1_tens.cuda(), detach=False)
                    actor_loss = -actor_loss.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # compute tau * tar_p + (1 - tau) * main_p
                    actor_critic.update_target(polyak)
            print('[{}] epoch is : {}, cumulative reward : {}'.format(datetime.now(), epoch, np.sum(np.array(cum_reward))))
            writer.add_scalar("reward/mean", np.mean(np.array(cum_reward)), epoch)
            writer.add_scalar("reward/sum", np.sum(np.array(cum_reward)), epoch)
            obs, ret, done, ep_ret, ep_len, cum_reward = env.reset(), 0, False, 0, 0, []
        
        if epoch % 100 == 0:
            actor_critic.save_weights(epoch)

if __name__ == '__main__':
    td3()


