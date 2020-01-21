import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from models import Actor, Critic, ActorCritic
from replay_buffer import ReplayBuffer
from Agent import TD3
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse


def td3(steps_per_epoch=5000, epochs=1000, gamma=0.99, polyak=0.995, pi_lr=1e-3,
        q_lr=1e-3, batch_size=100, start_steps=50000, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, max_ep_len=1000, policy_delay=2):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            act = actor_critic.get_action(obs_tens.to(device), act_noise).detach().cpu().numpy().reshape(-1)
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
                q1, q2 = actor_critic.q_function(obs1_tens.to(device), action=acts_tens.to(device))
                # compute r + gamma * (1 - d) * Q(s', mu_targ(s'))
                pi_targ = actor_critic.get_target_action(obs2_tens.to(device), target_noise, noise_clip)
                q_targ = actor_critic.compute_target(obs2_tens.to(device), pi_targ.to(device), gamma, rews_tens, done_tens)
                # compute (Q(s,a) - y(r, s', d))^2
                q_loss = (q1.cpu()-q_targ.cpu()).pow(2).mean() + (q2.cpu() - q_targ.cpu()).pow(2).mean()

                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                q_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.step()

                if j % policy_delay == 0:
                    # compute Q(s, mu(s))
                    actor_loss, _ = actor_critic.q_function(obs1_tens.to(device), detach=False)
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


def main(args):
    file_name = 'td3_lunalander'

    writer = SummaryWriter(log_dir="logs/{}_{}".format(file_name, 'numeric'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('LunarLanderContinuous-v2')

    max_action = float(env.action_space.high[0])    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(int(1e6), [state_dim], action_dim)

    policy = TD3(env, state_dim, action_dim, max_action)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            obs_tens = torch.from_numpy(state).float().reshape(1, -1).to(device)
            # action = np.clip(policy.select_action(obs_tens) + np.random.normal(0, max_action * args.expl_noise, size=3), -max_action, max_action)
            action = np.clip(policy.select_action(obs_tens.to(device)) + np.random.normal(0, max_action * args.expl_noise), -max_action, max_action)
        
        if t > int(6e5):
            env.render()
        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.store_transition(state, action, reward, next_state, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalar("reward", episode_reward, episode_num+1)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t % 100 == 0:
            policy.save(file_name)


if __name__ == '__main__':
    # td3()
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    main(args)


