import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from models import Actor, Critic
from replay_buffer import ReplayBuffer
from Agent import TD3
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from frame_stack import GrayFrameObsWrapper, FrameConvertWrapper, GrayFrameStackWrapper, ForPytorchWrapper 


def main(args):
    file_name = 'td3_pedulum_10fps_3framestack_1fc'

    writer = SummaryWriter(log_dir="logs/{}_{}".format(file_name, 'conv'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('Pendulum-v0')
    env = GrayFrameObsWrapper(env, 64, 64)
    env = FrameConvertWrapper(env)
    env = GrayFrameStackWrapper(env, _n_stack_frames=3)
    env = ForPytorchWrapper(env)
    env._max_episode_steps = 200
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    max_action = float(env.action_space.high[0])
    state_dim = [3, 64, 64]
    action_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(int(1e6), state_dim, action_dim)

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
            # action = np.clip(policy.select_action(obs_tens) + np.random.normal(0, max_action * args.expl_noise, size=3), -max_action, max_action)
            # action = np.clip(policy.select_action(obs_tens.to(device)) + np.random.normal(0, max_action * args.expl_noise), -max_action, max_action)
            action = policy.select_action(state.to(device))
        
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

        # if done:
        if env._max_episode_steps <= episode_timesteps:
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
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=4e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    main(args)


