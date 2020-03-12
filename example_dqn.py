import random
from collections import deque

import numpy as np
import gym
import torch

from insomnia import models


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make('CartPole-v0')
    dqn = models.dqn.DQN(env, cuda=True)  # if use gpu, cuda = False

    dqn.target_update()
    recent_10_episodes = deque(maxlen=10)
    for episode in range(3000):
        state = env.reset()
        episode_count = 0

        for timestep in range(200):
            action = dqn.decide_action(torch.tensor(state), episode)
            next_state, reward, done, _ = env.step(action)

            terminal = 0
            episode_count += 1

            if done:
                terminal = 1
                if timestep < 195:
                    reward = -1

            dqn.store_transition(state, action, reward, next_state, terminal)
            state = next_state
            dqn.model_update()

            if done:
                recent_10_episodes.append(episode_count+1)
                if episode % 10 == 0:
                    print('epoch:{}, timestep:{}, mean of recent 10 time steps:{}'.format(
                        episode, timestep, sum(list(recent_10_episodes)) / 10))
                break

        if sum(list(recent_10_episodes)) / 10 >= 200:
            print("Train Completed")
            # uncomment when save
            # dqn.save_model("PATH")
            break

        if episode % 10 == 0:
            dqn.target_update()


if __name__ == '__main__':
    main()
