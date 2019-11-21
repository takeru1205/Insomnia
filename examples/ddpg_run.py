import sys

sys.path.append('../')
from insomnia.models import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from insomnia.utils import FrameStackWrapper, FrameObsWrapper, ForPytorchWrapper
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

writer = SummaryWriter(log_dir='tmp/ddpg/logs/conv3-fc2-10000epochs')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('LunarLanderContinuous-v2')
env = FrameObsWrapper(env)
env = FrameStackWrapper(env)
env = ForPytorchWrapper(env)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[12,100,150], tau=0.001, env=env,
              batch_size=64, layer1_size=300, n_actions=2)

# agent.load_models()
np.random.seed(0)

score_history = []
i = 0
while True:
    state = env.reset()
    print(state.shape)
    new_state = deepcopy(state)
    done = False
    score = 0
    while not done:
        act = agent.choose_action(state.to(agent.device))
        print(act)
        state = new_state
        new_state, reward, done, info = env.step(act[0])
        # state = new_state
        # new_state = agent.get_screen()

        # new_state -= state # from PyTorch DQN tutorial

        agent.remember(state, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        # env.render()
    score_history.append(score)
    score_mean = np.mean(score_history[-100:])
    writer.add_scalar("score/reward", score, i)
    writer.add_scalar("score/avg-reward", score_mean, i)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % score_mean)
    if score_mean >= 50:
        print('============================================')
        print('========= This task completed !! ===========')
        print('============================================')
        agent.save_models()
        break
    i += 1

writer.close()
filename = 'LunarLander-alpha000025-beta00025-conv3-fc2.png'
plt.plot(score_history, label='learning-curve')
plt.savefig(filename)
plt.show()
