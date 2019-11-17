import sys

sys.path.append('../')
from insomnia.models import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='tmp/ddpg/logs')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[3,100,150], tau=0.001, env=env,
              batch_size=64, layer1_size=300, n_actions=2)

# agent.load_models()
np.random.seed(0)

score_history = []
for i in range(5000):
    env.reset()
    state = agent.get_screen()
    new_state = agent.get_screen()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(state)
        _, reward, done, info = env.step(act[0])
        state = new_state
        new_state = agent.get_screen()

        new_state -= state # from PyTorch DQN tutorial

        agent.remember(state, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        # env.render()
    score_history.append(score)
    writer.add_scalar("reward", score, i)
    writer.add_scalar("avg reward", np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plt.plot(score_history, label='learning-curve')
plt.savefig(filename)
plt.show()
