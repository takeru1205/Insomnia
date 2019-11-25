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

writer = SummaryWriter(log_dir='tmp/ddpg/logs/pendulum-conv3-fc2-4frame_stack-gaussian_noise')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')
env = FrameObsWrapper(env)
env = FrameStackWrapper(env)
env = ForPytorchWrapper(env)

state = env.reset()

agent = Agent(alpha=0.0005, beta=0.0003, input_dims=[12, 50, 75], tau=0.001,
              batch_size=16, layer1_size=300, n_actions=1)

# agent.load_models()
np.random.seed(0)

max_step = 400

score_history = []
i = 0
while True:
    state = env.reset()
    new_state = deepcopy(state)
    done = False
    score = 0
    current_step = 0
    while not done:
        act = agent.choose_action(new_state.to(agent.device))
        state = new_state
        new_state, reward, done, _ = env.step(act)
        # state = new_state
        # new_state = agent.get_screen()

        # new_state -= state # from PyTorch DQN tutorial

        agent.remember(state, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        current_step += 1
        if current_step >= max_step:
            break
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
