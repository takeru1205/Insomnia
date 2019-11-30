import sys

sys.path.append('../')
from insomnia.models import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from insomnia.utils import FrameStackWrapper, FrameObsWrapper, ForPytorchWrapper
from torch.utils.tensorboard import SummaryWriter
import torch
from copy import deepcopy
import logging
import datetime
from PIL import Image
import slackweb
from config import URL

logging.basicConfig(filename='tmp/ddpg/logger.log', level=logging.INFO)

config = 'lunarlaunder-gaussian_noise-128batch-sigma01-not-stop'

logging.info('{}'.format(datetime.datetime.now()))
logging.info('{}'.format(config))

log_dir = 'tmp/ddpg/logs/' + config
writer = SummaryWriter(log_dir=log_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

slack = slackweb.Slack(url=URL)
slack.notify(text='===============================')
slack.notify(text='======= Training Start!! ======')
slack.notify(text='===============================')

env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Pendulum-v0')
env = FrameObsWrapper(env, 100,100)
env = FrameStackWrapper(env)
env = ForPytorchWrapper(env)

state = env.reset()

agent = Agent(alpha=0.0001, beta=0.0001, input_dims=[12, 100, 100], tau=0.001,
              batch_size=128, layer1_size=300, n_actions=2)

# agent.load_models()
np.random.seed(0)

max_step = 500

score_history = []
i = 0
while True:
    state = env.reset()
    new_state = deepcopy(state)
    done = False
    score = 0
    current_step = 0
    logging.info('Episode {}'.format(i))
    while not done:
        act = agent.choose_action(new_state.to(agent.device))
        # print(act[0].shape)
        state = new_state
        new_state, reward, done, _ = env.step(act[0])

        # state = new_state
        # new_state = agent.get_screen()

        # new_state -= state # from PyTorch DQN tutorial
        logging.debug('action : {}'.format(act))
        agent.remember(state, act, reward, new_state, int(done))
        agent.learn(current_step)
        score += reward
        current_step += 1
    score_history.append(score)
    score_mean = np.mean(score_history[-100:])
    writer.add_scalar("score/reward", score, i)
    writer.add_scalar("score/avg_reward", score_mean, i)

    if i % 25 == 0:
        done = False
        agent.save_models()
        test_state = env.reset()
        test_score = 0
        test_steps = 0
        logging.info('Episode {} test play'.format(i))
        while not done:
            test_act = agent.test_action(test_state.to(agent.device))
            logging.info('step {}, action : {}'.format(test_steps, test_act[0]))
            test_state, reward, done, _ = env.step(test_act[0])
            test_score += reward
            test_steps += 1
        print('============================================')
        print('total score : {},   test steps : {}'.format(test_score, test_steps))
        print('============================================')
        writer.add_scalar("test/total-reward", test_score, i)
        if i % 100 == 0:
            slack_text = 'Episode : {} \n Test Score : {} \n Test Steps : {} \n 100Episodes Average Score : {}'.format(i, test_score, test_steps, score_mean)
            slack.notify(text=slack_text)

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % score_mean)
    logging.info('episode : {}, score : {}, 100 avg score : {}'.format(i, score, score_mean))

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
env.close()
