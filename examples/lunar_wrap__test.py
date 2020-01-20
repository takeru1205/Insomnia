import gym
import numpy as np
import os
import sys
sys.path.append('../')
from insomnia.utils import FrameStackWrapper, FrameObsWrapper, ForPytorchWrapper


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('LunarLanderContinuous-v2')
env = FrameObsWrapper(env)
env = FrameStackWrapper(env)
env = ForPytorchWrapper(env)

steps = 50
render = True

for i in range(10):
    obs = env.reset()
    for i in range(steps):
        if render:
            env.render()
        obs, reward, done, info = env.step([0,0])

