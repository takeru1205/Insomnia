import gym
import numpy as np
import torch
from collections import deque
import torchvision.transforms as T
from PIL import Image
import random
import cv2

class FrameObsWrapper(gym.Wrapper):
    def __init__(self, env, img_width, img_height):
        super().__init__(env)
        self.img_width = img_width
        self.img_height = img_height
    
    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        obs = self.env.render(mode='rgb_array')
        obs = cv2.resize(obs, (self.img_height, self.img_width), interpolation=cv2.INTER_AREA)
        obs = obs.transpose((2, 0, 1))
        obs = torch.Tensor(obs).to(torch.uint8)
        return obs

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        obs = cv2.resize(obs, (self.img_height, self.img_width), interpolation=cv2.INTER_AREA)
        obs = obs.transpose((2, 0, 1))
        obs = torch.Tensor(obs).to(torch.uint8)
        return obs, reward, done, info


class FrameConvertWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        frame = self.env.reset(**kwargs)
        return self._convert(frame)

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        frame = self._convert(frame)
        return frame, reward, done, info

    def _convert(self, frame):
        return 1 - frame / 255


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, _n_stack_frames=4):
        super().__init__(env)
        self._n_stack_frames = _n_stack_frames
        self._frames = deque(maxlen=_n_stack_frames)
    
    def reset(self, **kwargs):
        frame = self.env.reset(**kwargs)
        for _ in range(self._n_stack_frames):
            self._frames.append(frame)
        return torch.cat(list(self._frames))
    
    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self._frames.append(frame)
        obs = torch.cat(list(self._frames))
        return obs, reward, done, info


class ForPytorchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = obs.unsqueeze(0)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.unsqueeze(0)  # add axis to make batch
        return obs, reward, done, info
