import gym
import numpy as np
import torch
from collections import deque
import torchvision.transforms as T
from PIL import Image

class FrameObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.resize = T.Compose([T.ToPILImage(),
                            T.Resize((50, 75), interpolation=Image.CUBIC),
                            T.ToTensor()])
    
    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        obs = self.env.render(mode='rgb_array')
        obs = obs.transpose((2, 0, 1))
        torch.Tensor(obs.astype(np.uint8))
        obs = self.resize(obs)
        return obs

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        obs = obs.transpose((2, 0, 1))
        torch.Tensor(obs.astype(np.uint8))
        obs = self.resize(obs)
        return obs, reward, done, info

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
        obs = obs.unsqueeze(0)
        return obs, reward, done, info