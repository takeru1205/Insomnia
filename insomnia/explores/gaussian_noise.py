import numpy as np


class GaussianActionNoise(object):
    def __init__(self, mu, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

