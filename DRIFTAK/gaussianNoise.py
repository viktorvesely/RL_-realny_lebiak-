import numpy as np

class GaussianNoise:

    def __init__(self, std, num_action):
        self.n = num_action
        self.std = std
        self.mean = 0

    def __call__(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=self.n)

        