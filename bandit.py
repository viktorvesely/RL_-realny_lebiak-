import numpy as np

class Bandit():

    init_Q = 1

    def __init__(self, method, k, env):
        self.k = k
        self.Q = np.ones(k) * Bandit.init_Q
        self.update = getattr(self, method)
        self.env = env


    def alpha(self, t):
        return 1 / t

    def greedy(self, t):

        action = np.argmax(self.Q)
        reward = self.env.reward(action)
        self.Q[action] = self.Q[action] + self.alpha(t) * (reward - self.Q[action])