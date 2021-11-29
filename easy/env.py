import numpy as np
import random

class EnvGaussian:

    reward_min_mu = -5
    reward_max_mu = 5
    reward_min_sd = 1
    reward_max_sd = 4


    def __init__(self, k):
        
        self.k = k # number of actions
        self.gaussian = np.ones((self.k, 2)).T

        self.gaussian[0] = np.random.rand(self.k) * (EnvGaussian.reward_max_mu - EnvGaussian.reward_min_mu) + EnvGaussian.reward_min_mu
        self.gaussian[1] = np.random.rand(self.k) * (EnvGaussian.reward_max_sd - EnvGaussian.reward_min_sd) + EnvGaussian.reward_min_sd
        
        self.gaussian = self.gaussian.T

    def reward(self, action):
        
        pars = self.gaussian[action]

        return np.random.normal(
            loc = pars[0],
            scale = pars[1]
        )

    def getExpecteds(self):
        return self.gaussian.T[0]

    def bestAction(self):
        return np.argmax(self.gaussian.T[0])

    def evaluateAction(self, action):
        rankedActions = np.argsort(self.gaussian.T[0])[::-1]
        rank = np.where(rankedActions == action)
        return rank[0]

class EnvBernoulli:

    def __init__(self, k):
        
        self.k = k # number of actions
        self.p = np.random.rand(k)

    def reward(self, action):

        return 1 if random.random() < self.p[action] else 0

    def getExpecteds(self):
        return self.p

    def bestAction(self):
        return np.argmax(self.p)

    def evaluateAction(self, action):
        rankedActions = np.argsort(self.p)[::-1]
        rank = np.where(rankedActions == action)
        return rank[0]
    