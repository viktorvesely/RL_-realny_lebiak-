import numpy as np
import random

class EnvGaussian:
    """
    Gaussian environment chooses reward from a normal distribution
    """
    reward_min_mu = 0
    reward_max_mu = 1
    reward_min_sd = 0
    reward_max_sd = 0.025


    def __init__(self, k):
        
        self.k = k # number of actions
        self.gaussian = np.ones((self.k, 2)).T

        # Create means
        self.gaussian[0] = np.random.rand(self.k) * (EnvGaussian.reward_max_mu - EnvGaussian.reward_min_mu) + EnvGaussian.reward_min_mu
        
        # Create standard deviations
        self.gaussian[1] = np.random.rand(self.k) * (EnvGaussian.reward_max_sd - EnvGaussian.reward_min_sd) + EnvGaussian.reward_min_sd
        
        self.gaussian = self.gaussian.T

    def reward(self, action):
        """
        @param {action}: index of the action
        @return: reward sampled from a normal distribution
        """
        
        pars = self.gaussian[action]

        return np.random.normal(
            loc = pars[0],
            scale = pars[1]
        )

    def getExpecteds(self):
        """
        @return: means of the normal distributions
        """
        return self.gaussian.T[0]

    def bestAction(self):
        """
        @return: index of the action with the highest expected value
        """
        return np.argmax(self.gaussian.T[0])

    def evaluateAction(self, action):
        """
        @param {action}: index of the action 
        @return: rank of the action  (optimal action is 0, second optimal is 1 etc.)
        """
        rankedActions = np.argsort(self.gaussian.T[0])[::-1]
        rank = np.where(rankedActions == action)
        return rank[0]

class EnvBernoulli:
    """
    Bernoulli environment gives binary reward based on probability p 
    """

    def __init__(self, k):
        
        self.k = k # number of actions
        self.p = np.random.rand(k)

    def reward(self, action):
        """
        @param {action}: index of the action
        @return: 1 with a prability p and 0 otherwise
        """
        return 1 if random.random() < self.p[action] else 0

    def getExpecteds(self):
        """
        @return: p array
        """
        return self.p

    def bestAction(self):
        """
        @return: index of the action where the p is the highest
        """
        return np.argmax(self.p)

    def evaluateAction(self, action):
        """
        @param {action}: index of the action 
        @return: rank of the action  (optimal action is 0, second optimal is 1 etc.)
        """
        rankedActions = np.argsort(self.p)[::-1]
        rank = np.where(rankedActions == action)
        return rank[0]
    