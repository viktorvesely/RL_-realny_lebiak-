import numpy as np
import random

from enum import Enum

class BanditTypes(Enum):
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    UCB = 3
    AP = 4
    SOFT_MAX = 5
    

bt = BanditTypes

class Bandit():

    def __init__(self, method, k, env, extra=None):

        self.epsilon = None
        self.t = None
        self.qScales = 1
        self.k = k
        self.method = method
        self.extra = extra
        self.env = env

        self.algorithm = self.resolve_method()
        self.Q = np.random.rand(k) * self.qScales # TODO this is probably not corret but the algs perform better

    def resolve_method(self):
        m = self.method

        if m == bt.GREEDY:
            return self.greedy

        if m == bt.OPTIMISTIC:
            self.qScales = self.extra
            return self.greedy

        if m == bt.EPSILON_GREEDY:
            self.epsilon = self.extra
            return self.eps_greedy

        if m == bt.UCB:
            self.histogram = np.ones(self.k) # TODO is this correct? I think yes
            return self.UCB

        if m == bt.AP or m == bt.SOFT_MAX:
            self.H = np.zeros(self.k) 
            self.Q = np.zeros(self.k)
            self.PI = None

        if m == bt.AP:
            return self.AP

        if m == bt.SOFT_MAX:
            return self.soft_max
        
        return None

    def alpha(self, t):
        return 1 / t

    def increment(self, action, reward, t):
        self.Q[action] = self.Q[action] + self.alpha(t) * (reward - self.Q[action])

    def best_action(self):
        m = self.method
        t = self.t

        if m == bt.GREEDY or m == bt.OPTIMISTIC:
            return np.argmax(self.Q)

        if m == bt.EPSILON_GREEDY:
            return np.random.choice(self.k)  if random.random() < self.epsilon else np.argmax(self.Q)

        if m == bt.UCB:
            Q_adj = self.Q + np.sqrt(np.log(t) / self.histogram)
            return np.argmax(Q_adj)

        if m == bt.SOFT_MAX or m == bt.AP:
            stochastic = m == bt.SOFT_MAX
            sum_sm = np.sum(np.exp(self.H))

            self.PI = np.exp(self.H) / sum_sm
            return np.random.choice(self.k, p=self.PI) if stochastic else np.argmax(self.H)

    def update(self, t):
        self.t = t
        self.algorithm(t)

    def greedy(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        self.increment(action, reward, t)

    def UCB(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        self.histogram[action] += 1
        self.increment(action, reward, t)

    def soft_max(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        
        mask = np.zeros(self.k, np.bool)
        mask[action] = True
        
        self.H[mask] = self.H[mask] + self.alpha(t) * (reward - self.Q[mask]) * (1 - self.PI[mask])
        self.H[~mask] = self.H[~mask] - self.alpha(t) * (reward - self.Q[~mask]) * self.PI[~mask]

        self.increment(action, reward, t)

    def AP(self, t):
        self.soft_max(t)

    def eps_greedy(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        self.increment(action, reward, t)
