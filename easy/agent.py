import numpy as np
import random

from enum import Enum

class AgentTypes(Enum):
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    UCB = 3
    AP = 4
    SOFT_MAX_AP = 5
    SOFT_MAX_Q_VALUES = 6

at = AgentTypes

class Agent():

    def __init__(self, method, k, env, extra=None):

        self.t = None
        self.qScales = 1
        self.k = k
        self.method = method
        self.extra = extra
        self.env = env

        self.algorithm = self.resolve_method() 
        #self.Q = np.random.rand(k) * self.qScales
        self.Q = np.zeros(k) if method != at.OPTIMISTIC else self.Q
        #self.Q = np.ones(k)

    def resolve_method(self):
        m = self.method

        if m == at.GREEDY:
            return self.greedy

        if m == at.OPTIMISTIC:
            self.qScales = self.extra
            self.Q = np.ones(self.k) * self.qScales
            return self.greedy

        if m == at.EPSILON_GREEDY:
            return self.eps_greedy

        if m == at.UCB:
            self.histogram = 0.00001 * np.ones(self.k) # TODO is this correct? I think yes
            return self.UCB

        if m == at.AP or m == at.SOFT_MAX_AP or m == at.SOFT_MAX_Q_VALUES:
            self.H = np.zeros(self.k) 
            self.Q = np.zeros(self.k)
            self.PI = None
            return self.soft_max
        
        return None

    def epsilon(self, t):
        return 1 / (t ** (1/2)) 

    def alpha(self, t):
        return 1 / (t ** (1/2))

    def increment(self, action, reward, t):
        self.Q[action] = self.Q[action] + self.alpha(t) * (reward - self.Q[action])

    def best_action(self):
        m = self.method
        t = self.t

        if m == at.GREEDY or m == at.OPTIMISTIC:
            return self.argmax(self.Q)

        if m == at.EPSILON_GREEDY:
            return np.random.choice(self.k)  if random.random() < self.epsilon(t) else self.argmax(self.Q)

        if m == at.UCB:
            Q_adj = self.Q + np.sqrt(np.log(t) / self.histogram)
            return self.argmax(Q_adj)

        if m == at.SOFT_MAX_AP or m == at.AP or m == at.SOFT_MAX_Q_VALUES:

            H = self.H
            if m == at.SOFT_MAX_Q_VALUES:
                H = self.Q

            sum_sm = np.sum(np.exp(H))
            self.PI = np.exp(H) / sum_sm 

            deterministic = m == at.AP
            return self.argmax(H) if deterministic else np.random.choice(self.k, p=self.PI)
        
    def argmax(self, array):
        index = np.argwhere(array == np.amax(array))
        return np.random.choice(np.concatenate(index, axis=0))

    def update(self, t):
        self.t = t
        return self.algorithm(t)

    def greedy(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        self.increment(action, reward, t)
        return (action, reward)

    def UCB(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        self.histogram[action] += 1
        self.increment(action, reward, t)
        return (action, reward)

    def soft_max(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        
        mask = np.zeros(self.k, np.bool)
        mask[action] = True
        
        self.H[mask] = self.H[mask] + 0.1 * (reward - self.Q[mask]) * (1 - self.PI[mask])
        self.H[~mask] = self.H[~mask] - 0.1 * (reward - self.Q[~mask]) * self.PI[~mask]

        self.increment(action, reward, t)
        return (action, reward)

    def eps_greedy(self, t):
        action = self.best_action()
        reward = self.env.reward(action)
        self.increment(action, reward, t)
        return (action, reward)
