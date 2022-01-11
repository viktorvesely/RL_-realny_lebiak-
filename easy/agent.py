import numpy as np
import random

from enum import Enum

class AgentTypes(Enum):
    """
    Enum for the agent types
    """
    GREEDY = 0
    EPSILON_GREEDY = 1
    OPTIMISTIC = 2
    UCB = 3
    SOFT_MAX_AP = 4
    SOFT_MAX_Q_VALUES = 5

# Aliasing
at = AgentTypes

class Agent():
    """
    Class that defines all the algorithms
    """

    def __init__(self, method, k, env, extra=None):
        """
        @param method: Enum for the method type
        @param k: number of arms
        @param env: Object that defines the environment
        @param extra: Extra parameters 
        """

        self.k = k
        self.method = method
        self.extra = extra
        self.env = env

        # Simulation time
        self.t = None

        # Resolve algorithm function
        self.algorithm = self.resolve_method() 

        # Initialize Q values
        if method != at.OPTIMISTIC:
            self.Q = np.zeros(k)

    def resolve_method(self):
        """
        @return function which corresponds to the self.method attribute
        """
        m = self.method

        if m == at.GREEDY:
            return self.greedy

        if m == at.OPTIMISTIC:
            self.Q = np.ones(self.k) * self.extra
            return self.greedy

        if m == at.EPSILON_GREEDY:
            return self.eps_greedy

        if m == at.UCB:
            self.histogram = 0.00001 * np.ones(self.k) 
            return self.UCB

        if m == at.SOFT_MAX_AP or m == at.SOFT_MAX_Q_VALUES:
            self.H = np.zeros(self.k) 
            self.Q = np.zeros(self.k)
            self.PI = None
            return self.soft_max
        
        return None

    def epsilon(self, t):
        return 1 / (t ** (0.2)) 

    def alpha(self, t):
        """
        t ** (0.51) needs to be there, otherwise the greedy algorithms do not work
        0.51 was chosen so it satisfies both criteria defined in the lecture
        1) \sum_{t=1}{\infty} 1 / (t ** (0.51)) diverges to infinity
        2) \sum_{t=1}{\infty} (1 / (t ** (0.51)))^2 is less than infinity
        @return alpha parameter which determines exploration/exploitation
        """
        return 1 / (t ** (0.51))

    def increment(self, action, reward, t):
        """
        Iterative update rule for q values
        """
        self.Q[action] = self.Q[action] + self.alpha(t) * (reward - self.Q[action])

    def best_action(self):
        """
        @return the best action as defined by each algorithm
        """
        m = self.method
        t = self.t

        if m == at.GREEDY or m == at.OPTIMISTIC:
            return self.argmax(self.Q)

        if m == at.EPSILON_GREEDY:
            return np.random.choice(self.k)  if random.random() < self.epsilon(t) else self.argmax(self.Q)

        if m == at.UCB:
            Q_adj = self.Q + np.sqrt(np.log(t) / self.histogram)
            return self.argmax(Q_adj)

        if m == at.SOFT_MAX_AP or m == at.SOFT_MAX_Q_VALUES:
            H = self.H
            if m == at.SOFT_MAX_Q_VALUES:
                H = self.Q

            sum_sm = np.sum(np.exp(H * 10))
            self.PI = np.exp(H * 10) / sum_sm 

            return np.random.choice(self.k, p=self.PI)
        
    def argmax(self, array):
        """
        @return np.argmax if there is unique max value otherwise returns random choice
        from the max values
        """
        index = np.argwhere(array == np.amax(array))
        return np.random.choice(np.concatenate(index, axis=0))

    def update(self, t):
        """ 
        Function that progress the algorithms
        """
        self.t = t
        return self.algorithm(t)

    def greedy(self, t):
        """
        Selects greedily from Q values 
        """
        action = self.best_action()
        reward = self.env.reward(action)
        self.increment(action, reward, t)
        return (action, reward)

    def UCB(self, t):
        """
        Select greedily from (Q values + upper confidence bounds)
        """
        action = self.best_action()
        reward = self.env.reward(action)
        self.histogram[action] += 1
        self.increment(action, reward, t)
        return (action, reward)

    def soft_max(self, t):
        """
        Construct boltzmann distribution from Q values or Action rpeferences
        and uses that as policy
        """
        action = self.best_action()
        reward = self.env.reward(action)
        
        mask = np.zeros(self.k, np.bool)
        mask[action] = True
        
        # Update rule for the choosen action
        self.H[mask] = self.H[mask] + 0.1 * (reward - self.Q[mask]) * (1 - self.PI[mask])
        
        # Update rule for the other actions
        self.H[~mask] = self.H[~mask] - 0.1 * (reward - self.Q[~mask]) * self.PI[~mask]

        self.increment(action, reward, t)
        return (action, reward)

    def eps_greedy(self, t):
        """
        Selects random action with epsilon proability
        Selects greedily
        """
        action = self.best_action()
        reward = self.env.reward(action)
        self.increment(action, reward, t)
        return (action, reward)
