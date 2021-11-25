import numpy as np
from bandits import Bandit

INF = 99999999

class ExplorationMethod(Enum):
    GREEDY = 1
    EGREEDY = 2
    OIV = 3 # Optimistic initial values
    SOFT = 4 # Softmax policy
    UCB = 5 # Upper confidence bound
    AP = 6 # Action preferences


class Agent:
    
    def __init__(self, eMethod: ExplorationMethod, actionlist: np.ndarray, environment: Bandit) -> None:
        self.method = eMethod
        self.algorithm = self.selectMethod(self.method)
        self.actions = actionlist
        self.env = environment
        self.t = 1
        self.Q = np.zeros(self.actions.size)


    def selectMethod(self, eMethod):
        if eMethod == ExplorationMethod.GREEDY:
            return self.greedy

        elif eMethod == ExplorationMethod.EGREEDY:
            self.epsilon = 0.1
            return self.egreedy

        elif eMethod == ExplorationMethod.OIV:
            self.Q = np.ones(self.actions.size)*10
            return self.oiv
        
        elif eMethod == ExplorationMethod.SOFT:
            return self.soft
        
        elif eMethod == ExplorationMethod.UCB:
            self.c = 2
            self.Na = np.zeros(self.actions.size)
            self.U = np.ones(self.actions.size)*INF
            return self.ucb
        
        elif eMethod == ExplorationMethod.AP:
            self.H = np.zeros(self.actions.size)
            self.pi = self.getSoftmaxDistribution(np.zeros(self.actions.size))
            return self.ap
    
    def greedy(self):
        # if there are multiple best actions choose randomly
        return np.random.choice(np.argmax(self.Q))
    
    def egreedy(self):
        # exploration
        nActions = self.actions.size
        probabilities = np.ones(nActions) * self.epsilon/nActions
        # exploitation -- the 1 - e probability is evenly distributed across all the best options
        best = np.argmax(self.Q)
        bestprob = (1 - self.epsilon)/best.size
        probabilities[best] += bestprob
        return np.random.choice(self.actions, p=probabilities)

    def oiv(self):
        return self.greedy()
    
    def soft(self):
        softmax = self.getSoftmaxDistribution(self.Q)
        return np.random.choice(self.actions, p=softmax)
    
    def ucb(self):
        selected = np.random.choice(np.argmax(self.Q + self.U))
        #update uncertainties
        self.Na[selected] += 1
        self.U[selected] = self.c*np.sqrt(np.log(self.t)/self.Na[selected])
        return selected
    
    def ap(self):
        action = np.random.choice(self.actions, p=self.pi)
        reward = self.env.getReward(action)

        mask = np.zeros(self.actions.size, np.bool)
        mask[action] = True

        self.H[mask] += (1/self.t)*(reward - self.Q)*(1-self.pi[mask])
        self.H[~mask] -= (1/self.t)*(reward - self.Q)*(self.pi[~mask])

        self.pi = self.getSoftmaxDistribution(self.H)
        return np.random.choice(self.actions, p=self.pi)
        
    def getSoftmaxDistribution(self, array):
        exponents = np.exp(array)
        return exponents/np.sum(exponents)

    def incrementStep(self, action):
        self.Q[action] += (1/self.t)*(self.env.getReward(action) - self.Q[action])
        self.t += 1

    def makeAction(self):
        action = self.algorithm()
        self.incrementStep(action)
        return action