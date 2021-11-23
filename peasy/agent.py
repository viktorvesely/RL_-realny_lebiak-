import numpy as np
from bandits import Bandit

class ExplorationMethod(Enum):
    GREEDY = 1
    EGREEDY = 2
    OIV = 3 # Optimistic initial values
    SOFT = 4 # Softmax policy
    UCB = 5 # Upper confidence bound
    AP = 6 # Action preferences


class Agent:
    
    def __init__(self, eMethod: ExplorationMethod, actionlist: list) -> None:
        self.method = eMethod
        self.algorithm = self.selectMethod(self.method)
        self.actions = actionlist
        self.t = 1
        self.Q = np.zeros(len(self.actions))
        self.epsilon = 0.1

    def selectMethod(self, eMethod):
        if eMethod == ExplorationMethod.GREEDY:
            return self.greedy

        elif eMethod == ExplorationMethod.EGREEDY:
            return self.egreedy

        elif eMethod == ExplorationMethod.OIV:
            return self.oiv
        
        elif eMethod == ExplorationMethod.SOFT:
            return self.soft
        
        elif eMethod == ExplorationMethod.UCB:
            return self.ucb
        
        elif eMethod == ExplorationMethod.AP:
            return self.ap
    
    def greedy(self):
        # if there are multiple best actions choose randomly
        return np.random.choice(np.argmax(self.Q))
    
    def egreedy(self):
        # exploration
        nActions = len(self.actions)
        probabilities = np.ones(nActions) * self.epsilon/nActions
        # exploitation -- the 1 - e probability is evenly distributed across all the best options
        best = np.argmax(self.Q)
        bestprob = (1 - self.epsilon)/best.size
        probabilities[best] += bestprob
        return np.random.choice(np.arange(len(self.actions)), p=probabilities)

    def oiv(self):
        if self.t == 1:
            self.Q = np.ones(len(self.actions))*10
        return self.greedy()
    
    def soft(self):

    
    def incrementStep(self, action, bandit: Bandit):
        self.Q[action] += (1/self.t)*(bandit.getReward(action) - self.Q[action])
        self.t += 1