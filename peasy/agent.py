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
    
    def __init__(self, eMethod: ExplorationMethod, actionlist: list) -> None:
        self.method = eMethod
        self.algorithm = self.selectMethod(self.method)
        self.actions = actionlist
        self.t = 1
        self.Q = np.zeros(len(self.actions))

        # extra variables for particular algorithms
        self.epsilon = 0.1
        self.c = 2
        self.Na = np.zeros(len(self.actions))
        self.U = np.ones(len(self.actions))*INF

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
        exponents = np.exp(self.Q)
        softmax = exponents/np.sum(exponents)
        return np.random.choice(np.arange(len(self.actions)), p=softmax)
    
    def ucb(self):
        selected = np.random.choice(np.argmax(self.Q + self.U))
        #update uncertainties
        self.Na[selected] += 1
        self.U[selected] = self.c*np.sqrt(np.log(self.t)/self.Na[selected])
        return selected
    
    def incrementStep(self, action, bandit: Bandit):
        self.Q[action] += (1/self.t)*(bandit.getReward(action) - self.Q[action])
        self.t += 1

    def makeAction(self, bandit: Bandit):
        action = self.algorithm()
        self.incrementStep(action, bandit)
        return action