import random
import numpy as np
from enum import Enum

class BanditType(Enum):
    GAUSSIAN = 1
    BERNOULLI = 2

class Bandit():
    def __init__(self, btype: BanditType, k: int):
        self.type = btype
        self.nArms = k
        self.Arms = []

        if self.type == BanditType.BERNOULLI:
            for _ in range(0, k):
                self.Arms.append(random.random())
        elif self.type == BanditType.GAUSSIAN:
            for _ in range(0, k):
                mean = random.random()
                std = random.random()/2
                self.Arms.append(tuple([mean, std]))

    def getReward(self, action):
        if self.type == BanditType.BERNOULLI:
            r = random.random()
            return 1 if r <= self.Arms[action] else 0
        elif self.type == BanditType.GAUSSIAN:
            mean, std = self.Arms[action]
            return np.random.normal(mean, std)
        else:
            raise Exception("You should not reach this part, invalid BanditType")
    
    def getPossibleActions(self):
        return np.array(list(range(0, self.nArms)))
    
    def getExpectedValue(self, action):
        if self.type == BanditType.BERNOULLI:
            return self.Arms[action]
        elif self.type == BanditType.GAUSSIAN:
            mean, _ = self.Arms[action]
            return mean
        else:
            raise Exception("You should not reach this part, invalid BanditType")
    
    def getBestAction(self):
        actions = self.getPossibleActions()
        expected = []
        for action in actions:
            expected.append(self.getExpectedValue(action))
        return np.argmax(expected)