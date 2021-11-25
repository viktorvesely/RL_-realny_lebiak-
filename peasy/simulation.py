import numpy as np

from agent import Agent, ExplorationMethod
from bandits import Bandit, BanditType

def simulate(learningMethod: ExplorationMethod, btype: BanditType, N: int, T:int, k:int):

    resultsActions = np.zeros((N, T))
    resultsRewards = np.zeros((N, T))

    for i in range(0, N):
        bandit = Bandit(btype, k)
        player = Agent(learningMethod, bandit, btype)

        
        for j in range(0, T):
            action, reward = player.makeAction()
            bestAction = bandit.getBestAction()
            if bestAction.size != 1:
                print("Error, there is more than one optimal action")
            resultsActions[i,j] = action == bestAction()[0]
            resultsRewards[i,j] = reward

    return resultsActions, resultsRewards

def simulateAll(N: int, T:int, k:int):
    actions = {}
    rewards = {}

    for eMethod in ExplorationMethod:
        for bType in BanditType:
            actions[(eMethod, bType)], rewards[(eMethod, bType)] = simulate(eMethod, bType, N, T, k)
    
    return actions, rewards

def summarise(actions, rewards, N, T):
    summarisedActions = {}
    for key, value in actions.items():
        #in percents
        summarisedActions[key] = (np.mean(value, axis=0))*100

    summarisedRewards = {}
    for key, value in actions.items():
        summarisedRewards[key] = np.mean(value, axis=0)
    
    return summarisedActions, summarisedRewards

        
