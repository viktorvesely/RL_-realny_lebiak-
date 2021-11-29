import numpy as np
import matplotlib.pyplot as plt

from agent import Agent, ExplorationMethod
from bandits import Bandit, BanditType

def simulate(learningMethod: ExplorationMethod, btype: BanditType, N: int, T:int, k:int):

    resultsActions = np.zeros((N, T))
    resultsRewards = np.zeros((N, T))

    for i in range(0, N):
        bandit = Bandit(btype, k)
        player = Agent(learningMethod, bandit.getPossibleActions(),  bandit)

        
        for j in range(0, T):
            action, reward = player.makeAction()
            bestAction = bandit.getBestAction()
            if bestAction.size != 1:
                print("Error, there is more than one optimal action")
            resultsActions[i,j] = action == bestAction
            resultsRewards[i,j] = reward

    return resultsActions, resultsRewards

def simulateANDshow(learningMethod: ExplorationMethod, btype: BanditType, N: int, T:int, k:int):

    actions, rewards  = simulate(learningMethod, btype, N, T, k)

    actions = (np.mean(actions, axis=0))*100

    rewards = np.mean(rewards, axis=0)


    plt.rcParams["figure.figsize"] = (25,5)

    x = list(range(1,T+1))
    plot1 = plt.figure(1)
    plt.plot(x, actions, linewidth=0.5)
    plt.title(f"Percentage of optimal actions, {btype.name} distribution, {learningMethod.name}")
    plt.legend()

    plot2 = plt.figure(2)
    plt.plot(x, rewards, linewidth=0.5)
    plt.title(f"Average reward, {btype.name} distribution, {learningMethod.name}")
    plt.legend()


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
    for key, value in rewards.items():
        summarisedRewards[key] = np.mean(value, axis=0)
    
    return summarisedActions, summarisedRewards

        
