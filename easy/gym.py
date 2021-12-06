import numpy as np
import sys

from env import EnvBernoulli, EnvGaussian
from agent import Agent

def gains(method, environment, k=10, N=100, T=1000, extra=None):
        
    agents = [Agent(method, k, EnvBernoulli(k) if environment == "B" else EnvGaussian(k), extra=extra) for _ in range(N)]

    ts = np.arange(1, T + 1)

    accs = []
    rewards = []
    for t in ts:
        #sys.stdout.write('t=%s\r' % str(t / (T + 1) * 100))
        correct = 0
        reward = 0 
        for agent in agents:
            agent.update(t)
            best_action = agent.best_action()
            reward += agent.env.reward(best_action)
            correct += 1 if best_action == agent.env.bestAction() else 0

        rewards.append(reward / len(agents))
        accs.append(correct / len(agents))
            
    print()

    ranks = []
    for agent in agents:
        ranks.append(agent.env.evaluateAction(agent.best_action()))

    return (np.array(ranks), np.array(accs), np.array(rewards))
    
    

