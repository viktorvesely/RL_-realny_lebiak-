import numpy as np
import sys

from env import EnvBernoulli, EnvGaussian
from bandit import Bandit

def gains(method, environment, k=10, N=100, T=1000, extra=None):
        
    bandits = [Bandit(method, k, EnvBernoulli(k) if environment == "B" else EnvGaussian(k), extra=extra) for _ in range(N)]

    ts = np.arange(1, T + 1)

    accs = []
    for t in ts:
        sys.stdout.write('t=%s\r' % str(t / (T + 1) * 100))
        correct = 0
        for bandit in bandits:
            bandit.update(t)
            correct += 1 if bandit.best_action() == bandit.env.bestAction() else 0
        accs.append(correct / len(bandits))
            
    print()

    ranks = []
    for bandit in bandits:
        ranks.append(bandit.env.evaluateAction(bandit.best_action()))

    return (np.array(ranks), np.array(accs))
    
    

