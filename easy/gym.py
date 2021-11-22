import numpy as np
import sys

from env import EnvBernoulli, EnvGaussian
from bandit import Bandit

def gains(method, environment, k=10, N=100, T=1000, extra=None):
    env = None
    if environment == "B":
        env = EnvBernoulli(k)
    elif environment == "G":
        env = EnvGaussian(k)
        
    bandits = [Bandit(method, k, env, extra=extra) for _ in range(N)]

    ts = np.arange(1, T + 1)


    for t in ts:
        sys.stdout.write('t=%s\r' % str(t / (T + 1) * 100))
        for bandit in bandits:
            bandit.update(t)
    print()

    aStar = []
    for bandit in bandits:
        aStar.append(bandit.best_action())

    return (env.getExpecteds(), np.array(aStar))
    
    

