from simulation import simulateAll, summarise, simulateANDshow
from bandits import BanditType
from agent import ExplorationMethod
import matplotlib.pyplot as plt

# experiment runs
N = 100
# training iterations
T = 1000
# arms
K = 10


simulateANDshow(ExplorationMethod.GREEDY, BanditType.BERNOULLI, N=1, T=T, k=K)