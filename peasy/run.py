from peasy.simulation import simulateAll, summarise
import simulation

# experiment runs
N = 100
# training iterations
T = 1000
# arms
K = 10

actions, rewards = simulateAll(N= N, T= T, k= K)
summarisedA, summarisedR = summarise(actions, rewards, N, T)


