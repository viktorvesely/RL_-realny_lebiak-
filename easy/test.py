import gym
from bandit import BanditTypes as BT

method = BT.GREEDY
env = "G"
r = gym.gains(method, env, k=20, N=10, T=100)
print(r)