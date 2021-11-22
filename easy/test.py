import gym

r = gym.gains("greedy", k=20, N=10, T=100)
print(r)