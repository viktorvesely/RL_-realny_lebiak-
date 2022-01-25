import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
print("O")
print(env.observation_space)
print("A")
print(env.action_space.high)
print(env.action_space.low)