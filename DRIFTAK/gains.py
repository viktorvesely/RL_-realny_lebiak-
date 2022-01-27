import gym
import numpy as np
import torch

from buffer import Buffer
from drifter import Drifter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

n_episodes = 50
n_frames = 400
inspect = False

buffer_size = 10000
batch_size = 1

update_every = 10
sync_every = 10

env = gym.make("LunarLanderContinuous-v2")
action_space = env.action_space
state_space = env.observation_space

buffer = Buffer(buffer_size, state_space, action_space, batch_size)
drifter = Drifter()

for i_episode in range(n_episodes):
    state = env.reset()

    episodic_reward = 0
    rewards = []

    for t in range(n_frames):

        if inspect:
            env.render()
        
        action = drifter.makeAction(state)
        new_state, reward, done, _ = env.step(action)

        buffer.record(state, action, reward, new_state, done)
        episodic_reward += reward
        rewards.append(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

        drifter.learn(buffer())
    
    print(f"Total episodic reward: {episodic_reward} --- Average reward: {sum(rewards)/len(reward)}")

env.close()

def learn():
    pass
