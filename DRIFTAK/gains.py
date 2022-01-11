import gym
import numpy as np

from drifter import Drifter
from experiences import Buffer

n_episodes = 5
n_frames = 1000
inspect = True
experience_buffer_size = 1000
batch_size = 100

env = gym.make('CarRacing-v0')
buffer = Buffer(
    experience_buffer_size,
    batch_size,
    env.observation_space.shape,
    len(env.action_space.high)
)

def create_drifter():
    highs = np.array(env.action_space.high)
    lows = np.array(env.action_space.low)
    
    action_space = np.array([highs, lows])
    action_space = action_space.T
    
    return Drifter(action_space, env.observation_space.shape)

drifter = create_drifter()

for episode in range(n_episodes):
    state = env.reset()

    episodic_reward = 0

    for frame in range(n_frames):
        if inspect: 
            env.render()
        
        action = drifter(state)
        next_state, reward, done, _ = env.step(action)
        buffer.record(state, action, reward, next_state)

        episodic_reward += reward

        drifter.learn(buffer())
        drifter.sync_targets()
        state = next_state
        
    print(f"Episodic reward: {episodic_reward}")

env.close()