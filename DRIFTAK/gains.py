import gym
import numpy as np

from drifter import Drifter
from experiences import Buffer

n_episodes = 5
n_frames = 1000
inspect = True
experience_buffer_size = 1000
batch_size = 100

update_every = 100
sync_every = 200

env = gym.make('CarRacing-v0')
buffer = Buffer(
    experience_buffer_size,
    batch_size,
    env.observation_space.shape,
    len(env.action_space.high) - 1
)

def time_to(curr_frame, interval):
    return curr_frame % interval == 0

def create_drifter():
    highs = np.array(env.action_space.high)
    lows = np.array(env.action_space.low)
    
    action_space = np.array([[-1.0, 0.0], [1.0, 1.0]])
    action_space = action_space.T
    
    return Drifter(action_space, env.observation_space.shape)

drifter = create_drifter()

total_frames = 0

for episode in range(n_episodes):
    state = env.reset()

    episodic_reward = 0

    for frame in range(n_frames):
        total_frames += 1

        if inspect:
            env.render()
        
        action = drifter(state).numpy()

        try:
            next_state, reward, done, _ = env.step(np.append(action, 0))
        except TypeError:
            print(": )")

        buffer.record(state, action, reward, next_state)

        episodic_reward += reward

        if time_to(total_frames, update_every):
            drifter.learn(buffer())

        if time_to(total_frames, sync_every):
            drifter.sync_targets()
        state = next_state
        
    print(f"Episodic reward: {episodic_reward}")

env.close()