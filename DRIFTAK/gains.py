import gym
import numpy as np
from matplotlib import pyplot as plt

from drifter import Drifter
from experiences import Buffer

n_episodes = 10
n_frames = 500
inspect = True
experience_buffer_size = 10000
batch_size = 500

update_every = 90 
sync_every = 90

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
    # action_space = np.array([lows, highs])
    action_space = action_space.T
    
    return Drifter(action_space, env.observation_space.shape)

drifter = create_drifter()

total_frames = 0
loss = []


for episode in range(n_episodes):
    state = env.reset()

    episodic_reward = 0
    print(f"Epsiode {episode}/{n_episodes}")

    for frame in range(n_frames):
        total_frames += 1

        if inspect:
            env.render()
        
        action = drifter(state).numpy()

        try:
            #next_state, reward, done, _ = env.step(action)
            next_state, reward, done, _ = env.step(np.append(action, 0))

        except TypeError:
            print("I am the imfamous error : )")

        buffer.record(state, action, reward, next_state)

        episodic_reward += reward

        n_experiences = len(buffer)
        if time_to(total_frames, update_every) and n_experiences >= batch_size:
            actor_loss, critic_loss = drifter.learn(buffer())
            loss.append([actor_loss, critic_loss])

        if time_to(total_frames, sync_every) and n_experiences >= batch_size:
            drifter.sync_targets()
        state = next_state
        
    print(f"Episodic reward: {episodic_reward}")

env.close()

loss = np.array(loss).T
t = np.arange(len(loss[0]))
plt.plot(t, loss[0], label="actor", color="blue")
plt.plot(t, loss[1], label="critic", color="green")
plt.legend()

plt.show()

