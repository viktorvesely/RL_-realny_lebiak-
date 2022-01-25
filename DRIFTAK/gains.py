import gym
import numpy as np
from matplotlib import pyplot as plt
import sys 
import os.path

from drifter import Drifter
from experiences import Buffer

n_episodes = 500
n_frames = 800
inspect = False
experience_buffer_size = 10_000
batch_size = 1000

update_every = 2
sync_every = 2

env = gym.make('MountainCarContinuous-v0')
buffer = Buffer(
    experience_buffer_size,
    batch_size,
    env.observation_space.shape,
    len(env.action_space.high)
)

def time_to(curr_frame, interval):
    return curr_frame % interval == 0

def create_drifter():
    highs = np.array(env.action_space.high)
    lows = np.array(env.action_space.low)
    
    #action_space = np.array([[-1.0, 0.0], [1.0, 1.0]])
    action_space = np.array([lows, highs])
    
    return Drifter(action_space, env.observation_space.shape)

drifter = create_drifter()

loss = []

def learn():
    total_frames = 0
    for episode in range(n_episodes):
        state = env.reset()

        episodic_reward = 0
        print(f"Epsiode {episode}/{n_episodes}")

        for frame in range(n_frames):
            total_frames += 1

            if inspect:
                env.render()
            
            #state = mask.applyMask(state, episode, frame)
            #mask.showOneResult(state)
            
            action = drifter(state)

        
            next_state, reward, done, _ = env.step(action)
            buffer.record(state, action, reward, next_state)

            episodic_reward += reward

            n_experiences = len(buffer)
            if time_to(total_frames, update_every) and n_experiences >= batch_size:
                actor_loss, critic_loss = drifter.learn(buffer())
                #loss.append([actor_loss, critic_loss])

            if time_to(total_frames, sync_every) and n_experiences >= batch_size:
                drifter.sync_targets()
            state = next_state
        
        print(f"Episodic reward: {episodic_reward}")

    drifter.save_for_exploitation()
    env.close()

def exploit():
    drifter.load_for_exploitation()

    while True:
        state = env.reset()
        
        while True:

            env.render()

            action = drifter(state, training=False)
            next_state, reward, done, _ = env.step(action)

            if done:
                print("Task finished")
                break

            state = next_state


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "learn":
        print("[LEARNING]")
        learn()
    elif mode == "exploit":
        print("[EXPLOITING]")
        exploit()   
    else:
        print(f"'{mode}' is not correct mode of execution. Accapted values: 'learn' or 'exploit'")    

# loss = np.array(loss).T
# t = np.arange(len(loss[0]))
# plt.plot(t, loss[0], label="actor", color="blue")
# plt.plot(t, loss[1], label="critic", color="green")
# plt.legend()

# plt.show()

