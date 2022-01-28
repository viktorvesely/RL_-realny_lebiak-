import gym
import numpy as np
from matplotlib import pyplot as plt
import sys 
import os.path

from drifter import Drifter
from experiences import Buffer
from detective import Detective

n_episodes = 3500
n_frames = 500
inspect = False
experience_buffer_size = 100_000
batch_size = 128


env = gym.make('LunarLanderContinuous-v2')
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
            
            action = drifter(state, training=True)
        
            next_state, reward, done, _ = env.step(action)

            if done:
                break
            
            buffer.record(state, action, reward, next_state)
            detective.on_tick(state, action, reward, next_state)

            episodic_reward += reward

            n_experiences = len(buffer)
            if n_experiences >= experience_buffer_size:
                critic_loss, actor_loss = drifter.learn(buffer())
                drifter.sync_targets()
                detective.on_train(actor_loss, critic_loss)
        
            state = next_state
        
        print(f"Episodic reward: {episodic_reward}")
        detective.on_episode()
        drifter.on_episode()

    drifter.save_model()
    env.close()

def exploit():
    drifter.load_model()

    while True:
        state = env.reset()
        
        while True:

            env.render()

            action = drifter(state, training=False)
            next_state, reward, done, _ = env.step(action)
            detective.on_tick(state, action, reward, next_state)

            if done:
                print("Episode end")
                break

            state = next_state
        
        detective.on_episode()
        drifter.on_episode()


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "learn":
        print("[LEARNING]")
        detective = Detective(drifter, True)
        learn()
        detective.on_end()
    elif mode == "exploit":
        print("[EXPLOITING]")
        detective = Detective(drifter, False)
        exploit()
        detective.on_end()
    else:
        print(f"'{mode}' is not correct mode of execution. Accapted values: 'learn' or 'exploit'")    

