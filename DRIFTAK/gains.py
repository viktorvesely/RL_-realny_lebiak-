import gym
import numpy as np
from matplotlib import pyplot as plt
import sys 
import os

from drifter import Drifter
from experiences import Buffer
from detective import Detective
from Params import Params

pars = Params()
inspect = False

e_every_episode_print = 50
e_every_loss = 1
e_every_reward = 1

env = gym.make(pars.get('env'))

def time_to(curr_frame, interval):
    return curr_frame % interval == 0

def create_drifter():
    highs = np.array(env.action_space.high)
    lows = np.array(env.action_space.low)
    
    #action_space = np.array([[-1.0, 0.0], [1.0, 1.0]])
    action_space = np.array([lows, highs])
    
    return Drifter(action_space, env.observation_space.shape)

def learn():
    drifter = create_drifter()
    detective = Detective(drifter, True)

    experience_buffer_size = pars.get("experience_buffer_size")

    buffer = Buffer(
        experience_buffer_size,
        pars.get("batch_size"),
        env.observation_space.shape,
        len(env.action_space.high)
    )

    total_frames = 0
    n_episodes = pars.get("n_episodes")
    n_frames = pars.get("n_frames")
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
    detective.on_end()
    env.close()

def exploit():
    
    drifter = create_drifter()
    detective = Detective(drifter, False)

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

    detective.on_end()


def create_experiment(name):
    
    changed_name = False

    if os.path.isdir(f'./experiments/{name}'):
        changed_name = True
        name = f"{name}-1"    

    suffix = 2
    while os.path.isdir(f'./experiments/{name}'):
        name = name[:-2]
        name = f"{name}-{suffix}"
        suffix += 1 

    path = os.path.join(os.getcwd(), 'experiments', name)
    os.mkdir(path)
    pars.save(os.path.join(path, 'params.json'))
    
    return changed_name, name, path
    

def experiment(n_runs, name):

    original_name = name
    changed_name, name, path = create_experiment(name)

    if changed_name:
        print(f"Experiemnt {original_name} already exist, renaming to {name}")

    for run in range(n_runs):
        print(f"[{name}] run {run} / {n_runs}")
        
        drifter = create_drifter()

        experience_buffer_size = pars.get("experience_buffer_size")

        total_frames = 0
        n_episodes = pars.get("n_episodes")
        n_frames = pars.get("n_frames")

        actor_losses = []
        critic_losses = []
        episodic_rewards = []
        episodic_mean_rewards = []

        buffer = Buffer(
            experience_buffer_size,
            pars.get("batch_size"),
            env.observation_space.shape,
            len(env.action_space.high)
        )
       
        for episode in range(n_episodes):
            state = env.reset()

            episodic_reward = 0

            if episode % e_every_episode_print == 0:
                print(f"[{name}] Epsiode {episode}/{n_episodes}")

            frames_per_episode = 0

            for frame in range(n_frames):
                total_frames += 1
                frames_per_episode += 1

                if inspect:
                    env.render()
                
                action = drifter(state, training=True)
            
                next_state, reward, done, _ = env.step(action)

                if done:
                    break
                
                buffer.record(state, action, reward, next_state)

                episodic_reward += reward

                n_experiences = len(buffer)
                if n_experiences >= experience_buffer_size:
                    critic_loss, actor_loss = drifter.learn(buffer())
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    drifter.sync_targets()
            
                state = next_state
            
            if episode % e_every_episode_print == 0:
                print(f"Episodic reward: {episodic_reward}")
            

            episodic_rewards.append(episodic_reward)
            episodic_mean_rewards.append(episodic_reward / frames_per_episode)
            drifter.on_episode()
        
        drifter.save_model(os.path.join(path, f"actor-{run}.pt"))
        np.save(os.path.join(path, f"episodic_reward-{run}"), np.array(episodic_rewards))
        np.save(os.path.join(path, f"episodic_mean_reward-{run}"), np.array(episodic_mean_rewards))
        np.save(os.path.join(path, f"actor_loss-{run}"), np.array(actor_losses))
        np.save(os.path.join(path, f"critic_loss-{run}"), np.array(critic_losses))
        
        
    
    
    env.close()
    

if __name__ == "__main__":
    n_args = len(sys.argv)

    if n_args < 2:
        print("You need to call gains.py with at least one argument. Accepted options: 'learn', 'exploit', 'experiment'")
        exit()

    mode = sys.argv[1]    

    if mode == "learn":
        print("[LEARNING]")
        learn()
    elif mode == "exploit":
        print("[EXPLOITING]")
        exploit()
    elif mode == "experiment":
        if n_args < 4:
            print("Not enough arguments for exepriment mode. Please call:")
            print("python gains.py experiment [name] [num_iterations]")

        name = sys.argv[2]
        nRuns = int(sys.argv[3])
        print(f"[EXPERIMENT] running {name} experiment {nRuns} time(s)")
        experiment(nRuns, name)

    else:
        print(f"'{mode}' is not correct mode of execution. Accapted values: 'learn', 'exploit', 'experiment'")    

