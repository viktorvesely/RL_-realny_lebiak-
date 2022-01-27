import gym
import win32api

env = gym.make('CarRacing-v0')
env.reset()

run = True

while run:
    env.render()
    action = [0, 0, 0]


    if win32api.GetAsyncKeyState(ord('Q')):
        run = False

    if win32api.GetAsyncKeyState(ord('W')):
        action[1] = 0.8

    if win32api.GetAsyncKeyState(ord('D')):
        action[0] = 1

    if win32api.GetAsyncKeyState(ord('A')):
        action[0] = -1

    if win32api.GetAsyncKeyState(ord('S')):
        action[2] = 1
    

    observation, reward, done, info = env.step(action)
env.close()