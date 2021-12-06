import gym
from agent import AgentTypes as BT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

method = BT.SOFT_MAX_Q_VALUES
envName = "Bernoulli"
k = 10
N = 500
T = 500

name = method.name.lower()
env = envName[0]
ranks, trace, reward = gym.gains(method, env, k=k, N=N, T=T)


def trace_eval(trace, save=False, path="", title="Accuracy over time"):
    ts = np.arange(1, trace.size + 1)
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('t')
    ax1.set_ylabel('Accuracy')
    ax1.plot(ts, trace, color=color)
    ax1.title.set_text(title)
    ax1.set_ylim([0, 1])

    if save:
        plt.savefig(path)
        plt.close(fig)
    else:
        plt.show()

def ranks_eval(ranks, save=False, path="", title="Density over ranks"):
    ranks, counts = np.unique(ranks, return_counts=True)
    counts = counts / np.sum(counts)
    ranks = ranks + 1

    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Rank of action')
    ax1.set_ylabel('Portion of agents')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.bar(ranks, counts, color=color)
    ax1.title.set_text(title)
    ax1.set_ylim([0, 1])

    if save:
        plt.savefig(path)
        plt.close(fig)
    else:
        plt.show()


trace_eval(
        trace, 
        save=False, 
        path=f"./graphs2/{name}_{env}_ranks.png",
        title=f"Portion of agents over action rank \nfor {name} in {envName} environment"  
    )

ranks_eval(ranks)
#trace_eval(trace)
