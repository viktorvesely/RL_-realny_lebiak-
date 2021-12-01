import gym
from agent import AgentTypes as BT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

method = BT.SOFT_MAX
env = "G"
ranks, trace, reward = gym.gains(method, env, k=10, N=100, T=300)


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


ranks_eval(ranks)
#trace_eval(trace)
