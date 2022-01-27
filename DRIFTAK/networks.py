import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_shape, action_space):
        super(Actor, self).__init__()

        self.action_space = action_space
        self.state_shape = state_shape
        num_actions = self.num_actions()

        self.network_size = [128, 64, 32]

        self.layer1 = nn.Linear(state_shape[0], self.network_size[0])
        self.layer2 = nn.Linear(self.network_size[0], self.network_size[1]) 
        self.layer3 = nn.Linear(self.network_size[1], self.network_size[2])
        self.layer4 = nn.Linear(self.network_size[2], num_actions)

    
    def num_actions(self):
        return self.action_space.T.shape[0]

    def forward(self, states):

        x = states

        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        y = torch.tanh(self.layer4(x))

        return y

class Critic(nn.Module):
    def __init__(self, state_shape, action_space):
        super(Critic, self).__init__()

        self.state_shape = state_shape
        self.action_space = action_space
        num_actions = self.num_actions()

        self.network_size = [128, 64, 128, 64]

        self.layer1 = nn.Linear(state_shape[0], self.network_size[0])
        self.layer2 = nn.Linear(self.network_size[0], self.network_size[1]) 

        self.layer3 = nn.Linear(self.network_size[1] + num_actions, self.network_size[2])
        self.layer4 = nn.Linear(self.network_size[2], self.network_size[3])
        self.layer5 = nn.Linear(self.network_size[3], 1)

    
    def num_actions(self):
        return self.action_space.T.shape[0]

    def forward(self, states, actions):

        x = states

        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))

        x = torch.cat((x, actions), 1)
        x = F.leaky_relu(self.layer3(x))
        x = F.leaky_relu(self.layer4(x))
        y = self.layer5(x)

        return y