import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

class Actor(nn.Module):
    def __init__(self, state_shape, action_space):
        super(Actor, self).__init__()

        self.action_space = action_space
        self.state_shape = state_shape
        num_actions = self.num_actions()

        self.network_size = [64, 32, 16]

        self.layer1 = nn.Linear(state_shape[0], self.network_size[0])
        self.layer2 = nn.Linear(self.network_size[0], self.network_size[1])
        self.layer3 = nn.Linear(self.network_size[1], self.network_size[2])
        self.layer4 = nn.Linear(self.network_size[2], num_actions)

        nn.init.uniform_(self.layer4.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.layer4.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    
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

        self.network_size = [64, 32, 64, 32]

        self.layer1 = nn.Linear(state_shape[0], self.network_size[0])
        self.layer2 = nn.Linear(self.network_size[0], self.network_size[1]) 

        self.layer3 = nn.Linear(self.network_size[1] + num_actions, self.network_size[2])
        self.layer4 = nn.Linear(self.network_size[2], self.network_size[3])
        self.layer5 = nn.Linear(self.network_size[3], 1)

        nn.init.uniform_(self.layer5.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.layer5.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    
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