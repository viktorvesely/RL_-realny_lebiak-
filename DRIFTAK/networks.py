import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, inputs):
        super(Actor, self).__init__()

        self._fc1 = torch.nn.Linear(inputs, 400)
        self._bn1 = torch.nn.BatchNorm1d(400)
        self._relu1 = torch.nn.ReLU(inplace=True)

        self._fc2 = torch.nn.Linear(400, 300)
        self._bn2 = torch.nn.BatchNorm1d(300)
        self._relu2 = torch.nn.ReLU(inplace=True)

        self._fc3 = torch.nn.Linear(300, 1)
        self._tanh3 = torch.nn.Tanh()

        self._fc3.weight.data.uniform_(-0.003, 0.003)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x