# Simple softmax feedforward policy
# Author: Conor Igoe
# Date: July 8 2021

import torch
import numpy as np
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Policy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax()
        )

        self.num_inputs = num_inputs
        self.num_actions = num_actions

    def forward(self, x):
        return self.layers(torch.from_numpy(np.asarray(x)).float())

    def act(self, state):
        actions = list(range(self.num_actions))
        probabilities = self.forward(state).detach().numpy()
        return np.random.choice(actions, p=probabilities)

class Baseline(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Baseline, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.num_inputs = num_inputs

    def forward(self, x):
        return self.layers(torch.from_numpy(np.asarray(x)).float())
