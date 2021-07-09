# Simple buffer that also calculates & stores returns
# Author: Conor Igoe
# Date: July 8 2021

import torch
import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 
                            'next_state', 'done','rturn'))

class Buffer(object):

    def __init__(self, capacity, gamma=1):
        self.memory = deque([],maxlen=capacity)
        self.gamma = gamma
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def add_path(self, path):
        rturn = 0
        for state, action, reward, next_state, done in path[::-1]:
            rturn = reward + self.gamma * rturn
            self.push(state, action, reward, next_state, done, rturn)

    def sample(self, batch_size=1):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def reset(self):
        self.memory = deque([],maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)