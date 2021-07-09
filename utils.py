# Helper functions
# Author: Conor Igoe
# Date: July 8 2021

import random
import numpy as np
import torch
import collections

def set_random_seed(seed, env):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	env.seed(seed)

def evaluate(pi, env, num_evaluation_rollouts):
    rturns = 0
    with torch.no_grad():
        for i in range(num_evaluation_rollouts):
            state, done = env.reset(), False
            while not done:
                action = pi.act(state)
                next_state, reward, done, info = env.step(action)
                rturns += reward
                state = next_state

    return rturns / num_evaluation_rollouts	

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)    